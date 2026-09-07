"""AI features: fitness-coach chat + screenshot-to-session extraction.

Route handlers only; the LLM plumbing lives in :mod:`app.llm`.
"""

from __future__ import annotations

import base64
import difflib
import json
import logging
import re
from datetime import date, datetime, timedelta
from typing import Any

from fastapi import APIRouter, Depends, File, HTTPException, Request, UploadFile
from fastapi.responses import HTMLResponse, RedirectResponse, StreamingResponse
from sqlalchemy import delete as sa_delete
from sqlalchemy.orm import Session

from . import llm as llm_mod
from . import models
from .auth import get_current_user, get_db
from .web import (
    BODYWEIGHT_DEFAULT_KG,
    _form_str,
    CARDIO_ACTIVITY_TYPES,
    _cardio_load_factor,
    _iso_week_key,
    _parse_duration_min,
    render_page,
)

logger = logging.getLogger("trainlocks.ai")

router = APIRouter()

# How many persisted messages are replayed to the LLM each turn.
COACH_HISTORY_LIMIT = 40

COACH_SYSTEM_PROMPT = """You are an experienced strength & conditioning coach \
chatting with an athlete about their training. You are given real data from \
their training log (sessions, weekly load, bodyweight, top lifts). Ground \
your advice in that data when it is relevant; be concise, practical and \
encouraging. Use metric units (kg, km). If asked to plan a session or \
program, give concrete sets/reps/weights matched to their current level."""


# ---------------------------------------------------------------------------
# Coach chat data helpers
# ---------------------------------------------------------------------------

def _append_coach_message(db: Session, role: str, content: str) -> None:
    db.add(models.CoachChatMessage(role=role, content=content))
    db.commit()


def _coach_history(db: Session, limit: int = COACH_HISTORY_LIMIT) -> list[dict]:
    """The last ``limit`` messages in chronological order."""
    rows = (
        db.query(models.CoachChatMessage)
        .order_by(models.CoachChatMessage.created_at.desc(), models.CoachChatMessage.id.desc())
        .limit(limit)
        .all()
    )
    return [{"role": r.role, "content": r.content} for r in reversed(rows)]


def _fmt_duration_min(minutes: float) -> str:
    """Format decimal minutes as M:SS (e.g. 43.57 -> '43:34')."""
    total = round(minutes * 60)
    h, rem = divmod(total, 3600)
    m, s = divmod(rem, 60)
    if h:
        return f"{h}:{m:02d}:{s:02d}"
    return f"{m}:{s:02d}"


def _training_context(user: models.User, db: Session) -> str:
    """Deterministic training-data context for the coach system prompt."""
    now = datetime.now()
    lines: list[str] = []
    bodyweight = user.bodyweight or BODYWEIGHT_DEFAULT_KG

    recent = (
        db.query(models.WorkoutSession)
        .order_by(models.WorkoutSession.date.desc(), models.WorkoutSession.id.desc())
        .limit(10)
        .all()
    )
    if recent:
        lines.append("Recent sessions (oldest first):")
        for s in reversed(recent):
            parts = []
            for se in sorted(s.sets, key=lambda x: (x.exercise_id, x.set_number)):
                ex = se.exercise.name if se.exercise else "?"
                if se.weight is not None or not (se.exercise and se.exercise.is_bodyweight):
                    parts.append(f"{ex}: {se.weight if se.weight is not None else 0}kg x {se.reps}")
                else:
                    parts.append(f"{ex}: BW x {se.reps}")
            for c in s.cardio:
                bits = [c.activity_type]
                if c.distance_km:
                    bits.append(f"{c.distance_km}km")
                if c.duration_min:
                    bits.append(f"{_fmt_duration_min(c.duration_min)}min")
                parts.append(" ".join(bits))
            line_txt = "; ".join(parts) if parts else "(no sets recorded)"
            notes = f" — {s.notes}" if s.notes else ""
            session_date = s.date or date.today()
            lines.append(f"- {session_date.isoformat()}: {line_txt}{notes}")
    else:
        lines.append("No sessions logged yet.")

    week_start = now.date() - timedelta(days=now.date().weekday()) - timedelta(weeks=11)
    sessions_window = (
        db.query(models.WorkoutSession)
        .filter(models.WorkoutSession.date >= week_start)
        .all()
    )
    weekly_load: dict[str, float] = {}
    for s in sessions_window:
        key = _iso_week_key(s.date or date.today())
        for se in s.sets:
            is_bw = bool(se.exercise and se.exercise.is_bodyweight)
            weight = (bodyweight + (se.weight or 0.0)) if is_bw else (se.weight or 0.0)
            weekly_load[key] = weekly_load.get(key, 0.0) + weight * (se.reps or 0)
        for c in s.cardio:
            if c.distance_km:
                weekly_load[key] = weekly_load.get(
                    key, 0.0) + c.distance_km * _cardio_load_factor(c.activity_type)
    if weekly_load:
        lines.append("Weekly training load (kg volume, last 12 weeks):")
        lines.extend(f"- {k}: {round(v)}" for k, v in sorted(weekly_load.items()))

    if user.bodyweight:
        lines.append(f"Current bodyweight: {bodyweight}kg (self-reported)")
    else:
        lines.append(f"Bodyweight: unknown (assuming {BODYWEIGHT_DEFAULT_KG}kg for load math)")
    lines.append(f"Today's date: {now.date().isoformat()}")
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Coach chat routes
# ---------------------------------------------------------------------------

@router.get("/coach", response_class=HTMLResponse)
async def coach_page(
    request: Request,
    user: models.User = Depends(get_current_user),
    db: Session = Depends(get_db),
):
    history = _coach_history(db)
    return render_page(request, "coach.html", {"history": history, "user": user})


@router.post("/coach/send/stream")
async def coach_send_stream(
    request: Request,
    user: models.User = Depends(get_current_user),
    db: Session = Depends(get_db),
):
    """SSE stream of the coach's reply to a user message.

    Persists the user message immediately and the streamed assistant reply
    after the stream completes, so a refresh mid-stream never leaves an
    orphan question without an answer in the history.
    """
    body_raw = await request.body()
    try:
        body = json.loads(body_raw) if body_raw else {}
        if not isinstance(body, dict):
            raise ValueError
    except ValueError:
        raise HTTPException(status_code=400, detail="invalid JSON body")
    user_text = str(body.get("message") or "").strip()
    if not user_text:
        raise HTTPException(status_code=400, detail="message required")

    _append_coach_message(db, "user", user_text)

    history = _coach_history(db)
    system = f"{COACH_SYSTEM_PROMPT}\n\n# Athlete data\n{_training_context(user, db)}"
    messages: list[dict] = [{"role": "system", "content": system}] + history

    async def event_gen():
        final_text = ""
        last_evt: dict = {}
        try:
            async for evt in llm_mod.chat_stream(messages):
                last_evt = evt
                if evt.get("type") in ("delta", "done"):
                    final_text = evt.get("text") or final_text
                # Proper SSE framing: 'data:' line + blank line terminator.
                yield f"data: {json.dumps(evt)}\n\n"
        finally:
            # Runs on normal completion AND on client disconnect (Stop
            # button / refresh), which raises GeneratorExit at the yield.
            # Bookkeeping must live here or a disconnect leaves the user
            # message persisted with no assistant reply (the orphan the
            # docstring promises never to produce).
            if last_evt.get("type") == "done" and final_text.strip():
                _append_coach_message(db, "assistant", final_text)
            elif last_evt.get("type") == "error":
                # The turn failed: drop the just-persisted user message so the
                # history doesn't keep an unanswered question.
                last_user = (
                    db.query(models.CoachChatMessage)
                    .filter(models.CoachChatMessage.role == "user")
                    .order_by(models.CoachChatMessage.id.desc())
                    .first()
                )
                if last_user:
                    db.delete(last_user)
                    db.commit()

    return StreamingResponse(event_gen(), media_type="text/event-stream")


@router.get("/coach/history")
async def coach_history(
    user: models.User = Depends(get_current_user),
    db: Session = Depends(get_db),
):
    return {"messages": _coach_history(db)}


@router.delete("/coach/history")
async def coach_history_clear(
    user: models.User = Depends(get_current_user),
    db: Session = Depends(get_db),
):
    db.execute(sa_delete(models.CoachChatMessage))
    db.commit()
    return {"cleared": True}


# ---------------------------------------------------------------------------
# LLM model selection
# ---------------------------------------------------------------------------

@router.get("/api/llm/status")
async def llm_status(user: models.User = Depends(get_current_user)):
    backends = await llm_mod.list_backends()
    active = await llm_mod.current_backend()
    return {
        "backends": backends,
        "active_backend": active.get("name", ""),
        "active_model": active.get("model", ""),
    }


@router.post("/api/llm/select")
async def llm_select(request: Request, user: models.User = Depends(get_current_user)):
    body_raw = await request.body()
    try:
        body = json.loads(body_raw) if body_raw else {}
        if not isinstance(body, dict):
            raise ValueError
    except ValueError:
        raise HTTPException(status_code=400, detail="invalid JSON body")
    backend = str(body.get("backend") or "")
    model = body.get("model")
    model = str(model) if model is not None else None
    try:
        active = await llm_mod.select_backend(backend, model)
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    return {"active_backend": active.get("name", ""), "active_model": active.get("model", "")}


# ---------------------------------------------------------------------------
# AI session extraction (screenshot -> workout session)
# ---------------------------------------------------------------------------

def _extraction_system_prompt() -> str:
    """Extraction prompt; includes today's date so partial dates in the
    screenshot ("Wed 2. Sep") can be resolved to a full ISO date."""
    today = date.today()
    weekday = today.strftime("%A")
    return f"""You read workout screenshots (gym app logs, notes app workouts, \
whiteboard photos, watch/fitness-app summaries) and extract the training \
session as strict JSON. Respond with JSON only — no prose, no markdown \
fences. Schema:

{{"date": "YYYY-MM-DD or null",
 "exercises": [{{"name": "exercise name",
                "sets": [{{"reps": <int>, "weight_kg": <number or null>}}]}}],
 "cardio": [{{"activity_type": "running|swimming|cycling|walking|rowing|other",
             "distance_km": <number or null>, "duration_min": <number or null>,
             "notes": "string or null"}}],
 "notes": "session notes or null"}}

Rules:
- Today is {today.isoformat()} ({weekday}). If the screenshot shows a date or \
partial date (e.g. "Wed 2. Sep", "Sep 2", "yesterday"), resolve it to the \
most recent matching date in the past and output full YYYY-MM-DD. Only use \
null when no date information at all is visible.
- reps are integers; weight_kg is in kilograms (convert lb: /2.2046, round to 0.5).
- For bodyweight exercises set weight_kg to null.
- Cardio: duration_min is the workout time in minutes (convert H:MM:SS or \
MM:SS, e.g. 0:43:34 -> 43.57). Put extra metrics (pace, heart rate, \
elevation, calories, cadence, power, location, start time) into the cardio \
notes so they are preserved.
- Drop empty/zero rows; keep the exercise order from the screenshot."""


def _parse_json_loose(text: str) -> Any:
    """Best-effort JSON extraction from an LLM reply (handles fences/prose).

    Returns whatever the JSON parses to — the caller must validate the shape.
    """
    text = text.strip()
    fence = re.search(r"```(?:json)?\s*(\{.*\})\s*```", text, re.DOTALL)
    if fence:
        text = fence.group(1)
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        brace = re.search(r"\{.*\}", text, re.DOTALL)
        if brace:
            return json.loads(brace.group(0))
        raise


def _coerce_float(v: Any) -> float | None:
    """float(v) tolerating None/""/junk (LLM JSON values are untyped)."""
    try:
        return float(v) if v not in (None, "") else None
    except (TypeError, ValueError):
        return None


def _norm_name(name: str) -> str:
    return re.sub(r"[^a-z0-9]+", " ", name.lower()).strip()


_BODYWEIGHT_HINTS = ("pull up", "pullup", "pull-up", "push up", "pushup", "push-up",
                     "dip", "muscle up", "pistol squat", "planche", "human flag",
                     "handstand push", "chin up", "chinup", "chin-up")


def _match_or_create_exercises(
    db: Session, wanted: list[str]
) -> tuple[dict[str, models.Exercise], list[str]]:
    """Fuzzy-match wanted exercise names to DB rows; create missing ones.

    Returns (mapping wanted-name -> Exercise, list of newly created names).
    """
    existing = db.query(models.Exercise).all()
    by_norm: dict[str, models.Exercise] = {_norm_name(e.name): e for e in existing}
    by_lower = {e.name.lower(): e for e in existing}

    mapping: dict[str, models.Exercise] = {}
    created: list[str] = []
    for w in wanted:
        norm = _norm_name(w)
        if not norm:
            continue
        if norm in by_norm:
            mapping[w] = by_norm[norm]
            continue
        exact = by_lower.get(w.lower())
        if exact:
            mapping[w] = exact
            continue
        close = difflib.get_close_matches(norm, list(by_norm), n=1, cutoff=0.85)
        if close:
            mapping[w] = by_norm[close[0]]
            continue
        bw = any(t in norm for t in _BODYWEIGHT_HINTS)
        ex = models.Exercise(name=w.strip(), is_bodyweight=bw)
        db.add(ex)
        by_norm[norm] = ex
        by_lower[w.lower()] = ex
        created.append(w.strip())
        mapping[w] = ex
    return mapping, created


@router.get("/sessions/ai", response_class=HTMLResponse)
async def ai_session_page(
    request: Request,
    user: models.User = Depends(get_current_user),
):
    return render_page(request, "ai_session.html", {"user": user})


@router.post("/sessions/ai/extract", response_class=HTMLResponse)
async def ai_session_extract(
    request: Request,
    screenshot: UploadFile = File(...),
    user: models.User = Depends(get_current_user),
    db: Session = Depends(get_db),
):
    """Run vision extraction on an uploaded screenshot and render a review form.

    The review form posts to the regular ``POST /sessions/new`` flow, so
    confirming/editing reuses the existing session-creation code path.
    """
    # Size guard BEFORE reading the body into memory: a multi-GB upload
    # would otherwise be fully buffered just to be rejected.
    max_bytes = 10 * 1024 * 1024
    size = getattr(screenshot, "size", None)
    if size is not None and size > max_bytes:
        raise HTTPException(status_code=400, detail="file too large (max 10 MB)")
    raw = await screenshot.read()
    if not raw:
        raise HTTPException(status_code=400, detail="empty file")
    if len(raw) > max_bytes:
        raise HTTPException(status_code=400, detail="file too large (max 10 MB)")
    b64 = base64.b64encode(raw).decode("ascii")

    if not llm_mod.llm_enabled():
        # Feature is switched off — say so instead of letting the disabled
        # message fail JSON parsing further down (502 "try another model").
        return render_page(request, "ai_session.html", {
            "user": user,
            "extract_error": llm_mod.LLM_DISABLED_MSG,
        })

    if not await llm_mod.current_backend():
        # Switch is on but nothing is configured: chat() would return prose
        # that then fails JSON parsing with a misleading "try another model".
        return render_page(request, "ai_session.html", {
            "user": user,
            "extract_error": "No LLM backend configured — add one via LLM_BACKENDS.",
        })

    try:
        reply = await llm_mod.chat([
            {"role": "system", "content": _extraction_system_prompt()},
            {"role": "user",
             "content": "Extract the workout session from this screenshot as JSON.",
             "images": [b64]},
        ])
    except llm_mod.LLMBackendError as e:
        # Most common cause: the active model doesn't accept images (HTTP 400
        # from the backend). Render the upload page with the error instead of
        # an unhandled 500.
        msg = str(e)
        hint = ("This model may not support images. Pick a vision-capable "
                "model (e.g. glm-5.3-flash:cloud, gemma4, gpt-4o) from the "
                "dropdown and try again.")
        logger.warning("Screenshot extraction failed: %s", msg)
        # NOTE: 200 on purpose — a 502 would be intercepted by Cloudflare and
        # replaced with its own "Bad Gateway" page, hiding the helpful error.
        return render_page(request, "ai_session.html", {
            "user": user,
            "extract_error": f"{msg} — {hint}",
        })
    text = (reply.get("text") or "").strip()
    if not text:
        # NOTE: 200 + rendered error, not 502 — a 5xx would be intercepted by
        # Cloudflare and replaced with its own "Bad Gateway" page, hiding the
        # helpful error (same rationale as the LLMBackendError branch above).
        return render_page(request, "ai_session.html", {
            "user": user,
            "extract_error": "The AI returned an empty response — try again or pick another model.",
        })
    try:
        data = _parse_json_loose(text)
    except (json.JSONDecodeError, ValueError, RecursionError):
        return render_page(request, "ai_session.html", {
            "user": user,
            "extract_error": ("Could not parse the AI response as JSON — "
                              "try another model."),
        })
    # The reply comes from an LLM — the shape is untrusted. Anything but a
    # JSON object fails here rather than crashing the route with a 500.
    if not isinstance(data, dict):
        return render_page(request, "ai_session.html", {
            "user": user,
            "extract_error": ("The AI response was not a JSON object — "
                              "try another model."),
        })

    wanted: list[str] = []
    for e in data.get("exercises") or []:
        if not isinstance(e, dict):
            continue
        name = str(e.get("name") or "").strip()
        if name and name not in wanted:
            wanted.append(name)
    mapping, created = _match_or_create_exercises(db, wanted)
    db.commit()

    # Build the review form's structure: exercises resolved to DB ids with
    # their sets, so the form posts reps-<ex_id>-<set> fields that
    # POST /sessions/new already understands.
    review_exercises = []
    for name in wanted:
        ex = mapping.get(name)
        if ex is None:
            # e.g. a name that normalizes to nothing ("!!!") — matching
            # skipped it; skip it here too instead of raising KeyError.
            continue
        raw_sets = next(
            (e.get("sets") or [] for e in data.get("exercises") or []
             if isinstance(e, dict) and str(e.get("name") or "").strip() == name),
            [],
        )
        sets = []
        for s in raw_sets:
            if not isinstance(s, dict):
                continue
            try:
                reps = int(s.get("reps") or 0)
            except (TypeError, ValueError):
                reps = 0
            try:
                weight = _coerce_float(s.get("weight_kg"))
            except (TypeError, ValueError):
                weight = None
            if reps == 0 and weight is None:
                continue
            sets.append({"reps": reps, "weight": weight})
        review_exercises.append({
            "exercise": ex,
            "is_new": name.strip() in created,
            "sets": sets,
        })

    date_val = data.get("date")
    try:
        parsed_date = date.fromisoformat(str(date_val)) if date_val else date.today()
    except ValueError:
        parsed_date = date.today()

    # Normalize cardio entries for the review form (validate numerics so the
    # template can post them straight to the cardio flow on save).
    review_cardio = []
    for c in data.get("cardio") or []:
        if not isinstance(c, dict):
            continue
        try:
            dist = _coerce_float(c.get("distance_km"))
        except (TypeError, ValueError):
            dist = None
        try:
            dur = _coerce_float(c.get("duration_min"))
        except (TypeError, ValueError):
            dur = None
        if dist is None and dur is None:
            continue
        atype = str(c.get("activity_type") or "other").strip().lower()
        if atype not in CARDIO_ACTIVITY_TYPES:
            atype = "other"
        review_cardio.append({
            "activity_type": atype,
            "distance_km": dist,
            "duration_min": dur,
            "notes": str(c.get("notes") or ""),
        })

    return render_page(request, "ai_session_review.html", {
        "user": user,
        "review_exercises": review_exercises,
        "created_names": created,
        "ai_date": parsed_date,
        "ai_notes": data.get("notes") or "",
        "ai_cardio": review_cardio,
        "model_label": await llm_mod.current_model_label(),
    })


@router.post("/sessions/ai/save")
async def ai_session_save(
    request: Request,
    user: models.User = Depends(get_current_user),
    db: Session = Depends(get_db),
):
    """Save the reviewed AI session: creates the WorkoutSession with sets and
    any checked cardio activities (in one transaction), then redirects to the
    session list."""
    form = await request.form()
    date_str = _form_str(form.get("date"))
    if not date_str:
        raise HTTPException(status_code=400, detail="Date required")
    try:
        workout_date = date.fromisoformat(str(date_str))
    except ValueError:
        raise HTTPException(status_code=400, detail="Invalid date")

    workout = models.WorkoutSession(
        date=workout_date,
        notes=_form_str(form.get("notes")) or None,
    )
    db.add(workout)
    db.flush()

    # Sets — same reps-<ex_id>-<set> fields POST /sessions/new accepts.
    for key, value in form.items():
        if not key.startswith("reps-"):
            continue
        try:
            _, ex_id_str, set_num_str = key.split("-")
            ex_id = int(ex_id_str)
            set_num = int(set_num_str)
        except (ValueError, IndexError):
            continue
        weight_val = _form_str(form.get(f"weight-{ex_id}-{set_num}"))
        try:
            reps = int(_form_str(value)) if _form_str(value) else 0
        except ValueError:
            reps = 0
        try:
            weight = float(weight_val) if weight_val else None
        except ValueError:
            weight = None
        if reps == 0 and weight is None:
            continue
        db.add(models.SetEntry(
            session_id=workout.id,
            exercise_id=ex_id,
            set_number=set_num,
            reps=reps,
            weight=weight,
        ))

    # Cardio entries checked in the review form. The loop keys off the
    # always-submitted -type select (not the checkbox): unchecked checkboxes
    # are not POSTed, so an -include loop would stop at the first unchecked
    # entry and silently drop later checked ones.
    cardio_saved = 0
    idx = 0
    while f"cardio-{idx}-type" in form:
        include = _form_str(form.get(f"cardio-{idx}-include")) == "1"
        if include:
            activity_type = (_form_str(form.get(f"cardio-{idx}-type")) or "other").strip().lower()
            if activity_type not in CARDIO_ACTIVITY_TYPES:
                activity_type = "other"

            distance_km = _coerce_float(form.get(f"cardio-{idx}-distance"))
            try:
                duration_min = _parse_duration_min(_form_str(form.get(f"cardio-{idx}-duration")))
            except ValueError:
                duration_min = None
            if distance_km is not None and distance_km < 0:
                distance_km = None
            if duration_min is not None and duration_min < 0:
                duration_min = None
            if distance_km is not None or duration_min is not None:
                db.add(models.CardioActivity(
                    session_id=workout.id,
                    activity_type=activity_type,
                    distance_km=distance_km,
                    duration_min=duration_min,
                    notes=_form_str(form.get(f"cardio-{idx}-notes")) or None,
                ))
                cardio_saved += 1
        idx += 1

    db.commit()
    # Preserve AI provenance in the flash-less flow: redirect with a flag.
    return RedirectResponse(
        url=f"/sessions/{workout.id}?ai_saved=1&cardio={cardio_saved}",
        status_code=303,
    )