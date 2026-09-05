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

from fastapi import APIRouter, Depends, File, HTTPException, Request, UploadFile
from fastapi.responses import HTMLResponse, StreamingResponse
from sqlalchemy import delete as sa_delete
from sqlalchemy.orm import Session

from . import llm as llm_mod
from . import models
from .auth import get_current_user, get_db
from .web import BODYWEIGHT_DEFAULT_KG, _cardio_load_factor, _iso_week_key, render_page

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
                    bits.append(f"{round(c.duration_min)}min")
                parts.append(" ".join(bits))
            line_txt = "; ".join(parts) if parts else "(no sets recorded)"
            notes = f" — {s.notes}" if s.notes else ""
            lines.append(f"- {s.date.isoformat()}: {line_txt}{notes}")
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
        key = _iso_week_key(s.date)
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
    body = await request.json()
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
            pass

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
    body = await request.json()
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

EXTRACTION_SYSTEM_PROMPT = """You read workout screenshots (gym app logs, \
notes app workouts, whiteboard photos) and extract the training session as \
strict JSON. Respond with JSON only — no prose, no markdown fences. Schema:

{"date": "YYYY-MM-DD or null",
 "exercises": [{"name": "exercise name",
                "sets": [{"reps": <int>, "weight_kg": <number or null>}]}],
 "cardio": [{"activity_type": "running|swimming|cycling|walking|rowing|other",
             "distance_km": <number or null>, "duration_min": <number or null>,
             "notes": "string or null"}],
 "notes": "session notes or null"}

Rules:
- reps are integers; weight_kg is in kilograms (convert lb: /2.2046, round to 0.5).
- For bodyweight exercises set weight_kg to null.
- Drop empty/zero rows; keep the exercise order from the screenshot.
- If the date is visible use it, else null."""


def _parse_json_loose(text: str) -> dict:
    """Best-effort JSON extraction from an LLM reply (handles fences/prose)."""
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
    raw = await screenshot.read()
    if not raw:
        raise HTTPException(status_code=400, detail="empty file")
    if len(raw) > 10 * 1024 * 1024:
        raise HTTPException(status_code=400, detail="file too large (max 10 MB)")
    b64 = base64.b64encode(raw).decode("ascii")

    try:
        reply = await llm_mod.chat([
            {"role": "system", "content": EXTRACTION_SYSTEM_PROMPT},
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
        raise HTTPException(status_code=502, detail="LLM returned an empty response")
    try:
        data = _parse_json_loose(text)
    except (json.JSONDecodeError, ValueError):
        raise HTTPException(
            status_code=502,
            detail="Could not parse the AI response as JSON. Try another model.",
        )

    wanted: list[str] = []
    for e in data.get("exercises") or []:
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
        ex = mapping[name]
        raw_sets = next(
            (e.get("sets") or [] for e in data.get("exercises") or []
             if str(e.get("name") or "").strip() == name),
            [],
        )
        sets = []
        for s in raw_sets:
            try:
                reps = int(s.get("reps") or 0)
            except (TypeError, ValueError):
                reps = 0
            try:
                weight = float(s.get("weight_kg")) if s.get("weight_kg") is not None else None
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

    return render_page(request, "ai_session_review.html", {
        "user": user,
        "review_exercises": review_exercises,
        "created_names": created,
        "ai_date": parsed_date,
        "ai_notes": data.get("notes") or "",
        "ai_cardio": data.get("cardio") or [],
        "model_label": await llm_mod.current_model_label(),
    })