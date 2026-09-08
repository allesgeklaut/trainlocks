import os
import time
import argparse
import tempfile
import shutil
from datetime import date
from typing import Generator, Optional
import httpx
import json
import math
from urllib.parse import urlsplit
from dotenv import load_dotenv
from fastapi import Cookie, Depends, FastAPI, Form, HTTPException, Query, Request
from fastapi.responses import HTMLResponse, RedirectResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from sqlalchemy.orm import Session
from sqlalchemy import text as sa_text, func, Boolean

# 1. Load environment variables first so they can be used for DB setup or by other modules
load_dotenv()

# 2. Parse configuration (Command line arguments and Environment Variables)
parser = argparse.ArgumentParser(description='Training Log Dashboard')
parser.add_argument('--test-db', action='store_true', help='Use test database with sample data')
args, _ = parser.parse_known_args()

# Support both command line flag and environment variable for flexibility (e.g. in Docker)
use_test_db = args.test_db or os.getenv('USE_TEST_DB', '').lower() in ('true', '1', 'yes')

if use_test_db:
    # Create a temporary database file
    temp_db_fd, temp_db_path = tempfile.mkstemp(suffix='.db', prefix='training_log_test_')
    os.close(temp_db_fd)  # Close the file descriptor
    
    # Copy test database from tests/ folder to the temporary location
    test_template_path = os.path.join(os.path.dirname(__file__), '..', 'tests', 'test_database.db')
    if os.path.exists(test_template_path):
        shutil.copy2(test_template_path, temp_db_path)
    else:
        print(f"Warning: Test database template not found at {test_template_path}")

    # Update DATABASE_URL environment variable BEFORE importing app modules that use it
    os.environ['DATABASE_URL'] = f'sqlite:///{temp_db_path}'
    print(f"Using test database: {temp_db_path}")

# 3. Now import local application modules (which will now pick up the correct DATABASE_URL)
from . import models
from .ai import router as ai_router
from .auth import get_current_user, create_session_cookie, verify_password, COOKIE_NAME, SESSION_MAX_AGE
from .database import Base, SessionLocal, engine
from .web import (
    BODYWEIGHT_DEFAULT_KG,
    _form_str,
    CARDIO_ACTIVITY_TYPES,
    _cardio_json,
    _cardio_load_factor,
    _cardio_pace,
    _cardio_pace_unit,
    _iso_week_key,
    _parse_duration_min,
    render_page,
    templates,
)
from .load import (
    BW_LOAD_FACTORS,
    default_bw_load_factor,
    effective_load_kg,
    exercise_load_factor,
    is_bodyweight_name,
)

Base.metadata.create_all(bind=engine)

# ── Lightweight migration for existing databases ──────────────────────────
# The app has no Alembic; create_all only creates missing tables but won't
# add tables/columns to an existing DB file. Mimic it with an idempotent
# CREATE TABLE IF NOT EXISTS for the cardio table so pre-existing volumes
# (./training_log_data/training_log.db) pick it up on next start.
with engine.begin() as conn:
    conn.execute(sa_text(
        """
        CREATE TABLE IF NOT EXISTS cardio_activities (
            id INTEGER NOT NULL PRIMARY KEY,
            session_id INTEGER,
            activity_type VARCHAR NOT NULL,
            distance_km FLOAT,
            duration_min FLOAT,
            notes VARCHAR,
            CONSTRAINT fk_cardio_activities_session_id
                FOREIGN KEY(session_id) REFERENCES workout_sessions (id)
        )
        """
    ))
    # Add the index if the table was pre-existing (create_all won't have run).
    conn.execute(sa_text(
        "CREATE INDEX IF NOT EXISTS ix_cardio_activities_session_id "
        "ON cardio_activities (session_id)"
    ))
    # users.bodyweight powers bodyweight-exercise load scaling; the column
    # predates create_all on existing DBs, so add it if missing.
    try:
        conn.execute(sa_text("ALTER TABLE users ADD COLUMN bodyweight FLOAT"))
    except Exception:
        pass
    # coach_chat_messages backs the AI fitness-coach chat history; create_all
    # DOES add missing tables to existing DBs, but an explicit idempotent
    # CREATE keeps this migration block self-contained and matches the
    # cardio-table pattern above.
    conn.execute(sa_text(
        """
        CREATE TABLE IF NOT EXISTS coach_chat_messages (
            id INTEGER NOT NULL PRIMARY KEY,
            role VARCHAR(16) NOT NULL,
            content TEXT NOT NULL,
            created_at DATETIME
        )
        """
    ))
    conn.execute(sa_text(
        "CREATE INDEX IF NOT EXISTS ix_coach_chat_messages_created_at "
        "ON coach_chat_messages (created_at)"
    ))
    # Bodyweight load scaling (see app/load.py): exercises.bw_load_factor
    # stores the %BW each exercise moves (push-up 0.65, pull-up 1.0, …);
    # set_entries.assist_kg records machine/counterweight support
    # (supported dips, assisted pull-ups). Add both if missing.
    try:
        conn.execute(sa_text("ALTER TABLE exercises ADD COLUMN bw_load_factor FLOAT"))
    except Exception:
        pass
    try:
        conn.execute(sa_text("ALTER TABLE set_entries ADD COLUMN assist_kg FLOAT"))
    except Exception:
        pass
    try:
        conn.execute(sa_text("ALTER TABLE session_template_exercises ADD COLUMN prescription VARCHAR"))
    except Exception:
        pass
    try:
        conn.execute(sa_text("ALTER TABLE session_templates ADD COLUMN description TEXT"))
    except Exception:
        pass
    # Backfill research-default load factors for known bodyweight names.
    # Pattern order matters (BW_LOAD_FACTORS is most-specific first), so
    # evaluate in Python and update row-by-row; idempotent — only rows
    # whose factor is still NULL (or now matching a different pattern via
    # rename) are touched.
    for pattern, factor in BW_LOAD_FACTORS:
        conn.execute(
            sa_text(
                "UPDATE exercises SET bw_load_factor = :f "
                "WHERE bw_load_factor IS NULL AND is_bodyweight = 1 "
                "AND lower(name) LIKE :p"
            ),
            {"f": factor, "p": f"%{pattern}%"},
        )

app = FastAPI(title="Training Log Dashboard")
app.mount("/static", StaticFiles(directory="app/static"), name="static")
app.include_router(ai_router)


# ── Security headers + CSRF Origin check ────────────────────────────────────
#
# One middleware covers both:
#
# * Every response gets Cache-Control: no-store (authenticated HTML: training
#   data, chat history — shared caches/proxies must not retain it) and
#   X-Frame-Options: DENY (a third-party page embedding this app in an
#   invisible iframe is the clickjacking vector).
#
# * State-changing requests (POST/PUT/PATCH/DELETE) from browsers must carry
#   an Origin or Referer header matching the Host. SameSite=lax already
#   stops cross-site POSTs from carrying the session cookie, so this is
#   defense-in-depth — it closes the gap if cookie settings are ever relaxed
#   or a second user appears. Non-browser API clients (MCP, curl) send no
#   Origin/Referer at all and are unaffected: the check only REJECTS when a
#   cross-origin value is present.
@app.middleware("http")
async def security_headers_and_origin_check(request: Request, call_next):
    method = request.method.upper()
    if method in ("POST", "PUT", "PATCH", "DELETE"):
        origin = request.headers.get("origin", "")
        referer = request.headers.get("referer", "")
        host = request.headers.get("host", "")
        if origin or referer:
            src_host = ""
            if origin:
                try:
                    src_host = urlsplit(origin).netloc
                except ValueError:
                    src_host = ""
            if not src_host:
                try:
                    src_host = urlsplit(referer).netloc
                except ValueError:
                    src_host = ""
            # Accept exact host match and the common proxy variants
            # (X-Forwarded-Host behind Cloudflare/nginx, localhost:port).
            fwd_host = request.headers.get("x-forwarded-host", "").split(",")[0].strip()
            allowed = {host, fwd_host}
            # Strip a :port from the request host for scheme-default matches.
            if host and ":" in host:
                allowed.add(host.rsplit(":", 1)[0])
            if src_host and src_host not in allowed:
                return JSONResponse(
                    {"detail": "cross-origin request rejected"},
                    status_code=403,
                )
    response = await call_next(request)
    response.headers.setdefault("Cache-Control", "no-store")
    response.headers.setdefault("X-Frame-Options", "DENY")
    response.headers.setdefault("Referrer-Policy", "same-origin")
    return response


# Cache-buster for static assets: bumps on every app startup so proxies and
# browsers re-fetch CSS/JS after a rebuild.
STATIC_VERSION = str(int(time.time()))
templates.env.globals["static_version"] = STATIC_VERSION


def _cardio_pace_display(c) -> str:
    """Human pace string for templates (e.g. '6:00 /km', '2:00 /100m') or '—'."""
    p = _cardio_pace(c)
    if p is None:
        return "—"
    m = int(p)
    s = int(round((p - m) * 60))
    if s == 60:
        m += 1
        s = 0
    return f"{m}:{s:02d} /{_cardio_pace_unit(c)}"


templates.env.globals["_cardio_pace_display"] = _cardio_pace_display

# _parse_duration_min lives in web.py (shared with the AI router).


def _cardio_duration_display(c) -> str:
    """Human duration for templates ('45 min', '44:51') or '—'."""
    if c.duration_min is None:
        return "—"
    m, s = divmod(round(c.duration_min * 60), 60)
    return f"{m} min" if s == 0 else f"{m}:{s:02d}"


def _cardio_duration_value(c) -> str:
    """Prefill for the duration input ('45' or '44:51')."""
    if c.duration_min is None:
        return ""
    m, s = divmod(round(c.duration_min * 60), 60)
    return str(m) if s == 0 else f"{m}:{s:02d}"


templates.env.globals["_cardio_duration_display"] = _cardio_duration_display
templates.env.globals["_cardio_duration_value"] = _cardio_duration_value

FREE_EXERCISE_DB_URL = (
    "https://raw.githubusercontent.com/yuhonas/free-exercise-db/main/dist/exercises.json"
)
_exercise_db_cache: list = []
_exercise_db_failed_at: float = 0.0
_EXERCISE_DB_NEGATIVE_TTL = 60.0

STARTER_PLANS_PATH = os.path.join(os.path.dirname(__file__), "static", "plans", "starter_plans.json")


def get_db() -> Generator[Session, None, None]:
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()


def _prune_empty_session(db: Session, session: Optional[models.WorkoutSession]) -> None:
    """Delete a workout session that has neither sets nor cardio left.

    Sessions auto-created for cardio would otherwise linger as empty rows
    after the activity is deleted or moved to another date. Sessions with a
    template (explicitly created from /sessions/new) are never pruned.
    """
    if session is None or session.id is None or session.template_id is not None:
        return
    # Flush pending cardio deletes / session_id moves so the lazy-loaded
    # collections below reflect them (autoflush is off).
    db.flush()
    db.refresh(session)
    if not session.sets and not session.cardio:
        db.delete(session)


async def fetch_exercise_db() -> list:
    global _exercise_db_cache, _exercise_db_failed_at
    if _exercise_db_cache:
        return _exercise_db_cache
    # Negative cache: if the fetch failed recently, fail fast instead of
    # blocking every page load on the full HTTP timeout while offline.
    if _exercise_db_failed_at and time.monotonic() - _exercise_db_failed_at < _EXERCISE_DB_NEGATIVE_TTL:
        raise RuntimeError("Exercise database temporarily unavailable, retry shortly")
    try:
        async with httpx.AsyncClient(timeout=15.0) as client:
            r = await client.get(FREE_EXERCISE_DB_URL)
            r.raise_for_status()
            data = r.json()
    except Exception:
        _exercise_db_failed_at = time.monotonic()
        raise
    _exercise_db_cache = data
    return data


# ── PAGES ────────────────────────────────────────────────────────────────────

# ── AUTH ─────────────────────────────────────────────────────────────────────

@app.get("/login", response_class=HTMLResponse)
async def login_page(request: Request):
    # A boosted navigation whose session has expired lands here. The login
    # page is a standalone document — swapping it into #app-shell would
    # leave the dashboard chrome around it — so tell htmx to do a full
    # browser redirect instead.
    if "hx-request" in {k.lower() for k in request.headers}:
        return HTMLResponse(status_code=200, headers={"HX-Redirect": "/login"})
    return templates.TemplateResponse(request, "login.html", {"error": None})


@app.post("/login")
async def login(
    request: Request,
    username: str = Form(...),
    password: str = Form(...),
    db: Session = Depends(get_db),
):
    user = db.query(models.User).filter(models.User.username == username).first()
    if not user or not verify_password(password, user.hashed_password):
        return templates.TemplateResponse(request, "login.html", {"error": "Invalid credentials"})
    
    response = RedirectResponse(url="/", status_code=303)
    # Behind Cloudflare Tunnel the origin connection is plain HTTP and the
    # real client scheme arrives in X-Forwarded-Proto (set by cloudflared).
    # Direct plain-HTTP clients (MCP client, localhost) send no such header.
    served_over_tls = (
        request.headers.get("x-forwarded-proto", "").lower() == "https"
        or request.url.scheme == "https"
    )
    response.set_cookie(
        key=COOKIE_NAME,
        value=create_session_cookie(user.id),
        httponly=True,
        # Only mark Secure when actually served over TLS — a Secure cookie
        # over plain HTTP is ignored by strict cookie clients (browsers make
        # an exception for localhost origins, which masks this).
        secure=served_over_tls,
        max_age=SESSION_MAX_AGE,
        samesite="lax",
    )
    return response


@app.post("/logout")
async def logout(tl_session: Optional[str] = Cookie(default=None)):
    response = RedirectResponse(url="/login", status_code=303)
    response.delete_cookie(key=COOKIE_NAME)
    return response


# ── PROFILE ────────────────────────────────────────────────────────────────────

@app.get("/profile", response_class=HTMLResponse)
async def profile_page(request: Request, user: models.User = Depends(get_current_user)):
    return render_page(request, "profile.html", {"user": user})


@app.post("/profile")
async def profile_update(
    request: Request,
    user: models.User = Depends(get_current_user),
    db: Session = Depends(get_db),
):
    form = await request.form()
    raw = _form_str(form.get("bodyweight"))
    try:
        bodyweight = float(raw) if raw is not None and str(raw).strip() else None
    except ValueError:
        raise HTTPException(status_code=400, detail="bodyweight must be a number in kg")
    # Reject 0 (not a meaningful bodyweight) and nan/inf, which float() accepts
    # and which would poison downstream load math.
    if bodyweight is not None and not (bodyweight > 0 and math.isfinite(bodyweight)):
        raise HTTPException(status_code=400, detail="bodyweight must be a positive number in kg")
    # user comes from get_current_user's own session; persist via the route's.
    db_user = db.get(models.User, user.id)
    if db_user is not None:
        db_user.bodyweight = bodyweight
    db.commit()
    return RedirectResponse("/profile?saved=1", status_code=303)


# ── PAGES ────────────────────────────────────────────────────────────────────

def _dashboard_context(user: models.User, db: Session, weeks: int) -> dict:
    """Shared data assembly for the dashboard page and its htmx chart
    fragments so both always render identical numbers."""
    sessions = db.query(models.WorkoutSession).order_by(
        models.WorkoutSession.date.desc()).limit(5).all()
    exercises = db.query(models.Exercise).all()
    templates_db = db.query(models.SessionTemplate).all()

    # Training load data for dashboard chart (last `weeks` weeks)
    from datetime import timedelta
    from collections import defaultdict
    # Monday of the current week; chart spans exactly `weeks` Monday-aligned weeks.
    current_monday = date.today() - timedelta(days=date.today().weekday())
    week_start = current_monday - timedelta(weeks=weeks - 1)
    recent_sessions_full = (
        db.query(models.WorkoutSession)
        .filter(models.WorkoutSession.date >= week_start)
        .order_by(models.WorkoutSession.date)
        .all()
    )
    # Build weekly training load: sum(effective_load * reps) per ISO week.
    # Mirrors the lifetime-tonnage query below (bodyweight-exercise load =
    # BW * factor + added kg − assist, per app/load.py; unmeasured weight
    # counts as 0) so hero stats and chart tell the same story.
    weekly_load = defaultdict(float)
    bodyweight_kg = user.bodyweight or BODYWEIGHT_DEFAULT_KG
    for sess in recent_sessions_full:
        week_key = _iso_week_key(sess.date or date.today())
        for set_entry in sess.sets:
            weight = effective_load_kg(
                set_entry.exercise,
                set_entry.weight,
                set_entry.assist_kg,
                bodyweight_kg,
            )
            weekly_load[week_key] += weight * (set_entry.reps or 0)

    # Cardio load per ISO week: distance scaled by per-activity effort factor
    cardio_activities = (
        db.query(models.CardioActivity)
        .join(models.WorkoutSession, models.CardioActivity.session_id == models.WorkoutSession.id)
        .filter(models.WorkoutSession.date >= week_start)
        .all()
    )
    weekly_cardio_load = defaultdict(float)
    for a in cardio_activities:
        week_key = _iso_week_key(a.session.date if a.session else date.today())
        if a.distance_km:
            weekly_cardio_load[week_key] += a.distance_km * _cardio_load_factor(a.activity_type)

    # Dense week axis so weeks without sessions show as zero instead of
    # being skipped. ISO weeks (isocalendar) start on Monday. Exactly `weeks`
    # Monday-aligned buckets; empty series when there is no data so the
    # template's "No training data yet" empty state stays reachable.
    week_cursor = week_start
    all_weeks = []
    while week_cursor <= current_monday:
        all_weeks.append(_iso_week_key(week_cursor))
        week_cursor += timedelta(weeks=1)
    weekly_data = [
        {
            "date": k,
            "load": round(weekly_load.get(k, 0.0), 0),
            "cardio_load": round(weekly_cardio_load.get(k, 0.0), 1),
        }
        for k in all_weeks
    ] if (weekly_load or weekly_cardio_load) else []

    # Hero stats: this week's load, consecutive training weeks, lifetime tonnage.
    # Streak counts consecutive weeks (ending this week) with any logged activity,
    # computed over ALL history so it isn't capped by the 12-week chart window.
    this_week_key = _iso_week_key(current_monday)
    this_week_load = int(weekly_load.get(this_week_key, 0.0))

    week_keys = set(weekly_load) | set(weekly_cardio_load)
    week_streak = 0
    streak_cursor = current_monday
    while _iso_week_key(streak_cursor) in week_keys:
        week_streak += 1
        streak_cursor -= timedelta(weeks=1)

    # Same semantics as the weekly-load loop above: bodyweight exercises
    # contribute BW * factor + added kg − assist (app/load.py), unmeasured
    # weight counts as 0. Sessions carry no user_id (single-athlete data
    # model), so the viewer's own bodyweight is used rather than joining users.
    bodyweight_kg = user.bodyweight or BODYWEIGHT_DEFAULT_KG
    total_volume = (
        db.query(func.coalesce(func.sum(
            (
                func.coalesce(models.SetEntry.weight, 0.0)
                + func.coalesce(models.Exercise.is_bodyweight, False).cast(Boolean) * (
                    bodyweight_kg * func.coalesce(models.Exercise.bw_load_factor, 1.0)
                )
                - func.coalesce(models.SetEntry.assist_kg, 0.0)
            ) * func.coalesce(models.SetEntry.reps, 0)
        ), 0.0))
        .join(models.WorkoutSession, models.SetEntry.session_id == models.WorkoutSession.id)
        .join(models.Exercise, models.SetEntry.exercise_id == models.Exercise.id)
        .scalar() or 0.0
    )
    total_volume_t = f"{total_volume / 1000.0:,.1f}"

    return {
        "recent_sessions": sessions,
        "exercise_count": len(exercises),
        "template_count": len(templates_db),
        "session_count": db.query(models.WorkoutSession).count(),
        "weekly_activity": weekly_data,
        "week_streak": week_streak,
        "this_week_load": f"{this_week_load:,}",
        "total_volume_t": total_volume_t,
        "chart_weeks": weeks,
        "user": user,
    }


@app.get("/", response_class=HTMLResponse)
async def index(request: Request, user: models.User = Depends(get_current_user), db: Session = Depends(get_db)):
    return render_page(request, "index.html",
                                      _dashboard_context(user, db, weeks=12))


@app.get("/fragments/weekly-load", response_class=HTMLResponse)
async def fragment_weekly_load(
    request: Request,
    weeks: int = Query(default=12, ge=4, le=52),
    user: models.User = Depends(get_current_user),
    db: Session = Depends(get_db),
):
    """htmx fragment: the weekly-load chart card for the requested range."""
    ctx = _dashboard_context(user, db, weeks=weeks)
    return templates.TemplateResponse(request, "partials/weekly_load.html", ctx)


@app.get("/exercises", response_class=HTMLResponse)
async def list_exercises(request: Request, user: models.User = Depends(get_current_user), db: Session = Depends(get_db)):
    exercises = db.query(models.Exercise).order_by(models.Exercise.name).all()
    return render_page(request, "exercises.html", {"exercises": exercises, "user": user})


@app.post("/exercises")
async def create_exercise(
    name: str = Form(...),
    is_bodyweight: Optional[str] = Form(None),
    bw_load_percent: Optional[str] = Form(None),
    user: models.User = Depends(get_current_user),
    db: Session = Depends(get_db),
):
    existing = db.query(models.Exercise).filter(models.Exercise.name == name.strip()).first()
    if existing:
        raise HTTPException(status_code=400, detail="Exercise already exists")
    bodyweight = is_bodyweight == "1"
    # Empty field -> research default for known BW names (app/load.py), so
    # exercises created mid-session still scale correctly without a restart.
    factor = _parse_bw_load_percent(bw_load_percent)
    if bodyweight and factor is None:
        factor = default_bw_load_factor(name.strip())
    ex = models.Exercise(
        name=name.strip(),
        is_bodyweight=bodyweight,
        bw_load_factor=factor if bodyweight else None,
    )
    db.add(ex)
    db.commit()
    return RedirectResponse(url="/exercises", status_code=303)


@app.post("/exercises/{exercise_id}/edit")
async def edit_exercise(
    exercise_id: int,
    is_bodyweight: Optional[str] = Form(None),
    bw_load_percent: Optional[str] = Form(None),
    user: models.User = Depends(get_current_user),
    db: Session = Depends(get_db),
):
    """Update an exercise's bodyweight flag and load-scaling percentage.

    The name is intentionally not editable: it is the unique key other
    rows (set entries, template rows) reference by id, and renaming
    through this form has no demand.
    """
    ex = db.get(models.Exercise, exercise_id)
    if not ex:
        raise HTTPException(status_code=404)
    bodyweight = is_bodyweight == "1"
    factor = _parse_bw_load_percent(bw_load_percent)
    ex.is_bodyweight = bodyweight
    ex.bw_load_factor = factor if bodyweight else None
    db.commit()
    return RedirectResponse(url="/exercises", status_code=303)


def _parse_bw_load_percent(raw: Optional[str]) -> Optional[float]:
    """Parse a 'Bodyweight load %' form value (e.g. '65') into a factor.

    Empty/None -> None (use the research default). Rejects non-numeric
    and out-of-range values; 0 is allowed (plank-family holds).
    """
    if raw is None:
        return None
    raw = raw.strip()
    if not raw:
        return None
    try:
        percent = float(raw)
    except ValueError:
        raise HTTPException(status_code=400, detail="Bodyweight load % must be a number")
    if not (0 <= percent <= 200) or not math.isfinite(percent):
        raise HTTPException(status_code=400, detail="Bodyweight load % must be between 0 and 200")
    return percent / 100.0


@app.post("/exercises/{exercise_id}/delete")
async def delete_exercise(exercise_id: int, user: models.User = Depends(get_current_user), db: Session = Depends(get_db)):
    ex = db.get(models.Exercise, exercise_id)
    if not ex:
        raise HTTPException(status_code=404)
    # Refuse to delete an exercise that still has rows referencing it. FK
    # enforcement is off, so deleting here would orphan SetEntry /
    # SessionTemplateExercise rows that the UI can no longer address.
    referenced_by_sets = db.query(models.SetEntry).filter(
        models.SetEntry.exercise_id == exercise_id
    ).first()
    referenced_by_templates = db.query(models.SessionTemplateExercise).filter(
        models.SessionTemplateExercise.exercise_id == exercise_id
    ).first()
    if referenced_by_sets or referenced_by_templates:
        raise HTTPException(
            status_code=400,
            detail="Cannot delete: exercise is used in sessions or templates",
        )
    db.delete(ex)
    db.commit()
    return RedirectResponse(url="/exercises", status_code=303)


@app.get("/templates", response_class=HTMLResponse)
async def list_templates(request: Request, user: models.User = Depends(get_current_user), db: Session = Depends(get_db)):
    templates_db = db.query(models.SessionTemplate).order_by(models.SessionTemplate.name).all()
    exercises = db.query(models.Exercise).order_by(models.Exercise.name).all()
    return render_page(request, "templates.html", {
        "templates": templates_db, "exercises": exercises, "user": user
    })


@app.post("/templates")
async def create_template(name: str = Form(...), user: models.User = Depends(get_current_user), db: Session = Depends(get_db)):
    if db.query(models.SessionTemplate).filter(models.SessionTemplate.name == name.strip()).first():
        raise HTTPException(status_code=400, detail="Template already exists")
    tpl = models.SessionTemplate(name=name.strip())
    db.add(tpl)
    db.commit()
    return RedirectResponse(url="/templates", status_code=303)


@app.post("/templates/{template_id}/add_exercise")
async def add_exercise_to_template(
    template_id: int,
    exercise_id: int = Form(...),
    sets: int = Form(...),
    user: models.User = Depends(get_current_user),
    db: Session = Depends(get_db),
):
    tpl = db.get(models.SessionTemplate, template_id)
    if not tpl:
        raise HTTPException(status_code=404)
    max_order = max((te.order or 0 for te in tpl.exercises), default=0)
    te = models.SessionTemplateExercise(
        session_template_id=template_id, exercise_id=exercise_id,
        sets=sets, order=max_order + 1,
    )
    db.add(te)
    db.commit()
    return RedirectResponse(url="/templates", status_code=303)


@app.post("/templates/{template_id}/remove_exercise/{te_id}")
async def remove_template_exercise(template_id: int, te_id: int, user: models.User = Depends(get_current_user), db: Session = Depends(get_db)):
    te = db.get(models.SessionTemplateExercise, te_id)
    if te:
        db.delete(te)
        db.commit()
    return RedirectResponse(url="/templates", status_code=303)


@app.post("/templates/{template_id}/delete")
async def delete_template(template_id: int, user: models.User = Depends(get_current_user), db: Session = Depends(get_db)):
    tpl = db.get(models.SessionTemplate, template_id)
    if tpl:
        db.delete(tpl)
        db.commit()
    return RedirectResponse(url="/templates", status_code=303)


@app.get("/sessions", response_class=HTMLResponse)
async def list_sessions(request: Request, user: models.User = Depends(get_current_user), db: Session = Depends(get_db)):
    sessions = db.query(models.WorkoutSession).order_by(models.WorkoutSession.date.desc()).all()
    return render_page(request, "sessions.html", {"sessions": sessions, "user": user})


@app.get("/sessions/new", response_class=HTMLResponse)
async def new_session(request: Request, user: models.User = Depends(get_current_user), template_id: Optional[int] = None, db: Session = Depends(get_db)):
    templates_db = db.query(models.SessionTemplate).order_by(models.SessionTemplate.name).all()
    selected_template = db.get(models.SessionTemplate, template_id) if template_id else None
    return render_page(request, "new_session.html", {
        "templates": templates_db,
        "selected_template": selected_template, "today": date.today(), "user": user,
    })


@app.post("/sessions/new")
async def create_session(request: Request, user: models.User = Depends(get_current_user), db: Session = Depends(get_db)):
    form = await request.form()
    date_str = _form_str(form.get("date"))
    if not date_str:
        raise HTTPException(status_code=400, detail="Date required")
    try:
        workout_date = date.fromisoformat(date_str)
    except ValueError:
        raise HTTPException(status_code=400, detail="Invalid date")

    template_id = None
    if _form_str(form.get("template_id")):
        try:
            template_id = int(_form_str(form["template_id"]))
        except ValueError:
            raise HTTPException(status_code=400, detail="Invalid template id")
    template = db.get(models.SessionTemplate, template_id) if template_id else None

    workout = models.WorkoutSession(
        date=workout_date,
        template_id=template_id,
        notes=_form_str(form.get("notes")) or None,
    )
    db.add(workout)
    db.flush()

    # Persist only sets the user actually filled in. Template exercises that
    # were left blank are not stored, so they won't show up as 0-rep rows.
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
        assist_val = _form_str(form.get(f"assist-{ex_id}-{set_num}"))
        try:
            reps = int(_form_str(value)) if _form_str(value) else 0
        except ValueError:
            reps = 0
        try:
            weight = float(weight_val) if weight_val else None
        except ValueError:
            weight = None
        try:
            assist = float(assist_val) if assist_val else None
        except ValueError:
            assist = None

        if reps == 0 and weight is None and assist is None:
            continue

        db.add(models.SetEntry(
            session_id=workout.id,
            exercise_id=ex_id,
            set_number=set_num,
            reps=reps,
            weight=weight,
            assist_kg=assist,
        ))

    db.commit()
    # API clients (e.g. MCP server) get the new session id as JSON; the
    # browser UI keeps the redirect flow.
    accept = request.headers.get("accept", "").lower()
    if "application/json" in accept:
        return JSONResponse(
            {
                "id": workout.id,
                "date": str(workout.date),
                "template_id": workout.template_id,
            },
            status_code=201,
        )
    return RedirectResponse(url="/sessions", status_code=303)

# ---------------------------------------------------------------------------
# Session CRUD – delete & edit
# ---------------------------------------------------------------------------

@app.get("/sessions/{session_id}", response_class=HTMLResponse)
async def view_session(session_id: int, request: Request, user: models.User = Depends(get_current_user), db: Session = Depends(get_db)):
    sess = db.get(models.WorkoutSession, session_id)
    if not sess:
        raise HTTPException(status_code=404, detail="Session not found")
    return render_page(request, "view_session.html", {"session": sess, "user": user})

@app.post("/sessions/{session_id}/delete")
async def delete_session(session_id: int, user: models.User = Depends(get_current_user), db: Session = Depends(get_db)):
    sess = db.get(models.WorkoutSession, session_id)
    if not sess:
        raise HTTPException(status_code=404, detail="Session not found")
    db.delete(sess)
    db.commit()
    return RedirectResponse(url="/sessions", status_code=303)

@app.get("/sessions/edit/{session_id}", response_class=HTMLResponse)
async def edit_session_form(session_id: int, request: Request, user: models.User = Depends(get_current_user), db: Session = Depends(get_db)):
    sess = db.get(models.WorkoutSession, session_id)
    if not sess:
        raise HTTPException(status_code=404, detail="Session not found")

    # Build a dummy template structure for the form
    class DummyTemplate:
        def __init__(self, template_id, exercises):
            self.id = template_id
            self.exercises = exercises

    # Group sets by exercise.
    sets_by_exercise: dict[int, list[models.SetEntry]] = {}
    for se in sess.sets:
        if se.exercise_id is None:
            continue
        sets_by_exercise.setdefault(se.exercise_id, []).append(se)

    def make_dummy(exercise, set_numbers, order):
        return type("DummyTE", (), {
            "exercise": exercise,
            "sets": len(set_numbers),
            "set_numbers": set_numbers,
            "order": order,
        })()

    dummy_exercises = []
    template = db.get(models.SessionTemplate, sess.template_id) if sess.template_id else None
    if template:
        # Show every template exercise (in template order) so exercises
        # that were skipped when the session was logged can still be
        # added here. Exercises without sets render only their ghost row.
        for order_idx, te in enumerate(template.exercises, start=1):
            ex_sets = sets_by_exercise.get(te.exercise_id or 0, [])
            set_numbers = sorted({s.set_number or 0 for s in ex_sets})
            dummy_exercises.append(make_dummy(te.exercise, set_numbers, order_idx))
    else:
        # No template: fall back to the exercises that have sets, ordered
        # by their earliest set.
        first_seen: list[int] = []
        for se in sorted(sess.sets, key=lambda s: s.set_number or 0):
            if se.exercise_id is None:
                continue
            if se.exercise_id not in first_seen:
                first_seen.append(se.exercise_id)
        for order_idx, ex_id in enumerate(first_seen, start=1):
            ex_sets = sets_by_exercise[ex_id]
            # Render one row per actual set_number (they may have gaps
            # after a middle set was deleted), plus one blank row so new
            # sets can be added. Renumbering here would shift data onto
            # the wrong set.
            set_numbers = sorted({s.set_number or 0 for s in ex_sets})
            dummy_exercises.append(make_dummy(ex_sets[0].exercise, set_numbers, order_idx))
    dummy_template = DummyTemplate(sess.template_id, dummy_exercises)

    return render_page(request, "new_session.html", {
        "templates": [],
        "selected_template": dummy_template,
        "today": sess.date,
        "notes": sess.notes,
        "session_id": session_id,
        "existing_sets": {(se.exercise_id, se.set_number): se for se in sess.sets},
    })

@app.post("/sessions/edit/{session_id}")
async def edit_session(session_id: int, request: Request, user: models.User = Depends(get_current_user), db: Session = Depends(get_db)):
    sess = db.get(models.WorkoutSession, session_id)
    if not sess:
        raise HTTPException(status_code=404, detail="Session not found")

    form = await request.form()
    date_str = _form_str(form.get("date"))
    if not date_str:
        raise HTTPException(status_code=400, detail="Date required")
    try:
        sess.date = date.fromisoformat(date_str)
    except ValueError:
        raise HTTPException(status_code=400, detail="Invalid date")
    # Template is chosen at creation time and is not editable from here, so
    # sess.template_id is intentionally left untouched.
    sess.notes = _form_str(form.get("notes")) or None

    # Collect all (exercise_id, set_number) tuples being submitted.
    submitted_pairs = set()

    for key in form.keys():
        if key.startswith("reps-"):
            # reps-EX-SET format
            try:
                _, ex_id_str, set_num_str = key.split("-")
                submitted_pairs.add((int(ex_id_str), int(set_num_str)))
            except (ValueError, IndexError):
                continue

    # Track which existing rows should be deleted (user emptied both fields).
    existing_rows = db.query(models.SetEntry).filter(
        models.SetEntry.session_id == session_id
    ).all()
    rows_by_key = {(r.exercise_id, r.set_number): r for r in existing_rows}

    # Upsert submitted rows.
    for ex_id, set_num in submitted_pairs:
        reps_val = _form_str(form.get(f"reps-{ex_id}-{set_num}", ""))
        weight_val = _form_str(form.get(f"weight-{ex_id}-{set_num}", ""))
        assist_val = _form_str(form.get(f"assist-{ex_id}-{set_num}", ""))

        try:
            reps = int(reps_val) if reps_val else 0
        except ValueError:
            reps = 0

        try:
            weight = float(weight_val) if weight_val else None
        except ValueError:
            weight = None

        try:
            assist = float(assist_val) if assist_val else None
        except ValueError:
            assist = None

        existing = rows_by_key.get((ex_id, set_num))

        if reps == 0 and weight is None and assist is None:
            if existing:
                db.delete(existing)
            continue

        if existing:
            existing.reps = reps
            existing.weight = weight
            existing.assist_kg = assist
        else:
            db.add(models.SetEntry(
                session_id=session_id,
                exercise_id=ex_id,
                set_number=set_num,
                reps=reps,
                weight=weight,
                assist_kg=assist,
            ))

    db.commit()
    return RedirectResponse(url="/sessions", status_code=303)


@app.get("/progression", response_class=HTMLResponse)
async def progression(request: Request, user: models.User = Depends(get_current_user), exercise_id: Optional[int] = None, db: Session = Depends(get_db)):
    exercises = db.query(models.Exercise).order_by(models.Exercise.name).all()
    selected_exercise = db.get(models.Exercise, exercise_id) if exercise_id else None
    return render_page(request, "progression.html", {
        "exercises": exercises, "selected_exercise": selected_exercise, "user": user,
    })


# ── CARDIO (running / swimming) ──────────────────────────────────────────────

@app.get("/cardio", response_class=HTMLResponse)
async def cardio_page(
    request: Request,
    user: models.User = Depends(get_current_user),
    db: Session = Depends(get_db),
):
    activities = (
        db.query(models.CardioActivity)
        .join(models.WorkoutSession)
        .order_by(models.WorkoutSession.date.desc(), models.CardioActivity.id.desc())
        .limit(50)
        .all()
    )
    return render_page(request, "cardio.html", {
        "activities": activities, "today": date.today(), "user": user,
    })


@app.post("/cardio")
async def cardio_create(
    request: Request,
    user: models.User = Depends(get_current_user),
    db: Session = Depends(get_db),
):
    form = await request.form()
    activity_type = (_form_str(form.get("activity_type")) or "").strip().lower()
    if not activity_type:
        raise HTTPException(status_code=400, detail="activity_type is required")
    if activity_type not in CARDIO_ACTIVITY_TYPES:
        raise HTTPException(
            status_code=400,
            detail=f"activity_type must be one of: {', '.join(CARDIO_ACTIVITY_TYPES)}",
        )

    def _to_float(v):
        if v is None or v == "":
            return None
        try:
            return float(v)
        except (TypeError, ValueError):
            raise HTTPException(status_code=400, detail="invalid numeric value")

    distance_km = _to_float(_form_str(form.get("distance_km")))
    try:
        duration_min = _parse_duration_min(_form_str(form.get("duration_min")))
    except ValueError:
        raise HTTPException(status_code=400, detail="duration must be minutes (e.g. 45) or M:SS (e.g. 44:51)")
    if distance_km is not None and distance_km < 0:
        raise HTTPException(status_code=400, detail="distance must be >= 0")
    if duration_min is not None and duration_min < 0:
        raise HTTPException(status_code=400, detail="duration must be >= 0")
    if (distance_km is None or distance_km == 0) and duration_min is None:
        raise HTTPException(status_code=400, detail="enter a distance or a duration")

    date_str = _form_str(form.get("date")) or date.today().isoformat()
    try:
        activity_date = date.fromisoformat(date_str)
    except ValueError:
        raise HTTPException(status_code=400, detail="Invalid date")

    sess = (
        db.query(models.WorkoutSession)
        .filter(models.WorkoutSession.date == activity_date)
        .first()
    )
    if not sess:
        sess = models.WorkoutSession(date=activity_date)
        db.add(sess)
        db.flush()

    db.add(models.CardioActivity(
        session_id=sess.id,
        activity_type=activity_type,
        distance_km=distance_km,
        duration_min=duration_min,
        notes=_form_str(form.get("notes")) or None,
    ))
    db.commit()
    return RedirectResponse(url="/cardio?created=1", status_code=303)


@app.get("/cardio/{cardio_id}/edit", response_class=HTMLResponse)
async def cardio_edit_page(
    cardio_id: int,
    request: Request,
    user: models.User = Depends(get_current_user),
    db: Session = Depends(get_db),
):
    c = db.get(models.CardioActivity, cardio_id)
    if not c:
        raise HTTPException(status_code=404, detail="Cardio activity not found")
    return render_page(request, "cardio_edit.html", {
        "activity": c, "today": date.today(), "user": user,
    })


@app.post("/cardio/{cardio_id}")
async def cardio_update(
    cardio_id: int,
    request: Request,
    user: models.User = Depends(get_current_user),
    db: Session = Depends(get_db),
):
    c = db.get(models.CardioActivity, cardio_id)
    if not c:
        raise HTTPException(status_code=404, detail="Cardio activity not found")

    form = await request.form()
    activity_type = (_form_str(form.get("activity_type")) or "").strip().lower()
    if not activity_type:
        raise HTTPException(status_code=400, detail="activity_type is required")
    if activity_type not in CARDIO_ACTIVITY_TYPES:
        raise HTTPException(
            status_code=400,
            detail=f"activity_type must be one of: {', '.join(CARDIO_ACTIVITY_TYPES)}",
        )

    def _to_float(v):
        if v is None or v == "":
            return None
        try:
            return float(v)
        except (TypeError, ValueError):
            raise HTTPException(status_code=400, detail="invalid numeric value")

    distance_km = _to_float(_form_str(form.get("distance_km")))
    try:
        duration_min = _parse_duration_min(_form_str(form.get("duration_min")))
    except ValueError:
        raise HTTPException(status_code=400, detail="duration must be minutes (e.g. 45) or M:SS (e.g. 44:51)")
    if distance_km is not None and distance_km < 0:
        raise HTTPException(status_code=400, detail="distance must be >= 0")
    if duration_min is not None and duration_min < 0:
        raise HTTPException(status_code=400, detail="duration must be >= 0")
    if (distance_km is None or distance_km == 0) and duration_min is None:
        raise HTTPException(status_code=400, detail="enter a distance or a duration")

    date_str = _form_str(form.get("date")) or (c.session.date.isoformat() if c.session and c.session.date else date.today().isoformat())
    try:
        activity_date = date.fromisoformat(date_str)
    except ValueError:
        raise HTTPException(status_code=400, detail="Invalid date")

    old_session = c.session
    sess = (
        db.query(models.WorkoutSession)
        .filter(models.WorkoutSession.date == activity_date)
        .first()
    )
    if not sess:
        sess = models.WorkoutSession(date=activity_date)
        db.add(sess)
        db.flush()

    c.session_id = sess.id
    c.activity_type = activity_type
    c.distance_km = distance_km
    c.duration_min = duration_min
    c.notes = _form_str(form.get("notes")) or None
    # Moving the activity to another date can leave its previous auto-created
    # session empty; prune it so /sessions doesn't accumulate orphan rows.
    _prune_empty_session(db, old_session)
    db.commit()
    return RedirectResponse(url="/cardio", status_code=303)


@app.post("/cardio/{cardio_id}/delete")
async def cardio_delete(
    cardio_id: int,
    user: models.User = Depends(get_current_user),
    db: Session = Depends(get_db),
):
    c = db.get(models.CardioActivity, cardio_id)
    if not c:
        raise HTTPException(status_code=404, detail="Cardio activity not found")
    owning_session = c.session
    db.delete(c)
    _prune_empty_session(db, owning_session)
    db.commit()
    return RedirectResponse(url="/cardio", status_code=303)


# ── JSON API for chart data ───────────────────────────────────────────────────

@app.get("/api/progression/{exercise_id}")
async def progression_data(exercise_id: int, user: models.User = Depends(get_current_user), db: Session = Depends(get_db)):
    sessions = (
        db.query(models.WorkoutSession)
        .join(models.SetEntry)
        .filter(models.SetEntry.exercise_id == exercise_id)
        .order_by(models.WorkoutSession.date)
        .all()
    )
    bodyweight_kg = user.bodyweight or BODYWEIGHT_DEFAULT_KG
    rows = []
    for sess in sessions:
        sets = [s for s in sess.sets if s.exercise_id == exercise_id]
        if not sets:
            continue
        exercise = sets[0].exercise
        is_bodyweight = bool(exercise and exercise.is_bodyweight)
        # Effective load per set (app/load.py): bodyweight lifts count
        # BW * factor + added kg − assist; weighted lifts count the
        # logged weight only.
        base_kg = (
            bodyweight_kg * exercise_load_factor(exercise)
            if is_bodyweight else 0.0
        )

        # Weighted entries: sets with an added/logged weight. Bodyweight
        # sets may still carry assist (supported) without added weight.
        weighted_sets = [s for s in sets if s.weight is not None]
        bw_sets = [s for s in sets if s.weight is None and is_bodyweight]

        # Every session here has at least one set, so there is always a
        # chartable metric: weighted exercises use their added weight,
        # bodyweight exercises always produce an effective load.
        if weighted_sets:
            has_weight = True
            top_weight = max((s.weight or 0.0) for s in weighted_sets)
        elif bw_sets:
            has_weight = True
            top_weight = 0.0
        else:
            # Weighted exercise whose sets were logged without a weight
            # (e.g. weight forgotten). Keep the session visible in history
            # but flag it so the charts can exclude it.
            has_weight = False
            top_weight = 0.0

        # Effective (chartable) load and Epley estimated 1RM over the best
        # set: w * (1 + reps/30). Bodyweight rows use effective load so the
        # chart shows real load instead of a flat zero line.
        if has_weight:
            chartable = sets if is_bodyweight else weighted_sets
            eff_loads = [
                effective_load_kg(
                    exercise, s.weight, s.assist_kg, bodyweight_kg
                )
                for s in chartable
            ]
            eff_top = max(eff_loads)
            est_1rm = max(
                e * (1.0 + (s.reps or 0) / 30.0)
                for e, s in zip(eff_loads, chartable)
            )
        else:
            eff_top = 0.0
            est_1rm = 0.0

        # Volume is tonnage in kg: effective load × reps for every set
        # (bodyweight rows included — real tonnage, not a rep count).
        volume = sum(
            effective_load_kg(exercise, s.weight, s.assist_kg, bodyweight_kg)
            * (s.reps or 0)
            for s in sets
        )

        total_reps = sum(s.reps or 0 for s in sets)
        rows.append({
            "date": str(sess.date),
            "top_weight": round(top_weight, 2),
            "effective_top_weight": round(eff_top, 1),
            "est_1rm": round(est_1rm, 1),
            "volume": round(volume, 2),
            "total_reps": total_reps,
            "has_weight": has_weight,
            "is_bodyweight": is_bodyweight,
            "bw_load_factor": exercise_load_factor(exercise) if is_bodyweight else None,
        })
    return JSONResponse({"data": rows})


@app.get("/api/exercises")
async def api_list_exercises(user: models.User = Depends(get_current_user), db: Session = Depends(get_db)):
    exercises = db.query(models.Exercise).order_by(models.Exercise.name).all()
    return [
        {
            "id": e.id,
            "name": e.name,
            "is_bodyweight": bool(e.is_bodyweight),
            "bw_load_factor": exercise_load_factor(e) if e.is_bodyweight else None,
        }
        for e in exercises
    ]


@app.get("/api/templates")
async def api_list_templates(user: models.User = Depends(get_current_user), db: Session = Depends(get_db)):
    templates = db.query(models.SessionTemplate).order_by(models.SessionTemplate.name).all()
    return [
        {
            "id": t.id,
            "name": t.name,
            "description": t.description,
            "exercises": [
                {
                    "exercise_id": te.exercise_id,
                    "name": te.exercise.name if te.exercise else "",
                    "sets": te.sets,
                    "prescription": te.prescription,
                    "order": te.order,
                }
                for te in t.exercises
            ],
        }
        for t in templates
    ]


@app.get("/api/sessions")
async def api_list_sessions(
    limit: int = Query(20, ge=1, le=500),
    user: models.User = Depends(get_current_user),
    db: Session = Depends(get_db),
):
    sessions = (
        db.query(models.WorkoutSession)
        .order_by(models.WorkoutSession.date.desc(), models.WorkoutSession.id.desc())
        .limit(limit)
        .all()
    )
    return [
        {
            "id": s.id,
            "date": str(s.date),
            "template_id": s.template_id,
            "template_name": s.template.name if s.template else None,
            "notes": s.notes,
            "cardio": [_cardio_json(c) for c in s.cardio],
        }
        for s in sessions
    ]


@app.get("/api/sessions/{session_id}")
async def api_get_session(
    session_id: int,
    user: models.User = Depends(get_current_user),
    db: Session = Depends(get_db),
):
    sess = db.get(models.WorkoutSession, session_id)
    if not sess:
        raise HTTPException(status_code=404, detail="Session not found")
    sets = (
        db.query(models.SetEntry)
        .filter(models.SetEntry.session_id == session_id)
        .order_by(models.SetEntry.exercise_id, models.SetEntry.set_number)
        .all()
    )
    return {
        "id": sess.id,
        "date": str(sess.date),
        "template_id": sess.template_id,
        "template_name": sess.template.name if sess.template else None,
        "notes": sess.notes,
        "sets": [
            {
                "exercise_id": s.exercise_id,
                "exercise": s.exercise.name if s.exercise else "",
                "set_number": s.set_number,
                "reps": s.reps,
                "weight": s.weight,
                "assist_kg": s.assist_kg,
            }
            for s in sets
        ],
        "cardio": [_cardio_json(c) for c in sess.cardio],
    }


# ── Cardio CRUD (JSON API) ───────────────────────────────────────────────────

@app.get("/api/cardio")
async def api_list_cardio(
    limit: int = Query(50, ge=1, le=500),
    user: models.User = Depends(get_current_user),
    db: Session = Depends(get_db),
):
    """List cardio activities, newest first (by their session date)."""
    rows = (
        db.query(models.CardioActivity)
        .join(models.WorkoutSession)
        .order_by(models.WorkoutSession.date.desc(), models.CardioActivity.id.desc())
        .limit(limit)
        .all()
    )
    return [
        dict(_cardio_json(c), date=str(c.session.date if c.session else ""))
        for c in rows
    ]


@app.get("/api/cardio/{cardio_id}")
async def api_get_cardio(
    cardio_id: int,
    user: models.User = Depends(get_current_user),
    db: Session = Depends(get_db),
):
    c = db.get(models.CardioActivity, cardio_id)
    if not c:
        raise HTTPException(status_code=404, detail="Cardio activity not found")
    return dict(_cardio_json(c), date=str(c.session.date if c.session else ""))


@app.post("/api/cardio")
async def api_create_cardio(
    request: Request,
    user: models.User = Depends(get_current_user),
    db: Session = Depends(get_db),
):
    """Create a cardio activity.

    Accepts JSON (recommended for API/MCP clients) or form data. Attaches to
    the session for ``date`` (creating one if none exists for that date) or to
    an explicit ``session_id``.
    """
    content_type = request.headers.get("content-type", "")
    if "application/json" in content_type:
        body = await request.json()
        get = lambda k, d=None: body.get(k, d)
    else:
        form = await request.form()
        def get(k, d=None):
            v = form.get(k, d)
            return v if isinstance(v, str) else None

    activity_type = (get("activity_type") or "").strip().lower()
    if not activity_type:
        raise HTTPException(status_code=400, detail="activity_type is required")
    if activity_type not in CARDIO_ACTIVITY_TYPES:
        raise HTTPException(
            status_code=400,
            detail=f"activity_type must be one of: {', '.join(CARDIO_ACTIVITY_TYPES)}",
        )

    def _to_float(v):
        if v is None or v == "":
            return None
        try:
            return float(v)
        except (TypeError, ValueError):
            raise HTTPException(
                status_code=400, detail=f"invalid numeric value: {v!r}"
            )

    distance_km = _to_float(get("distance_km"))
    try:
        duration_min = _parse_duration_min(get("duration_min"))
    except ValueError:
        raise HTTPException(
            status_code=400, detail="duration_min must be minutes (e.g. 45) or M:SS (e.g. 44:51)"
        )
    if distance_km is not None and distance_km < 0:
        raise HTTPException(status_code=400, detail="distance_km must be >= 0")
    if duration_min is not None and duration_min < 0:
        raise HTTPException(status_code=400, detail="duration_min must be >= 0")
    if distance_km is None and duration_min is None:
        raise HTTPException(
            status_code=400, detail="at least one of distance_km or duration_min is required"
        )
    if distance_km == 0 and duration_min is None:
        raise HTTPException(status_code=400, detail="distance_km is 0 but duration_min is missing")

    session_id = get("session_id")
    if session_id is not None and session_id != "":
        try:
            session_id = int(session_id)
        except (TypeError, ValueError):
            raise HTTPException(status_code=400, detail="invalid session_id")
        sess = db.get(models.WorkoutSession, session_id)
        if not sess:
            raise HTTPException(status_code=404, detail="Session not found")
    else:
        # Attach to (or create) a session for the given date.
        date_str = get("date") or date.today().isoformat()
        try:
            activity_date = date.fromisoformat(date_str)
        except ValueError:
            raise HTTPException(status_code=400, detail="Invalid date")
        sess = (
            db.query(models.WorkoutSession)
            .filter(models.WorkoutSession.date == activity_date)
            .first()
        )
        if not sess:
            sess = models.WorkoutSession(date=activity_date)
            db.add(sess)
            db.flush()

    notes = get("notes") or None
    c = models.CardioActivity(
        session_id=sess.id,
        activity_type=activity_type,
        distance_km=distance_km,
        duration_min=duration_min,
        notes=notes,
    )
    db.add(c)
    db.commit()
    db.refresh(c)
    return JSONResponse(dict(_cardio_json(c), date=str(sess.date)), status_code=201)


@app.delete("/api/cardio/{cardio_id}")
async def api_delete_cardio(
    cardio_id: int,
    user: models.User = Depends(get_current_user),
    db: Session = Depends(get_db),
):
    c = db.get(models.CardioActivity, cardio_id)
    if not c:
        raise HTTPException(status_code=404, detail="Cardio activity not found")
    owning_session = c.session
    db.delete(c)
    _prune_empty_session(db, owning_session)
    db.commit()
    return JSONResponse({"deleted": cardio_id})


# ── BROWSE EXERCISES (free-exercise-db) ──────────────────────────────────────

@app.get("/browse/exercises", response_class=HTMLResponse)
async def browse_exercises(
    request: Request, user: models.User = Depends(get_current_user),
    q: str = "",
    category: str = "",
    muscle: str = "",
    equipment: str = "",
    level: str = "",
):
    try:
        data = await fetch_exercise_db()
        error = None
    except Exception as e:
        data = []
        error = str(e)

    # Build filter options from full dataset
    all_categories = sorted({ex.get("category", "") for ex in data if ex.get("category")})
    all_muscles = sorted({m for ex in data for m in (ex.get("primaryMuscles") or [])})
    all_equipment = sorted({ex.get("equipment", "") for ex in data if ex.get("equipment")})
    all_levels = ["beginner", "intermediate", "expert"]

    # Filter
    filtered = data
    if q:
        ql = q.lower()
        filtered = [ex for ex in filtered if ql in ex.get("name", "").lower()]
    if category:
        filtered = [ex for ex in filtered if ex.get("category") == category]
    if muscle:
        filtered = [ex for ex in filtered if muscle in (ex.get("primaryMuscles") or [])]
    if equipment:
        filtered = [ex for ex in filtered if ex.get("equipment") == equipment]
    if level:
        filtered = [ex for ex in filtered if ex.get("level") == level]

    return render_page(request, "browse_exercises.html", {
        "exercises": filtered[:200],  # cap at 200 for perf
        "total": len(filtered),
        "all_categories": all_categories,
        "all_muscles": all_muscles,
        "all_equipment": all_equipment,
        "all_levels": all_levels,
        "q": q, "category": category, "muscle": muscle,
        "equipment": equipment, "level": level,
        "error": error, "user": user,
    })


@app.post("/browse/exercises/import")
async def import_exercises(request: Request, user: models.User = Depends(get_current_user), db: Session = Depends(get_db)):
    form = await request.form()
    names = form.getlist("exercise_names")
    bw_set = set(form.getlist("bw_names"))
    imported = 0
    for name in names:
        if not name:
            continue
        name = _form_str(name)
        existing = db.query(models.Exercise).filter(models.Exercise.name == name).first()
        if not existing:
            bodyweight = name in bw_set
            factor = default_bw_load_factor(name) if bodyweight else None
            db.add(models.Exercise(
                name=name,
                is_bodyweight=bodyweight,
                bw_load_factor=factor,
            ))
            imported += 1
    db.commit()
    return RedirectResponse(url=f"/exercises?imported={imported}", status_code=303)


# ── BROWSE PLANS ─────────────────────────────────────────────────────────────

@app.get("/browse/plans", response_class=HTMLResponse)
async def browse_plans(request: Request, user: models.User = Depends(get_current_user), db: Session = Depends(get_db)):
    with open(STARTER_PLANS_PATH) as f:
        plans = json.load(f)
    existing_templates = {t.name for t in db.query(models.SessionTemplate).all()}
    existing_exercises = {e.name for e in db.query(models.Exercise).all()}
    return render_page(request, "browse_plans.html", {
        "plans": plans,
        "existing_templates": existing_templates,
        "existing_exercises": existing_exercises,
        "user": user,
    })


@app.post("/browse/plans/import")
async def import_plan(
    request: Request,
    user: models.User = Depends(get_current_user),
    db: Session = Depends(get_db),
):
    form = await request.form()
    plan_id = _form_str(form.get("plan_id"))

    with open(STARTER_PLANS_PATH) as f:
        plans = json.load(f)

    plan = next((p for p in plans if p["id"] == plan_id), None)
    if not plan:
        raise HTTPException(status_code=404, detail="Plan not found")

    # Create or find each exercise. New exercises detect their bodyweight
    # status from the name (hints + research load factor, app/load.py) so
    # imported plans scale correctly without manual fixes.
    order = 1
    exercise_ids = []
    for item in plan["exercises"]:
        ex = db.query(models.Exercise).filter(models.Exercise.name == item["name"]).first()
        if not ex:
            bodyweight = is_bodyweight_name(item["name"])
            ex = models.Exercise(
                name=item["name"],
                is_bodyweight=bodyweight,
                bw_load_factor=default_bw_load_factor(item["name"]) if bodyweight else None,
            )
            db.add(ex)
            db.flush()
        exercise_ids.append((ex.id, item.get("sets"), item.get("prescription"), order))
        order += 1

    # Create template (avoid duplicate names)
    base_name = plan["name"]
    name = base_name
    i = 2
    while db.query(models.SessionTemplate).filter(models.SessionTemplate.name == name).first():
        name = f"{base_name} ({i})"
        i += 1

    tpl = models.SessionTemplate(name=name, description=plan.get("description"))
    db.add(tpl)
    db.flush()

    for ex_id, sets, prescription, ord_ in exercise_ids:
        db.add(models.SessionTemplateExercise(
            session_template_id=tpl.id,
            exercise_id=ex_id,
            sets=sets,
            prescription=prescription,
            order=ord_,
        ))

    db.commit()
    return RedirectResponse(url="/templates?imported=1", status_code=303)
