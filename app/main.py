import os
import time
import argparse
import tempfile
import shutil
from datetime import date
from typing import Generator, Optional
import httpx
import json
from dotenv import load_dotenv
from fastapi import Cookie, Depends, FastAPI, Form, HTTPException, Query, Request
from fastapi.responses import HTMLResponse, RedirectResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from sqlalchemy.orm import Session
from sqlalchemy import text as sa_text

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
from .auth import get_current_user, create_session_cookie, verify_password, COOKIE_NAME, SESSION_MAX_AGE
from .database import Base, SessionLocal, engine

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

app = FastAPI(title="Training Log Dashboard")
app.mount("/static", StaticFiles(directory="app/static"), name="static")
templates = Jinja2Templates(directory="app/templates")
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

# Dashboard cardio load: per-activity effort multiplier applied to distance (km).
CARDIO_LOAD_FACTOR = {"running": 1.0, "swimming": 3.0}
CARDIO_LOAD_DEFAULT_FACTOR = 1.0


def _cardio_load_factor(activity_type: str) -> float:
    return CARDIO_LOAD_FACTOR.get((activity_type or "").lower(), CARDIO_LOAD_DEFAULT_FACTOR)


def _parse_duration_min(value) -> float | None:
    """Parse a form duration: minutes ('45', '44.85') or M:SS / H:MM:SS ('44:51').

    Raises ValueError for non-empty values that don't match.
    """
    if value is None:
        return None
    if isinstance(value, (int, float)):
        return float(value)
    value = value.strip()
    if not value:
        return None
    parts = value.split(":")
    if len(parts) in (2, 3) and all(p.strip().isdigit() for p in parts):
        nums = [int(p) for p in parts]
        if len(nums) == 2:
            h, m, s = 0, nums[0], nums[1]
        else:
            h, m, s = nums
        if not (0 <= m < 60 and 0 <= s < 60):
            raise ValueError("invalid duration")
        return h * 60 + m + s / 60
    try:
        return float(value)
    except ValueError:
        raise ValueError("invalid duration")


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


# ── PAGES ────────────────────────────────────────────────────────────────────

@app.get("/", response_class=HTMLResponse)
async def index(request: Request, user: models.User = Depends(get_current_user), db: Session = Depends(get_db)):
    sessions = db.query(models.WorkoutSession).order_by(models.WorkoutSession.date.desc()).limit(5).all()
    exercises = db.query(models.Exercise).all()
    templates_db = db.query(models.SessionTemplate).all()
    
    # Training load data for dashboard chart (last 12 weeks)
    from datetime import timedelta
    from collections import defaultdict
    twelve_weeks_ago = date.today() - timedelta(weeks=12)
    recent_sessions_full = (
        db.query(models.WorkoutSession)
        .filter(models.WorkoutSession.date >= twelve_weeks_ago)
        .order_by(models.WorkoutSession.date)
        .all()
    )
    # Build weekly training load: sum(weight * reps) per ISO week
    weekly_load = defaultdict(float)
    for sess in recent_sessions_full:
        iso_cal = sess.date.isocalendar()
        week_key = f"{iso_cal[0]}-W{iso_cal[1]:02d}"
        for set_entry in sess.sets:
            weight = set_entry.weight if set_entry.weight is not None else 1.0
            weekly_load[week_key] += weight * set_entry.reps

    # Cardio load per ISO week: distance scaled by per-activity effort factor
    cardio_activities = (
        db.query(models.CardioActivity)
        .join(models.WorkoutSession, models.CardioActivity.session_id == models.WorkoutSession.id)
        .filter(models.WorkoutSession.date >= twelve_weeks_ago)
        .all()
    )
    weekly_cardio_load = defaultdict(float)
    weekly_cardio_min = defaultdict(float)
    for a in cardio_activities:
        iso = a.session.date.isocalendar()
        week_key = f"{iso[0]}-W{iso[1]:02d}"
        if a.distance_km:
            weekly_cardio_load[week_key] += a.distance_km * _cardio_load_factor(a.activity_type)
        if a.duration_min:
            weekly_cardio_min[week_key] += a.duration_min

    all_weeks = sorted(set(weekly_load) | set(weekly_cardio_load) | set(weekly_cardio_min))
    weekly_data = [
        {
            "date": k,
            "load": round(weekly_load.get(k, 0.0), 0),
            "cardio_load": round(weekly_cardio_load.get(k, 0.0), 1),
            "cardio_min": round(weekly_cardio_min.get(k, 0.0), 0),
        }
        for k in all_weeks
    ]
    
    return templates.TemplateResponse(request, "index.html", {
        "recent_sessions": sessions,
        "exercise_count": len(exercises),
        "template_count": len(templates_db),
        "session_count": db.query(models.WorkoutSession).count(),
        "weekly_activity": weekly_data,
        "user": user,
    })


@app.get("/exercises", response_class=HTMLResponse)
async def list_exercises(request: Request, user: models.User = Depends(get_current_user), db: Session = Depends(get_db)):
    exercises = db.query(models.Exercise).order_by(models.Exercise.name).all()
    return templates.TemplateResponse(request, "exercises.html", {"exercises": exercises, "user": user})


@app.post("/exercises")
async def create_exercise(
    name: str = Form(...),
    is_bodyweight: Optional[str] = Form(None),
    user: models.User = Depends(get_current_user),
    db: Session = Depends(get_db),
):
    existing = db.query(models.Exercise).filter(models.Exercise.name == name.strip()).first()
    if existing:
        raise HTTPException(status_code=400, detail="Exercise already exists")
    ex = models.Exercise(name=name.strip(), is_bodyweight=(is_bodyweight == "1"))
    db.add(ex)
    db.commit()
    return RedirectResponse(url="/exercises", status_code=303)


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
    return templates.TemplateResponse(request, "templates.html", {
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
    max_order = max((te.order for te in tpl.exercises), default=0)
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
    return templates.TemplateResponse(request, "sessions.html", {"sessions": sessions, "user": user})


@app.get("/sessions/new", response_class=HTMLResponse)
async def new_session(request: Request, user: models.User = Depends(get_current_user), template_id: Optional[int] = None, db: Session = Depends(get_db)):
    templates_db = db.query(models.SessionTemplate).order_by(models.SessionTemplate.name).all()
    selected_template = db.get(models.SessionTemplate, template_id) if template_id else None
    return templates.TemplateResponse(request, "new_session.html", {
        "templates": templates_db,
        "selected_template": selected_template, "today": date.today(), "user": user,
    })


@app.post("/sessions/new")
async def create_session(request: Request, user: models.User = Depends(get_current_user), db: Session = Depends(get_db)):
    form = await request.form()
    date_str = form.get("date")
    if not date_str:
        raise HTTPException(status_code=400, detail="Date required")
    try:
        workout_date = date.fromisoformat(date_str)
    except ValueError:
        raise HTTPException(status_code=400, detail="Invalid date")

    template_id = None
    if form.get("template_id"):
        try:
            template_id = int(form["template_id"])
        except ValueError:
            raise HTTPException(status_code=400, detail="Invalid template id")
    template = db.get(models.SessionTemplate, template_id) if template_id else None

    workout = models.WorkoutSession(
        date=workout_date,
        template_id=template_id,
        notes=form.get("notes") or None,
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

        weight_val = form.get(f"weight-{ex_id}-{set_num}")
        try:
            reps = int(value) if value else 0
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
    return templates.TemplateResponse(request, "view_session.html", {"session": sess, "user": user})

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
            ex_sets = sets_by_exercise.get(te.exercise_id, [])
            set_numbers = sorted({s.set_number for s in ex_sets})
            dummy_exercises.append(make_dummy(te.exercise, set_numbers, order_idx))
    else:
        # No template: fall back to the exercises that have sets, ordered
        # by their earliest set.
        first_seen: list[int] = []
        for se in sorted(sess.sets, key=lambda s: s.set_number):
            if se.exercise_id not in first_seen:
                first_seen.append(se.exercise_id)
        for order_idx, ex_id in enumerate(first_seen, start=1):
            ex_sets = sets_by_exercise[ex_id]
            # Render one row per actual set_number (they may have gaps
            # after a middle set was deleted), plus one blank row so new
            # sets can be added. Renumbering here would shift data onto
            # the wrong set.
            set_numbers = sorted({s.set_number for s in ex_sets})
            dummy_exercises.append(make_dummy(ex_sets[0].exercise, set_numbers, order_idx))
    dummy_template = DummyTemplate(sess.template_id, dummy_exercises)

    return templates.TemplateResponse(request, "new_session.html", {
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
    date_str = form.get("date")
    if not date_str:
        raise HTTPException(status_code=400, detail="Date required")
    try:
        sess.date = date.fromisoformat(date_str)
    except ValueError:
        raise HTTPException(status_code=400, detail="Invalid date")
    # Template is chosen at creation time and is not editable from here, so
    # sess.template_id is intentionally left untouched.
    sess.notes = form.get("notes") or None

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
        reps_val = form.get(f"reps-{ex_id}-{set_num}", "")
        weight_val = form.get(f"weight-{ex_id}-{set_num}", "")

        try:
            reps = int(reps_val) if reps_val else 0
        except ValueError:
            reps = 0

        try:
            weight = float(weight_val) if weight_val else None
        except ValueError:
            weight = None

        existing = rows_by_key.get((ex_id, set_num))

        if reps == 0 and weight is None:
            if existing:
                db.delete(existing)
            continue

        if existing:
            existing.reps = reps
            existing.weight = weight
        else:
            db.add(models.SetEntry(
                session_id=session_id,
                exercise_id=ex_id,
                set_number=set_num,
                reps=reps,
                weight=weight,
            ))

    db.commit()
    return RedirectResponse(url="/sessions", status_code=303)


@app.get("/progression", response_class=HTMLResponse)
async def progression(request: Request, user: models.User = Depends(get_current_user), exercise_id: Optional[int] = None, db: Session = Depends(get_db)):
    exercises = db.query(models.Exercise).order_by(models.Exercise.name).all()
    selected_exercise = db.get(models.Exercise, exercise_id) if exercise_id else None
    return templates.TemplateResponse(request, "progression.html", {
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
    return templates.TemplateResponse(request, "cardio.html", {
        "activities": activities, "today": date.today(), "user": user,
    })


@app.post("/cardio")
async def cardio_create(
    request: Request,
    user: models.User = Depends(get_current_user),
    db: Session = Depends(get_db),
):
    form = await request.form()
    activity_type = (form.get("activity_type") or "").strip().lower()
    if not activity_type:
        raise HTTPException(status_code=400, detail="activity_type is required")

    def _to_float(v):
        if v is None or v == "":
            return None
        try:
            return float(v)
        except (TypeError, ValueError):
            raise HTTPException(status_code=400, detail="invalid numeric value")

    distance_km = _to_float(form.get("distance_km"))
    try:
        duration_min = _parse_duration_min(form.get("duration_min"))
    except ValueError:
        raise HTTPException(status_code=400, detail="duration must be minutes (e.g. 45) or M:SS (e.g. 44:51)")
    if duration_min is not None and duration_min < 0:
        raise HTTPException(status_code=400, detail="duration must be >= 0")
    if (distance_km is None or distance_km == 0) and duration_min is None:
        raise HTTPException(status_code=400, detail="enter a distance or a duration")

    date_str = form.get("date") or date.today().isoformat()
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
        notes=form.get("notes") or None,
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
    return templates.TemplateResponse(request, "cardio_edit.html", {
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
    activity_type = (form.get("activity_type") or "").strip().lower()
    if not activity_type:
        raise HTTPException(status_code=400, detail="activity_type is required")

    def _to_float(v):
        if v is None or v == "":
            return None
        try:
            return float(v)
        except (TypeError, ValueError):
            raise HTTPException(status_code=400, detail="invalid numeric value")

    distance_km = _to_float(form.get("distance_km"))
    try:
        duration_min = _parse_duration_min(form.get("duration_min"))
    except ValueError:
        raise HTTPException(status_code=400, detail="duration must be minutes (e.g. 45) or M:SS (e.g. 44:51)")
    if duration_min is not None and duration_min < 0:
        raise HTTPException(status_code=400, detail="duration must be >= 0")
    if (distance_km is None or distance_km == 0) and duration_min is None:
        raise HTTPException(status_code=400, detail="enter a distance or a duration")

    date_str = form.get("date") or c.session.date.isoformat()
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

    c.session_id = sess.id
    c.activity_type = activity_type
    c.distance_km = distance_km
    c.duration_min = duration_min
    c.notes = form.get("notes") or None
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
    db.delete(c)
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
    rows = []
    for sess in sessions:
        sets = [s for s in sess.sets if s.exercise_id == exercise_id]
        if not sets:
            continue
        # Separate weighted sets from bodyweight-only sets.
        weighted_sets = [s for s in sets if s.weight is not None]
        bw_sets = [s for s in sets if s.weight is None and s.exercise.is_bodyweight]
        
        # For bodyweight-only sessions (no weight at all), still include them
        # using total_reps as the primary metric.
        if weighted_sets:
            has_weight = True
            top_weight = max((s.weight or 0.0) for s in weighted_sets)
            volume = sum((s.weight or 0.0) * s.reps for s in weighted_sets)
        elif bw_sets:
            # Bodyweight exercise with no added weight – use reps as metric.
            has_weight = True
            top_weight = 0.0
            volume = sum(s.reps for s in bw_sets)  # use volume column for total reps when BW
        else:
            # Weighted exercise whose sets were logged without a weight
            # (e.g. weight forgotten). Keep the session visible in history
            # but flag it so the charts can exclude it.
            has_weight = False
            top_weight = 0.0
            volume = 0.0
        
        total_reps = sum(s.reps for s in sets)
        rows.append({
            "date": str(sess.date),
            "top_weight": round(top_weight, 2),
            "volume": round(volume, 2),
            "total_reps": total_reps,
            "has_weight": has_weight,
        })
    return JSONResponse({"data": rows})


@app.get("/api/exercises")
async def api_list_exercises(user: models.User = Depends(get_current_user), db: Session = Depends(get_db)):
    exercises = db.query(models.Exercise).order_by(models.Exercise.name).all()
    return [
        {"id": e.id, "name": e.name, "is_bodyweight": bool(e.is_bodyweight)}
        for e in exercises
    ]


@app.get("/api/templates")
async def api_list_templates(user: models.User = Depends(get_current_user), db: Session = Depends(get_db)):
    templates = db.query(models.SessionTemplate).order_by(models.SessionTemplate.name).all()
    return [
        {
            "id": t.id,
            "name": t.name,
            "exercises": [
                {
                    "exercise_id": te.exercise_id,
                    "name": te.exercise.name,
                    "sets": te.sets,
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
                "exercise": s.exercise.name,
                "set_number": s.set_number,
                "reps": s.reps,
                "weight": s.weight,
            }
            for s in sets
        ],
        "cardio": [_cardio_json(c) for c in sess.cardio],
    }


# ── Cardio helpers ────────────────────────────────────────────────────────────

def _cardio_pace_unit(c: models.CardioActivity) -> str:
    """Distance unit the pace is expressed against (swimming: per 100 m)."""
    return "100m" if (c.activity_type or "").lower() == "swimming" else "km"


def _cardio_pace(c: models.CardioActivity) -> Optional[float]:
    """Minutes per pace unit (per km, or per 100 m for swimming), or None
    when we lack the distance or duration."""
    if not c.distance_km or not c.duration_min:
        return None
    if c.distance_km <= 0:
        return None
    if _cardio_pace_unit(c) == "100m":
        return round(c.duration_min / (c.distance_km * 10), 2)
    return round(c.duration_min / c.distance_km, 2)


def _cardio_json(c: models.CardioActivity) -> dict:
    return {
        "id": c.id,
        "session_id": c.session_id,
        "activity_type": c.activity_type,
        "distance_km": c.distance_km,
        "duration_min": c.duration_min,
        "pace": _cardio_pace(c),
        "pace_unit": _cardio_pace_unit(c),
        "notes": c.notes,
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
    return [dict(_cardio_json(c), date=str(c.session.date)) for c in rows]


@app.get("/api/cardio/{cardio_id}")
async def api_get_cardio(
    cardio_id: int,
    user: models.User = Depends(get_current_user),
    db: Session = Depends(get_db),
):
    c = db.get(models.CardioActivity, cardio_id)
    if not c:
        raise HTTPException(status_code=404, detail="Cardio activity not found")
    return dict(_cardio_json(c), date=str(c.session.date))


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
        get = lambda k, d=None: form.get(k, d)

    activity_type = (get("activity_type") or "").strip().lower()
    if not activity_type:
        raise HTTPException(status_code=400, detail="activity_type is required")

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
    db.delete(c)
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

    return templates.TemplateResponse(request, "browse_exercises.html", {
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
        existing = db.query(models.Exercise).filter(models.Exercise.name == name).first()
        if not existing:
            db.add(models.Exercise(name=name, is_bodyweight=(name in bw_set)))
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
    return templates.TemplateResponse(request, "browse_plans.html", {
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
    plan_id = form.get("plan_id")

    with open(STARTER_PLANS_PATH) as f:
        plans = json.load(f)

    plan = next((p for p in plans if p["id"] == plan_id), None)
    if not plan:
        raise HTTPException(status_code=404, detail="Plan not found")

    # Create or find each exercise
    order = 1
    exercise_ids = []
    for item in plan["exercises"]:
        ex = db.query(models.Exercise).filter(models.Exercise.name == item["name"]).first()
        if not ex:
            ex = models.Exercise(name=item["name"], is_bodyweight=False)
            db.add(ex)
            db.flush()
        exercise_ids.append((ex.id, item["sets"], order))
        order += 1

    # Create template (avoid duplicate names)
    base_name = plan["name"]
    name = base_name
    i = 2
    while db.query(models.SessionTemplate).filter(models.SessionTemplate.name == name).first():
        name = f"{base_name} ({i})"
        i += 1

    tpl = models.SessionTemplate(name=name)
    db.add(tpl)
    db.flush()

    for ex_id, sets, ord_ in exercise_ids:
        db.add(models.SessionTemplateExercise(
            session_template_id=tpl.id,
            exercise_id=ex_id,
            sets=sets,
            order=ord_,
        ))

    db.commit()
    return RedirectResponse(url="/templates?imported=1", status_code=303)
