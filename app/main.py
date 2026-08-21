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
from fastapi import Cookie, Depends, FastAPI, Form, HTTPException, Request
from fastapi.responses import HTMLResponse, RedirectResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from sqlalchemy.orm import Session

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
app = FastAPI(title="Training Log Dashboard")
app.mount("/static", StaticFiles(directory="app/static"), name="static")
templates = Jinja2Templates(directory="app/templates")
# Cache-buster for static assets: bumps on every app startup so proxies and
# browsers re-fetch CSS/JS after a rebuild.
STATIC_VERSION = str(int(time.time()))
templates.env.globals["static_version"] = STATIC_VERSION

FREE_EXERCISE_DB_URL = (
    "https://raw.githubusercontent.com/yuhonas/free-exercise-db/main/dist/exercises.json"
)
_exercise_db_cache: list = []

STARTER_PLANS_PATH = os.path.join(os.path.dirname(__file__), "static", "plans", "starter_plans.json")


def get_db() -> Generator[Session, None, None]:
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()


async def fetch_exercise_db() -> list:
    global _exercise_db_cache
    if _exercise_db_cache:
        return _exercise_db_cache
    async with httpx.AsyncClient(timeout=15.0) as client:
        r = await client.get(FREE_EXERCISE_DB_URL)
        r.raise_for_status()
        _exercise_db_cache = r.json()
    return _exercise_db_cache


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
    response.set_cookie(
        key=COOKIE_NAME,
        value=create_session_cookie(user.id),
        httponly=True,
        secure=True,
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

    weekly_data = [
        {"date": k, "load": round(v, 0)}
        for k, v in sorted(weekly_load.items())
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
    
    template_id = int(form["template_id"]) if form.get("template_id") else None
    template = db.get(models.SessionTemplate, template_id) if template_id else None
    
    workout = models.WorkoutSession(
        date=date.fromisoformat(date_str),
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
        def __init__(self, exercises):
            self.exercises = exercises

    # Group sets by exercise and order exercises by their earliest set.
    sets_by_exercise: dict[int, list[models.SetEntry]] = {}
    order_counter = 0
    exercise_order: dict[int, int] = {}
    for se in sorted(sess.sets, key=lambda s: s.set_number):
        if se.exercise_id not in sets_by_exercise:
            sets_by_exercise[se.exercise_id] = []
            exercise_order[se.exercise_id] = order_counter
            order_counter += 1
        sets_by_exercise[se.exercise_id].append(se)

    dummy_exercises = []
    for ex_id, ex_sets in sets_by_exercise.items():
        # `sets` is the count of distinct set_numbers; we render one row per
        # set plus one blank row so the "Add set" button has somewhere to start.
        distinct_set_nums = {s.set_number for s in ex_sets}
        set_count = len(distinct_set_nums)
        dummy_exercises.append(
            type("DummyTE", (), {
                "exercise": ex_sets[0].exercise,
                "sets": set_count,
                "order": exercise_order[ex_id],
            })())
    dummy_template = DummyTemplate(dummy_exercises)

    return templates.TemplateResponse(request, "new_session.html", {
        "templates": [],
        "selected_template": dummy_template,
        "today": sess.date,
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
    sess.date = date.fromisoformat(date_str)
    sess.template_id = int(form["template_id"]) if form.get("template_id") else None
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
            top_weight = max((s.weight or 0.0) for s in weighted_sets)
            volume = sum((s.weight or 0.0) * s.reps for s in weighted_sets)
        elif bw_sets:
            # Bodyweight exercise with no added weight – use reps as metric.
            top_weight = 0.0
            volume = sum(s.reps for s in bw_sets)  # use volume column for total reps when BW
        else:
            continue
        
        total_reps = sum(s.reps for s in sets)
        rows.append({
            "date": str(sess.date),
            "top_weight": round(top_weight, 2),
            "volume": round(volume, 2),
            "total_reps": total_reps,
        })
    return JSONResponse({"data": rows})


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
