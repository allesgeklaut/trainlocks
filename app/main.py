from datetime import date
from typing import Generator, Optional
from fastapi import Depends, FastAPI, Form, HTTPException, Request
from fastapi.responses import HTMLResponse, RedirectResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from sqlalchemy.orm import Session
import httpx, json, os
from dotenv import load_dotenv

# Load environment variables from a .env file if present. This allows the
# ``DATABASE_URL`` defined there to be read by ``app.database``.
load_dotenv()

from . import models
from .database import Base, SessionLocal, engine

Base.metadata.create_all(bind=engine)
app = FastAPI(title="Training Log Dashboard")
app.mount("/static", StaticFiles(directory="app/static"), name="static")
templates = Jinja2Templates(directory="app/templates")

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

@app.get("/", response_class=HTMLResponse)
async def index(request: Request, db: Session = Depends(get_db)):
    sessions = db.query(models.WorkoutSession).order_by(models.WorkoutSession.date.desc()).limit(5).all()
    exercises = db.query(models.Exercise).all()
    templates_db = db.query(models.SessionTemplate).all()
    return templates.TemplateResponse("index.html", {
        "request": request,
        "recent_sessions": sessions,
        "exercise_count": len(exercises),
        "template_count": len(templates_db),
        "session_count": db.query(models.WorkoutSession).count(),
    })


@app.get("/exercises", response_class=HTMLResponse)
async def list_exercises(request: Request, db: Session = Depends(get_db)):
    exercises = db.query(models.Exercise).order_by(models.Exercise.name).all()
    return templates.TemplateResponse("exercises.html", {"request": request, "exercises": exercises})


@app.post("/exercises")
async def create_exercise(
    name: str = Form(...),
    is_bodyweight: Optional[str] = Form(None),
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
async def delete_exercise(exercise_id: int, db: Session = Depends(get_db)):
    ex = db.query(models.Exercise).get(exercise_id)
    if not ex:
        raise HTTPException(status_code=404)
    db.delete(ex)
    db.commit()
    return RedirectResponse(url="/exercises", status_code=303)


@app.get("/templates", response_class=HTMLResponse)
async def list_templates(request: Request, db: Session = Depends(get_db)):
    templates_db = db.query(models.SessionTemplate).order_by(models.SessionTemplate.name).all()
    exercises = db.query(models.Exercise).order_by(models.Exercise.name).all()
    return templates.TemplateResponse("templates.html", {
        "request": request, "templates": templates_db, "exercises": exercises
    })


@app.post("/templates")
async def create_template(name: str = Form(...), db: Session = Depends(get_db)):
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
    db: Session = Depends(get_db),
):
    tpl = db.query(models.SessionTemplate).get(template_id)
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
async def remove_template_exercise(template_id: int, te_id: int, db: Session = Depends(get_db)):
    te = db.query(models.SessionTemplateExercise).get(te_id)
    if te:
        db.delete(te)
        db.commit()
    return RedirectResponse(url="/templates", status_code=303)


@app.post("/templates/{template_id}/delete")
async def delete_template(template_id: int, db: Session = Depends(get_db)):
    tpl = db.query(models.SessionTemplate).get(template_id)
    if tpl:
        db.delete(tpl)
        db.commit()
    return RedirectResponse(url="/templates", status_code=303)


@app.get("/sessions", response_class=HTMLResponse)
async def list_sessions(request: Request, db: Session = Depends(get_db)):
    sessions = db.query(models.WorkoutSession).order_by(models.WorkoutSession.date.desc()).all()
    return templates.TemplateResponse("sessions.html", {"request": request, "sessions": sessions})


@app.get("/sessions/new", response_class=HTMLResponse)
async def new_session(request: Request, template_id: Optional[int] = None, db: Session = Depends(get_db)):
    templates_db = db.query(models.SessionTemplate).order_by(models.SessionTemplate.name).all()
    selected_template = db.query(models.SessionTemplate).get(template_id) if template_id else None
    return templates.TemplateResponse("new_session.html", {
        "request": request, "templates": templates_db,
        "selected_template": selected_template, "today": date.today(),
    })


@app.post("/sessions/new")
async def create_session(request: Request, db: Session = Depends(get_db)):
    form = await request.form()
    date_str = form.get("date")
    if not date_str:
        raise HTTPException(status_code=400, detail="Date required")
    workout = models.WorkoutSession(
        date=date.fromisoformat(date_str),
        template_id=int(form["template_id"]) if form.get("template_id") else None,
        notes=form.get("notes") or None,
    )
    db.add(workout)
    db.flush()
    for key, value in form.items():
        if not key.startswith("reps-") or not value:
            continue
        try:
            _, ex_id_str, set_num_str = key.split("-")
            reps = int(value)
        except (ValueError, AttributeError):
            continue
        weight_val = form.get(f"weight-{ex_id_str}-{set_num_str}")
        db.add(models.SetEntry(
            session_id=workout.id,
            exercise_id=int(ex_id_str),
            set_number=int(set_num_str),
            reps=reps,
            weight=float(weight_val) if weight_val else None,
        ))
    db.commit()
    return RedirectResponse(url="/sessions", status_code=303)


@app.get("/progression", response_class=HTMLResponse)
async def progression(request: Request, exercise_id: Optional[int] = None, db: Session = Depends(get_db)):
    exercises = db.query(models.Exercise).order_by(models.Exercise.name).all()
    selected_exercise = db.query(models.Exercise).get(exercise_id) if exercise_id else None
    return templates.TemplateResponse("progression.html", {
        "request": request, "exercises": exercises, "selected_exercise": selected_exercise,
    })


# ── JSON API for chart data ───────────────────────────────────────────────────

@app.get("/api/progression/{exercise_id}")
async def progression_data(exercise_id: int, db: Session = Depends(get_db)):
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
        top_weight = max((s.weight or 0.0) for s in sets)
        volume = sum((s.weight or 0.0) * s.reps for s in sets)
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
    request: Request,
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

    return templates.TemplateResponse("browse_exercises.html", {
        "request": request,
        "exercises": filtered[:200],  # cap at 200 for perf
        "total": len(filtered),
        "all_categories": all_categories,
        "all_muscles": all_muscles,
        "all_equipment": all_equipment,
        "all_levels": all_levels,
        "q": q, "category": category, "muscle": muscle,
        "equipment": equipment, "level": level,
        "error": error,
    })


@app.post("/browse/exercises/import")
async def import_exercises(request: Request, db: Session = Depends(get_db)):
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
async def browse_plans(request: Request, db: Session = Depends(get_db)):
    with open(STARTER_PLANS_PATH) as f:
        plans = json.load(f)
    existing_templates = {t.name for t in db.query(models.SessionTemplate).all()}
    existing_exercises = {e.name for e in db.query(models.Exercise).all()}
    return templates.TemplateResponse("browse_plans.html", {
        "request": request,
        "plans": plans,
        "existing_templates": existing_templates,
        "existing_exercises": existing_exercises,
    })


@app.post("/browse/plans/import")
async def import_plan(
    request: Request,
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
