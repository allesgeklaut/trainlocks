from datetime import date

import pytest
from fastapi.testclient import TestClient

# DATABASE_URL is set by conftest.py before the app is imported.
from app.main import app
from app.database import Base, SessionLocal, engine
from app import models
from app.auth import COOKIE_NAME, create_session_cookie


@pytest.fixture(scope="function")
def client():
    # Create tables for the test database.
    Base.metadata.create_all(bind=engine)
    # Seed a user and build a signed session cookie so authenticated
    # endpoints don't redirect to /login.
    db = SessionLocal()
    if not db.query(models.User).filter_by(username="tester").first():
        db.add(models.User(username="tester",
                           hashed_password="$2b$12$" + "x" * 53))
        db.commit()
    cookie = create_session_cookie(1)
    with TestClient(app) as c:
        c.cookies.set(COOKIE_NAME, cookie)
        yield c
    # Drop tables after tests.
    Base.metadata.drop_all(bind=engine)


def _seed(client):
    """Create the exercises, template and session used by several tests."""
    client.post("/exercises", data={"name": "Bench Press", "is_bodyweight": "0"})
    client.post("/exercises", data={"name": "Pull Ups", "is_bodyweight": "1"})
    client.post("/templates", data={"name": "Upper Body"})
    db = SessionLocal()
    bench = db.query(models.Exercise).filter_by(name="Bench Press").first()
    pull = db.query(models.Exercise).filter_by(name="Pull Ups").first()
    tpl = db.query(models.SessionTemplate).filter_by(name="Upper Body").first()
    db.add(models.SessionTemplateExercise(
        session_template_id=tpl.id, exercise_id=bench.id, sets=5, order=1))
    db.add(models.SessionTemplateExercise(
        session_template_id=tpl.id, exercise_id=pull.id, sets=3, order=2))
    db.commit()
    payload = {
        "date": "2026-08-25",
        "template_id": str(tpl.id),
        "notes": "API test session",
        f"reps-{bench.id}-1": "10",
        f"weight-{bench.id}-1": "80",
        f"reps-{bench.id}-2": "8",
        f"weight-{bench.id}-2": "82.5",
        f"reps-{pull.id}-1": "6",
    }
    resp = client.post("/sessions/new", data=payload,
                       headers={"Accept": "application/json"})
    return bench.id, pull.id, tpl.id, resp


def test_unauthenticated_api_returns_json_401():
    with TestClient(app) as anon:
        resp = anon.get("/api/sessions", follow_redirects=False)
    assert resp.status_code == 401
    assert resp.headers["content-type"].startswith("application/json")
    assert resp.json()["detail"] == "Not authenticated"


def test_unauthenticated_page_still_redirects_to_login():
    with TestClient(app) as anon:
        resp = anon.get("/sessions", follow_redirects=False)
    assert resp.status_code == 303
    assert resp.headers["location"] == "/login"


def test_api_exercises(client):
    client.post("/exercises", data={"name": "Bench Press", "is_bodyweight": "0"})
    client.post("/exercises", data={"name": "Pull Ups", "is_bodyweight": "1"})
    resp = client.get("/api/exercises")
    assert resp.status_code == 200
    data = resp.json()
    assert [e["name"] for e in data] == ["Bench Press", "Pull Ups"]
    assert data[0]["is_bodyweight"] is False
    assert data[1]["is_bodyweight"] is True
    assert all(isinstance(e["id"], int) for e in data)


def test_api_templates(client):
    client.post("/exercises", data={"name": "Bench Press", "is_bodyweight": "0"})
    client.post("/templates", data={"name": "Upper Body"})
    db = SessionLocal()
    bench = db.query(models.Exercise).filter_by(name="Bench Press").first()
    tpl = db.query(models.SessionTemplate).filter_by(name="Upper Body").first()
    db.add(models.SessionTemplateExercise(
        session_template_id=tpl.id, exercise_id=bench.id, sets=5, order=1))
    db.commit()
    resp = client.get("/api/templates")
    assert resp.status_code == 200
    data = resp.json()
    assert len(data) == 1
    t = data[0]
    assert t["name"] == "Upper Body"
    assert t["exercises"] == [{
        "exercise_id": bench.id, "name": "Bench Press", "sets": 5, "order": 1,
    }]


def test_api_sessions_list_and_detail(client):
    bench_id, pull_id, tpl_id, resp = _seed(client)
    assert resp.status_code == 201
    session_id = resp.json()["id"]

    resp = client.get("/api/sessions")
    assert resp.status_code == 200
    sessions = resp.json()
    assert len(sessions) == 1
    s = sessions[0]
    assert s["id"] == session_id
    assert s["date"] == "2026-08-25"
    assert s["template_id"] == tpl_id
    assert s["template_name"] == "Upper Body"
    assert s["notes"] == "API test session"

    resp = client.get(f"/api/sessions/{session_id}")
    assert resp.status_code == 200
    detail = resp.json()
    assert detail["id"] == session_id
    assert detail["template_name"] == "Upper Body"
    assert len(detail["sets"]) == 3
    bench_sets = [x for x in detail["sets"] if x["exercise_id"] == bench_id]
    assert [(x["set_number"], x["reps"], x["weight"]) for x in bench_sets] == [
        (1, 10, 80.0), (2, 8, 82.5),
    ]
    pull_sets = [x for x in detail["sets"] if x["exercise_id"] == pull_id]
    assert pull_sets[0] == {
        "exercise_id": pull_id, "exercise": "Pull Ups",
        "set_number": 1, "reps": 6, "weight": None,
    }


def test_api_session_not_found(client):
    resp = client.get("/api/sessions/99999")
    assert resp.status_code == 404


def test_create_session_json_for_api_clients(client):
    bench_id, _, _, resp = _seed(client)
    assert resp.status_code == 201
    body = resp.json()
    assert body["date"] == "2026-08-25"
    assert isinstance(body["id"], int)


def test_create_session_browser_still_redirects(client):
    client.post("/exercises", data={"name": "Bench Press", "is_bodyweight": "0"})
    db = SessionLocal()
    bench = db.query(models.Exercise).filter_by(name="Bench Press").first()
    db.close()
    resp = client.post("/sessions/new", data={
        "date": date.today().isoformat(),
        f"reps-{bench.id}-1": "5",
    }, follow_redirects=False)
    assert resp.status_code == 303
    assert resp.headers["location"] == "/sessions"
