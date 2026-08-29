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


# ---------- Cardio API ----------

def _cardio_payload(**overrides):
    payload = {
        "activity_type": "running",
        "distance_km": 5,
        "duration_min": 30,
    }
    payload.update(overrides)
    return payload


def test_api_cardio_create_list_detail_delete(client):
    resp = client.post("/api/cardio", json=_cardio_payload())
    assert resp.status_code == 201
    body = resp.json()
    assert body["activity_type"] == "running"
    assert body["distance_km"] == 5
    assert body["duration_min"] == 30
    assert body["pace"] == pytest.approx(6.0)
    assert body["pace_unit"] == "km"
    assert "id" in body and "session_id" in body

    resp = client.get("/api/cardio")
    assert resp.status_code == 200
    items = resp.json()
    assert len(items) == 1
    assert items[0]["id"] == body["id"]

    resp = client.get(f"/api/cardio/{body['id']}")
    assert resp.status_code == 200
    assert resp.json()["activity_type"] == "running"

    resp = client.delete(f"/api/cardio/{body['id']}")
    assert resp.status_code == 200

    assert client.get("/api/cardio").json() == []
    assert client.get(f"/api/cardio/{body['id']}").status_code == 404
    assert client.delete(f"/api/cardio/{body['id']}").status_code == 404


def test_api_cardio_requires_type_and_values(client):
    assert client.post("/api/cardio", json={}).status_code == 400
    assert client.post("/api/cardio", json={"activity_type": "running"}).status_code == 400
    assert client.post("/api/cardio", json={"activity_type": "", "distance_km": 5}).status_code == 400
    assert client.post("/api/cardio", json={"activity_type": "running", "distance_km": "abc"}).status_code == 400
    assert client.post("/api/cardio", json={"activity_type": "running", "duration_min": -1}).status_code == 400


def test_api_cardio_invalid_date(client):
    resp = client.post("/api/cardio", json=_cardio_payload(date="nope"))
    assert resp.status_code == 400


def test_api_cardio_requires_session(client):
    # no session exists yet for today; cardio create must find-or-create it
    before = client.get("/api/sessions").json()
    client.post("/api/cardio", json=_cardio_payload())
    after = client.get("/api/sessions").json()
    assert len(after) == len(before) + 1


def test_api_cardio_attached_to_session_by_date(client):
    before = len(client.get("/api/sessions").json())
    resp = client.post("/api/cardio", json=_cardio_payload(distance_km=1, duration_min=20))
    first_id = resp.json()["session_id"]
    resp = client.post("/api/cardio", json=_cardio_payload(activity_type="swimming", distance_km=1, duration_min=20))
    second_id = resp.json()["session_id"]
    # same date -> same underlying session
    assert first_id == second_id
    assert len(client.get("/api/sessions").json()) == before + 1

    resp = client.get(f"/api/sessions/{first_id}")
    assert resp.status_code == 200
    cardio = resp.json()["cardio"]
    assert [a["activity_type"] for a in cardio] == ["running", "swimming"]


def test_api_cardio_swimming_pace_per_100m(client):
    resp = client.post("/api/cardio", json=_cardio_payload(activity_type="swimming", distance_km=1, duration_min=20))
    assert resp.status_code == 201
    body = resp.json()
    assert body["activity_type"] == "swimming"
    # 1 km in 20 min => 10 x 100m => 2.0 min per 100m
    assert body["pace"] == pytest.approx(2.0)
    assert body["pace_unit"] == "100m"
    # detail endpoint agrees
    detail = client.get(f"/api/cardio/{body['id']}").json()
    assert detail["pace"] == pytest.approx(2.0)
    assert detail["pace_unit"] == "100m"


def test_api_session_list_includes_cardio(client):
    client.post("/api/cardio", json=_cardio_payload())
    resp = client.get("/api/sessions")
    assert resp.status_code == 200
    sessions = resp.json()
    assert len(sessions) == 1
    assert len(sessions[0]["cardio"]) == 1
    assert sessions[0]["cardio"][0]["activity_type"] == "running"


def test_api_cardio_bad_delete(client):
    assert client.delete("/api/cardio/99999").status_code == 404


def test_cardio_edit_prefills_existing_values(client):
    resp = client.post("/api/cardio", json=_cardio_payload(notes="morning tempo run"))
    assert resp.status_code == 201
    cardio_id = resp.json()["id"]

    resp = client.get(f"/cardio/{cardio_id}/edit")
    assert resp.status_code == 200
    html = resp.text
    assert "morning tempo run" in html
    assert 'value="5.0"' in html
    assert 'value="30.0"' in html
    assert '<option value="running" selected' in html


def test_cardio_edit_saves_without_losing_notes(client):
    resp = client.post("/api/cardio", json=_cardio_payload(notes="morning tempo run"))
    cardio_id = resp.json()["id"]

    resp = client.post(
        f"/cardio/{cardio_id}",
        data={
            "activity_type": "running",
            "date": "2026-08-29",
            "distance_km": "5",
            "duration_min": "30",
            "notes": "morning tempo run, felt easy",
        },
        follow_redirects=False,
    )
    assert resp.status_code == 303
    assert resp.headers["location"] == "/cardio"

    detail = client.get(f"/api/cardio/{cardio_id}").json()
    assert detail["notes"] == "morning tempo run, felt easy"
    assert detail["distance_km"] == 5
    assert detail["duration_min"] == 30
    assert detail["activity_type"] == "running"


def test_cardio_edit_bad_id_and_validation(client):
    assert client.get("/cardio/99999/edit").status_code == 404
    assert client.post("/cardio/99999", data={}).status_code == 404
    resp = client.post("/api/cardio", json=_cardio_payload())
    cardio_id = resp.json()["id"]
    # no distance and no duration -> 400, existing values untouched
    assert client.post(f"/cardio/{cardio_id}", data={
        "activity_type": "running",
    }).status_code == 400
    assert client.get(f"/api/cardio/{cardio_id}").json()["notes"] is None


def test_sessions_overview_shows_cardio_notes(client):
    resp = client.post("/api/cardio", json=_cardio_payload(
        activity_type="swimming", notes="swam at the pool, 6x200m"))
    assert resp.status_code == 201
    cardio_id = resp.json()["id"]

    resp = client.get("/sessions")
    assert resp.status_code == 200
    assert "swam at the pool, 6x200m" in resp.text
    assert f"/cardio/{cardio_id}/edit" in resp.text

    resp = client.get("/")
    assert resp.status_code == 200
    assert "swam at the pool, 6x200m" in resp.text
    assert f"/cardio/{cardio_id}/edit" in resp.text
