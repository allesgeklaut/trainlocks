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
    # invalid mm:ss -> 400
    assert client.post("/api/cardio", json={"activity_type": "running", "duration_min": "12:75"}).status_code == 400


def test_api_cardio_duration_minutes_seconds(client):
    resp = client.post("/api/cardio", json={"activity_type": "running", "duration_min": "44:51"})
    assert resp.status_code == 201
    # 44:51 -> 44 + 51/60 minutes
    assert resp.json()["duration_min"] == pytest.approx(44 + 51 / 60)


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
    assert 'value="30"' in html
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


def test_cardio_form_duration_minutes_seconds(client):
    resp = client.post("/cardio", data={
        "activity_type": "Running", "duration_min": "44:51",
    }, follow_redirects=False)
    assert resp.status_code == 303
    resp = client.get("/cardio")
    assert "44:51" in resp.text


def test_cardio_form_duration_validation(client):
    for bad in ("12:75", "abc", "-5"):
        resp = client.post("/cardio", data={
            "activity_type": "running", "duration_min": bad,
        }, follow_redirects=False)
        assert resp.status_code == 400


def test_cardio_edit_prefills_duration_seconds(client):
    client.post("/cardio", data={
        "activity_type": "running", "duration_min": "44:51",
    }, follow_redirects=False)
    resp = client.get("/cardio/1/edit")
    assert 'value="44:51"' in resp.text


def test_dashboard_cardio_load_scales_by_activity(client):
    # 5 km run (x1) + 1 km swim (x3) -> weekly cardio load of 8.0
    client.post("/api/cardio", json=_cardio_payload(activity_type="running", distance_km=5))
    client.post("/api/cardio", json=_cardio_payload(activity_type="swimming", distance_km=1))
    resp = client.get("/")
    assert resp.status_code == 200
    assert '"cardio_load": 8.0' in resp.text
    # ECharts axis title (title-case matches the chart config)
    assert "Cardio load" in resp.text


def test_dashboard_bodyweight_load_scaling(client):
    bench_id, pull_id, tpl_id, resp = _seed(client)
    assert resp.status_code == 201
    # Bench: 80*10 + 82.5*8 = 1460 (recorded weight).
    # Pull-ups (bodyweight, no added weight): 6 reps x 80 default bodyweight = 480.
    resp = client.get("/")
    assert resp.status_code == 200
    assert '"load": 1940.0' in resp.text

    # Adding 10 kg (vest/belt) raises the pull-up set to (80 + 10) x 6 = 540.
    db = SessionLocal()
    sess = db.query(models.WorkoutSession).filter_by(date=date(2026, 8, 25)).first()
    pull_set = [s for s in sess.sets if s.exercise_id == pull_id][0]
    pull_set.weight = 10
    db.commit()
    db.close()
    resp = client.get("/")
    assert resp.status_code == 200
    assert '"load": 2000.0' in resp.text


def test_profile_bodyweight(client):
    bench_id, pull_id, tpl_id, resp = _seed(client)
    assert resp.status_code == 201
    resp = client.get("/profile")
    assert resp.status_code == 200
    assert 'name="bodyweight"' in resp.text

    # Setting bodyweight to 78 changes the pull-up set from 6 x 80 to 6 x 78 = 468.
    resp = client.post("/profile", data={"bodyweight": "78"}, follow_redirects=False)
    assert resp.status_code == 303
    resp = client.get("/profile")
    assert 'value="78.0"' in resp.text
    resp = client.get("/")
    assert '"load": 1928.0' in resp.text


def test_sessions_overview_shows_cardio_notes(client):
    resp = client.post("/api/cardio", json=_cardio_payload(
        activity_type="swimming", notes="swam at the pool, 6x200m"))
    assert resp.status_code == 201
    cardio_id = resp.json()["id"]

    resp = client.get("/sessions")
    assert resp.status_code == 200
    assert "swam at the pool, 6x200m" in resp.text
    assert f"/cardio/{cardio_id}/edit" in resp.text
    # cardio-only session: the row's main Edit button opens the cardio edit
    assert f'<a href="/cardio/{cardio_id}/edit" class="btn-icon" title="Edit">' in resp.text

    resp = client.get("/")
    assert resp.status_code == 200
    assert "swam at the pool, 6x200m" in resp.text


def test_sessions_overview_edit_button_routing(client):
    # Strength session + cardio: main Edit button stays on the session edit.
    bench_id, _, _, resp = _seed(client)
    assert resp.status_code == 201
    session_id = resp.json()["id"]
    client.post("/api/cardio", json=_cardio_payload(date="2026-08-25"))

    resp = client.get("/sessions")
    assert resp.status_code == 200
    assert f'/sessions/edit/{session_id}" class="btn-icon" title="Edit"' in resp.text


def test_view_session_edit_button_routing(client):
    # Cardio-only session: detail page Edit button opens the cardio edit.
    resp = client.post("/api/cardio", json=_cardio_payload(notes="morning tempo run"))
    cardio_id = resp.json()["id"]
    session_id = resp.json()["session_id"]

    resp = client.get(f"/sessions/{session_id}")
    assert resp.status_code == 200
    assert f'<a href="/cardio/{cardio_id}/edit" class="btn btn-primary">Edit</a>' in resp.text


# ---------- Orphaned session pruning ----------

def test_api_cardio_delete_prunes_orphaned_session(client):
    """Deleting the only cardio activity must also remove the session that
    was auto-created for it, instead of leaving an empty row behind."""
    resp = client.post("/api/cardio", json=_cardio_payload())
    body = resp.json()
    session_id = body["session_id"]

    assert client.delete(f"/api/cardio/{body['id']}").status_code == 200
    assert client.get(f"/api/sessions/{session_id}").status_code == 404
    assert client.get("/api/sessions").json() == []


def test_form_cardio_delete_prunes_orphaned_session(client):
    resp = client.post("/api/cardio", json=_cardio_payload())
    body = resp.json()
    session_id = body["session_id"]

    resp = client.post(f"/cardio/{body['id']}/delete", follow_redirects=False)
    assert resp.status_code == 303
    assert client.get(f"/api/sessions/{session_id}").status_code == 404


def test_cardio_delete_keeps_session_with_sets(client):
    """A session created explicitly (with sets) must survive the deletion of
    its cardio activity — pruning only targets auto-created empty ones."""
    bench_id, _, _, resp = _seed(client)
    session_id = resp.json()["id"]

    resp = client.post("/api/cardio", json=_cardio_payload(session_id=session_id))
    cardio_id = resp.json()["id"]
    client.delete(f"/api/cardio/{cardio_id}")

    assert client.get(f"/api/sessions/{session_id}").status_code == 200


def test_form_cardio_delete_keeps_session_with_sets(client):
    bench_id, _, _, resp = _seed(client)
    session_id = resp.json()["id"]

    resp = client.post("/api/cardio", json=_cardio_payload(session_id=session_id))
    cardio_id = resp.json()["id"]
    client.post(f"/cardio/{cardio_id}/delete", follow_redirects=False)

    assert client.get(f"/api/sessions/{session_id}").status_code == 200


def test_cardio_redate_prunes_old_auto_session(client):
    """Re-dating a cardio activity to a day with no other sets must remove
    the now-empty session on the old date and keep the new one."""
    resp = client.post("/api/cardio", json=_cardio_payload(date="2026-08-25"))
    body = resp.json()
    old_session_id = body["session_id"]

    resp = client.post(f"/cardio/{body['id']}", data={
        "activity_type": "running",
        "date": "2026-08-26",
        "distance_km": "5",
        "duration_min": "30",
    }, follow_redirects=False)
    assert resp.status_code == 303

    assert client.get(f"/api/sessions/{old_session_id}").status_code == 404
    sessions = client.get("/api/sessions").json()
    assert len(sessions) == 1
    assert sessions[0]["date"] == "2026-08-26"


def test_cardio_same_date_update_keeps_session(client):
    """Saving an edit without changing the date must not delete the session."""
    resp = client.post("/api/cardio", json=_cardio_payload(date="2026-08-25"))
    body = resp.json()
    session_id = body["session_id"]

    resp = client.post(f"/cardio/{body['id']}", data={
        "activity_type": "running",
        "date": "2026-08-25",
        "distance_km": "6",
        "duration_min": "35",
    }, follow_redirects=False)
    assert resp.status_code == 303
    assert client.get(f"/api/sessions/{session_id}").status_code == 200


# ---------- Validation tightening ----------

def test_cardio_activity_type_allowlist(client):
    # Unknown types are rejected by both the API and the form route.
    assert client.post(
        "/api/cardio", json=_cardio_payload(activity_type="frog jumps")
    ).status_code == 400
    assert client.post(
        "/cardio",
        data={"activity_type": "frog jumps", "distance_km": "5", "duration_min": "30"},
        follow_redirects=False,
    ).status_code == 400
    # Every documented type is accepted; input is case-insensitive.
    for t in ("running", "swimming", "cycling", "walking", "rowing", "other"):
        assert client.post(
            "/api/cardio", json=_cardio_payload(activity_type=t.upper())
        ).status_code == 201


def test_cardio_form_rejects_negative_distance(client):
    """POST /cardio must apply the same distance_km >= 0 check as the API."""
    resp = client.post("/cardio", data={
        "activity_type": "running", "distance_km": "-5", "duration_min": "30",
    }, follow_redirects=False)
    assert resp.status_code == 400


def test_profile_bodyweight_rejects_zero_nan_inf(client):
    """0, NaN and inf are not meaningful bodyweights and would poison the
    load math (0 via the `or` default, NaN by propagating)."""
    for raw in ("0", "0.0", "-1", "nan", "inf", "-inf", "Infinity"):
        resp = client.post("/profile", data={"bodyweight": raw}, follow_redirects=False)
        assert resp.status_code == 400, f"bodyweight={raw!r} was accepted"

    # A valid value still saves.
    resp = client.post("/profile", data={"bodyweight": "72.5"}, follow_redirects=False)
    assert resp.status_code == 303
    assert 'value="72.5"' in client.get("/profile").text


def test_normal_page_loads_are_full_documents(client):
    """Without HX-Request, pages render the full base.html layout."""
    html = client.get("/").text
    assert "<html" in html
    assert 'hx-boost="true"' in html
    assert 'id="app-shell"' in html


def test_htmx_requests_get_shell_fragments(client):
    """Boosted navigation must swap #app-shell only: the response carries
    the shell plus page <title>, but no <html>/<head> chrome, no fixed
    body chrome (topbar/overlay/indicator) and no base scripts — htmx
    would insert everything else at the swap point and duplicate it."""
    resp = client.get("/exercises", headers={"HX-Request": "true"})
    html = resp.text
    assert resp.status_code == 200
    assert "<html" not in html
    assert "<title>" in html and "Exercises" in html
    assert 'id="app-shell"' in html
    assert "mobile-topbar" not in html
    assert "boost-indicator" not in html
    assert "sidebar-overlay" not in html
    # hx-boost itself must not appear in the fragment head — the only
    # occurrence is the logout form's deliberate hx-boost="false" opt-out.
    assert 'hx-boost="true"' not in html
    assert 'hx-boost="false"' in html


def test_htmx_fragments_carry_page_scripts_inside_the_shell(client):
    """Page scripts/styles live inside #app-shell so a swap replaces them
    together with the content instead of accumulating as body children."""
    html = client.get("/", headers={"HX-Request": "true"}).text
    shell_start = html.index('id="app-shell"')
    assert html.rindex("<script>") > shell_start  # count-up script is inside


def test_boosted_redirect_to_login_uses_hx_redirect(client):
    """An expired session behind hx-boost must end in an HX-Redirect so
    htmx performs a full browser navigation to the standalone login page.
    The auth dependency 303s to /login; htmx's XHR transparently follows
    the redirect (re-sending HX-Request), and login_page then answers
    with the header that makes htmx replace the whole page."""
    with TestClient(app) as anon:
        resp = anon.get("/sessions", headers={"HX-Request": "true"},
                        follow_redirects=True)
    assert resp.status_code == 200
    assert resp.headers["HX-Redirect"] == "/login"
