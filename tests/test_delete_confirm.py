"""Wiring tests: delete/save confirmations must go through htmx's
htmx:confirm event (fired before the request is issued) instead of
submit-time handlers, which run after htmx has already sent the request."""

import pytest
from fastapi.testclient import TestClient

# DATABASE_URL is set by conftest.py before the app is imported.
from app.main import app
from app.database import Base, SessionLocal, engine
from app import models
from app.auth import COOKIE_NAME, create_session_cookie

from test_json_api import _seed


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


def test_session_delete_uses_htmx_confirm(client):
    _, _, _, resp = _seed(client)
    assert resp.status_code == 201
    session_id = resp.json()["id"]

    resp = client.get(f"/sessions/{session_id}")
    assert resp.status_code == 200
    html = resp.text
    assert 'data-confirm="Delete this session?"' in html
    assert "onsubmit=" not in html


def test_base_layout_uses_htmx_confirm_listener(client):
    resp = client.get("/")
    assert resp.status_code == 200
    html = resp.text
    assert "htmx:confirm" in html
    assert "document.addEventListener('submit'" not in html


def test_exercise_delete_uses_data_confirm(client):
    client.post("/exercises", data={"name": "Deadlift", "is_bodyweight": "0"})
    resp = client.get("/exercises")
    assert resp.status_code == 200
    assert 'data-confirm="Delete Deadlift?"' in resp.text


def test_new_session_guard_uses_htmx_confirm(client):
    resp = client.get("/sessions/new")
    assert resp.status_code == 200
    html = resp.text
    assert "htmx:confirm" in html
    assert "form.addEventListener('submit'" not in html
