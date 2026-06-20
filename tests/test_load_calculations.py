import os
import tempfile
from datetime import date
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

# Create a temporary database for testing
test_db_fd, test_db_path = tempfile.mkstemp(suffix=".db")
os.close(test_db_fd)
os.environ["DATABASE_URL"] = f"sqlite:///{test_db_path}"

# Import after setting env var so that the engine is created correctly.
from app.main import app
from app.database import Base, SessionLocal, engine
from app import models

@pytest.fixture(scope="function")
def client():
    # Create tables for the test database.
    Base.metadata.create_all(bind=engine)
    with TestClient(app) as c:
        yield c
    # Drop tables after tests.
    Base.metadata.drop_all(bind=engine)

def teardown_module():
    """Clean up the test database file after all tests."""
    if os.path.exists(test_db_path):
        try:
            os.remove(test_db_path)
        except OSError:
            pass

def test_load_calculations_with_varied_weights(client):
    """Create a session with multiple sets of different weights and verify
    the progression API returns correct top weight, volume and total reps."""
    # Create exercise.
    client.post("/exercises", data={"name": "Bench Press", "is_bodyweight": "0"})
    # Retrieve the newly created exercise from the database.
    db = SessionLocal()
    ex = db.query(models.Exercise).filter_by(name="Bench Press" ).first()
    ex_id = ex.id

    # Create a session.
    payload = {
        "date": date.today().isoformat(),
        "template_id": "",
        "notes": "Test session",
        f"reps-{ex_id}-1": "10",
        f"weight-{ex_id}-1": "50",
        f"reps-{ex_id}-2": "8",
        f"weight-{ex_id}-2": "60",
        f"reps-{ex_id}-3": "6",
        f"weight-{ex_id}-3": "70",
    }
    client.post("/sessions/new", data=payload)

    # Get progression data.
    resp = client.get(f"/api/progression/{ex_id}")
    assert resp.status_code == 200
    data = resp.json()["data"]
    assert len(data) == 1
    row = data[0]
    assert row["top_weight"] == 70.0
    # Volume: (50*10)+(60*8)+(70*6)=500+480+420=1400
    assert row["volume"] == 1400.0
    assert row["total_reps"] == 24

def test_bodyweight_no_weight_shows_rep_count(client):
    """Body‑weight exercise with no weight should show rep count in volume and total_reps."""
    # Create body‑weight exercise.
    client.post("/exercises", data={"name": "Push Ups", "is_bodyweight": "1"})
    db = SessionLocal()
    ex = db.query(models.Exercise).filter_by(name="Push Ups" ).first()
    ex_id = ex.id

    # Create a session with reps but no weight.
    payload = {
        "date": date.today().isoformat(),
        "template_id": "",
        "notes": "BW test",
        f"reps-{ex_id}-1": "15",
        # No weight field
    }
    client.post("/sessions/new", data=payload)

    resp = client.get(f"/api/progression/{ex_id}")
    assert resp.status_code == 200
    data = resp.json()["data"]
    # Bodyweight sets with no weight should appear with reps in volume and total_reps.
    assert len(data) == 1
    row = data[0]
    assert row["top_weight"] == 0.0
    assert row["volume"] == 15.0
    assert row["total_reps"] == 15


# ── Edit Session Tests ────────────────────────────────────────────────────

def test_edit_form_pre_populates_reps_and_weight(client):
    """When editing a session, the form should pre-populate with existing reps and weight."""
    # Create exercise and session.
    client.post("/exercises", data={"name": "Squat", "is_bodyweight": "0"})
    db = SessionLocal()
    ex = db.query(models.Exercise).filter_by(name="Squat").first()
    ex_id = ex.id

    payload = {
        "date": date.today().isoformat(),
        "template_id": "",
        "notes": "Test session",
        f"reps-{ex_id}-1": "10",
        f"weight-{ex_id}-1": "100",
        f"reps-{ex_id}-2": "8",
        f"weight-{ex_id}-2": "105",
    }
    client.post("/sessions/new", data=payload)

    # Get the session ID.
    sess = db.query(models.WorkoutSession).order_by(models.WorkoutSession.id.desc()).first()
    session_id = sess.id

    # Open the edit form.
    resp = client.get(f"/sessions/edit/{session_id}")
    assert resp.status_code == 200
    html = resp.text

    # Verify that the input values are pre-populated in the HTML.
    assert f'name="reps-{ex_id}-1"' in html
    assert f'value="10"' in html
    assert f'name="weight-{ex_id}-1"' in html
    assert f'value="100' in html  # float as 100.0 or 100
    assert f'name="reps-{ex_id}-2"' in html
    assert f'value="8"' in html
    assert f'name="weight-{ex_id}-2"' in html
    assert f'value="105' in html


def test_edit_session_upserts_does_not_wipe_unrelated_rows(client):
    """When editing a session, only submitted rows should be updated/created; unsubmitted rows should be preserved."""
    # Create exercise and session with 3 sets.
    client.post("/exercises", data={"name": "Deadlift", "is_bodyweight": "0"})
    db = SessionLocal()
    ex = db.query(models.Exercise).filter_by(name="Deadlift").first()
    ex_id = ex.id

    payload = {
        "date": date.today().isoformat(),
        "template_id": "",
        f"reps-{ex_id}-1": "5",
        f"weight-{ex_id}-1": "150",
        f"reps-{ex_id}-2": "5",
        f"weight-{ex_id}-2": "150",
        f"reps-{ex_id}-3": "5",
        f"weight-{ex_id}-3": "150",
    }
    client.post("/sessions/new", data=payload)

    sess = db.query(models.WorkoutSession).order_by(models.WorkoutSession.id.desc()).first()
    session_id = sess.id

    # Edit: only update set 1 and 2, don't touch set 3.
    edit_payload = {
        "date": date.today().isoformat(),
        "template_id": "",
        f"reps-{ex_id}-1": "6",
        f"weight-{ex_id}-1": "160",
        f"reps-{ex_id}-2": "6",
        f"weight-{ex_id}-2": "160",
        # Set 3 not submitted
    }
    resp = client.post(f"/sessions/edit/{session_id}", data=edit_payload)
    # TestClient follows redirects by default, so we should get a 200 on the /sessions page
    assert resp.status_code == 200

    # Verify all 3 sets still exist in the database.
    db.expire_all()
    sets = db.query(models.SetEntry).filter(models.SetEntry.session_id == session_id).all()
    assert len(sets) == 3

    # Verify sets 1 and 2 are updated.
    s1 = db.query(models.SetEntry).filter(
        models.SetEntry.session_id == session_id,
        models.SetEntry.set_number == 1
    ).first()
    assert s1.reps == 6
    assert s1.weight == 160.0

    s2 = db.query(models.SetEntry).filter(
        models.SetEntry.session_id == session_id,
        models.SetEntry.set_number == 2
    ).first()
    assert s2.reps == 6
    assert s2.weight == 160.0

    # Verify set 3 is unchanged.
    s3 = db.query(models.SetEntry).filter(
        models.SetEntry.session_id == session_id,
        models.SetEntry.set_number == 3
    ).first()
    assert s3.reps == 5
    assert s3.weight == 150.0


# The following test was removed because the application no longer supports a remove checkbox.


def test_edit_session_adds_new_set(client):
    """When a new set row is submitted (via the 'Add set' button), it should be persisted."""
    # Create exercise and session with 1 set.
    client.post("/exercises", data={"name": "Leg Press", "is_bodyweight": "0"})
    db = SessionLocal()
    ex = db.query(models.Exercise).filter_by(name="Leg Press").first()
    ex_id = ex.id

    payload = {
        "date": date.today().isoformat(),
        "template_id": "",
        f"reps-{ex_id}-1": "12",
        f"weight-{ex_id}-1": "200",
    }
    client.post("/sessions/new", data=payload)

    sess = db.query(models.WorkoutSession).order_by(models.WorkoutSession.id.desc()).first()
    session_id = sess.id

    # Edit: keep set 1, and add a new set 2 (simulating the "Add set" button).
    edit_payload = {
        "date": date.today().isoformat(),
        "template_id": "",
        f"reps-{ex_id}-1": "12",
        f"weight-{ex_id}-1": "200",
        f"reps-{ex_id}-2": "10",
        f"weight-{ex_id}-2": "210",
    }
    resp = client.post(f"/sessions/edit/{session_id}", data=edit_payload)
    # TestClient follows redirects by default, so we should get a 200 on the /sessions page
    assert resp.status_code == 200

    # Verify both sets exist.
    db.expire_all()
    sets = db.query(models.SetEntry).filter(models.SetEntry.session_id == session_id).all()
    assert len(sets) == 2

    s2 = db.query(models.SetEntry).filter(
        models.SetEntry.session_id == session_id,
        models.SetEntry.set_number == 2
    ).first()
    assert s2 is not None
    assert s2.reps == 10
    assert s2.weight == 210.0