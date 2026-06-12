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

@pytest.fixture(scope="function"},)
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

def test_bodyweight_no_weight_shows_no_load(client):
    """Body‑weight exercise should not show a load of 0 when no weight is added."""
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
    # Since no weight was added, the exercise should not appear in progression.
    assert len(data) == 0