import os
import tempfile
from datetime import date
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from app.auth import hash_password, verify_password

# Create a temporary database for testing
test_db_fd, test_db_path = tempfile.mkstemp(suffix=".db")
os.close(test_db_fd)
os.environ["DATABASE_URL"] = f"sqlite:///{test_db_path}"

# Import after setting env var so that the engine is created correctly.
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
    assert row["has_weight"] is True

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


def test_template_session_skips_unfilled_exercises(client):
    """When logging from a template, exercises the user left blank should not
    create 0-rep / None-weight rows in the database."""
    # Create two exercises.
    client.post("/exercises", data={"name": "Bench Press", "is_bodyweight": "0"})
    client.post("/exercises", data={"name": "Rows", "is_bodyweight": "0"})
    db = SessionLocal()
    bench = db.query(models.Exercise).filter_by(name="Bench Press").first()
    rows = db.query(models.Exercise).filter_by(name="Rows").first()

    # Create a template with both exercises, 3 sets each.
    client.post("/templates", data={"name": "Upper A"})
    tpl = db.query(models.SessionTemplate).filter_by(name="Upper A").first()
    client.post(f"/templates/{tpl.id}/add_exercise",
                data={"exercise_id": bench.id, "sets": 3})
    client.post(f"/templates/{tpl.id}/add_exercise",
                data={"exercise_id": rows.id, "sets": 3})

    # Submit a session from the template: fill in Bench sets 1 & 2, leave
    # Bench set 3 and all Rows sets empty (browser sends blank values).
    payload = {
        "date": date.today().isoformat(),
        "template_id": str(tpl.id),
        "notes": "",
        f"reps-{bench.id}-1": "10",
        f"weight-{bench.id}-1": "50",
        f"reps-{bench.id}-2": "8",
        f"weight-{bench.id}-2": "60",
        f"reps-{bench.id}-3": "",
        f"weight-{bench.id}-3": "",
        f"reps-{rows.id}-1": "",
        f"weight-{rows.id}-1": "",
        f"reps-{rows.id}-2": "",
        f"weight-{rows.id}-2": "",
        f"reps-{rows.id}-3": "",
        f"weight-{rows.id}-3": "",
        # Ghost row (set 4) left blank for bench too.
        f"reps-{bench.id}-4": "",
        f"weight-{bench.id}-4": "",
    }
    resp = client.post("/sessions/new", data=payload)
    assert resp.status_code == 200  # follows redirect to /sessions

    sess = db.query(models.WorkoutSession).order_by(models.WorkoutSession.id.desc()).first()
    sets = db.query(models.SetEntry).filter(models.SetEntry.session_id == sess.id).all()
    # Only the two filled-in Bench sets should exist — no 0-rep phantoms.
    assert len(sets) == 2
    bench_sets = sorted(
        db.query(models.SetEntry).filter(
            models.SetEntry.session_id == sess.id,
            models.SetEntry.exercise_id == bench.id
        ).all(), key=lambda s: s.set_number)
    assert [s.set_number for s in bench_sets] == [1, 2]
    assert bench_sets[0].reps == 10 and bench_sets[0].weight == 50.0
    assert bench_sets[1].reps == 8 and bench_sets[1].weight == 60.0
    # No Rows sets at all.
    assert db.query(models.SetEntry).filter(
        models.SetEntry.session_id == sess.id,
        models.SetEntry.exercise_id == rows.id
    ).count() == 0


def test_edit_session_deletes_emptied_set(client):
    """When editing a session, clearing both reps and weight of an existing
    set should delete that row rather than leaving a 0-rep phantom behind."""
    # Create exercise and session with 2 filled sets.
    client.post("/exercises", data={"name": "OHP", "is_bodyweight": "0"})
    db = SessionLocal()
    ex = db.query(models.Exercise).filter_by(name="OHP").first()
    ex_id = ex.id

    payload = {
        "date": date.today().isoformat(),
        "template_id": "",
        f"reps-{ex_id}-1": "10",
        f"weight-{ex_id}-1": "40",
        f"reps-{ex_id}-2": "8",
        f"weight-{ex_id}-2": "45",
    }
    client.post("/sessions/new", data=payload)

    sess = db.query(models.WorkoutSession).order_by(models.WorkoutSession.id.desc()).first()
    session_id = sess.id
    assert db.query(models.SetEntry).filter(
        models.SetEntry.session_id == session_id).count() == 2

    # Edit: keep set 1, blank out set 2 (browser sends empty strings).
    edit_payload = {
        "date": date.today().isoformat(),
        "template_id": "",
        f"reps-{ex_id}-1": "10",
        f"weight-{ex_id}-1": "40",
        f"reps-{ex_id}-2": "",
        f"weight-{ex_id}-2": "",
    }
    resp = client.post(f"/sessions/edit/{session_id}", data=edit_payload)
    assert resp.status_code == 200

    # Only set 1 should remain; set 2 should be deleted, not a 0-rep phantom.
    db.expire_all()
    sets = db.query(models.SetEntry).filter(
        models.SetEntry.session_id == session_id).all()
    assert len(sets) == 1
    assert sets[0].set_number == 1
    assert sets[0].reps == 10
    assert sets[0].weight == 40.0


# ── Gapped set numbers (middle set deleted) ──────────────────────────────

def test_edit_form_with_gapped_set_numbers_preserves_all_sets(client):
    """A session whose set numbers have a gap (middle set deleted) must render
    a row for *each actual* set number, with the ghost row at max+1. A no-op
    save must not delete the rows that sit 'after' the gap.

    Previously the form rendered rows 1..len(distinct), so after deleting set 2
    the form showed rows 1, 2 plus a ghost numbered 3 — the ghost collided with
    real set 3 and a plain save silently deleted set 3's data."""
    client.post("/exercises", data={"name": "Bench Press", "is_bodyweight": "0"})
    db = SessionLocal()
    ex = db.query(models.Exercise).filter_by(name="Bench Press").first()
    ex_id = ex.id

    payload = {
        "date": date.today().isoformat(),
        "template_id": "",
        f"reps-{ex_id}-1": "10", f"weight-{ex_id}-1": "50",
        f"reps-{ex_id}-2": "8",  f"weight-{ex_id}-2": "55",
        f"reps-{ex_id}-3": "6",  f"weight-{ex_id}-3": "60",
    }
    client.post("/sessions/new", data=payload)
    session_id = db.query(models.WorkoutSession).order_by(
        models.WorkoutSession.id.desc()).first().id

    # Delete the middle set (2) via edit.
    client.post(f"/sessions/edit/{session_id}", data={
        "date": date.today().isoformat(),
        "template_id": "",
        f"reps-{ex_id}-1": "10", f"weight-{ex_id}-1": "50",
        f"reps-{ex_id}-2": "",   f"weight-{ex_id}-2": "",
        f"reps-{ex_id}-3": "6",  f"weight-{ex_id}-3": "60",
    })
    db.expire_all()
    nums = sorted(s.set_number for s in db.query(models.SetEntry)
                  .filter(models.SetEntry.session_id == session_id).all())
    assert nums == [1, 3]

    # Open the edit form: it must show rows for set 1 AND set 3, and the ghost
    # row must be numbered 4 (max+1), not collide with the existing set 3.
    html = client.get(f"/sessions/edit/{session_id}").text
    assert f'name="reps-{ex_id}-1"' in html
    assert f'name="reps-{ex_id}-3"' in html
    assert f'name="reps-{ex_id}-4"' in html  # ghost row
    assert f'name="reps-{ex_id}-2"' not in html  # no re-created gap row

    # No-op save (resubmit the exact rows the form rendered, ghost blank) must
    # not delete set 3.
    client.post(f"/sessions/edit/{session_id}", data={
        "date": date.today().isoformat(),
        "template_id": "",
        f"reps-{ex_id}-1": "10", f"weight-{ex_id}-1": "50",
        f"reps-{ex_id}-3": "6",  f"weight-{ex_id}-3": "60",
        f"reps-{ex_id}-4": "",   f"weight-{ex_id}-4": "",
    })
    db.expire_all()
    remaining = db.query(models.SetEntry).filter(
        models.SetEntry.session_id == session_id).all()
    assert sorted(s.set_number for s in remaining) == [1, 3]
    s3 = next(s for s in remaining if s.set_number == 3)
    assert s3.reps == 6 and s3.weight == 60.0


# ── Edit form offers un-logged template exercises ─────────────────────────

def test_edit_form_offers_template_exercises_not_yet_logged(client):
    """When a session was logged from a template but only some of its
    exercises were filled in, the edit form must still show the remaining
    template exercises (with just a ghost row) so they can be added later."""
    client.post("/exercises", data={"name": "Bench Press", "is_bodyweight": "0"})
    client.post("/exercises", data={"name": "Rows", "is_bodyweight": "0"})
    client.post("/templates", data={"name": "Push Day"})
    db = SessionLocal()
    bench = db.query(models.Exercise).filter_by(name="Bench Press").first()
    row = db.query(models.Exercise).filter_by(name="Rows").first()
    tpl = db.query(models.SessionTemplate).filter_by(name="Push Day").first()
    bench_id, row_id, tpl_id = bench.id, row.id, tpl.id
    db.close()

    client.post(f"/templates/{tpl_id}/add_exercise",
                data={"exercise_id": bench_id, "sets": 3})
    client.post(f"/templates/{tpl_id}/add_exercise",
                data={"exercise_id": row_id, "sets": 2})

    # Log only Bench Press from the template.
    client.post("/sessions/new", data={
        "date": date.today().isoformat(),
        "template_id": str(tpl_id),
        f"reps-{bench_id}-1": "10", f"weight-{bench_id}-1": "50",
    })
    db = SessionLocal()
    session_id = db.query(models.WorkoutSession).order_by(
        models.WorkoutSession.id.desc()).first().id
    db.close()

    html = client.get(f"/sessions/edit/{session_id}").text
    # Bench shows its real set 1 plus the ghost row.
    assert f'name="reps-{bench_id}-1"' in html
    assert f'class="ghost-row" name="reps-{bench_id}-2"' in html
    # Rows was not logged yet: it must appear with only a ghost row (1),
    # no fabricated set rows.
    assert f'class="ghost-row" name="reps-{row_id}-1"' in html
    assert f'name="reps-{row_id}-2"' not in html
    # Template order: Bench (1.) before Rows (2.).
    assert html.index("1. Bench Press") < html.index("2. Rows")

    # Add a Rows set through the form (as if typed into the ghost row).
    client.post(f"/sessions/edit/{session_id}", data={
        "date": date.today().isoformat(),
        "template_id": str(tpl_id),
        f"reps-{bench_id}-1": "10", f"weight-{bench_id}-1": "50",
        f"reps-{row_id}-1": "8",  f"weight-{row_id}-1": "40",
    })
    db = SessionLocal()
    row_sets = db.query(models.SetEntry).filter_by(exercise_id=row_id).all()
    assert len(row_sets) == 1
    assert row_sets[0].set_number == 1
    assert row_sets[0].reps == 8 and row_sets[0].weight == 40.0
    bench_sets = db.query(models.SetEntry).filter_by(exercise_id=bench_id).all()
    assert len(bench_sets) == 1 and bench_sets[0].reps == 10
    db.close()


# ── Progression: weight-less sets on a weighted exercise ───────────────────

def test_progression_includes_weightless_session_on_weighted_exercise(client):
    """A weighted exercise logged with reps but no weight (weight forgotten)
    should still appear in progression history, flagged has_weight=False so the
    charts can exclude it — rather than vanishing silently."""
    client.post("/exercises", data={"name": "Bench Press", "is_bodyweight": "0"})
    db = SessionLocal()
    ex = db.query(models.Exercise).filter_by(name="Bench Press").first()
    ex_id = ex.id

    payload = {
        "date": date.today().isoformat(),
        "template_id": "",
        f"reps-{ex_id}-1": "10",  # weight intentionally omitted
    }
    client.post("/sessions/new", data=payload)

    resp = client.get(f"/api/progression/{ex_id}")
    assert resp.status_code == 200
    data = resp.json()["data"]
    assert len(data) == 1
    row = data[0]
    assert row["has_weight"] is False
    assert row["top_weight"] == 0.0
    assert row["volume"] == 0.0
    assert row["total_reps"] == 10


def test_progression_flagged_on_bodyweight_rows(client):
    """Bodyweight rows (reps-as-metric) are chartable and must be flagged
    has_weight=True."""
    client.post("/exercises", data={"name": "Push Ups", "is_bodyweight": "1"})
    db = SessionLocal()
    ex = db.query(models.Exercise).filter_by(name="Push Ups").first()
    ex_id = ex.id
    client.post("/sessions/new", data={
        "date": date.today().isoformat(),
        "template_id": "",
        f"reps-{ex_id}-1": "15",
    })
    row = client.get(f"/api/progression/{ex_id}").json()["data"][0]
    assert row["has_weight"] is True
    assert row["volume"] == 15.0


# ── Deleting an in-use exercise ────────────────────────────────────────────

def test_delete_exercise_used_in_session_is_rejected(client):
    """Deleting an exercise that still has SetEntry rows must be refused (400)
    so we don't orphan rows the UI can no longer address."""
    client.post("/exercises", data={"name": "Squat", "is_bodyweight": "0"})
    db = SessionLocal()
    ex = db.query(models.Exercise).filter_by(name="Squat").first()
    ex_id = ex.id

    client.post("/sessions/new", data={
        "date": date.today().isoformat(),
        "template_id": "",
        f"reps-{ex_id}-1": "5", f"weight-{ex_id}-1": "100",
    })

    resp = client.post(f"/exercises/{ex_id}/delete")
    assert resp.status_code == 400
    db.expire_all()
    assert db.query(models.Exercise).filter_by(id=ex_id).first() is not None


def test_delete_exercise_used_in_template_is_rejected(client):
    """Deleting an exercise referenced by a template must be refused (400)."""
    client.post("/exercises", data={"name": "Rows", "is_bodyweight": "0"})
    client.post("/templates", data={"name": "Pull Day"})
    db = SessionLocal()
    ex = db.query(models.Exercise).filter_by(name="Rows").first()
    ex_id = ex.id
    tpl = db.query(models.SessionTemplate).filter_by(name="Pull Day").first()
    tpl_id = tpl.id

    client.post(f"/templates/{tpl_id}/add_exercise",
                data={"exercise_id": ex_id, "sets": 3})

    resp = client.post(f"/exercises/{ex_id}/delete")
    assert resp.status_code == 400
    db.expire_all()
    assert db.query(models.Exercise).filter_by(id=ex_id).first() is not None


def test_delete_unused_exercise_succeeds(client):
    """An exercise with no referencing rows still deletes fine."""
    client.post("/exercises", data={"name": "Curls", "is_bodyweight": "0"})
    db = SessionLocal()
    ex = db.query(models.Exercise).filter_by(name="Curls").first()
    ex_id = ex.id
    resp = client.post(f"/exercises/{ex_id}/delete")
    assert resp.status_code == 200
    db.expire_all()
    assert db.query(models.Exercise).filter_by(id=ex_id).first() is None


# ── Input validation ───────────────────────────────────────────────────────

def test_create_session_rejects_malformed_date(client):
    resp = client.post("/sessions/new", data={"date": "13/13/2025", "template_id": ""})
    assert resp.status_code == 400


def test_create_session_rejects_malformed_template_id(client):
    resp = client.post("/sessions/new", data={
        "date": date.today().isoformat(), "template_id": "not-an-int"})
    assert resp.status_code == 400


def test_edit_session_rejects_malformed_date(client):
    client.post("/sessions/new", data={
        "date": date.today().isoformat(), "template_id": ""})
    db = SessionLocal()
    sess = db.query(models.WorkoutSession).order_by(
        models.WorkoutSession.id.desc()).first()
    resp = client.post(f"/sessions/edit/{sess.id}", data={"date": "garbage"})
    assert resp.status_code == 400


def test_verify_password_with_overlong_password_returns_false():
    """bcrypt raises ValueError for passwords >72 bytes; verify_password must
    convert that into a plain False instead of a 500."""
    hashed = hash_password("real-password")
    assert verify_password("x" * 100, hashed) is False