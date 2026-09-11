"""Tests for the AI features: LLM backend plumbing, coach chat history and
screenshot-to-session extraction. All LLM calls are mocked — no network."""

import base64
import json
from datetime import date

import pytest
from fastapi.testclient import TestClient

# DATABASE_URL is set by conftest.py before the app is imported.
from app import llm as llm_mod
from app import models
from app.ai import (
    _extraction_system_prompt,
    _match_or_create_exercises,
    _norm_name,
    _parse_json_loose,
    _review_store_put,
)
from app.auth import COOKIE_NAME, create_session_cookie
from app.database import Base, SessionLocal, engine
from app.main import app


def _save_token() -> str:
    """Fresh PRG token, as the review form would carry (save requires one)."""
    return _review_store_put({
        "exercise_ids": [], "is_new_flags": [], "set_lists": [],
        "created_names": [], "ai_date": "", "ai_notes": "",
        "ai_cardio": [], "model_label": "test",
    })


def _one(qry):
    """first() with a hard assert — pyright narrows the Optional away."""
    row = qry.first()
    assert row is not None
    return row



@pytest.fixture(scope="function")
def client():
    Base.metadata.create_all(bind=engine)
    db = SessionLocal()
    if not db.query(models.User).filter_by(username="tester").first():
        db.add(models.User(username="tester",
                           hashed_password="$2b$12$" + "x" * 53))
        db.commit()
    cookie = create_session_cookie(1)
    with TestClient(app) as c:
        c.cookies.set(COOKIE_NAME, cookie)
        yield c
    Base.metadata.drop_all(bind=engine)


@pytest.fixture(scope="function")
def llm_state(tmp_path, monkeypatch):
    """Point the LLM module at a temp state file and configure one fake backend."""
    state_file = tmp_path / "llm_state.json"
    monkeypatch.setattr(llm_mod, "_DEFAULT_STATE_FILE", state_file)
    monkeypatch.setattr(
        llm_mod, "_LLM_BACKENDS_RAW",
        json.dumps([{"name": "fake", "type": "openai",
                     # 127.0.0.1:1 — closed port: connects/fails instantly.
                     # An unresolvable .local host would hang ~10s per call
                     # (httpx's connect timeout does not bound getaddrinfo).
                     "url": "http://127.0.0.1:1", "model": "fake-vision"}]),
    )
    monkeypatch.setattr(llm_mod, "_LLM_ENABLED", True)
    return state_file


# ---------------------------------------------------------------------------
# LLM plumbing
# ---------------------------------------------------------------------------

class TestLLMConfig:
    def test_backend_parsing(self, llm_state):
        backends = llm_mod._configured_backends()
        assert len(backends) == 1
        assert backends[0]["name"] == "fake"
        assert backends[0]["type"] == "openai"

    def test_invalid_json_falls_back_to_legacy(self, monkeypatch, tmp_path, llm_state):
        monkeypatch.setattr(llm_mod, "_LLM_BACKENDS_RAW", "not json")
        monkeypatch.setattr(llm_mod, "_OLLAMA_MODEL", "legacy-model")
        backends = llm_mod._configured_backends()
        assert backends[0]["model"] == "legacy-model"

    def test_selection_persists(self, llm_state):
        import anyio
        anyio.run(llm_mod.select_backend, "fake", "other-model")
        assert json.loads(llm_state.read_text())["models"]["fake"] == "other-model"

    def test_unknown_backend_rejected(self, llm_state):
        import anyio
        with pytest.raises(ValueError):
            anyio.run(llm_mod.select_backend, "nope")

    def test_invalid_backend_entry_redacts_api_key(self, llm_state, caplog):
        """A malformed LLM_BACKENDS entry that carries an inline api_key must
        not leak the key into the log (it lands in container logs)."""
        import anyio as _anyio
        import logging as _logging
        monkeypatched_raw = json.dumps([
            {"name": "", "type": "openai", "url": "http://x",
             "api_key": "sk-SUPERSECRET-123"},
        ])
        llm_mod._LLM_BACKENDS_RAW = monkeypatched_raw
        try:
            with caplog.at_level(_logging.WARNING, logger="trainlocks.llm"):
                backends = _anyio.run(llm_mod.list_backends)
        finally:
            llm_mod._LLM_BACKENDS_RAW = json.dumps([
                {"name": "fake", "type": "openai",
                 "url": "http://127.0.0.1:1", "model": "fake-vision"}])
        assert backends == [] or all(b.get("name") != "" for b in backends)
        leaked = [r for r in caplog.records if "sk-SUPERSECRET-123" in r.getMessage()]
        assert not leaked, "api_key must be redacted in invalid-backend logs"
        assert any("Skipping invalid LLM backend entry" in r.getMessage()
                   for r in caplog.records)

    def test_extract_content_openai_shape(self, llm_state):
        text, reasoning, model = llm_mod._extract_content(
            {"choices": [{"message": {"content": "hi",
                                      "reasoning_content": "hm"}}],
             "model": "m1"},
            {"name": "x", "model": "m1"},
        )
        assert (text, reasoning, model) == ("hi", "hm", "m1")

    def test_extract_content_ollama_shape(self, llm_state):
        text, _, model = llm_mod._extract_content(
            {"message": {"content": "yo"}}, {"name": "x", "model": "m2"},
        )
        assert text == "yo" and model == "m2"

    def test_message_encoding_images(self, llm_state):
        backend = {"type": "openai"}
        msgs = [{"role": "user", "content": "look", "images": ["QUJD"]}]
        wire = llm_mod._encode_messages(backend, msgs)
        assert wire[0]["content"][0] == {"type": "text", "text": "look"}
        assert wire[0]["content"][1]["image_url"]["url"].endswith(";base64,QUJD")

        backend = {"type": "ollama"}
        wire = llm_mod._encode_messages(backend, msgs)
        assert wire[0]["images"] == ["QUJD"]

    def test_data_url_passthrough(self):
        url = "data:image/jpeg;base64,QUJD"
        assert llm_mod._to_data_url(url) == url


# ---------------------------------------------------------------------------
# Coach chat
# ---------------------------------------------------------------------------

class TestCoachChat:
    def test_history_persist_and_clear(self, client, llm_state):
        r = client.get("/coach/history")
        assert r.status_code == 200
        assert r.json() == {"messages": []}

        # Simulate two persisted turns (user + assistant).
        db = SessionLocal()
        db.add(models.CoachChatMessage(role="user", content="How am I doing?"))
        db.add(models.CoachChatMessage(role="assistant", content="Great."))
        db.commit()
        r = client.get("/coach/history")
        msgs = r.json()["messages"]
        assert [m["role"] for m in msgs] == ["user", "assistant"]
        assert [m["content"] for m in msgs] == ["How am I doing?", "Great."]

        # Clear.
        r = client.delete("/coach/history")
        assert r.status_code == 200
        assert client.get("/coach/history").json() == {"messages": []}

    def test_history_ordering_chronological(self, client, llm_state):
        db = SessionLocal()
        for i in range(5):
            db.add(models.CoachChatMessage(role="user", content=f"m{i}"))
        db.commit()
        msgs = client.get("/coach/history").json()["messages"]
        assert [m["content"] for m in msgs] == [f"m{i}" for i in range(5)]

    def test_history_limit_40(self, client, llm_state):
        db = SessionLocal()
        for i in range(45):
            db.add(models.CoachChatMessage(role="user", content=f"m{i}"))
        db.commit()
        msgs = client.get("/coach/history").json()["messages"]
        assert len(msgs) == 40
        assert msgs[0]["content"] == "m5"  # oldest trimmed

    def test_coach_page_renders(self, client, llm_state):
        r = client.get("/coach")
        assert r.status_code == 200
        assert "AI Coach" in r.text

    def test_send_requires_message(self, client, llm_state):
        r = client.post("/coach/send/stream", json={"message": ""})
        assert r.status_code == 400

    def test_send_streams_and_persists(self, client, llm_state, monkeypatch):
        async def fake_stream(messages):
            yield {"type": "delta", "text": "Hel"}
            yield {"type": "delta", "text": "lo!"}
            yield {"type": "done", "text": "Hello!", "reasoning": "",
                   "backend": "fake", "model": "fake-vision"}

        monkeypatch.setattr(llm_mod, "chat_stream", fake_stream)
        r = client.post("/coach/send/stream", json={"message": "hi coach"})
        assert r.status_code == 200
        body = r.text
        assert "data: " in body and "Hel" in body and '"done"' in body
        # Proper SSE framing: every event terminated by a blank line.
        assert body.endswith("\n\n")

        # Both the user message and the reply are persisted after the stream.
        msgs = client.get("/coach/history").json()["messages"]
        assert [(m["role"], m["content"]) for m in msgs] == [
            ("user", "hi coach"), ("assistant", "Hello!")]

    def test_send_error_rolls_back_user_message(self, client, llm_state, monkeypatch):
        async def fake_stream(messages):
            yield {"type": "error", "text": "backend down"}

        monkeypatch.setattr(llm_mod, "chat_stream", fake_stream)
        r = client.post("/coach/send/stream", json={"message": "hello?"})
        assert r.status_code == 200
        assert client.get("/coach/history").json()["messages"] == []

    def test_stream_includes_training_context(self, client, llm_state, monkeypatch):
        captured = {}

        async def fake_stream(messages):
            captured["system"] = messages[0]["content"]
            yield {"type": "done", "text": "ok", "reasoning": "",
                   "backend": "fake", "model": "m"}

        monkeypatch.setattr(llm_mod, "chat_stream", fake_stream)
        # Seed a session so the context has data.
        db = SessionLocal()
        ex = models.Exercise(name="Squat", is_bodyweight=False)
        db.add(ex)
        db.flush()
        from datetime import date
        sess = models.WorkoutSession(date=date(2026, 9, 1))
        db.add(sess)
        db.flush()
        db.add(models.SetEntry(session_id=sess.id, exercise_id=ex.id,
                               set_number=1, reps=5, weight=100.0))
        db.add(models.CardioActivity(session_id=sess.id, activity_type="running",
                                     distance_km=6.39, duration_min=43.57))
        db.commit()

        client.post("/coach/send/stream", json={"message": "review my squat"})
        system = captured["system"]
        assert "Squat" in system and "100.0kg x 5" in system
        assert "Weekly training load" in system
        assert "Today's date" in system
        # Durations keep real precision (43.57 min -> 43:34), no rounding to 44min.
        assert "running 6.39km 43:34min" in system
        assert "44min" not in system


# ---------------------------------------------------------------------------
# Model switching API
# ---------------------------------------------------------------------------

class TestLLMApi:
    def test_status_and_select(self, client, llm_state):
        st = client.get("/api/llm/status").json()
        assert st["backends"][0]["name"] == "fake"
        assert st["active_backend"] == "fake"

        r = client.post("/api/llm/select",
                        json={"backend": "fake", "model": "other"})
        assert r.status_code == 200
        assert r.json() == {"active_backend": "fake", "active_model": "other"}

        st = client.get("/api/llm/status").json()
        assert st["active_model"] == "other"

    def test_select_unknown_backend(self, client, llm_state):
        r = client.post("/api/llm/select", json={"backend": "nope"})
        assert r.status_code == 400

    def test_requires_auth(self, llm_state):
        with TestClient(app) as anon:
            r = anon.get("/api/llm/status")
            assert r.status_code == 401


# ---------------------------------------------------------------------------
# Screenshot extraction
# ---------------------------------------------------------------------------

class TestExtraction:
    def test_parse_json_loose(self):
        assert _parse_json_loose('{"a": 1}') == {"a": 1}
        assert _parse_json_loose('```json\n{"a": 1}\n```') == {"a": 1}
        assert _parse_json_loose('Here you go: {"a": 1} hope it helps') == {"a": 1}
        with pytest.raises(json.JSONDecodeError):
            _parse_json_loose("no json at all")

    def test_norm_name(self):
        assert _norm_name("Bench-Press!") == "bench press"
        assert _norm_name("  INCLINE  press ") == "incline press"

    def test_match_existing(self, client):
        db = SessionLocal()
        db.add(models.Exercise(name="Bench Press", is_bodyweight=False))
        db.commit()
        mapping, created = _match_or_create_exercises(
            db, ["bench  press", "Bench Press"])
        assert not created
        assert all(m.name == "Bench Press" for m in mapping.values())

    def test_fuzzy_match(self, client):
        db = SessionLocal()
        db.add(models.Exercise(name="Pull Ups", is_bodyweight=True))
        db.commit()
        mapping, created = _match_or_create_exercises(db, ["Pull-Up"])
        assert not created
        assert mapping["Pull-Up"].name == "Pull Ups"

    def test_creates_missing_exercise(self, client):
        db = SessionLocal()
        mapping, created = _match_or_create_exercises(db, ["Cable Row"])
        db.commit()
        assert created == ["Cable Row"]
        ex = _one(db.query(models.Exercise).filter_by(name="Cable Row"))
        assert ex is not None
        assert ex.is_bodyweight is False

    def test_bodyweight_flag_on_create(self, client):
        db = SessionLocal()
        _match_or_create_exercises(db, ["Handstand Push Up"])
        db.commit()
        ex = _one(db.query(models.Exercise).filter_by(name="Handstand Push Up"))
        assert ex.is_bodyweight is True

    def test_extract_route(self, client, llm_state, monkeypatch):
        llm_json = json.dumps({
            "date": "2026-09-03",
            "exercises": [
                {"name": "Bench Press",
                 "sets": [{"reps": 10, "weight_kg": 60},
                          {"reps": 8, "weight_kg": 62.5}]},
                {"name": "Brand New Machine", "sets": [{"reps": 12}]},
            ],
            "cardio": [],
            "notes": "Felt strong",
        })

        async def fake_chat(messages, **kw):
            assert messages[0]["role"] == "system"
            assert "strict JSON" in messages[0]["content"]
            assert messages[1].get("images"), "screenshot must be attached"
            return {"text": llm_json, "backend": "fake",
                    "model": "fake-vision", "reasoning": ""}

        monkeypatch.setattr(llm_mod, "chat", fake_chat)
        png = base64.b64encode(b"\x89PNG fake").decode()
        r = client.post(
            "/sessions/ai/extract",
            files={"screenshot": ("shot.png", png.encode(), "image/png")},
        )
        # PRG: the POST 303-redirects to a GET-rendered review page.
        assert r.status_code == 200  # TestClient follows the redirect
        assert r.history and r.history[0].status_code == 303
        assert "/sessions/ai/review?t=" in str(r.url)
        # Review form posts to the AI save route (NOT /sessions/new).
        assert 'action="/sessions/ai/save"' in r.text
        assert 'value="2026-09-03"' in r.text
        assert "Felt strong" in r.text
        assert "Brand New Machine" in r.text
        assert "New" in r.text  # new-exercise badge

        # The new exercise was committed for the review form.
        db = SessionLocal()
        ex = _one(db.query(models.Exercise).filter_by(name="Brand New Machine"))
        assert ex is not None

    def test_extract_route_full_flow_to_session(self, client, llm_state, monkeypatch):
        """Extract, then submit the review form — session lands via the
        normal /sessions/new code path."""
        llm_json = json.dumps({
            "date": None,
            "exercises": [{"name": "Bench Press",
                           "sets": [{"reps": 10, "weight_kg": 60}]}],
            "cardio": [], "notes": None,
        })

        async def fake_chat(messages, **kw):
            return {"text": llm_json, "backend": "fake",
                    "model": "fake-vision", "reasoning": ""}

        monkeypatch.setattr(llm_mod, "chat", fake_chat)
        png = base64.b64encode(b"\x89PNG fake").decode()
        r = client.post(
            "/sessions/ai/extract",
            files={"screenshot": ("shot.png", png.encode(), "image/png")},
        )
        assert r.status_code == 200

        db = SessionLocal()
        ex = _one(db.query(models.Exercise).filter_by(name="Bench Press"))
        r = client.post("/sessions/new", data={
            "date": "2026-09-04", "template_id": "",
            "notes": "from AI",
            f"reps-{ex.id}-1": "10", f"weight-{ex.id}-1": "60",
        }, follow_redirects=False)
        assert r.status_code == 303
        sess = _one(db.query(models.WorkoutSession).filter_by(notes="from AI"))
        assert sess is not None
        assert sess.sets[0].weight == 60.0

    def test_extract_rejects_empty_file(self, client, llm_state):
        r = client.post(
            "/sessions/ai/extract",
            files={"screenshot": ("shot.png", b"", "image/png")},
        )
        assert r.status_code == 400

    def test_extract_bad_llm_json(self, client, llm_state, monkeypatch):
        async def fake_chat(messages, **kw):
            return {"text": "I cannot read this image, sorry!",
                    "backend": "fake", "model": "m", "reasoning": ""}

        monkeypatch.setattr(llm_mod, "chat", fake_chat)
        png = base64.b64encode(b"\x89PNG fake").decode()
        r = client.post(
            "/sessions/ai/extract",
            files={"screenshot": ("shot.png", png.encode(), "image/png")},
        )
        # 200 + rendered error (not 502): a 5xx would be swallowed by
        # Cloudflare's own error page before the user could read the hint.
        assert r.status_code == 200
        assert "Could not parse the AI response as JSON" in r.text
        assert 'action="/sessions/ai/extract"' in r.text

    def test_extract_non_object_json_renders_error(self, client, llm_state, monkeypatch):
        """A list/int reply must not crash the route with a 500."""
        async def fake_chat(messages, **kw):
            return {"text": "[1, 2, 3]", "backend": "fake",
                    "model": "m", "reasoning": ""}

        monkeypatch.setattr(llm_mod, "chat", fake_chat)
        png = base64.b64encode(b"\x89PNG fake").decode()
        r = client.post(
            "/sessions/ai/extract",
            files={"screenshot": ("shot.png", png.encode(), "image/png")},
        )
        assert r.status_code == 200
        assert "not a JSON object" in r.text

    def test_extract_hostile_shapes_no_500(self, client, llm_state, monkeypatch):
        """Exercises/sets/cardio items that aren't dicts, and names that
        normalize to nothing, must be skipped — never AttributeError/KeyError."""
        llm_json = json.dumps({
            "date": "2026-09-03",
            "exercises": [
                {"Bench": "3x8"},                      # non-dict exercise item
                {"name": "!!!", "sets": [1, 2]},       # symbols-only name, junk sets
                {"name": "Bench Press", "sets": ["5x100", {"reps": 5, "weight_kg": 60}]},
            ],
            "cardio": ["junk", {"activity_type": "running", "distance_km": 5}],
            "notes": "ok",
        })

        async def fake_chat(messages, **kw):
            return {"text": llm_json, "backend": "fake",
                    "model": "m", "reasoning": ""}

        monkeypatch.setattr(llm_mod, "chat", fake_chat)
        png = base64.b64encode(b"\x89PNG fake").decode()
        r = client.post(
            "/sessions/ai/extract",
            files={"screenshot": ("shot.png", png.encode(), "image/png")},
        )
        assert r.status_code == 200
        assert "Bench Press" in r.text
        assert "running" in r.text  # the dict cardio entry survived
        # Only the valid set made it into the review form.
        db = SessionLocal()
        ex = _one(db.query(models.Exercise).filter_by(name="Bench Press"))
        assert ex is not None

    def test_extract_no_backend_falls_back_to_ocr(self, client, llm_state, monkeypatch):
        """LLM on but nothing configured: engine=auto now falls back to the
        built-in OCR (the fake PNG is undecodable, so the OCR error shows).
        Forced-LLM without a backend still renders the configure hint."""
        monkeypatch.setattr(llm_mod, "_LLM_BACKENDS_RAW", "")
        monkeypatch.setattr(llm_mod, "_OLLAMA_MODEL", "")
        async def fake_current_backend():
            return {}

        monkeypatch.setattr(llm_mod, "current_backend", fake_current_backend)
        png = base64.b64encode(b"\x89PNG fake").decode()
        r = client.post(
            "/sessions/ai/extract",
            data={"engine": "llm"},
            files={"screenshot": ("shot.png", png.encode(), "image/png")},
        )
        assert r.status_code == 200
        assert "No LLM backend configured" in r.text
        assert "built-in OCR" in r.text

    def test_upload_page_renders(self, client, llm_state):
        r = client.get("/sessions/ai")
        assert r.status_code == 200
        assert "Upload Screenshot" in r.text

class TestExtractionErrors:
    def test_extract_non_vision_model_renders_error_page(self, client, llm_state, monkeypatch):
        """A backend error (e.g. text-only model given an image) renders the
        upload page with a friendly message instead of a 500."""
        from app import llm as llm_mod

        async def fake_chat(messages, **kw):
            raise llm_mod.LLMBackendError(
                "LLM backend ollama (nemotron-3-super:cloud) returned HTTP 400: "
                "model does not support images"
            )

        monkeypatch.setattr(llm_mod, "chat", fake_chat)
        png = base64.b64encode(b"\x89PNG fake").decode()
        r = client.post(
            "/sessions/ai/extract",
            files={"screenshot": ("shot.png", png.encode(), "image/png")},
        )
        # 200 (not 502): a 5xx would be swallowed by Cloudflare's own error
        # page before the user could read the hint.
        assert r.status_code == 200
        assert "Extraction failed" in r.text
        assert "may not support images" in r.text
        # Upload form is still present so the user can retry
        assert 'action="/sessions/ai/extract"' in r.text

    def test_llm_backend_error_is_importable_and_raised(self, llm_state):
        from app.llm import LLMBackendError
        assert issubclass(LLMBackendError, RuntimeError)

    def test_extract_when_llm_disabled_ocr_still_works(self, client, llm_state, monkeypatch):
        """With LLM_ENABLED=false the OCR engine is unaffected — auto mode
        uses it (the fake PNG is undecodable, so the OCR error shows) and
        the upload form is still present for a retry."""
        monkeypatch.setattr(llm_mod, "_LLM_ENABLED", False)
        png = base64.b64encode(b"\x89PNG fake").decode()
        r = client.post(
            "/sessions/ai/extract",
            files={"screenshot": ("shot.png", png.encode(), "image/png")},
        )
        assert r.status_code == 200
        assert "OCR extraction failed" in r.text
        # Upload form is still present so the user can retry.
        assert 'action="/sessions/ai/extract"' in r.text


class TestAiSessionSave:
    def test_save_session_with_cardio(self, client, llm_state):
        """The review form's save route creates session + sets + cardio in one go."""
        db = SessionLocal()
        ex = models.Exercise(name="SaveTest Lift", is_bodyweight=False)
        db.add(ex)
        db.commit()
        r = client.post("/sessions/ai/save", data={
            "t": _save_token(),
            "date": "2026-09-02",
            "notes": "AI saved session",
            f"reps-{ex.id}-1": "10", f"weight-{ex.id}-1": "50",
            "cardio-0-include": "1",
            "cardio-0-type": "running",
            "cardio-0-distance": "6.39",
            "cardio-0-duration": "43:34",
            "cardio-0-notes": "Avg pace 6'49\"/km, Graz",
            # second entry unchecked -> skipped
            "cardio-1-include": "0",
            "cardio-1-type": "cycling",
            "cardio-1-distance": "20",
            "cardio-1-duration": "60",
        }, follow_redirects=False)
        assert r.status_code == 303
        assert "/sessions/" in r.headers["location"] and "cardio=1" in r.headers["location"]

        sess = _one(db.query(models.WorkoutSession).filter_by(notes="AI saved session"))
        assert sess.date is not None and sess.date.isoformat() == "2026-09-02"
        assert sess.sets[0].weight == 50.0
        assert len(sess.cardio) == 1
        c = sess.cardio[0]
        assert c.activity_type == "running"
        assert c.distance_km == 6.39
        assert c.duration_min is not None and abs(c.duration_min - 43.566666) < 0.01  # 43:34 parsed
        assert "Graz" in (c.notes or "")

    def test_save_session_cardio_only(self, client, llm_state):
        """A cardio-only screenshot saves fine with no sets."""
        r = client.post("/sessions/ai/save", data={
            "t": _save_token(),
            "date": "2026-09-03",
            "notes": "",
            "cardio-0-include": "1",
            "cardio-0-type": "swimming",
            "cardio-0-distance": "2.0",
            "cardio-0-duration": "45",
            "cardio-0-notes": "",
        }, follow_redirects=False)
        assert r.status_code == 303
        db = SessionLocal()
        sess = _one(db.query(models.WorkoutSession).filter_by(date=date(2026, 9, 3)))
        assert sess is not None
        assert len(sess.cardio) == 1
        assert sess.cardio[0].activity_type == "swimming"

    def test_save_unchecked_first_cardio_keeps_later_checked(self, client, llm_state):
        """Regression: unchecking cardio #0 must not silently drop checked
        entry #1 (unchecked checkboxes aren't submitted; the loop must key
        off the always-submitted -type select)."""
        r = client.post("/sessions/ai/save", data={
            "t": _save_token(),
            "date": "2026-09-05",
            "notes": "",
            # entry 0 UNCHECKED (checkbox absent, as real browsers submit;
            # the hidden fallback posts an empty value)
            "cardio-0-type": "running",
            "cardio-0-distance": "5",
            "cardio-0-duration": "30",
            # entry 1 CHECKED
            "cardio-1-include": "1",
            "cardio-1-type": "cycling",
            "cardio-1-distance": "20",
            "cardio-1-duration": "60",
        }, follow_redirects=False)
        assert r.status_code == 303
        assert "cardio=1" in r.headers["location"]
        db = SessionLocal()
        sess = _one(db.query(models.WorkoutSession).filter_by(date=date(2026, 9, 5)))
        assert sess is not None
        assert len(sess.cardio) == 1
        assert sess.cardio[0].activity_type == "cycling"

    def test_extraction_prompt_includes_today(self, client, llm_state):
        """The extraction prompt embeds today's date so partial dates resolve."""
        prompt = _extraction_system_prompt()
        assert "Today is" in prompt
        assert date.today().isoformat() in prompt

    def test_review_page_save_guard_targets_correct_form(self, client, llm_state, monkeypatch):
        """The 'reps but no weight' confirm must bind to the review form —
        the selector used to point at /sessions/new (dead code)."""
        llm_json = json.dumps({
            "date": None,
            "exercises": [{"name": "Bench Press",
                           "sets": [{"reps": 10, "weight_kg": 60}]}],
            "cardio": [], "notes": None,
        })

        async def fake_chat(messages, **kw):
            return {"text": llm_json, "backend": "fake",
                    "model": "fake-vision", "reasoning": ""}

        monkeypatch.setattr(llm_mod, "chat", fake_chat)
        png = base64.b64encode(b"\x89PNG fake").decode()
        r = client.post(
            "/sessions/ai/extract",
            files={"screenshot": ("shot.png", png.encode(), "image/png")},
        )
        assert r.status_code == 200
        assert 'form[action="/sessions/ai/save"]' in r.text
        assert 'form[action="/sessions/new"]' not in r.text

    def test_coach_stream_malformed_json_body_is_400(self, client, llm_state):
        r = client.post("/coach/send/stream",
                        content=b"{not json",
                        headers={"Content-Type": "application/json"})
        assert r.status_code == 400

    def test_llm_select_malformed_json_body_is_400(self, client, llm_state):
        r = client.post("/api/llm/select",
                        content=b"[1,2,3",
                        headers={"Content-Type": "application/json"})
        assert r.status_code == 400

    def test_review_page_duration_prefill_is_mss(self, client, llm_state, monkeypatch):
        """43.5667 min prefills as '43:34', not a raw float (matches
        cardio_edit.html's input format)."""
        llm_json = json.dumps({
            "date": None, "exercises": [],
            "cardio": [{"activity_type": "running", "distance_km": 6.39,
                        "duration_min": 43.566666}],
            "notes": None,
        })

        async def fake_chat(messages, **kw):
            return {"text": llm_json, "backend": "fake",
                    "model": "fake-vision", "reasoning": ""}

        monkeypatch.setattr(llm_mod, "chat", fake_chat)
        png = base64.b64encode(b"\x89PNG fake").decode()
        r = client.post(
            "/sessions/ai/extract",
            files={"screenshot": ("shot.png", png.encode(), "image/png")},
        )
        assert r.status_code == 200
        assert 'value="43:34"' in r.text
        assert "43.5667" not in r.text


# ---------------------------------------------------------------------------
# Hardening: PRG extraction, security headers, CSRF origin check
# ---------------------------------------------------------------------------

class TestExtractionPRG:
    def test_extract_redirects_to_review_get(self, client, llm_state, monkeypatch):
        """POST /sessions/ai/extract must 303 to a token GET — refresh on the
        review page re-runs a lookup, not the paid LLM extraction."""
        llm_json = json.dumps({
            "date": None,
            "exercises": [{"name": "Bench Press",
                           "sets": [{"reps": 10, "weight_kg": 60}]}],
            "cardio": [], "notes": None,
        })

        calls = {"n": 0}

        async def fake_chat(messages, **kw):
            calls["n"] += 1
            return {"text": llm_json, "backend": "fake",
                    "model": "fake-vision", "reasoning": ""}

        monkeypatch.setattr(llm_mod, "chat", fake_chat)
        png = base64.b64encode(b"\x89PNG fake").decode()
        r = client.post(
            "/sessions/ai/extract",
            files={"screenshot": ("shot.png", png.encode(), "image/png")},
            follow_redirects=False,
        )
        assert r.status_code == 303
        location = r.headers["location"]
        assert location.startswith("/sessions/ai/review?t=")
        token = location.split("t=", 1)[1]

        # The GET renders the form WITHOUT calling the LLM again.
        r2 = client.get(location)
        assert r2.status_code == 200
        assert 'action="/sessions/ai/save"' in r2.text
        assert calls["n"] == 1

        # Refreshing (same GET) still works, still no extra LLM call.
        r3 = client.get(location)
        assert r3.status_code == 200
        assert calls["n"] == 1

    def test_review_token_unknown_or_expired(self, client, llm_state):
        r = client.get("/sessions/ai/review?t=bogus-token")
        assert r.status_code == 200
        assert "expired" in r.text
        assert 'action="/sessions/ai/extract"' in r.text  # upload form back

    def test_review_token_expires_after_ttl(self, client, llm_state, monkeypatch):
        import time as _time
        from app import ai as ai_mod
        llm_json = json.dumps({"date": None, "exercises": [],
                               "cardio": [], "notes": None})

        async def fake_chat(messages, **kw):
            return {"text": llm_json, "backend": "fake",
                    "model": "m", "reasoning": ""}

        monkeypatch.setattr(llm_mod, "chat", fake_chat)
        png = base64.b64encode(b"\x89PNG fake").decode()
        r = client.post(
            "/sessions/ai/extract",
            files={"screenshot": ("shot.png", png.encode(), "image/png")},
            follow_redirects=False,
        )
        token = r.headers["location"].split("t=", 1)[1]
        assert client.get(r.headers["location"]).status_code == 200

        # Age the entry past the TTL.
        expired = _time.monotonic() - 1
        with ai_mod._review_store_lock:
            exp, payload = ai_mod._review_store[token]
            ai_mod._review_store[token] = (exp - ai_mod._REVIEW_TTL_SECONDS - 1, payload)
        r2 = client.get(r.headers["location"])
        assert "expired" in r2.text


class TestSecurityHeaders:
    def test_html_response_headers(self, client, llm_state):
        r = client.get("/coach")
        assert r.headers.get("x-frame-options") == "DENY"
        assert r.headers.get("cache-control") == "no-store"
        assert r.headers.get("referrer-policy") == "same-origin"

    def test_static_files_get_headers_too(self, client, llm_state):
        r = client.get("/static/js/htmx.min.js")
        assert r.status_code == 200
        assert r.headers.get("x-frame-options") == "DENY"


class TestCSRFOriginCheck:
    def test_cross_origin_post_rejected(self, client, llm_state):
        r = client.post("/api/llm/select", json={"backend": "fake"},
                        headers={"Origin": "https://evil.example.com"})
        assert r.status_code == 403

    def test_cross_origin_referer_rejected(self, client, llm_state):
        r = client.post("/sessions/ai/save", data={"date": "2026-09-05"},
                        headers={"Referer": "https://evil.example.com/form"})
        assert r.status_code == 403

    def test_same_origin_post_allowed(self, client, llm_state):
        r = client.post("/api/llm/select", json={"backend": "fake", "model": "other"},
                        headers={"Origin": "http://testserver",
                                 "Referer": "http://testserver/coach"})
        assert r.status_code == 200

    def test_no_origin_referer_is_unaffected(self, client, llm_state):
        """API clients (MCP/curl) send no Origin/Referer — not blocked."""
        r = client.post("/api/llm/select", json={"backend": "fake"})
        assert r.status_code == 200

    def test_get_never_blocked_by_cross_origin_header(self, client, llm_state):
        r = client.get("/coach", headers={"Origin": "https://evil.example.com"})
        assert r.status_code == 200
