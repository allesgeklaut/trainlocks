"""Tests for the baked-in OCR extraction: layout parser, date resolution,
set-text parsing and the engine-selection wiring in the extract route.

The real RapidOCR engine is never loaded in tests — ocr_image is mocked;
the layout parser is tested directly with synthetic OcrLine fixtures."""

import json
from datetime import date, timedelta

import pytest
from fastapi.testclient import TestClient

from app import llm as llm_mod
from app import models
from app.ai import _match_or_create_exercises
from app.auth import COOKIE_NAME, create_session_cookie
from app.database import Base, SessionLocal, engine
from app.main import app
from app.ocr import (
    OCRError,
    OcrLine,
    _parse_date,
    _parse_set_text,
    parse_ocr_layout,
)


def _one(qry):
    row = qry.first()
    assert row is not None
    return row


def _line(text: str, y: float, x0: float = 20.0, x1: float | None = None,
          h: float = 24.0, score: float = 0.99) -> OcrLine:
    return OcrLine(text=text, score=score, x0=x0, y0=y,
                   x1=x1 if x1 is not None else x0 + len(text) * 10.0,
                   y1=y + h)


@pytest.fixture(scope="function")
def client(tmp_path, monkeypatch):
    """Same harness as test_ai.py, with the LLM pointed at a dead port."""
    Base.metadata.create_all(bind=engine)
    db = SessionLocal()
    if not db.query(models.User).filter_by(username="tester").first():
        db.add(models.User(username="tester",
                           hashed_password="$2b$12$" + "x" * 53))
        db.commit()
    db.close()
    state_file = tmp_path / "llm_state.json"
    monkeypatch.setattr(llm_mod, "_DEFAULT_STATE_FILE", state_file)
    monkeypatch.setattr(
        llm_mod, "_LLM_BACKENDS_RAW",
        json.dumps([{"name": "fake", "type": "openai",
                     "url": "http://127.0.0.1:1", "model": "fake-vision"}]),
    )
    monkeypatch.setattr(llm_mod, "_LLM_ENABLED", True)
    cookie = create_session_cookie(1)
    with TestClient(app) as c:
        c.cookies.set(COOKIE_NAME, cookie)
        yield c
    Base.metadata.drop_all(bind=engine)


def _mock_ocr_route(monkeypatch, data: dict):
    """Replace the OCR engine + parser with a canned payload."""
    import app.ai as ai_mod
    from app import ocr as ocr_mod

    async def fake_extract(raw, today):
        return data

    monkeypatch.setattr(ai_mod, "_ocr_extract", fake_extract)


# ---------------------------------------------------------------------------
# Set-text parsing
# ---------------------------------------------------------------------------

class TestParseSetText:
    def test_kg_first(self):
        sets = _parse_set_text("80kg x 10, 80kg x 9")
        assert sets == [
            {"reps": 10, "weight_kg": 80.0, "assist_kg": None,
             "duration_seconds": None},
            {"reps": 9, "weight_kg": 80.0, "assist_kg": None,
             "duration_seconds": None},
        ]

    def test_weight_first(self):
        sets = _parse_set_text("10 x 80kg")
        assert sets == [{"reps": 10, "weight_kg": 80.0, "assist_kg": None,
                         "duration_seconds": None}]

    def test_plain_sets_x_reps(self):
        sets = _parse_set_text("3 x 8")
        # "3 x 8" without any unit = 3 sets of 8 reps → expanded per set.
        assert len(sets) == 3
        assert all(s["reps"] == 8 and s["weight_kg"] is None for s in sets)

    def test_lb_conversion(self):
        sets = _parse_set_text("135lb x 5")
        assert sets[0]["weight_kg"] == 61.0  # 135 lb = 61.2 kg -> 61.0

    def test_hold_time_colon(self):
        sets = _parse_set_text("1:30 hold")
        assert sets == [{"reps": None, "weight_kg": None, "assist_kg": None,
                         "duration_seconds": 90}]

    def test_hold_seconds(self):
        sets = _parse_set_text("45s")
        assert sets[0]["duration_seconds"] == 45

    def test_hold_sets_x_seconds(self):
        sets = _parse_set_text("3 x 60s")
        assert len(sets) == 3
        assert all(s["duration_seconds"] == 60 for s in sets)

    def test_assist(self):
        sets = _parse_set_text("8 x -25kg")
        assert sets[0]["reps"] == 8 and sets[0]["assist_kg"] == 25.0

    def test_bare_reps(self):
        sets = _parse_set_text("12")
        assert sets[0]["reps"] == 12

    def test_decimal_weight(self):
        sets = _parse_set_text("62,5kg x 8")
        assert sets[0]["weight_kg"] == 62.5

    def test_junk_returns_empty(self):
        assert _parse_set_text("no numbers here") == []


# ---------------------------------------------------------------------------
# Date resolution
# ---------------------------------------------------------------------------

class TestParseDate:
    TODAY = date(2026, 9, 10)

    def test_iso(self):
        assert _parse_date("2026-09-02", self.TODAY) == "2026-09-02"

    def test_weekday_month_day(self):
        # "Wed 2. Sep" -> most recent Sep 2 in the past
        assert _parse_date("Wed 2. Sep", self.TODAY) == "2026-09-02"

    def test_month_day(self):
        assert _parse_date("Sep 2", self.TODAY) == "2026-09-02"

    def test_day_dot_month_dot_year(self):
        assert _parse_date("02.09.2026", self.TODAY) == "2026-09-02"

    def test_future_rolls_back_a_year(self):
        assert _parse_date("Sep 15", self.TODAY) == "2025-09-15"

    def test_slash_dates_both_orders(self):
        got = _parse_date("9/2/26", self.TODAY)
        assert got in ("2026-09-02", "2026-02-09")

    def test_none_when_nothing(self):
        assert _parse_date("Bench Press 80kg x 10", self.TODAY) is None

    def test_too_old_ignored(self):
        assert _parse_date("Sep 2 2020", self.TODAY) is None


# ---------------------------------------------------------------------------
# Layout parsing
# ---------------------------------------------------------------------------

class TestParseOcrLayout:
    def test_table_layout(self):
        lines = [
            _line("Workout Log", 20),
            _line("Bench Press", 80, x0=20, x1=150),
            _line("80kg x 10", 80, x0=350, x1=470),
            _line("Pull Ups", 140, x0=20, x1=120),
            _line("3 x 8", 140, x0=350, x1=410),
        ]
        data = parse_ocr_layout(lines, date(2026, 9, 10))
        assert [e["name"] for e in data["exercises"]] == ["Bench Press", "Pull Ups"]
        assert data["exercises"][0]["sets"][0]["reps"] == 10
        assert data["exercises"][0]["sets"][0]["weight_kg"] == 80.0
        assert data["exercises"][1]["sets"][0]["reps"] == 8

    def test_inline_layout_values_below_name(self):
        lines = [
            _line("Bench Press", 80, x0=20, x1=150),
            _line("80kg x 10, 80kg x 9", 120, x0=40, x1=280),
        ]
        data = parse_ocr_layout(lines, date(2026, 9, 10))
        assert len(data["exercises"]) == 1
        assert data["exercises"][0]["name"] == "Bench Press"
        assert len(data["exercises"][0]["sets"]) == 2

    def test_hold_row(self):
        lines = [
            _line("Plank", 80, x0=20, x1=100),
            _line("1:30 hold", 120, x0=40, x1=160),
        ]
        data = parse_ocr_layout(lines, date(2026, 9, 10))
        assert data["exercises"][0]["sets"][0]["duration_seconds"] == 90
        assert data["exercises"][0]["sets"][0]["reps"] is None

    def test_cardio_row(self):
        lines = [_line("Running 6.4 km 43:34", 80)]
        data = parse_ocr_layout(lines, date(2026, 9, 10))
        assert data["cardio"][0]["activity_type"] == "running"
        assert data["cardio"][0]["distance_km"] == 6.4
        assert abs(data["cardio"][0]["duration_min"] - 43.57) < 0.01

    def test_notes_become_pending_then_dropped(self):
        # A trailing words-only line with no numbers and no pending name
        # becomes an empty exercise that is dropped (not a note).
        lines = [
            _line("Bench Press", 80, x0=20, x1=150),
            _line("80kg x 10", 80, x0=350, x1=470),
            _line("Felt strong today", 200),
        ]
        data = parse_ocr_layout(lines, date(2026, 9, 10))
        assert len(data["exercises"]) == 1
        assert data["notes"] is None

    def test_empty_input(self):
        data = parse_ocr_layout([], date(2026, 9, 10))
        assert data == {"date": None, "exercises": [], "cardio": [],
                        "notes": None}

    def test_exercises_without_valid_sets_dropped(self):
        lines = [
            _line("Warmup", 80),
            _line("some chat text", 120),
        ]
        data = parse_ocr_layout(lines, date(2026, 9, 10))
        assert data["exercises"] == []


# ---------------------------------------------------------------------------
# Engine selection in the extract route
# ---------------------------------------------------------------------------

class TestEngineSelection:
    PNG = b"\x89PNG fake"

    def _post(self, client, engine: str | None = None):
        files = {"screenshot": ("shot.png", self.PNG, "image/png")}
        data = {} if engine is None else {"engine": engine}
        return client.post("/sessions/ai/extract", files=files, data=data,
                           follow_redirects=False)

    def test_forced_ocr_uses_ocr(self, client, monkeypatch):
        seen = {}

        async def fake_extract(raw, today):
            seen["called"] = True
            return {"date": None,
                    "exercises": [{"name": "Bench Press",
                                   "sets": [{"reps": 10, "weight_kg": 60.0,
                                             "assist_kg": None,
                                             "duration_seconds": None}]}],
                    "cardio": [], "notes": ""}

        import app.ai as ai_mod
        monkeypatch.setattr(ai_mod, "_ocr_extract", fake_extract)
        chat_calls = []

        async def fail_chat(messages, **kw):
            chat_calls.append(1)
            raise AssertionError("LLM must not be called for forced OCR")

        monkeypatch.setattr(llm_mod, "chat", fake_chat := fail_chat)

        r = self._post(client, "ocr")
        assert r.status_code == 303
        assert seen.get("called") is True
        assert not chat_calls
        token = r.headers["location"].split("t=", 1)[1]
        review = client.get(f"/sessions/ai/review?t={token}")
        assert "built-in OCR" in review.text
        assert "Bench Press" in review.text

    def test_forced_llm_with_backend(self, client, monkeypatch):
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
        r = self._post(client, "llm")
        assert r.status_code == 303
        token = r.headers["location"].split("t=", 1)[1]
        review = client.get(f"/sessions/ai/review?t={token}")
        assert "fake-vision" in review.text  # model label on review page

    def test_forced_llm_without_backend_renders_error(self, client, monkeypatch):
        async def fake_current_backend():
            return {}

        monkeypatch.setattr(llm_mod, "current_backend", fake_current_backend)
        r = self._post(client, "llm")
        assert r.status_code == 200
        assert "No LLM backend configured" in r.text
        assert "built-in OCR" in r.text

    def test_auto_without_backend_falls_back_to_ocr(self, client, monkeypatch):
        async def fake_current_backend():
            return {}

        monkeypatch.setattr(llm_mod, "current_backend", fake_current_backend)
        seen = {}

        async def fake_extract(raw, today):
            seen["called"] = True
            return {"date": None,
                    "exercises": [{"name": "Squat",
                                   "sets": [{"reps": 5, "weight_kg": 100.0,
                                             "assist_kg": None,
                                             "duration_seconds": None}]}],
                    "cardio": [], "notes": ""}

        import app.ai as ai_mod
        monkeypatch.setattr(ai_mod, "_ocr_extract", fake_extract)
        r = self._post(client, "auto")
        assert r.status_code == 303
        assert seen.get("called") is True

    def test_auto_with_backend_prefers_llm(self, client, monkeypatch):
        llm_json = json.dumps({"date": None, "exercises": [], "cardio": [],
                               "notes": None})
        async def fake_chat(messages, **kw):
            return {"text": llm_json, "backend": "fake",
                    "model": "fake-vision", "reasoning": ""}

        monkeypatch.setattr(llm_mod, "chat", fake_chat)
        import app.ai as ai_mod

        async def fail_extract(raw, today):
            raise AssertionError("OCR must not run when the LLM is configured")

        monkeypatch.setattr(ai_mod, "_ocr_extract", fail_extract)
        r = self._post(client, "auto")
        assert r.status_code == 303

    def test_forced_ocr_with_no_exercises_renders_hint(self, client, monkeypatch):
        async def fake_extract(raw, today):
            return {"date": None, "exercises": [], "cardio": [], "notes": ""}

        import app.ai as ai_mod
        monkeypatch.setattr(ai_mod, "_ocr_extract", fake_extract)
        r = self._post(client, "ocr")
        assert r.status_code == 200
        assert "OCR found no exercises" in r.text
        assert "LLM engine" in r.text

    def test_ocr_engine_error_renders_friendly_message(self, client, monkeypatch):
        async def fail_extract(raw, today):
            raise OCRError("Could not decode the image")

        import app.ai as ai_mod
        monkeypatch.setattr(ai_mod, "_ocr_extract", fail_extract)
        r = self._post(client, "ocr")
        assert r.status_code == 200
        assert "OCR extraction failed" in r.text
        assert "Could not decode the image" in r.text

    def test_invalid_engine_value_falls_back_to_auto(self, client, monkeypatch):
        llm_json = json.dumps({"date": None, "exercises": [], "cardio": [],
                               "notes": None})
        async def fake_chat(messages, **kw):
            return {"text": llm_json, "backend": "fake",
                    "model": "fake-vision", "reasoning": ""}

        monkeypatch.setattr(llm_mod, "chat", fake_chat)
        r = self._post(client, "bogus")
        # 'auto' with a configured backend -> LLM path -> 303 review
        assert r.status_code == 303

    def test_ocr_route_full_flow_to_session(self, client, monkeypatch):
        """Forced-OCR extraction then saving via the review form."""
        async def fake_extract(raw, today):
            return {"date": "2026-09-08",
                    "exercises": [{"name": "Bench Press",
                                   "sets": [{"reps": 10, "weight_kg": 60.0,
                                             "assist_kg": None,
                                             "duration_seconds": None}]}],
                    "cardio": [], "notes": "ocr session"}

        import app.ai as ai_mod
        monkeypatch.setattr(ai_mod, "_ocr_extract", fake_extract)
        r = self._post(client, "ocr")
        token = r.headers["location"].split("t=", 1)[1]
        review = client.get(f"/sessions/ai/review?t={token}")
        assert review.status_code == 200
        db = SessionLocal()
        ex = _one(db.query(models.Exercise).filter_by(name="Bench Press"))
        r = client.post("/sessions/ai/save", data={
            "date": "2026-09-08", "notes": "ocr session",
            f"reps-{ex.id}-1": "10", f"weight-{ex.id}-1": "60",
        }, follow_redirects=False)
        assert r.status_code == 303
        sess = _one(db.query(models.WorkoutSession).filter_by(
            notes="ocr session"))
        assert sess.sets[0].weight == 60.0

    def test_engine_picker_on_upload_page(self, client):
        r = client.get("/sessions/ai")
        assert r.status_code == 200
        assert 'id="engine"' in r.text
        assert 'value="auto"' in r.text
        assert 'value="ocr"' in r.text
        assert 'value="llm"' in r.text


# ---------------------------------------------------------------------------
# Real engine smoke test (only when models are already cached — offline-safe)
# ---------------------------------------------------------------------------

class TestRealEngine:
    def test_ocr_image_on_synthetic_screenshot(self):
        """Runs the actual engine; skipped when the model files aren't
        cached locally (CI / no network)."""
        try:
            import pathlib
            import rapidocr
            model_dir = pathlib.Path(rapidocr.__file__).parent / "models"
            if not any(model_dir.glob("*.onnx")):
                pytest.skip("OCR models not cached")
        except Exception:
            pytest.skip("rapidocr not fully installed")

        png = _render_workout_png()
        from app.ocr import ocr_image
        lines = ocr_image(png)
        texts = [l.text for l in lines]
        assert any("Bench" in t for t in texts)
        data = parse_ocr_layout(lines, date.today())
        names = [e["name"] for e in data["exercises"]]
        assert any("Bench" in n for n in names)


def _render_workout_png() -> bytes:
    import io

    from PIL import Image, ImageDraw, ImageFont

    img = Image.new("RGB", (720, 400), (255, 255, 255))
    d = ImageDraw.Draw(img)
    try:
        font = ImageFont.truetype(
            "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf", 24)
    except OSError:
        font = ImageFont.load_default()
    d.text((20, 20), "Bench Press", fill=(0, 0, 0), font=font)
    d.text((40, 60), "80kg x 10", fill=(0, 0, 0), font=font)
    buf = io.BytesIO()
    img.save(buf, format="PNG")
    return buf.getvalue()