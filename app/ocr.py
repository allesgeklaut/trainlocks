"""Baked-in OCR extraction: RapidOCR (PP-OCRv5, ONNX Runtime, CPU).

Provides a zero-setup fallback for screenshot-to-session extraction — no LLM
backend required. Models (~10 MB) download on first use into the rapidocr
package dir and are pre-fetched at Docker build time.

The layout parser groups OCR lines into workout-log rows and maps them onto
the same JSON schema the LLM extraction prompt produces, so both engines
share the downstream review-form flow untouched:

    {"date": ..., "exercises": [{"name", "sets": [{"reps", "weight_kg",
     "assist_kg", "duration_seconds"}]}], "cardio": [...], "notes": ...}
"""

from __future__ import annotations

import io
import logging
import re
import threading
from typing import Any

logger = logging.getLogger("trainlocks.ocr")

# ---------------------------------------------------------------------------
# Engine singleton (first call downloads/loads models; keep one warm copy)
# ---------------------------------------------------------------------------

_engine: Any = None
_engine_lock = threading.Lock()


class OCRError(RuntimeError):
    """The OCR engine could not process the image."""


def _get_engine() -> Any:
    global _engine
    if _engine is None:
        with _engine_lock:
            if _engine is None:
                from rapidocr import LangRec, ModelType, OCRVersion, RapidOCR

                _engine = RapidOCR(params={
                    # English recognizer is smaller and more accurate for
                    # latin-only workout logs than the default bilingual one.
                    "Rec.ocr_version": OCRVersion.PPOCRV5,
                    "Rec.model_type": ModelType.MOBILE,
                    "Rec.lang_type": LangRec.EN,
                })
    return _engine


def ocr_image(raw: bytes) -> list[OcrLine]:
    """Run OCR on raw image bytes; return lines ordered top-to-bottom.

    Raises OCRError when the image cannot be decoded/processed.
    """
    try:
        import numpy as np

        arr = np.asarray(bytearray(raw), dtype=np.uint8)
        import cv2

        img = cv2.imdecode(arr, cv2.IMREAD_COLOR)
        if img is None:
            raise OCRError("Could not decode the image (PNG/JPEG/WebP?).")
        result = _get_engine()(img)
    except OCRError:
        raise
    except Exception as e:  # engine internals raise plain Exception subclasses
        logger.warning("OCR engine failed: %s", e)
        raise OCRError(f"OCR failed: {e}") from e

    lines: list[OcrLine] = []
    boxes = getattr(result, "boxes", None)
    txts = getattr(result, "txts", None)
    scores = getattr(result, "scores", None)
    if boxes is None or txts is None:
        return lines
    if scores is None:
        scores = [1.0] * len(txts)
    for box, text, score in zip(boxes, txts, scores):
        # box: 4x2 array of corner points (clockwise from top-left).
        xs = [float(p[0]) for p in box]
        ys = [float(p[1]) for p in box]
        lines.append(OcrLine(
            text=str(text), score=float(score),
            x0=min(xs), y0=min(ys), x1=max(xs), y1=max(ys),
        ))
    lines.sort(key=lambda l: (l.y0, l.x0))
    return lines


class OcrLine:
    """One detected text region with its bounding box (pixels)."""

    __slots__ = ("text", "score", "x0", "y0", "x1", "y1")

    def __init__(self, text: str, score: float,
                 x0: float, y0: float, x1: float, y1: float) -> None:
        self.text = text
        self.score = score
        self.x0 = x0
        self.y0 = y0
        self.x1 = x1
        self.y1 = y1

    @property
    def height(self) -> float:
        return self.y1 - self.y0


# ---------------------------------------------------------------------------
# Layout parsing: OCR lines -> workout JSON (same schema as the LLM path)
# ---------------------------------------------------------------------------

# Column separator between the exercise-name column and the set-values column
# in table-style logs: a gap larger than ~1.2× the line height.
_COL_GAP_FACTOR = 1.2

_REPS_SET_RE = re.compile(
    r"(\d+)\s*(?:x|×|sets?|reps?)\s*(\d+)|(\d+)\s*(?:x|×)\s*(\d+)", re.IGNORECASE)
# "10 x 80kg" | "10x80" | "5 × 60 kg"
WEIGHT_FIRST_RE = re.compile(
    r"(\d+(?:[.,]\d+)?)\s*(?:kg|kgs|kilos?)?\s*(?:x|×)\s*(\d+)\s*(?:kg|kgs)?"
    r"(?:\s*(?:@\s*(\d+(?:[.,]\d+)?))?)?", re.IGNORECASE)
# "80kg x 10" | "80 x 10" | "62.5kg x 9"
KG_FIRST_RE = re.compile(
    r"(\d+(?:[.,]\d+)?)\s*(?:kg|kgs|lb|lbs)\s*(?:x|×)\s*(\d+)", re.IGNORECASE)
LB_HINT_RE = re.compile(r"\b(lb|lbs)\b", re.IGNORECASE)
# "3 x 8" plain sets x reps (no weight anywhere)
PLAIN_SETS_RE = re.compile(r"(\d+)\s*(?:x|×)\s*(\d+)")
# "1:30" hold duration -> seconds; "45s" / "90 sec"
HOLD_TIME_RE = re.compile(r"(?:(\d+):(\d{1,2}))|(?:(\d+)\s*(?:s|sec|secs|seconds)\b)",
                          re.IGNORECASE)
ASSIST_RE = re.compile(r"(-\s*(\d+(?:[.,]\d+)?)\s*kg)|(\+\s*(\d+(?:[.,]\d+)?)\s*kg)",
                       re.IGNORECASE)
# Date patterns: "Wed 2. Sep", "Sep 2", "2026-09-02", "02.09.2026", "9/2/26"
_MONTHS = {m.lower(): i + 1 for i, m in enumerate(
    ["January", "February", "March", "April", "May", "June", "July",
     "August", "September", "October", "November", "December"])}
_MONTHS.update({m[:3].lower(): i + 1 for i, m in enumerate(
    ["January", "February", "March", "April", "May", "June", "July",
     "August", "September", "October", "November", "December"])})
_DATE_ISO_RE = re.compile(r"\b(\d{4})-(\d{1,2})-(\d{1,2})\b")
_DATE_DOT_RE = re.compile(r"\b(\d{1,2})\.(\d{1,2})\.(\d{2,4})\b")
_DATE_SLASH_RE = re.compile(r"\b(\d{1,2})/(\d{1,2})/(\d{2,4})\b")
_DATE_MON_RE = re.compile(
    r"\b(" + "|".join(_MONTHS) + r")\.?\s*(\d{1,2})(?:st|nd|rd|th)?\b"
    r"|\b(\d{1,2})(?:st|nd|rd|th)?\.?\s+(" + "|".join(_MONTHS) + r")\b",
    re.IGNORECASE)
_DATE_DOW_RE = re.compile(
    r"\b(mon|tue|tues|wed|thu|thur|thurs|fri|sat|sun)[a-z]*\.?\s*"
    r"(\d{1,2})(?:st|nd|rd|th)?\.?\s*(" + "|".join(_MONTHS) + r")?\.?",
    re.IGNORECASE)

# Lines that start a cardio row.
CARDIO_HINTS = ("running", "run", "cycling", "cycle", "bike", "swimming",
                "swim", "rowing", "row", "walking", "walk", "treadmill",
                "elliptical", "hike", "swim")
KM_RE = re.compile(r"(\d+(?:[.,]\d+)?)\s*(?:km|kilometers?|kilometres?)\b",
                   re.IGNORECASE)
DURATION_RE = re.compile(r"(?:(\d{1,2}):(\d{2})(?::(\d{2}))?|(\d+(?:[.,]\d+)?)\s*(?:min|mins|minutes)\b)",
                         re.IGNORECASE)


def _to_float(token: str | None) -> float | None:
    if token is None:
        return None
    try:
        return float(token.replace(",", "."))
    except ValueError:
        return None


def _lb_to_kg(lb: float) -> float:
    return round(lb / 2.2046 * 2) / 2  # to 0.5 kg, matching the LLM prompt


def _parse_date(text: str, today: Any) -> str | None:
    """Resolve any date-ish text in ``text`` to a past YYYY-MM-DD."""
    from datetime import date as _date, timedelta

    def clamp_past(y: int, m: int, d: int) -> str | None:
        try:
            dt = _date(y, m, d)
        except ValueError:
            return None
        if dt > today:
            dt = dt.replace(year=dt.year - 1)
        if (today - dt) > timedelta(days=730):
            return None
        return dt.isoformat()

    m = _DATE_ISO_RE.search(text)
    if m:
        got = clamp_past(int(m.group(1)), int(m.group(2)), int(m.group(3)))
        if got:
            return got
    m = _DATE_DOW_RE.search(text)
    if m:
        dow, day, mon = m.groups()
        month = _MONTHS.get((mon or "").lower())
        if month:
            got = clamp_past(today.year, month, int(day))
            if got:
                return got
    m = _DATE_MON_RE.search(text)
    if m:
        mon, day, _d2, mon2 = m.groups()
        month = _MONTHS.get((mon or mon2 or "").lower())
        day_num = int(day or _d2 or 0)
        if month and day_num:
            got = clamp_past(today.year, month, day_num)
            if got:
                return got
    m = _DATE_DOT_RE.search(text)
    if m:
        d, mo, y = int(m.group(1)), int(m.group(2)), int(m.group(3))
        if y < 100:
            y += 2000
        got = clamp_past(y, mo, d)  # D.M.Y (European style)
        if got:
            return got
    m = _DATE_SLASH_RE.search(text)
    if m:
        a, b, y = int(m.group(1)), int(m.group(2)), int(m.group(3))
        if y < 100:
            y += 2000
        # Ambiguous M/D vs D/M: try both, prefer the one in the past.
        for mo, d in ((a, b), (b, a)):
            got = clamp_past(y, mo, d)
            if got:
                return got
    return None


def _parse_set_text(text: str) -> list[dict[str, Any]]:
    """Parse set values from a text chunk into set dicts.

    Handles "80kg x 10", "10 x 80kg", "3 x 8", "1:30", "45s", "12 x 60s",
    assisted loads ("-25kg"), multiple sets per line ("80kg x 10, 80kg x 9").
    """
    sets: list[dict[str, Any]] = []
    is_lb = bool(LB_HINT_RE.search(text))
    chunks = [c.strip() for c in re.split(r"[,;]+", text) if c.strip()]
    for chunk in chunks:
        chunk_lb = is_lb or bool(LB_HINT_RE.search(chunk))
        s: dict[str, Any] = {"reps": None, "weight_kg": None,
                             "assist_kg": None, "duration_seconds": None}

        # Timed holds: "1:30", "45s", "90 sec", "3 x 60s"
        hold = HOLD_TIME_RE.search(chunk)
        plain = PLAIN_SETS_RE.search(chunk)
        hold_like = hold is not None and (
            "hold" in chunk.lower()
            or re.search(r"\b(?:s|sec|secs|seconds)\b", chunk, re.IGNORECASE) is not None
            or (hold.group(1) is not None and ":" in chunk
                and not re.search(r"\d+\s*(?:x|×)\s*\d+", chunk))
        )
        if hold is not None and hold_like:
            if hold.group(1) is not None:
                s["duration_seconds"] = int(hold.group(1)) * 60 + int(hold.group(2))
            else:
                s["duration_seconds"] = int(hold.group(3))
            if plain and re.search(
                    rf"{re.escape(plain.group(1))}\s*(?:x|×)\s*(\d+)\s*(?:s|sec)\b",
                    chunk, re.IGNORECASE):
                for _ in range(int(plain.group(1))):
                    sets.append(dict(s))
            else:
                sets.append(s)
            continue

        # KG-first: "80kg x 10"
        m = KG_FIRST_RE.search(chunk)
        if m:
            weight = _to_float(m.group(1))
            reps = int(m.group(2))
            if chunk_lb and weight is not None:
                weight = _lb_to_kg(weight)
            s["reps"], s["weight_kg"] = reps, weight
            _apply_assist(chunk, s)
            sets.append(s)
            continue

        # Weight-first: "10 x 80kg" (also catches "10 x 80")
        m = WEIGHT_FIRST_RE.search(chunk)
        if m:
            reps = int(m.group(2))
            weight = _to_float(m.group(3))
            if weight is None:
                # "10 x 80" — second number is weight only if "kg"/"lb" hints
                # appear somewhere; otherwise it's sets x reps.
                if chunk_lb or re.search(r"\bkg\b", chunk, re.IGNORECASE):
                    weight = _to_float(chunk.split("x")[-1].split("×")[-1])
                else:
                    weight = None
            if chunk_lb and weight is not None:
                weight = _lb_to_kg(weight)
            s["reps"], s["weight_kg"] = reps, weight
            _apply_assist(chunk, s)
            sets.append(s)
            continue

        # Plain "3 x 8"
        m = PLAIN_SETS_RE.search(chunk)
        if m:
            sets.append({"reps": int(m.group(2)), "weight_kg": None,
                         "assist_kg": None, "duration_seconds": None})
            continue

        # Bare reps: "12" (machine row style)
        m = re.fullmatch(r"\d{1,3}", chunk)
        if m:
            sets.append({"reps": int(chunk), "weight_kg": None,
                         "assist_kg": None, "duration_seconds": None})
    return sets


def _apply_assist(chunk: str, s: dict[str, Any]) -> None:
    m = ASSIST_RE.search(chunk)
    if m:
        val = _to_float(m.group(2) or m.group(4))
        if val is not None:
            s["assist_kg"] = val


def _looks_cardio(text: str) -> bool:
    low = text.lower()
    return any(h in low for h in CARDIO_HINTS) and not re.search(
        r"\d+\s*(?:kg|lb)", low)


def _cardio_type(text: str) -> str:
    low = text.lower()
    for atype, hints in (
        ("running", ("running", "run ", "run", "treadmill")),
        ("cycling", ("cycling", "cycle", "bike", "biking")),
        ("swimming", ("swimming", "swim")),
        ("rowing", ("rowing", "row ")),
        ("walking", ("walking", "walk", "hike")),
    ):
        if any(h in low for h in hints):
            return atype
    return "other"


def parse_ocr_layout(lines: list[OcrLine], today: Any) -> dict[str, Any]:
    """Group OCR lines into workout-log rows and map them to the JSON schema.

    Two layouts are recognised per visual row:
    - table style: exercise name (left column) + set values (right column)
    - inline style: "Bench Press 80kg x 10" in one line, or the values on the
      line directly below the name.
    """
    out: dict[str, Any] = {"date": None, "exercises": [],
                           "cardio": [], "notes": None}
    if not lines:
        return out

    # Date from any early line.
    for line in lines[:8]:
        got = _parse_date(line.text, today)
        if got:
            out["date"] = got
            break

    med_h = sorted(l.height for l in lines)[len(lines) // 2] if lines else 20.0
    col_gap = med_h * _COL_GAP_FACTOR

    # Group lines into visual rows by vertical overlap of y-ranges.
    rows: list[list[OcrLine]] = []
    for line in lines:
        if rows and _overlaps(rows[-1][-1], line):
            rows[-1].append(line)
        else:
            rows.append([line])
    for row in rows:
        row.sort(key=lambda l: l.x0)

    exercises: list[dict[str, Any]] = []
    pending_name: str | None = None

    def flush_pending() -> None:
        nonlocal pending_name
        if pending_name is not None:
            exercises.append({"name": pending_name, "sets": []})
            pending_name = None

    for row in rows:
        text = " ".join(l.text for l in row).strip()
        if not text:
            continue

        # Cardio row?
        if _looks_cardio(text):
            flush_pending()
            c = _parse_cardio_row(row)
            if c:
                out["cardio"].append(c)
            continue

        # Table style: leftmost cell is a name, the rest carry set values.
        if len(row) >= 2 and (row[1].x0 - row[0].x1) > col_gap:
            name = row[0].text.strip()
            set_text = " ".join(l.text for l in row[1:])
            parsed = _parse_set_text(set_text)
            if parsed:
                flush_pending()
                exercises.append({"name": name, "sets": parsed})
                continue
            # No numbers on the right — treat whole row as a name line.
            flush_pending()
            pending_name = _clean_name(text)
            continue

        # Inline style: numbers inside a single line.
        parsed = _parse_set_text(text)
        has_num = bool(parsed)
        if pending_name is not None and has_num:
            exercises.append({"name": pending_name, "sets": parsed})
            pending_name = None
            continue
        if has_num and _has_unit_or_sep(text):
            # Values line without a preceding name — attach to nothing;
            # keep as name-only row if it also has words.
            if re.search(r"[A-Za-z]{3,}", text) and not _is_values_only(text):
                flush_pending()
                pending_name = _clean_name(text)
            continue
        # Pure words -> exercise name (or session note).
        flush_pending()
        pending_name = _clean_name(text)

    flush_pending()

    # Drop empty rows the way the LLM path does.
    for e in exercises:
        e["sets"] = [s for s in e["sets"]
                     if (s.get("reps") or 0) > 0 or s.get("weight_kg") is not None
                     or s.get("assist_kg") is not None
                     or s.get("duration_seconds") is not None]
    out["exercises"] = [e for e in exercises if e["sets"]]
    return out


def _overlaps(a: OcrLine, b: OcrLine) -> bool:
    """Do two lines vertically overlap enough to be one visual row?"""
    inter = min(a.y1, b.y1) - max(a.y0, b.y0)
    return inter > 0.4 * min(a.height, b.height)


def _is_values_only(text: str) -> bool:
    stripped = re.sub(r"[\d.,:x×kgsecore@+\-\s/]+", "", text, flags=re.IGNORECASE)
    return len(stripped) <= 2


def _has_unit_or_sep(text: str) -> bool:
    return bool(re.search(r"(kg|lb|x|×|:|sec|min)", text, re.IGNORECASE))


def _clean_name(text: str) -> str:
    # Strip trailing colons and set-count prefixes like "1." / "1)".
    return re.sub(r"^\s*\d+[.)]\s*", "", text).strip(" :\t-")


def _parse_cardio_row(row: list[OcrLine]) -> dict[str, Any] | None:
    text = " ".join(l.text for l in row)
    low = text.lower()
    atype = _cardio_type(text)
    dist = None
    dur = None
    m = KM_RE.search(text)
    if m:
        dist = _to_float(m.group(1))
    m = DURATION_RE.search(text)
    if m:
        if m.group(1) is not None:
            h = int(m.group(1))
            mm = int(m.group(2))
            sec = int(m.group(3) or 0)
            dur = round(h * 60 + mm + sec / 60, 2)
        else:
            dur = _to_float(m.group(4))
    if dist is None and dur is None:
        return None
    return {"activity_type": atype, "distance_km": dist,
            "duration_min": dur, "notes": text if atype == "other" else None}