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
# \b doesn't help when "lb" is glued to digits ("135lb") — match unit tokens
# with an optional word boundary in front and require a boundary after.
LB_HINT_RE = re.compile(r"(?:\b|(?<=\d))lbs?\b", re.IGNORECASE)
# "10 x 80kg" (unit after second number)
REPS_FIRST_UNIT_RE = re.compile(
    r"(\d+)\s*(?:x|×)\s*(-?\d+(?:[.,]\d+)?)\s*(?:kg|kgs|kilos?|lb|lbs)\b",
    re.IGNORECASE)
# "10 x 80" — reps x bare value (sign kept: "8 x -25kg" assisted)
REPS_FIRST_RE = re.compile(r"(\d+)\s*(?:x|×)\s*(-?\d+(?:[.,]\d+)?)")
# "80kg x 10" | "135lb x 5" — weight-with-unit first
UNIT_FIRST_RE = re.compile(
    r"(\d+(?:[.,]\d+)?)\s*(?:kg|kgs|kilos?|lb|lbs)\b\s*(?:x|×)\s*(\d+)",
    re.IGNORECASE)
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

# ── Screenshot summary-card extraction (Apple Watch / fitness-app cards) ─────
#
# These screenshots are a grid of label/value pairs, usually two columns
# ("Workout Time | Distance" / "0:50:38 | 6,42KM"), sometimes one label per
# row. Labels sit either in the same visual row as their value or in the row
# directly above. Anything not matching a metric label is context (title,
# subtitle, location, time-of-day range) and goes into the notes.

# Label (lowercased, stripped) -> internal metric key. Unmatched labels are
# ignored so stray UI text ("Workout Details >", tab bars) is never captured.
_METRIC_LABELS: dict[str, str] = {
    "workout time": "time", "duration": "time", "active time": "time",
    "distance": "distance", "total distance": "distance",
    "avg pace": "pace", "average pace": "pace", "pace": "pace",
    "avg heart rate": "hr", "average heart rate": "hr",
    "heart rate": "hr", "avg hr": "hr",
    "avg power": "power", "average power": "power",
    "avg cadence": "cadence", "average cadence": "cadence",
    "active kilocalories": "active_kcal", "active calories": "active_kcal",
    "total kilocalories": "total_kcal", "total calories": "total_kcal",
    "calories": "total_kcal", "energy": "total_kcal",
    "elevation gain": "elevation", "elevation asc": "elevation",
    "elevation": "elevation",
    "avg speed": "speed", "average speed": "speed",
    "avg stride length": "stride",
}
# A value cell: number+unit, bare time, or pace ("7'53"/KM", "5:12 /km").
_METRIC_VALUE_RE = re.compile(
    r"\d+(?:[.,]\d+)?\s*(?:km|kcal|cal|w(?:att)?s?|spm|bpm|km/h|min/km|m/km"
    r"|mi|ft|mm|kg|l|ml|floz|oz)\b"
    r"|\d+'\d+\"?\s*/\s*km"
    r"|\d+(?:[.,]\d+)?\s*/\s*km"
    r"|\d{1,2}:\d{2}(?::\d{2})?"
    r"|\d+(?:[.,]\d+)?%?",
    re.IGNORECASE)
# Time-of-day window the workout happened in ("06:52-07:43").
_TIME_OF_DAY_RE = re.compile(r"\b\d{1,2}:\d{2}\s*[-–]\s*\d{1,2}:\d{2}\b")
# UI noise never kept as context/notes: status bar, tab bar, nav rows, page
# dots, battery percent, clock-only rows.
_STATUS_NOISE_RE = re.compile(
    r"^(?:wlan|wifi|lte|5g|4g|call|battery|\d+\s*%)\b|"
    r"^\d{1,2}:\d{2}$|^\d+\s*%?$|^[+·•]+$|"
    r"^(?:summary|fitness\+?|workout(?:\s+details\s*>?)?|sharing|done|done\.)$",
    re.IGNORECASE)
# Human order for the notes line.
_METRIC_ORDER = ("time", "distance", "pace", "hr", "power", "cadence",
                 "speed", "active_kcal", "total_kcal", "elevation", "stride")
_METRIC_NOTES_LABEL = {
    "time": "time", "distance": "distance", "pace": "avg pace", "hr": "avg HR",
    "power": "avg power", "cadence": "avg cadence", "speed": "avg speed",
    "active_kcal": "active kcal", "total_kcal": "total kcal",
    "elevation": "elevation", "stride": "avg stride",
}


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
        # If the text carries an explicit 4-digit year, honour it.
        ym = re.search(r"\b(20\d{2})\b", text)
        year = int(ym.group(1)) if ym else today.year
        month = _MONTHS.get((mon or mon2 or "").lower())
        day_num = int(day or _d2 or 0)
        if month and day_num:
            got = clamp_past(year, month, day_num)
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
    # Split on set separators, but not inside decimal numbers ("62,5kg").
    # "10, 80" between digits stays one chunk only when it reads as a
    # decimal (digit,digit with single digits); ", " (comma+space) always
    # separates. Heuristic: a comma followed by whitespace or a digit-pair
    # with a space splits; "62,5" (no space) does not.
    chunks = [c.strip() for c in re.split(r",\s|;\s*|(?<!\d),(?!\d)", text)
              if c.strip()]
    for chunk in chunks:
        chunk_lb = is_lb or bool(LB_HINT_RE.search(chunk))
        s: dict[str, Any] = {"reps": None, "weight_kg": None,
                             "assist_kg": None, "duration_seconds": None}

        # Timed holds: "1:30", "45s", "90 sec", "3 x 60s"
        hold = HOLD_TIME_RE.search(chunk)
        plain = PLAIN_SETS_RE.search(chunk)
        hold_like = hold is not None and (
            "hold" in chunk.lower()
            or re.search(r"\b(?:sec|secs|seconds)\b", chunk, re.IGNORECASE) is not None
            or re.search(r"(?<=\d)\s*s\b", chunk, re.IGNORECASE) is not None
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

        # Weight-first: "80kg x 10" / "135lb x 5"
        m = UNIT_FIRST_RE.search(chunk)
        if m:
            weight = _to_float(m.group(1))
            reps = int(m.group(2))
            if chunk_lb and weight is not None:
                weight = _lb_to_kg(weight)
            s["reps"], s["weight_kg"] = reps, weight
            _apply_assist(chunk, s)
            sets.append(s)
            continue

        # Reps-first with unit: "10 x 80kg" / "8 x -25kg" (assisted)
        m = REPS_FIRST_UNIT_RE.search(chunk)
        if m:
            reps = int(m.group(1))
            weight = _to_float(m.group(2))
            if weight is not None and weight < 0:
                s["assist_kg"] = abs(weight)
                weight = None
            elif chunk_lb and weight is not None:
                weight = _lb_to_kg(weight)
            s["reps"], s["weight_kg"] = reps, weight
            _apply_assist(chunk, s)
            sets.append(s)
            continue

        # Reps-first bare: "10 x 80" / "8 x -25kg" — second number is the
        # per-set load only if the chunk carries a kg/lb hint anywhere;
        # otherwise it's sets x reps ("3 x 8" = 3 sets of 8).
        m = REPS_FIRST_RE.search(chunk)
        if m:
            weight = _to_float(m.group(2)) if (
                chunk_lb or re.search(r"\bkg\b", chunk, re.IGNORECASE)) else None
            if weight is None:
                # No unit: sets x reps — expand to one entry per set. But a
                # negative second number is never a set count: "8 x -25kg"
                # (unit present) is handled below; a bare "8 x -3" is junk.
                for _ in range(int(m.group(1))):
                    sets.append({"reps": int(m.group(2)), "weight_kg": None,
                                 "assist_kg": None, "duration_seconds": None})
                continue
            reps = int(m.group(1))
            if weight < 0:
                # Negative load on a reps-first row is an assisted load.
                s["assist_kg"] = abs(weight)
                weight = None
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
    # Phone screenshots carry a status bar (carrier/clock/battery) at the top
    # and a tab bar at the bottom — drop both bands entirely.
    max_y = max(l.y1 for l in lines)
    top_band = max_y * 0.05
    bottom_band = max_y * 0.95

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

    # Summary-card metrics (Apple-Watch-style label/value grids) become the
    # session notes; their rows must not leak in as exercise names. A row is
    # a metric row when it carries a known metric label; bare value rows
    # only count when they carry a unit (kg/... — not hold times or dates,
    # which belong to the workout rows).
    metrics, context = _extract_card_metrics(rows)
    metric_rows: set[int] = set()
    for i, row in enumerate(rows):
        text = " ".join(l.text for l in row).strip()
        if not text or _is_noise_row(text) or _looks_cardio(text):
            continue
        if any(c.text.lower().strip() in _METRIC_LABELS for c in row):
            metric_rows.add(i)
            continue
        if (len(row) <= 2
                and all(
                    _METRIC_VALUE_RE.search(c.text)
                    and not re.search(r"(?:kg|lb|x|×)", c.text, re.IGNORECASE)
                    for c in row)
                and any(_METRIC_VALUE_RE.search(c.text) for c in row)
                and any(re.search(r"[A-Za-z]{3}", c.text) is None
                        for c in row)):
            metric_rows.add(i)

    def flush_pending() -> None:
        nonlocal pending_name
        if pending_name is not None:
            exercises.append({"name": pending_name, "sets": []})
            pending_name = None

    for idx, row in enumerate(rows):
        text = " ".join(l.text for l in row).strip()
        if not text:
            continue
        if idx in metric_rows:
            continue
        # Status-bar / tab-bar bands (full-width UI, never workout data).
        if row[0].y1 < top_band or row[0].y0 > bottom_band:
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

    # Notes: summary-card context (title/subtitle/location/time) + metrics.
    title = next((t for t in context
                  if _cardio_type(t) != "other"
                  or re.search(r"(?i)\b(run|ride|swim|walk|row|hike|session)\b", t)),
                 None)
    out["notes"] = format_metric_notes(
        metrics, [t for t in context if t != title], title=title) or None

    # Summary-card screenshots with a cardio title ("Outdoor Run") and no
    # strength rows are a cardio session — synthesize the cardio entry from
    # the captured metrics so the review form shows the cardio flow.
    if metrics and not out["exercises"] and not out["cardio"] and title:
        atype = _cardio_type(title)
        if atype != "other":
            time_v = metrics.get("time")
            dist_v = metrics.get("distance")
            dur = None
            dist = None
            if time_v:
                tm = DURATION_RE.search(time_v)
                if tm:
                    if tm.group(3) is not None:
                        dur = round(int(tm.group(1)) * 60 + int(tm.group(2))
                                    + int(tm.group(3)) / 60, 2)
                    else:
                        dur = round(int(tm.group(1)) + int(tm.group(2)) / 60, 2)
            if dist_v:
                km = KM_RE.search(dist_v)
                if km:
                    dist = _to_float(km.group(1))
            if dist is not None or dur is not None:
                out["cardio"].append({
                    "activity_type": atype, "distance_km": dist,
                    "duration_min": dur,
                    "notes": " ".join(t for t in context if t != title) or None,
                })
    return out


def _overlaps(a: OcrLine, b: OcrLine) -> bool:
    """Do two lines vertically overlap enough to be one visual row?"""
    inter = min(a.y1, b.y1) - max(a.y0, b.y0)
    return inter > 0.4 * min(a.height, b.height)


def _is_values_only(text: str) -> bool:
    stripped = re.sub(r"[\d.,:x×kgsecore@+\-\s/]+", "", text, flags=re.IGNORECASE)
    # "Graz"-like words made only of charset letters count as values only
    # when the original also carries a digit/separator.
    return len(stripped) <= 2 and bool(
        re.search(r"[\d.,:x×@+\-/]", text))


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
            # "1:43:34" = H:MM:SS; bare "43:34" = MM:SS (43.57 min).
            if m.group(3) is not None:
                h, mm, sec = int(m.group(1)), int(m.group(2)), int(m.group(3))
                dur = round(h * 60 + mm + sec / 60, 2)
            else:
                dur = round(int(m.group(1)) + int(m.group(2)) / 60, 2)
        else:
            dur = _to_float(m.group(4))
    if dist is None and dur is None:
        return None
    return {"activity_type": atype, "distance_km": dist,
            "duration_min": dur, "notes": text if atype == "other" else None}


# ---------------------------------------------------------------------------
# Summary-card extraction: label/value grid + context -> notes enrichment
# ---------------------------------------------------------------------------

def _is_noise_row(text: str) -> bool:
    """Status bar / tab bar / page dots / nav rows — never context.
    Time-of-day ranges ("06:52-07:43") are context, not noise."""
    if _TIME_OF_DAY_RE.search(text):
        return False
    return bool(_STATUS_NOISE_RE.search(text)) or not re.search(
        r"[A-Za-z]{2,}", text)


def _extract_card_metrics(
        rows: list[list[OcrLine]]) -> tuple[dict[str, str], list[str]]:
    """Pull structured metric values + context lines from a workout-summary
    screenshot (Apple Watch card style).

    Labels sit either in the same visual row as their value (two-column
    grids) or in the row directly above (single-column lists). Returns
    (metric_key -> raw value text, context lines kept for the notes).
    """
    metrics: dict[str, str] = {}
    context: list[str] = []
    max_y = max((l.y1 for row in rows for l in row), default=0.0)

    def capture(label: str, key: str, value: str | None) -> None:
        if value and _METRIC_VALUE_RE.search(value) and key not in metrics:
            metrics[key] = value.strip()

    def is_noise_cell(text: str) -> bool:
        return _is_noise_row(text.strip())

    def is_noise_row_row(row: list[OcrLine]) -> bool:
        """A visual row is noise when every cell individually is noise —
        catches grouped tab bars ("Summary | Fitness+ | Workout | Sharing")."""
        return bool(row) and all(is_noise_cell(c.text) for c in row)

    for i, row in enumerate(rows):
        text = " ".join(l.text for l in row).strip()
        if not text or is_noise_row_row(row):
            continue
        if _looks_cardio(text) or _parse_set_text(text):
            # Real workout rows are handled by the layout parser — but
            # cardio-like titles without any numbers ("Outdoor Run",
            # "Easy run") carry no data, so keep them as context instead
            # of silently dropping them.
            if _looks_cardio(text) and not _parse_set_text(text) \
                    and not KM_RE.search(text) and not DURATION_RE.search(text):
                context.append(text)
            continue
        # Status-bar / tab-bar bands: skip from context as well.
        if row[0].y1 < max_y * 0.05 or row[0].y0 > max_y * 0.95:
            continue
        row_is_label = any(
            c.text.lower().strip() in _METRIC_LABELS for c in row)
        next_row = rows[i + 1] if i + 1 < len(rows) else []
        for j, cell in enumerate(row):
            key = _METRIC_LABELS.get(cell.text.lower().strip())
            if key is None:
                continue
            value: str | None = None
            # Value in the same row, to the right of the label.
            if j + 1 < len(row) and _METRIC_VALUE_RE.search(row[j + 1].text):
                value = row[j + 1].text
            # ...or in the row directly below, same column (aligned grids).
            elif (len(next_row) == len(row)
                  and _METRIC_VALUE_RE.search(next_row[j].text)):
                value = next_row[j].text
            elif (len(next_row) == 1
                  and _METRIC_VALUE_RE.search(next_row[0].text)
                  and len(row) == 1):
                value = next_row[0].text
            capture(cell.text, key, value)
        # Value-only rows whose label row was above are captured by the
        # label loop; anything else with words is context.
        if not row_is_metric_only(row, metrics):
            low = text.lower()
            if (not _TIME_OF_DAY_RE.search(text)
                    and not any(c.text.lower().strip() in _METRIC_LABELS
                                for c in row)
                    and not _is_values_only(text)):
                context.append(text)

    # Time-of-day rows ("06:52-07:43") — keep verbatim, they read naturally.
    for row in rows:
        text = " ".join(l.text for l in row).strip()
        if text and not is_noise_row_row(row) and _TIME_OF_DAY_RE.search(text) \
                and text not in context:
            context.append(text)

    return metrics, context


def row_is_metric_only(row: list[OcrLine], metrics: dict[str, str]) -> bool:
    """True when every cell in the row is either a known label or a value
    that looks like a metric (so it shouldn't repeat as context)."""
    for cell in row:
        low = cell.text.lower().strip()
        if low in _METRIC_LABELS or _METRIC_VALUE_RE.search(cell.text):
            continue
        if re.search(r"[A-Za-z]{2,}", cell.text):
            return False
    return True


def format_metric_notes(metrics: dict[str, str], context: list[str],
                        title: str | None = None) -> str:
    """Compose the notes text: context first, then metrics in fixed order."""
    bits: list[str] = []
    if title:
        bits.append(title)
    bits.extend(context)
    metric_bits = []
    for key in _METRIC_ORDER:
        if key in metrics:
            metric_bits.append(
                f"{_METRIC_NOTES_LABEL[key]} {metrics[key]}")
    for key, val in metrics.items():
        if key not in _METRIC_ORDER:
            metric_bits.append(f"{key} {val}")
    if metric_bits:
        bits.append(", ".join(metric_bits))
    return " — ".join(b for b in bits if b) or ""