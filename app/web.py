"""Shared web helpers used by both main.py and the AI router.

Kept separate to avoid a circular import (main.py includes ai.router,
while ai.py needs render_page and the load-factor helpers).
"""

from __future__ import annotations

from datetime import date
from typing import Optional

from fastapi import Request
from fastapi.templating import Jinja2Templates

templates = Jinja2Templates(directory="app/templates")

# Fallback for bodyweight-exercise load scaling when the user hasn't set a
# bodyweight in their profile.
BODYWEIGHT_DEFAULT_KG = 80.0
CARDIO_LOAD_DEFAULT_FACTOR = 1.0
# Dashboard cardio load: per-activity effort multiplier applied to distance (km).
CARDIO_LOAD_FACTOR = {"running": 1.0, "swimming": 3.0}
# Canonical activity types accepted by the form and API routes; must match
# the <option> lists in cardio.html / cardio_edit.html.
CARDIO_ACTIVITY_TYPES = ("running", "swimming", "cycling", "walking", "rowing", "other")


def render_page(request: Request, template_name: str, context: dict,
                status_code: int = 200):
    """Render a page response, as a full document or an htmx fragment.

    Boosted navigation swaps #app-shell (sidebar + main + page scripts).
    If htmx received a full document, it would insert *all* of the
    response body's children at the swap point (htmx P() puts the entire
    body into the fragment) — duplicating the fixed topbar/overlay/indicator
    and re-running base scripts on every navigation. So htmx requests are
    answered with a shell-only layout (base_shell.html): title + page
    styles + #app-shell. Full documents (base.html) are only sent to
    normal loads, which the browser renders from scratch.

    History restores fetch the URL without an HX-Request header, so they
    correctly receive full documents.
    """
    is_htmx = "hx-request" in {k.lower() for k in request.headers}
    # Per-request context (not app.state): concurrent renders must not
    # race on the layout choice.
    context.setdefault("layout", "base_shell.html" if is_htmx else "base.html")
    return templates.TemplateResponse(request, template_name, context,
                                      status_code=status_code)


def _cardio_load_factor(activity_type: str) -> float:
    return CARDIO_LOAD_FACTOR.get((activity_type or "").lower(), CARDIO_LOAD_DEFAULT_FACTOR)


def _ai_duration_value(minutes) -> str:
    """Prefill for the AI review form's duration input ('45' or '43:34').

    Lives on the templates env as a global so ai_session_review.html can
    format decimal minutes the same way cardio_edit.html does.
    """
    if minutes is None:
        return ""
    total = round(float(minutes) * 60)
    m, s = divmod(total, 60)
    return str(m) if s == 0 else f"{m}:{s:02d}"


templates.env.globals["_ai_duration_value"] = _ai_duration_value


def _parse_duration_min(value) -> float | None:
    """Parse a form duration: minutes ('45', '44.85') or M:SS / H:MM:SS ('44:51').

    Raises ValueError for non-empty values that don't match.
    """
    if value is None:
        return None
    if isinstance(value, (int, float)):
        return float(value)
    value = value.strip()
    if not value:
        return None
    parts = value.split(":")
    if len(parts) in (2, 3) and all(p.strip().isdigit() for p in parts):
        nums = [int(p) for p in parts]
        if len(nums) == 2:
            h, m, s = 0, nums[0], nums[1]
        else:
            h, m, s = nums
        if not (0 <= m < 60 and 0 <= s < 60):
            raise ValueError("invalid duration")
        return h * 60 + m + s / 60
    try:
        return float(value)
    except ValueError:
        raise ValueError("invalid duration")


def _iso_week_key(d: date) -> str:
    """Canonical chart bucket key for a date, e.g. '2026-W36' (ISO, Monday-aligned)."""
    iso = d.isocalendar()
    return f"{iso[0]}-W{iso[1]:02d}"


def _cardio_pace_unit(c) -> str:
    """Distance unit the pace is expressed against (swimming: per 100 m)."""
    return "100m" if (c.activity_type or "").lower() == "swimming" else "km"


def _cardio_pace(c) -> Optional[float]:
    """Minutes per pace unit (per km, or per 100 m for swimming), or None
    when we lack the distance or duration."""
    if not c.distance_km or not c.duration_min:
        return None
    if c.distance_km <= 0:
        return None
    if _cardio_pace_unit(c) == "100m":
        return round(c.duration_min / (c.distance_km * 10), 2)
    return round(c.duration_min / c.distance_km, 2)


def _cardio_json(c) -> dict:
    return {
        "id": c.id,
        "session_id": c.session_id,
        "activity_type": c.activity_type,
        "distance_km": c.distance_km,
        "duration_min": c.duration_min,
        "pace": _cardio_pace(c),
        "pace_unit": _cardio_pace_unit(c),
        "notes": c.notes,
    }