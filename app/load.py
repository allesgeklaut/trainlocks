"""Single source of truth for bodyweight-exercise load scaling.

Every place that turns a ``SetEntry`` into a chartable load (dashboard
weekly load, lifetime tonnage SQL, progression/e1RM, AI-coach context)
uses :func:`effective_load_kg` so the semantics cannot drift apart.

Model (per set)::

    effective_load = max(0, bodyweight * factor + added_kg - assist_kg)

* ``factor`` scales how much of the body is actually moved against
  gravity (a push-up is ~65 % BW per force-plate research, a pull-up is
  ~100 %). Stored per exercise as ``exercises.bw_load_factor``; the
  research defaults below backfill existing rows at startup.
* ``assist_kg`` captures machine/counterweight support (supported dips,
  assisted pull-ups) and is subtracted, clamped at zero.

References for the default factors:
- Ebben et al. 2011 (force plates, push-up variants)
- Suprak, Dawes & Stephenson 2011 (JSCR, push-up % body mass)
- Padulo et al. 2018 systematic review (41 push-up variants)
- Melrose & Dawes 2014 (inverted row % BW by body angle)
- ExRx.net body-segment analysis (squat/split squat/step-up/pistol)
"""

from __future__ import annotations

from .models import Exercise

# Fallback for bodyweight-exercise load scaling when the user hasn't set a
# bodyweight in their profile.
BODYWEIGHT_DEFAULT_KG = 80.0

# Name-pattern -> % of bodyweight used as load, checked in order (first
# match wins). Case-insensitive substring match on the exercise name.
# Most-specific patterns first: "knee push up" before "push up".
BW_LOAD_FACTORS: tuple[tuple[str, float], ...] = (
    ("knee push", 0.49),
    ("kneeling push", 0.49),
    ("hands elevated push", 0.50),
    ("incline push", 0.50),
    ("feet elevated push", 0.72),
    ("decline push", 0.72),
    ("feet elevated pike", 0.77),
    ("pike push", 0.66),
    ("push up", 0.65),
    ("pushup", 0.65),
    ("push-up", 0.65),
    ("inverted row", 0.70),
    ("ring row", 0.70),
    ("bodyweight row", 0.70),
    ("australian pull", 0.70),
    ("handstand push", 1.00),
    ("muscle up", 1.00),
    ("pull up", 1.00),
    ("pullup", 1.00),
    ("pull-up", 1.00),
    ("chin up", 1.00),
    ("chinup", 1.00),
    ("chin-up", 1.00),
    ("dip", 1.00),
    ("pistol squat", 0.89),
    ("pistol", 0.89),
    ("step up", 0.89),
    ("stepup", 0.89),
    ("split squat", 0.85),
    ("bodyweight squat", 0.77),
    ("lunge", 0.85),
    ("squat", 0.77),
    ("plank", 0.00),
    ("handstand hold", 1.00),
    ("planche", 1.00),
    ("human flag", 1.00),
)

# Name-pattern -> is_bodyweight guess, checked in order (first match wins).
# Used by AI screenshot extraction and plan import for newly created
# exercises. Moved here from ai.py so all creation paths share it.
_BODYWEIGHT_HINTS = (
    "pull up", "pullup", "pull-up", "push up", "pushup", "push-up",
    "dip", "muscle up", "pistol squat", "planche", "human flag",
    "handstand push", "chin up", "chinup", "chin-up", "plank",
    "inverted row", "ring row", "bodyweight row", "bodyweight squat",
    "lunge", "step up",
)


def bw_name_matches(name: str, patterns) -> bool:
    """Case-insensitive substring match: first matching pattern wins."""
    norm = (name or "").lower()
    return any(p in norm for p in patterns)


def default_bw_load_factor(name: str) -> float | None:
    """Research default factor for an exercise name, or None if unknown."""
    norm = (name or "").lower()
    for pattern, factor in BW_LOAD_FACTORS:
        if pattern in norm:
            return factor
    return None


def is_bodyweight_name(name: str) -> bool:
    """True when the name looks like a bodyweight exercise."""
    return bw_name_matches(name, _BODYWEIGHT_HINTS)


def exercise_load_factor(exercise: Exercise | None) -> float:
    """Effective %BW factor for an exercise row (default 1.0 for BW exercises)."""
    if exercise is None or not exercise.is_bodyweight:
        return 0.0
    stored = getattr(exercise, "bw_load_factor", None)
    return float(stored) if stored is not None else 1.0


def effective_load_kg(exercise: Exercise | None,
                      weight: float | None,
                      assist: float | None = None,
                      bodyweight: float | None = None) -> float:
    """Chartable load for one set, in kg.

    Bodyweight exercises: ``BW * factor + added - assist`` (clamped >= 0).
    Weighted exercises: just ``weight`` (the logged bar/dumbbell load).
    ``bodyweight`` falls back to BODYWEIGHT_DEFAULT_KG when unset.
    """
    if exercise is not None and exercise.is_bodyweight:
        bw = float(bodyweight) if bodyweight is not None else BODYWEIGHT_DEFAULT_KG
        factor = exercise_load_factor(exercise)
        added = float(weight) if weight is not None else 0.0
        assist_kg = float(assist) if assist is not None else 0.0
        return max(0.0, bw * factor + added - assist_kg)
    return float(weight) if weight is not None else 0.0