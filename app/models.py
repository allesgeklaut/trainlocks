from sqlalchemy import Boolean, Date, DateTime, Float, ForeignKey, Integer, String, Text
from sqlalchemy.orm import DeclarativeBase, Mapped, mapped_column, relationship
from datetime import date as date_type, datetime, timezone
from .database import Base


class Exercise(Base):
    __tablename__ = "exercises"
    id: Mapped[int] = mapped_column(Integer, primary_key=True, index=True)
    name: Mapped[str] = mapped_column(String, unique=True, index=True)
    is_bodyweight: Mapped[bool] = mapped_column(Boolean, default=False)
    # % of bodyweight used as load for this exercise (0..1). NULL means
    # "use the research default" (1.0 for pull-ups/dips, 0.65 for push-ups,
    # …) — see app/load.py. NULL is preserved so unknown exercises keep
    # the legacy full-bodyweight behavior until backfilled.
    bw_load_factor: Mapped[float | None] = mapped_column(Float, nullable=True)
    # Isometric hold exercise (plank, L-sit, hang, handstand hold): sets are
    # logged as seconds under tension instead of reps. Progression charts
    # plot total hold time; tonnage doesn't apply.
    is_hold: Mapped[bool] = mapped_column(Boolean, default=False, nullable=False)


class SessionTemplate(Base):
    __tablename__ = "session_templates"
    id: Mapped[int] = mapped_column(Integer, primary_key=True, index=True)
    name: Mapped[str] = mapped_column(String, unique=True, index=True)
    # Free-text description from imported plans (goal, level, progression
    # notes). Rendered on the templates page and as a session-form hint.
    description: Mapped[str | None] = mapped_column(Text, nullable=True)
    exercises: Mapped[list["SessionTemplateExercise"]] = relationship(
        "SessionTemplateExercise",
        back_populates="session_template",
        cascade="all, delete-orphan",
        order_by="SessionTemplateExercise.order",
    )


class SessionTemplateExercise(Base):
    __tablename__ = "session_template_exercises"
    id: Mapped[int] = mapped_column(Integer, primary_key=True, index=True)
    session_template_id: Mapped[int | None] = mapped_column(Integer, ForeignKey("session_templates.id"))
    exercise_id: Mapped[int | None] = mapped_column(Integer, ForeignKey("exercises.id"))
    sets: Mapped[int | None] = mapped_column(Integer)
    order: Mapped[int | None] = mapped_column(Integer)
    # Set/rep scheme label from imported plans, e.g. "5x3+" or "3x8-12".
    prescription: Mapped[str | None] = mapped_column(String, nullable=True)
    session_template: Mapped["SessionTemplate | None"] = relationship("SessionTemplate", back_populates="exercises")
    exercise: Mapped["Exercise | None"] = relationship("Exercise")


class WorkoutSession(Base):
    __tablename__ = "workout_sessions"
    id: Mapped[int] = mapped_column(Integer, primary_key=True, index=True)
    date: Mapped[date_type | None] = mapped_column(Date, index=True)
    template_id: Mapped[int | None] = mapped_column(Integer, ForeignKey("session_templates.id"), nullable=True)
    notes: Mapped[str | None] = mapped_column(String, nullable=True)
    template: Mapped["SessionTemplate | None"] = relationship("SessionTemplate")
    sets: Mapped[list["SetEntry"]] = relationship("SetEntry", back_populates="session", cascade="all, delete-orphan")
    cardio: Mapped[list["CardioActivity"]] = relationship("CardioActivity", back_populates="session", cascade="all, delete-orphan", order_by="CardioActivity.id")


class CardioActivity(Base):
    __tablename__ = "cardio_activities"
    id: Mapped[int] = mapped_column(Integer, primary_key=True, index=True)
    session_id: Mapped[int | None] = mapped_column(Integer, ForeignKey("workout_sessions.id"), index=True)
    activity_type: Mapped[str] = mapped_column(String, nullable=False)
    distance_km: Mapped[float | None] = mapped_column(Float, nullable=True)
    duration_min: Mapped[float | None] = mapped_column(Float, nullable=True)
    # AI extraction pours screenshot metrics (pace, HR, elevation, …) into
    # notes — unbounded text, so Column(Text) not VARCHAR.
    notes: Mapped[str | None] = mapped_column(Text, nullable=True)
    session: Mapped["WorkoutSession | None"] = relationship("WorkoutSession", back_populates="cardio")


class SetEntry(Base):
    __tablename__ = "set_entries"
    id: Mapped[int] = mapped_column(Integer, primary_key=True, index=True)
    session_id: Mapped[int | None] = mapped_column(Integer, ForeignKey("workout_sessions.id"))
    exercise_id: Mapped[int | None] = mapped_column(Integer, ForeignKey("exercises.id"))
    set_number: Mapped[int | None] = mapped_column(Integer)
    reps: Mapped[int | None] = mapped_column(Integer)
    # Isometric holds (planks, L-sits, hangs): reps stores the hold time in
    # SECONDS and load is tracked as time-under-tension, not tonnage. NULL
    # for normal rep-based sets. Driven by the exercise-level is_hold flag.
    duration_seconds: Mapped[int | None] = mapped_column(Integer, nullable=True)
    weight: Mapped[float | None] = mapped_column(Float, nullable=True)
    # Counterweight support for BW exercises (supported dips, assisted
    # pull-up machines): kg of bodyweight the machine removes, subtracted
    # from the effective load. NULL = no support.
    assist_kg: Mapped[float | None] = mapped_column(Float, nullable=True)
    session: Mapped["WorkoutSession"] = relationship("WorkoutSession", back_populates="sets")
    exercise: Mapped["Exercise | None"] = relationship("Exercise")


class User(Base):
    __tablename__ = "users"
    id: Mapped[int] = mapped_column(Integer, primary_key=True, index=True)
    username: Mapped[str] = mapped_column(String, unique=True, index=True, nullable=False)
    hashed_password: Mapped[str] = mapped_column(String, nullable=False)
    bodyweight: Mapped[float | None] = mapped_column(Float, nullable=True)


class CoachChatMessage(Base):
    """Persistent conversation history for the fitness-coach chat.

    Stored so the active chat session survives page reloads and the LLM can
    be fed the full prior context on every turn.
    """

    __tablename__ = "coach_chat_messages"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, index=True)
    # user | assistant
    role: Mapped[str] = mapped_column(String(16), nullable=False)
    content: Mapped[str] = mapped_column(Text, nullable=False)
    # Timezone-aware UTC; stored naive in SQLite for consistency with the
    # rest of the schema (ordering is only ever compared within this column).
    created_at: Mapped[datetime | None] = mapped_column(
        DateTime,
        default=lambda: datetime.now(timezone.utc).replace(tzinfo=None),
        index=True,
    )