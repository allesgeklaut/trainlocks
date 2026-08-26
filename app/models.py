from sqlalchemy import Column, Integer, String, Boolean, ForeignKey, Date, Float
from sqlalchemy.orm import relationship
from .database import Base


class Exercise(Base):
    __tablename__ = "exercises"
    id = Column(Integer, primary_key=True, index=True)
    name = Column(String, unique=True, index=True)
    is_bodyweight = Column(Boolean, default=False)


class SessionTemplate(Base):
    __tablename__ = "session_templates"
    id = Column(Integer, primary_key=True, index=True)
    name = Column(String, unique=True, index=True)
    exercises = relationship(
        "SessionTemplateExercise",
        back_populates="session_template",
        cascade="all, delete-orphan",
        order_by="SessionTemplateExercise.order",
    )


class SessionTemplateExercise(Base):
    __tablename__ = "session_template_exercises"
    id = Column(Integer, primary_key=True, index=True)
    session_template_id = Column(Integer, ForeignKey("session_templates.id"))
    exercise_id = Column(Integer, ForeignKey("exercises.id"))
    sets = Column(Integer)
    order = Column(Integer)
    session_template = relationship("SessionTemplate", back_populates="exercises")
    exercise = relationship("Exercise")


class WorkoutSession(Base):
    __tablename__ = "workout_sessions"
    id = Column(Integer, primary_key=True, index=True)
    date = Column(Date, index=True)
    template_id = Column(Integer, ForeignKey("session_templates.id"), nullable=True)
    notes = Column(String, nullable=True)
    template = relationship("SessionTemplate")
    sets = relationship("SetEntry", back_populates="session", cascade="all, delete-orphan")


class SetEntry(Base):
    __tablename__ = "set_entries"
    id = Column(Integer, primary_key=True, index=True)
    session_id = Column(Integer, ForeignKey("workout_sessions.id"))
    exercise_id = Column(Integer, ForeignKey("exercises.id"))
    set_number = Column(Integer)
    reps = Column(Integer)
    weight = Column(Float, nullable=True)
    session = relationship("WorkoutSession", back_populates="sets")
    exercise = relationship("Exercise")


class User(Base):
    __tablename__ = "users"
    id = Column(Integer, primary_key=True, index=True)
    username = Column(String, unique=True, index=True, nullable=False)
    hashed_password = Column(String, nullable=False)
