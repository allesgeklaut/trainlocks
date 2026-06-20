#!/usr/bin/env python3
"""
Script to generate a test database with sample data for testing the application.
This creates a properly initialized SQLite database with exercises and sessions 
that match the expectations of the existing tests.
"""

import os
import sys
import tempfile
from datetime import date, timedelta

# Add the app directory to Python path so we can import models
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'app'))

from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker
from app.database import Base
from app.models import Exercise, WorkoutSession, SetEntry

def create_test_database(output_path=None):
    """Create a test database with sample data."""
    
    # If no output path provided, default to tests/test_database.db
    if output_path is None:
        script_dir = os.path.dirname(os.path.abspath(__file__))
        temp_db_path = os.path.join(script_dir, 'tests', 'test_database.db')
    else:
        temp_db_path = output_path
    
    print(f"Creating test database at: {temp_db_path}")
    
    # Create engine and session
    engine = create_engine(f'sqlite:///{temp_db_path}')
    SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)
    
    # Drop and recreate all tables to ensure clean state
    Base.metadata.drop_all(bind=engine)
    Base.metadata.create_all(bind=engine)
    
    # Create a session
    db = SessionLocal()
    
    try:
        # Create some sample exercises
        bench_press = Exercise(name="Bench Press", is_bodyweight=False)
        db.add(bench_press)
        
        squat = Exercise(name="Squat", is_bodyweight=False)
        db.add(squat)
        
        push_ups = Exercise(name="Push Ups", is_bodyweight=True)
        db.add(push_ups)
        
        deadlift = Exercise(name="Deadlift", is_bodyweight=False)
        db.add(deadlift)
        
        db.commit()
        
        # Get exercise IDs
        bench_press_id = bench_press.id
        squat_id = squat.id
        push_ups_id = push_ups.id
        deadlift_id = deadlift.id
        
        # Create sample sessions with data for testing progression calculations
        
        # Session 1: Bench Press + Push Ups (push day)
        session1_date = date.today() - timedelta(days=5)
        session1 = WorkoutSession(
            date=session1_date,
            notes="Test session 1 - push day"
        )
        db.add(session1)
        db.flush()
        
        # Bench Press sets
        set1_1 = SetEntry(
            session_id=session1.id,
            exercise_id=bench_press_id,
            set_number=1,
            reps=10,
            weight=50.0
        )
        db.add(set1_1)
        
        set1_2 = SetEntry(
            session_id=session1.id,
            exercise_id=bench_press_id,
            set_number=2,
            reps=8,
            weight=60.0
        )
        db.add(set1_2)
        
        set1_3 = SetEntry(
            session_id=session1.id,
            exercise_id=bench_press_id,
            set_number=3,
            reps=6,
            weight=70.0
        )
        db.add(set1_3)
        
        # Push Ups sets (bodyweight - should not appear in progression charts)
        set1_4 = SetEntry(
            session_id=session1.id,
            exercise_id=push_ups_id,
            set_number=1,
            reps=20,
            weight=None
        )
        db.add(set1_4)
        
        set1_5 = SetEntry(
            session_id=session1.id,
            exercise_id=push_ups_id,
            set_number=2,
            reps=15,
            weight=None
        )
        db.add(set1_5)
        
        # Session 2: Squat + Deadlift (leg day)
        session2_date = date.today() - timedelta(days=3)
        session2 = WorkoutSession(
            date=session2_date,
            notes="Test session 2 - leg day"
        )
        db.add(session2)
        db.flush()
        
        # Squat sets
        set2_1 = SetEntry(
            session_id=session2.id,
            exercise_id=squat_id,
            set_number=1,
            reps=10,
            weight=80.0
        )
        db.add(set2_1)
        
        set2_2 = SetEntry(
            session_id=session2.id,
            exercise_id=squat_id,
            set_number=2,
            reps=8,
            weight=90.0
        )
        db.add(set2_2)
        
        # Deadlift sets
        set2_3 = SetEntry(
            session_id=session2.id,
            exercise_id=deadlift_id,
            set_number=1,
            reps=5,
            weight=140.0
        )
        db.add(set2_3)
        
        set2_4 = SetEntry(
            session_id=session2.id,
            exercise_id=deadlift_id,
            set_number=2,
            reps=5,
            weight=150.0
        )
        db.add(set2_4)
        
        # Session 3: Push Ups only (bodyweight day - should not appear in progression)
        session3_date = date.today() - timedelta(days=1)
        session3 = WorkoutSession(
            date=session3_date,
            notes="Test bodyweight session"
        )
        db.add(session3)
        db.flush()
        
        set3_1 = SetEntry(
            session_id=session3.id,
            exercise_id=push_ups_id,
            set_number=1,
            reps=25,
            weight=None
        )
        db.add(set3_1)
        
        set3_2 = SetEntry(
            session_id=session3.id,
            exercise_id=push_ups_id,
            set_number=2,
            reps=20,
            weight=None
        )
        db.add(set3_2)
        
        # Session 4: Deadlift + Bench Press (heavy day)
        session4_date = date.today() - timedelta(days=2)
        session4 = WorkoutSession(
            date=session4_date,
            notes="Test deadlift + bench session"
        )
        db.add(session4)
        db.flush()
        
        # Deadlift sets
        set4_1 = SetEntry(
            session_id=session4.id,
            exercise_id=deadlift_id,
            set_number=1,
            reps=5,
            weight=150.0
        )
        db.add(set4_1)
        
        set4_2 = SetEntry(
            session_id=session4.id,
            exercise_id=deadlift_id,
            set_number=2,
            reps=5,
            weight=160.0
        )
        db.add(set4_2)
        
        set4_3 = SetEntry(
            session_id=session4.id,
            exercise_id=deadlift_id,
            set_number=3,
            reps=5,
            weight=170.0
        )
        db.add(set4_3)
        
        # Bench Press sets
        set4_4 = SetEntry(
            session_id=session4.id,
            exercise_id=bench_press_id,
            set_number=1,
            reps=8,
            weight=60.0
        )
        db.add(set4_4)
        
        set4_5 = SetEntry(
            session_id=session4.id,
            exercise_id=bench_press_id,
            set_number=2,
            reps=6,
            weight=70.0
        )
        db.add(set4_5)
        
        db.commit()
        print("Test database created successfully with sample data")
        return temp_db_path
        
    except Exception as e:
        db.rollback()
        print(f"Error creating test database: {e}")
        raise
    finally:
        db.close()

if __name__ == "__main__":
    # If argument is provided, use it as the output path
    if len(sys.argv) > 1:
        db_path = create_test_database(sys.argv[1])
    else:
        db_path = create_test_database()
    print(f"Test database path: {db_path}")
