#!/usr/bin/env python3
"""
Create or update the admin user.
Usage: uv run python seed_user.py <username> <password>
"""
import sys, os
sys.path.insert(0, os.path.dirname(__file__))

from app.database import Base, engine, SessionLocal
from app import models
from app.auth import hash_password

Base.metadata.create_all(bind=engine)

if len(sys.argv) != 3:
    print("Usage: python seed_user.py <username> <password>")
    sys.exit(1)

username, password = sys.argv[1], sys.argv[2]
db = SessionLocal()

existing = db.query(models.User).filter(models.User.username == username).first()
if existing:
    existing.hashed_password = hash_password(password)
    print(f"Updated password for user '{username}'")
else:
    db.add(models.User(username=username, hashed_password=hash_password(password)))
    print(f"Created user '{username}'")

db.commit()
db.close()
