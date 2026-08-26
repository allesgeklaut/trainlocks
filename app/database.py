"""Database configuration.

The original implementation used a hard‑coded absolute path
``sqlite:////data/training_log.db`` which works inside the Docker
container where ``/data`` is a mounted volume, but fails when the
application is run locally because that path does not exist.  To make the
project runnable both inside Docker and on a developer's machine we read a
``DATABASE_URL`` environment variable (commonly defined in a ``.env``
file).  If the variable is not set we fall back to a SQLite database file
located in the project directory (``sqlite:///./training_log.db``).

The ``check_same_thread`` argument is required only for SQLite connections.
We therefore add it conditionally based on the URL scheme.
"""

import os
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker, declarative_base

# Use environment variable if provided, otherwise default to a local file.
SQLALCHEMY_DATABASE_URL = os.getenv(
    "DATABASE_URL", "sqlite:///./training_log.db"
)

# ``connect_args`` is needed for SQLite to allow usage with FastAPI's
# threaded model. For other databases it should be omitted.
connect_args = {"check_same_thread": False} if SQLALCHEMY_DATABASE_URL.startswith("sqlite") else {}

engine = create_engine(SQLALCHEMY_DATABASE_URL, connect_args=connect_args)
SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)
Base = declarative_base()
