"""Shared test environment.

The app binds its database engine at import time, so ``DATABASE_URL`` must
be fixed before any test module imports ``app``.  Both test modules share
the same temporary SQLite file created here; per-test isolation is handled
by the ``client`` fixtures (create/drop tables) in each module.
"""
import os
import tempfile

_db_fd, _db_path = tempfile.mkstemp(suffix=".db")
os.close(_db_fd)
os.environ["DATABASE_URL"] = f"sqlite:///{_db_path}"


def pytest_sessionfinish(session):
    try:
        os.remove(_db_path)
    except OSError:
        pass
