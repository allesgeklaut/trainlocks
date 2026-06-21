# Training Log Dashboard

FastAPI + SQLite strength training log with progression charts.

## Features

- Define exercises (bodyweight or weighted)
- Create session templates with ordered exercises and set counts
- Log sessions from a template — enter reps and weight per set
- Progression view per exercise:
  - Line chart: top weight per session
  - Bar chart: total volume (weight × reps) per session
  - Data table
- Light / dark mode

## Run with Docker Compose (recommended)

    docker compose up -d

Open http://localhost:8000

Data is persisted in a named Docker volume (`training_log_data`).

## Run locally without Docker (using uv)

  uv sync [--frozen]
  # Create a /data directory or adjust SQLALCHEMY_DATABASE_URL in app/database.py
  mkdir -p /data
  uv run uvicorn app.main:app --reload

## Stack

- **Backend**: FastAPI, SQLAlchemy, SQLite
- **Frontend**: Jinja2 templates, vanilla JS, Chart.js 4
- **Fonts**: Satoshi (Fontshare CDN)


## Browse and import open data

The app now includes:

- `/browse/exercises` — browse and filter the open `free-exercise-db` catalog, then import selected exercises into your local SQLite database.
- `/browse/plans` — import bundled starter templates such as Push/Pull/Legs and Upper/Lower.

Notes:

- Exercise browsing fetches data from the public `free-exercise-db` JSON source at runtime, then caches it in memory while the app is running.
- Starter plans are bundled locally in `app/static/plans/starter_plans.json`.

## Authentication

The app uses session-based login (bcrypt passwords, signed cookies).

### First-time setup — create your user

```bash
# Locally
uv run python seed_user.py johannes yourpassword

# In Docker (after starting the container)
docker exec -it training-log python seed_user.py johannes yourpassword
```

### Change password

Run `seed_user.py` again with the same username — it updates the hash.

### Secret key

Set a strong `SECRET_KEY` in `docker-compose.yml` (or a `.env` file):

```bash
python -c "import secrets; print(secrets.token_hex(32))"
```

Never use the default `change-me-in-production-please` value in production.
