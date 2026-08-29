# Training Log Dashboard

FastAPI + SQLite strength training log with progression charts.

## Features

- Define exercises (bodyweight or weighted)
- Create session templates with ordered exercises and set counts
- Log sessions from a template — enter reps and weight per set
- Log cardio activities (running, swimming, …) with distance, duration and pace
- Progression view per exercise:
  - Line chart: top weight per session
  - Bar chart: total volume (weight × reps) per session
  - Data table
- Light / dark mode

## Run with Docker Compose (recommended)

1. Generate a secret key and create a `.env` file:

       python -c "import secrets; print('SECRET_KEY=' + secrets.token_hex(32))" > .env

2. Build and start the container:

       docker compose up -d

3. Open http://localhost:8004 (port `8004` is mapped by default; adjust the
   `ports` section in `docker-compose.yml` if needed).

Data is persisted in a bind-mounted volume at `./training_log_data/`.

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

## Cardio

- `/cardio` — log running, swimming and other endurance activities. Each entry
  stores activity type, distance (km), duration (minutes) and computes pace
  (min/km). A session can mix strength sets and cardio on the same date.
- Cardio is also exposed through the JSON API (`POST/GET/DELETE /api/cardio`)
  and the MCP server (`log_cardio`, `list_cardio`, `get_cardio`, `delete_cardio`).

## Authentication

The app uses session-based login (bcrypt passwords, signed cookies).

### First-time setup — create your user

**Option A — Locally with `uv` (recommended):**

The seed script lives at the repository root, not inside the Docker image.
If you run it without a `DATABASE_URL`, it writes to `./training_log.db` in
the project directory — **not** the database the container uses
(`./training_log_data/training_log.db`).  To seed the correct database:

```bash
# Run from the project root, pointing at the container's bind-mounted DB
DATABASE_URL=sqlite:///./training_log_data/training_log.db \
  uv run python seed_user.py johannes yourpassword
```

You can then remove the stray local database if one was created:

```bash
rm -f ./training_log.db
```

**Option B — Inside the container:**

The default `Dockerfile` does **not** copy `seed_user.py` into the image.
To use this method, add `COPY seed_user.py ./` to the Dockerfile, rebuild,
and then:

```bash
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
