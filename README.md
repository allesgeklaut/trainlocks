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
  uv run uvicron app.main:app --reload

## Stack

- **Backend**: FastAPI, SQLAlchemy, SQLite
- **Frontend**: Jinja2 templates, vanilla JS, Chart.js 4
- **Fonts**: Satoshi (Fontshare CDN)
