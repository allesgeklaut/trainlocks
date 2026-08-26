FROM ghcr.io/astral-sh/uv:python3.12-bookworm-slim

WORKDIR /app

# Run as non-root
RUN useradd --create-home --uid 1000 trainlocks

COPY pyproject.toml uv.lock ./
RUN uv sync --frozen --no-dev

COPY app/ ./app/

VOLUME ["/data"]
EXPOSE 8000

USER trainlocks

CMD ["/app/.venv/bin/uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]