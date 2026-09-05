# Agents instructions
# General
* Use `docker` / `docker compose` for containers (podman is not installed)
* Only use `uv` and `pyproject.toml` for dependency management and running the app locally
* `uv` is installed and you can execute it

# Playwright (browser testing)
* The trainlocks dev server for browser tests: `http://127.0.0.1:8004`
  (docker container `training-log`); test credentials are in
  `/opt/secrets/trainlocks.env` (`TRAINLOGS_USERNAME` / `TRAINLOGS_PASSWORD`);
  set the signed `tl_session` cookie via `page.context().addCookies()` to log in.
