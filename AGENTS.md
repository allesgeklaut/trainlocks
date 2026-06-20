# Agents instructions
# General
* Use `podman` instead of `docker`
* Only use `uv` and `pyproject.toml` for dependency management and running the app locally
* `uv` is installed and you can execute it

## Tool calling
You are an expert coding assistant. Be direct and concise. Use code blocks for all code.

Tool calling rules:
- Always close </think> BEFORE emitting any tool call.
- Call tools silently — do NOT narrate or explain before calling a tool.
- Do NOT call a tool if you are uncertain of a required parameter — ask instead.
- After a tool returns a result, use it directly. Do not call the same tool twice for the same input.
- If a tool call fails, report the exact error. Do not silently retry.