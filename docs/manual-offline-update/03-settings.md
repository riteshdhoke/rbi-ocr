# 03 — Enable open extraction

## `app/settings.py`

Find `class Settings`. Immediately before its `SLM_MODEL` field, add this class field
with four leading spaces:

```python
    EXTRACT_ALL_FIELDS: bool = True
```

## `.env` on the target machine

Edit the existing keys (do not add conflicting duplicate definitions):

```dotenv
PROVIDER_PROFILE=cpu_node
SYNTHETIC_ONLY=false
HF_TOKEN=
EXTRACT_ALL_FIELDS=true
CELERY_TASK_ALWAYS_EAGER=true
```

Keep the existing local model paths, SQLite URL and thread caps. Do not copy a token
or database path from another machine. No remote language model is added by this update.

## `.env.example`

Add `EXTRACT_ALL_FIELDS=true` to document the new setting. The template can retain
its original laptop defaults; actual offline operation uses the target's `.env`.

Checkpoint (Windows PowerShell; on Linux/macOS replace `.venv\Scripts\python.exe` with `.venv/bin/python`):

```text
.venv\Scripts\python.exe -c "from app.settings import Settings; s=Settings(); print(s.PROVIDER_PROFILE.value, s.EXTRACT_ALL_FIELDS, s.hf_token_present)"
```

Expected: `cpu_node True False`.
