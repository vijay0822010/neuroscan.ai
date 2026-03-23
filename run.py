"""
run.py — NeuroScan AI entry point.
Railway calls: python run.py
PORT is automatically injected by Railway as an environment variable.
"""
import os
import sys
from pathlib import Path

# Load .env for local dev (Railway uses env vars set in dashboard, not .env)
_env = Path(__file__).resolve().parent / ".env"
if _env.exists():
    try:
        from dotenv import load_dotenv
        load_dotenv(dotenv_path=_env, override=False)
    except ImportError:
        pass

import uvicorn

# Railway injects PORT automatically — must read it here
port = int(os.environ.get("PORT", 8000))
host = os.environ.get("HOST", "0.0.0.0")

print(f"[NeuroScan] Starting on {host}:{port}", flush=True)
print(f"[NeuroScan] Python {sys.version}", flush=True)

uvicorn.run(
    "app.main:app",
    host=host,
    port=port,
    log_level="info",
    # No reload in production
    reload=False,
    # Single worker — Railway scales with multiple containers not workers
    workers=1,
)
