"""
app/core/config.py — settings loaded from environment variables.
On Railway: set all vars in the Variables dashboard.
Locally: use .env file next to run.py.
"""
from __future__ import annotations
import os
from pathlib import Path
from functools import lru_cache

# Load .env for local development only
_HERE         = Path(__file__).resolve()
_PROJECT_ROOT = _HERE.parent.parent.parent
_ENV_FILE     = _PROJECT_ROOT / ".env"

try:
    from dotenv import load_dotenv
    if _ENV_FILE.exists():
        load_dotenv(dotenv_path=_ENV_FILE, override=False)
    else:
        for _p in [Path.cwd(), *Path.cwd().parents]:
            if (_p / ".env").exists():
                load_dotenv(dotenv_path=_p / ".env", override=False)
                break
except ImportError:
    pass

from pydantic_settings import BaseSettings


class Settings(BaseSettings):
    # Groq API
    GROQ_API_KEY: str = "your_groq_api_key_here"
    GROQ_API_URL: str = "https://api.groq.com/openai/v1/chat/completions"
    GROQ_MODEL:   str = "meta-llama/llama-4-scout-17b-16e-instruct"

    # Application
    APP_NAME:    str = "NeuroScan AI"
    APP_VERSION: str = "2.1.0"
    HOST:        str = "0.0.0.0"
    PORT:        int = 8000

    # File handling — use /tmp on Railway (writable), local path otherwise
    MAX_UPLOAD_MB: int = 50
    UPLOAD_DIR:    str = "/tmp/neuroscan_uploads"
    REPORT_DIR:    str = "/tmp/neuroscan_reports"

    # Model hyperparameters
    RESNET_DIM:    int   = 512
    WAV2VEC2_DIM:  int   = 768
    FUSION_BIAS:   float = -0.312
    FUSION_WEIGHT: float =  3.847
    FUSION_W_IMG:  float =  0.55
    FUSION_W_AUD:  float =  0.45

    model_config = {
        "env_file":          str(_ENV_FILE) if _ENV_FILE.exists() else None,
        "env_file_encoding": "utf-8",
        "extra":             "ignore",
    }


@lru_cache()
def get_settings() -> Settings:
    return Settings()

settings = get_settings()
