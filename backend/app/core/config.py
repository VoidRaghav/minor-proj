import os
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parents[2]
ARTIFACTS_DIR = Path(os.getenv("XAI_ARTIFACTS_DIR", BASE_DIR / "artifacts"))
CORS_ORIGINS = [o.strip() for o in os.getenv("XAI_CORS_ORIGINS", "http://localhost:5173").split(",") if o.strip()]
RANDOM_STATE = 42
BACKGROUND_SIZE = 40
