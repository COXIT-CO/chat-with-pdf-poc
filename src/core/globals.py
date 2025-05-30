import os
from pathlib import Path
from importlib.metadata import version


VERSION = version("chat-with-pdf-poc")


PROCESSING_STRATEGY = os.environ.get("PROCESSING_STRATEGY")
assert PROCESSING_STRATEGY in ["openai_fs", "local_fs"]

SAVE_STRATEGY = os.environ.get("SAVE_STRATEGY")
assert SAVE_STRATEGY in ["redis", "milvus", ""]

EMBEDDING_MODEL = os.environ.get("EMBEDDING_MODEL")

if SAVE_STRATEGY and PROCESSING_STRATEGY != "local_fs":
    raise Exception(f"SAVE_STRATEGY must be left empty if PROCESSING_STRATEGY != local_fs, got {SAVE_STRATEGY=}; {PROCESSING_STRATEGY=}")

if not SAVE_STRATEGY and PROCESSING_STRATEGY == "local_fs":
    raise Exception(f"SAVE_STRATEGY must be defined if PROCESSING_STRATEGY == local_fs, got {SAVE_STRATEGY=}; {PROCESSING_STRATEGY=}")

if not EMBEDDING_MODEL and PROCESSING_STRATEGY == "local_fs":
    raise Exception(f"EMBEDDING_MODEL must be defined if PROCESSING_STRATEGY == local_fs, got {EMBEDDING_MODEL=}; {PROCESSING_STRATEGY=}")


BASE_DIR: Path = Path(__file__).resolve().parent.parent.parent
FILES_DIR = BASE_DIR / "files"
LOGS_DIR = BASE_DIR / "logs"
TELEMETRY_DIR = BASE_DIR / "telemetry"

# === EVALUATION ===
EVALUATIONS_DIR = BASE_DIR / "evaluations"
EVALUATIONS_DIR.mkdir(parents=True, exist_ok=True)

DATASETS_DIR = BASE_DIR / "datasets"
DATASETS_DIR.mkdir(parents=True, exist_ok=True)
# ===

DB_DIR = BASE_DIR / "db"
DB_DIR.mkdir(parents=True, exist_ok=True)

CONFIGS_DIR = BASE_DIR / "configs"
CONFIGS_DIR.mkdir(parents=True, exist_ok=True)

ASSETS = BASE_DIR / "assets"
ASSETS_CSS = ASSETS / "css"