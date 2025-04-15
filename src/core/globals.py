from pathlib import Path
from importlib.metadata import version


EMBEDDING_MODEL = "text-embedding-3-small"
VERSION = version("chat-with-pdf-poc")

# openai_fs or local_fs
PROCESSING_STRATEGY = "local_fs"

BASE_DIR: Path = Path(__file__).resolve().parent.parent.parent
FILES_DIR = BASE_DIR / "files"
LOGS_DIR = BASE_DIR / "logs"
TELEMETRY_DIR = BASE_DIR / "telemetry"
