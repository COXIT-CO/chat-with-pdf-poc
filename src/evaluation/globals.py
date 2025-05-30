import os

from core.globals import DB_DIR


EVAL_USER_ID: int = 1337_8413211
DB_EVAL_DIR = DB_DIR / "eval"
DB_EVAL_DIR.mkdir(parents=True, exist_ok=True)

EVAL_CHAT_ENDPOINT = os.environ.get("EVAL_CHAT_ENDPOINT")
EVAL_CHAT_ENDPOINT_API_KEY = os.environ.get("EVAL_CHAT_ENDPOINT_API_KEY")
