from dataclasses import dataclass
from datetime import datetime
from enum import Enum
from typing import Dict, Optional, Union, Any

from pydantic import BaseModel

from core.globals import VERSION


class RequestStatus(Enum):
    OK = "ok"
    NOT_OK = "not_ok"


@dataclass
class RequestResult:
    event: str
    status: RequestStatus
    ts_created: float
    duration_seconds: float
    status_code: Optional[int] = None
    attributes: Optional[Dict[str, Any]] = None
    error_message: Optional[str] = None


class TelemetryScope(Enum):
    W_PROCESSOR = "w_processor"


class TeleItemStatus(Enum):
    INFO = "info"
    SUCCESS = "success"
    FAILURE = "failure"


TeleItem = Union['TeleWProcessor']


class TeleWProcessor(BaseModel):
    version: str = "v0"
    proc_strategy: str
    event: str

    status: TeleItemStatus
    error_message: Optional[str] = None
    error_recoverable: Optional[bool] = None

    user_id: Optional[int] = None
    file_name: Optional[str] = None
    file_name_orig: Optional[str] = None
    vector_store: Optional[str] = None
    file_id: Optional[str] = None

    attributes: Optional[Dict[str, Any]] = None

    duration_seconds: Optional[float] = None
    timestamp: datetime = datetime.now()

    app_version: str = VERSION

    def to_dict(self) -> Dict[str, Any]:
        """Convert the model to a dictionary."""
        return {
            "version": self.version,
            "proc_strategy": self.proc_strategy,
            "event": self.event,
            "status": self.status.value,
            "error_message": self.error_message,
            "error_recoverable": self.error_recoverable,
            "user_id": self.user_id,
            "file_name": self.file_name,
            "file_name_orig": self.file_name_orig,
            "vector_store": self.vector_store,
            "file_id": self.file_id,
            "attributes": self.attributes,
            "duration_seconds": self.duration_seconds,
            "timestamp": self.timestamp.isoformat() if self.timestamp else None,
            "app_version": self.app_version
        }

    def write(self, writer):
        writer.write(self)
