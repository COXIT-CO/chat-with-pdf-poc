import sys
import logging

from datetime import datetime
from pathlib import Path
from termcolor import colored

from core.globals import BASE_DIR

__all__ = ['init_logger', 'info', 'error', 'warn', 'debug', 'exception']

logger = logging.getLogger("CWPP")
info = logger.info
error = logger.error
warn = logger.warning
debug = logger.debug
exception = logger.exception

FMT = '%(asctime)s %(levelname)s [%(filename)s] %(message)s'
DATE_FMT = '%Y%m%d %H:%M:%S'


# List of filenames to exclude from logging
EXCLUDED_FILENAMES = [
    '_base_client.py',
    '_trace.py',
    'inotify_buffer.py',
]


def init_logger(debug_on: bool) -> None:
    """Initialize the application logger with console and file handlers.

    Args:
        debug_on: Whether to enable debug logging
    """

    class ExcludeFilter(logging.Filter):
        def filter(self, record: logging.LogRecord) -> bool:
            # Return False to exclude the record from logging
            return record.filename not in EXCLUDED_FILENAMES

    class ColoredConsoleHandler(logging.Handler):
        def emit(self, record: logging.LogRecord) -> None:
            # Skip debug messages if debug mode is off
            if not debug_on and record.levelno == logging.DEBUG:
                return

            # Color the level name based on severity
            level_colors = {
                logging.ERROR: 'red',
                logging.WARNING: 'yellow'
            }

            # Save original level name
            original_levelname = record.levelname

            # Apply color if needed
            if record.levelno in level_colors:
                record.levelname = colored(record.levelname, level_colors[record.levelno])

            # Format and output the log entry
            log_entry = self.format(record)
            record.levelname = original_levelname

            # Write to stderr and flush
            sys.stderr.write(f"{log_entry}\n")
            sys.stderr.flush()

    log_path = BASE_DIR / "logs"
    log_path.mkdir(parents=True, exist_ok=True)

    class DailyFileHandler(logging.Handler):
        def __init__(self, log_dir, encoding='utf-8'):
            super().__init__()
            self.log_dir = Path(log_dir)
            self.encoding = encoding
            self.current_date = None
            self.file_handler = None
            # Initialize with the current file handler
            self._update_file_handler()

        def _update_file_handler(self):
            today = datetime.now().strftime("%Y%m%d")

            if today != self.current_date or self.file_handler is None:
                if self.file_handler:
                    self.file_handler.close()

                log_file_path = self.log_dir / f"{today}.log"
                self.file_handler = logging.FileHandler(
                    filename=log_file_path,
                    encoding=self.encoding
                )
                self.file_handler.setFormatter(self.formatter)
                self.current_date = today

        def emit(self, record):
            self._update_file_handler()
            self.file_handler.emit(record)

        def setFormatter(self, formatter):
            super().setFormatter(formatter)
            if self.file_handler:
                self.file_handler.setFormatter(formatter)

    file_handler = DailyFileHandler(log_path, encoding="utf-8")

    file_handler.setFormatter(logging.Formatter(
        FMT, DATE_FMT
    ))

    console_handler = ColoredConsoleHandler()

    # Add the exclude filter to both handlers
    exclude_filter = ExcludeFilter()
    console_handler.addFilter(exclude_filter)
    file_handler.addFilter(exclude_filter)

    logging.basicConfig(
        level=logging.DEBUG if debug_on else logging.INFO,
        format=FMT,
        datefmt=DATE_FMT,
        handlers=[console_handler, file_handler]
    )