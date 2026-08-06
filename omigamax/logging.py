"""JSONL logging to the logs/ directory (UTF-8 encoded, one JSON object per line).

All file I/O uses explicit ``encoding="utf-8"`` because the default Windows
code page (cp936) would garble Chinese log messages.
"""

import json
import logging
from pathlib import Path

DEFAULT_LOG_FILE = "omigamax.jsonl"


def get_logs_dir() -> Path:
    """Return the absolute path to the project logs/ directory, creating it if needed."""
    logs_dir = Path(__file__).resolve().parent.parent / "logs"
    logs_dir.mkdir(parents=True, exist_ok=True)
    return logs_dir


class JsonlFormatter(logging.Formatter):
    """Format log records as single-line JSON objects (UTF-8 safe)."""

    def format(self, record: logging.LogRecord) -> str:
        payload = {
            "timestamp": self.formatTime(record, "%Y-%m-%dT%H:%M:%S%z"),
            "level": record.levelname,
            "logger": record.name,
            "message": record.getMessage(),
        }
        if record.exc_info:
            payload["exc_info"] = self.formatException(record.exc_info)
        return json.dumps(payload, ensure_ascii=False)


def setup_logging(filename: str = DEFAULT_LOG_FILE) -> logging.Logger:
    """Configure the ``omigamax`` logger to append JSONL under logs/ (UTF-8)."""
    logs_dir = get_logs_dir()
    logger = logging.getLogger("omigamax")
    # Avoid stacking duplicate handlers if setup_logging is called repeatedly.
    for handler in list(logger.handlers):
        if getattr(handler, "name", None) == "omigamax_jsonl":
            logger.removeHandler(handler)
    handler = logging.FileHandler(logs_dir / filename, encoding="utf-8")
    handler.name = "omigamax_jsonl"
    handler.setFormatter(JsonlFormatter())
    logger.setLevel(logging.INFO)
    logger.addHandler(handler)
    logger.propagate = False
    logger.info("logging_initialized")
    return logger
