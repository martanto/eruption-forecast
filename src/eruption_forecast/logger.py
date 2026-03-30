"""Centralised logging configuration for the eruption-forecast package.

This module sets up a package-wide `loguru` logger with three output sinks:

- **Console** (stderr): coloured output at ``INFO`` level by default.
- **General log file** (``logs/forecast_YYYY-MM-DD.log``): ``DEBUG`` and above,
  rotated daily, retained for 30 days, compressed to ZIP.
- **Error log file** (``logs/errors_YYYY-MM-DD.log``): ``ERROR`` and above,
  rotated daily, retained for 90 days, compressed to ZIP.

All file writes use ``enqueue=True`` for safe use inside ``joblib`` worker
processes. If the ``DISABLE_LOGGING`` environment variable is set to ``"1"``
(inherited by child processes), no handlers are registered.

Key exports:
    - ``logger``: The configured `loguru` ``Logger`` instance — import this
      directly via ``from eruption_forecast.logger import logger``.
    - ``get_logger()``: Returns the same logger instance.
    - ``enable_logging()`` / ``disable_logging()``: Toggle all handlers at
      runtime; also update the ``DISABLE_LOGGING`` env var so worker processes
      inherit the setting.
    - ``set_log_level(level)``: Change the console sink level dynamically.
    - ``set_log_directory(log_dir)``: Redirect file sinks to a new directory.
    - ``DEFAULT_LOG_DIR``: Resolved path to the active log directory
      (``<cwd>/logs`` by default).
"""

import os
import sys

from loguru import logger

from eruption_forecast.utils.pathutils import ensure_dir


# Retention periods for log files.
_GENERAL_LOG_RETENTION = "30 days"
_ERROR_LOG_RETENTION = "90 days"

# Tracks whether logging is currently enabled.
_logging_enabled: bool = True

DEFAULT_LOG_DIR = ensure_dir(os.path.join(os.getcwd(), "logs"))

_FILE_FORMAT = (
    "{time:YYYY-MM-DD HH:mm:ss} | {level: <8} | {name}:{function}:{line} - {message}"
)

_CONSOLE_FORMAT = (
    "<green>{time:YYYY-MM-DD HH:mm:ss}</green> | <level>{level: <8}</level> | "
    "<cyan>{name}</cyan>:<cyan>{function}</cyan>:<cyan>{line}</cyan> - <level>{message}</level>"
)

logger.remove()


def _configure_handlers(log_dir: str, console_level: str = "INFO") -> None:
    """Remove all existing handlers and re-add console + file handlers.

    Centralises handler configuration so that module-level setup, set_log_level(),
    and set_log_directory() all use the same retention periods and formats.

    Args:
        log_dir (str): Directory path for log file output.
        console_level (str, optional): Log level for the console handler.
            Defaults to "INFO".
    """
    logger.remove()

    logger.add(
        sys.stderr,
        format=_CONSOLE_FORMAT,
        level=console_level.upper(),
        colorize=True,
    )

    logger.add(
        os.path.join(log_dir, "forecast_{time:YYYY-MM-DD}.log"),
        rotation="00:00",
        retention=_GENERAL_LOG_RETENTION,
        compression="zip",
        format=_FILE_FORMAT,
        level="DEBUG",
        enqueue=True,
    )

    logger.add(
        os.path.join(log_dir, "errors_{time:YYYY-MM-DD}.log"),
        rotation="00:00",
        retention=_ERROR_LOG_RETENTION,
        compression="zip",
        format=_FILE_FORMAT,
        level="ERROR",
        enqueue=True,
    )


# Skip if the parent process disabled logging (env var is inherited by workers).
if os.environ.get("DISABLE_LOGGING") != "1":
    _configure_handlers(DEFAULT_LOG_DIR)


def get_logger():
    """Return the package-wide loguru logger instance.

    Returns:
        loguru.Logger: The configured logger instance with console and file handlers.
    """
    return logger


def set_log_level(level: str) -> None:
    """Change the console log level dynamically.

    Removes all existing handlers and re-adds them with the new console level.
    File handlers retain their original levels.

    Args:
        level (str): Desired log level for the console handler. One of
            ``"DEBUG"``, ``"INFO"``, ``"WARNING"``, ``"ERROR"``, or
            ``"CRITICAL"``. Case-insensitive.
    """
    _configure_handlers(DEFAULT_LOG_DIR, console_level=level)


def set_log_directory(log_dir: str) -> None:
    """Change the log file directory dynamically.

    Updates the global ``DEFAULT_LOG_DIR``, creates the directory if needed,
    then reconfigures all handlers to write to the new location.

    Args:
        log_dir (str): Absolute or relative path to the new log directory.
            Created automatically if it does not exist.
    """
    global DEFAULT_LOG_DIR
    DEFAULT_LOG_DIR = ensure_dir(os.path.abspath(log_dir))
    _configure_handlers(DEFAULT_LOG_DIR)
    logger.info(f"Log directory changed to: {DEFAULT_LOG_DIR}")


def disable_logging() -> None:
    """Disable all logging output globally.

    Remove all active loguru handlers so no messages are written to the
    console or log files. Call :func:`enable_logging` to restore handlers.
    """
    global _logging_enabled
    _logging_enabled = False
    os.environ["DISABLE_LOGGING"] = "1"
    logger.remove()


def enable_logging() -> None:
    """Re-enable logging after a previous :func:`disable_logging` call.

    Restore console and file handlers using the current ``DEFAULT_LOG_DIR``.
    Has no effect if logging is already enabled.
    """
    global _logging_enabled
    if not _logging_enabled:
        _logging_enabled = True
        os.environ.pop("DISABLE_LOGGING", None)
        _configure_handlers(DEFAULT_LOG_DIR)
