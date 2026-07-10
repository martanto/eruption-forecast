"""Centralised logging configuration for the eruption-forecast package.

This module sets up a package-wide `loguru` logger with the following output
sinks:

- **Console** (stderr): coloured output at ``INFO`` level by default. Every
  record is shown, regardless of category.
- **General log file** (``logs/forecast_YYYY-MM-DD.log``): ``DEBUG`` and above,
  rotated daily, retained for 30 days, compressed to ZIP. Only *uncategorised*
  records land here — records tagged with a known category are routed to
  their dedicated file instead (see below).
- **Error log file** (``logs/errors_YYYY-MM-DD.log``): ``ERROR`` and above,
  rotated daily, retained for 90 days, compressed to ZIP. Same exclusion
  rule as the general log.
- **Per-category log files** (``logs/{category}_YYYY-MM-DD.log``): one file
  per registered category. A record is routed here when it was emitted via
  ``logger.bind(category={category}).warning(...)`` (or a similar level).
  The default installation registers the ``telegram`` category so Telegram
  notification failures land in ``telegram_YYYY-MM-DD.log`` rather than
  polluting the general log. Additional categories can be added at runtime
  via :func:`register_error_category`.

All file writes use ``enqueue=True`` for safe use inside ``joblib`` worker
processes. If the ``DISABLE_LOGGING`` environment variable is set to ``"1"``
(inherited by child processes), no handlers are registered.

Key exports:
    - ``logger``: The configured `loguru` ``Logger`` instance — import this
      directly via ``from eruption_forecast.logger import logger``.
    - ``get_logger()``: Returns the same logger instance.
    - ``get_category_logger(category)``: Returns a category-bound logger so
      callers can emit records that get routed to a dedicated file.
    - ``register_error_category(name, level, retention)``: Registers a new
      category sink at runtime.
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
from typing import TYPE_CHECKING
from collections.abc import Callable

from loguru import logger

from eruption_forecast.utils.pathutils import ensure_dir


if TYPE_CHECKING:
    from loguru import Record


# Retention periods for the general and error log files.
_GENERAL_LOG_RETENTION = "30 days"
_ERROR_LOG_RETENTION = "90 days"

# Tracks whether logging is currently enabled.
_logging_enabled: bool = True

# Per-category error sinks. Keyed by category name; each entry stores the
# minimum level and retention for that category's log file. Records emitted
# via ``logger.bind(category=<name>)`` are routed to
# ``logs/{name}_YYYY-MM-DD.log`` and excluded from the general and error
# log files.
_ERROR_CATEGORIES: dict[str, dict[str, str]] = {
    "telegram": {"level": "WARNING", "retention": _ERROR_LOG_RETENTION},
}

DEFAULT_LOG_DIR = ensure_dir(os.path.join(os.getcwd(), "logs"))

_FILE_FORMAT = (
    "{time:YYYY-MM-DD HH:mm:ss} | {level: <8} | {name}:{function}:{line} - {message}"
)

_CONSOLE_FORMAT = (
    "<green>{time:YYYY-MM-DD HH:mm:ss}</green> | <level>{level: <8}</level> | "
    "<cyan>{name}</cyan>:<cyan>{function}</cyan>:<cyan>{line}</cyan> - <level>{message}</level>"
)

logger.remove()


def _category_filter(name: str) -> "Callable[[Record], bool]":
    """Return a sink filter that accepts only records tagged with ``name``.

    The category name is captured via a default argument to avoid the
    late-binding pitfall when multiple filters share the same closure scope.

    Args:
        name (str): Category name to match against ``record["extra"]["category"]``.

    Returns:
        Callable[[Record], bool]: A predicate suitable for loguru's ``filter=`` parameter.
    """

    def _accept(record: "Record", _name: str = name) -> bool:
        return record["extra"].get("category") == _name

    return _accept


def _uncategorized_filter(record: "Record") -> bool:
    """Return ``True`` when ``record`` has no known category tag.

    Used by the general and error log sinks so records that belong to a
    registered category are routed exclusively to their dedicated file.

    Args:
        record (Record): Loguru record dictionary.

    Returns:
        bool: ``True`` if the record's ``extra.category`` is unset or not a
        registered category name.
    """
    return record["extra"].get("category") not in _ERROR_CATEGORIES


def _configure_handlers(log_dir: str, console_level: str = "INFO") -> None:
    """Remove all existing handlers and re-add console + file handlers.

    Centralises handler configuration so that module-level setup, set_log_level(),
    set_log_directory(), and register_error_category() all use the same retention
    periods and formats.

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
        filter=_uncategorized_filter,
    )

    logger.add(
        os.path.join(log_dir, "errors_{time:YYYY-MM-DD}.log"),
        rotation="00:00",
        retention=_ERROR_LOG_RETENTION,
        compression="zip",
        format=_FILE_FORMAT,
        level="ERROR",
        enqueue=True,
        filter=_uncategorized_filter,
    )

    for name, cfg in _ERROR_CATEGORIES.items():
        logger.add(
            os.path.join(log_dir, f"{name}_{{time:YYYY-MM-DD}}.log"),
            rotation="00:00",
            retention=cfg["retention"],
            compression="zip",
            format=_FILE_FORMAT,
            level=cfg["level"].upper(),
            enqueue=True,
            filter=_category_filter(name),
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


def get_category_logger(category: str):
    """Return a logger bound to ``category`` so its records land in a dedicated file.

    Records emitted via the returned logger carry ``extra["category"] = category``,
    which routes them to ``logs/{category}_YYYY-MM-DD.log`` when the category
    is registered (either at import time or via :func:`register_error_category`)
    and excludes them from ``forecast_*.log`` / ``errors_*.log``. Records for
    unregistered categories fall through the exclusion filter and land in the
    general log — call :func:`register_error_category` first to create the
    dedicated sink.

    Args:
        category (str): Category name to bind on every emitted record.

    Returns:
        loguru.Logger: A category-bound loguru logger.
    """
    return logger.bind(category=category)


def register_error_category(
    name: str,
    level: str = "WARNING",
    retention: str = _ERROR_LOG_RETENTION,
) -> None:
    """Register (or update) a per-category error log file.

    After this call, records emitted via
    ``logger.bind(category=name).warning(...)`` (or any level ≥ ``level``)
    are routed to ``{DEFAULT_LOG_DIR}/{name}_YYYY-MM-DD.log`` and excluded
    from the general and error log files.

    Re-registering an existing category updates its ``level`` and
    ``retention``. All sinks are rebuilt so no duplicate handlers are
    installed for the same category.

    Args:
        name (str): Category name. Used as the log file prefix and as the
            value expected in ``record["extra"]["category"]``.
        level (str): Minimum record level for the sink. Case-insensitive.
            Defaults to ``"WARNING"``.
        retention (str): Retention passed through to loguru
            (e.g. ``"90 days"``). Defaults to the error-log retention.
    """
    _ERROR_CATEGORIES[name] = {"level": level, "retention": retention}
    _configure_handlers(DEFAULT_LOG_DIR)


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
