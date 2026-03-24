"""
Centralized logger module using loguru for the package.

This module provides a single configured logger instance that all modules can import and use.
It configures console output with colors, daily rotating log files with compression, and separate
error-only log files for better debugging and monitoring.

The logger configuration includes:
- Console output with colored formatting
- Daily rotating general log files (retained for 30 days)
- Daily rotating error-only log files (retained for 90 days)
- Thread-safe logging with queue-based handlers
- Automatic log compression (zip format)

Examples:
    >>> from eruption_forecast.logger import logger
    >>> logger.info("Processing started")
    >>> logger.error("An error occurred")
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

# Define default log directory
DEFAULT_LOG_DIR = ensure_dir(os.path.join(os.getcwd(), "logs"))


# File log format shared across all file handlers.
_FILE_FORMAT = (
    "{time:YYYY-MM-DD HH:mm:ss} | {level: <8} | {name}:{function}:{line} - {message}"
)

# Console format with colour codes.
_CONSOLE_FORMAT = (
    "<green>{time:YYYY-MM-DD HH:mm:ss}</green> | <level>{level: <8}</level> | "
    "<cyan>{name}</cyan>:<cyan>{function}</cyan>:<cyan>{line}</cyan> - <level>{message}</level>"
)

# Init default handler
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


# Initial handler setup at import time.
# Skip if the parent process disabled logging (env var is inherited by workers).
if os.environ.get("DISABLE_LOGGING") != "1":
    _configure_handlers(DEFAULT_LOG_DIR)


def get_logger():
    """Get the configured logger instance.

    Returns the package-wide loguru Logger instance with pre-configured console
    and file handlers.

    Returns:
        loguru.Logger: The configured logger instance with console and file handlers.

    Examples:
        >>> from eruption_forecast.logger import get_logger
        >>> logger = get_logger()
        >>> logger.info("Application started")
    """
    return logger


def set_log_level(level: str) -> None:
    """Change the console log level dynamically.

    Removes all existing handlers and re-adds them with the new console level.
    File handlers retain their original levels (DEBUG for the general log,
    ERROR for the error log). This allows you to control console verbosity
    without affecting what gets written to files.

    Args:
        level (str): Desired log level for the console handler. One of
            "DEBUG", "INFO", "WARNING", "ERROR", or "CRITICAL". Case-insensitive.

    Returns:
        None

    Examples:
        >>> from eruption_forecast.logger import set_log_level
        >>> set_log_level("DEBUG")  # Show all debug messages in console
        >>> set_log_level("WARNING")  # Only show warnings and errors in console
    """
    _configure_handlers(DEFAULT_LOG_DIR, console_level=level)


def set_log_directory(log_dir: str) -> None:
    """Change the log file directory dynamically.

    Updates the global DEFAULT_LOG_DIR, creates the directory if needed, then
    reconfigures all handlers to write log files to the new location. All future
    log files will be written to this directory.

    Args:
        log_dir (str): Absolute or relative path to the new log directory.
            The directory is created automatically if it does not exist.

    Returns:
        None

    Examples:
        >>> from eruption_forecast.logger import set_log_directory
        >>> set_log_directory("output/logs")  # Use relative path
        >>> set_log_directory("/var/log/eruption_forecast")  # Use absolute path
    """
    global DEFAULT_LOG_DIR
    DEFAULT_LOG_DIR = ensure_dir(os.path.abspath(log_dir))
    _configure_handlers(DEFAULT_LOG_DIR)
    logger.info(f"Log directory changed to: {DEFAULT_LOG_DIR}")


def disable_logging() -> None:
    """Disable all logging output globally.

    Removes all active loguru handlers so no messages are written to the
    console or log files. Call enable_logging() to restore handlers.

    Returns:
        None

    Examples:
        >>> from eruption_forecast.logger import disable_logging
        >>> disable_logging()  # Silence all output
    """
    global _logging_enabled
    _logging_enabled = False
    os.environ["DISABLE_LOGGING"] = "1"
    logger.remove()


def enable_logging() -> None:
    """Re-enable logging after a previous disable_logging() call.

    Restores console and file handlers using the current DEFAULT_LOG_DIR.
    Has no effect if logging is already enabled.

    Returns:
        None

    Examples:
        >>> from eruption_forecast.logger import enable_logging
        >>> enable_logging()  # Restore output
    """
    global _logging_enabled
    if not _logging_enabled:
        _logging_enabled = True
        os.environ.pop("DISABLE_LOGGING", None)
        _configure_handlers(DEFAULT_LOG_DIR)
