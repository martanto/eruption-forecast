"""
Centralized logger module using loguru for the package.

This module provides a single configured logger instance that all modules can import and use.
It configures console output with colors, daily rotating log files with compression, and separate
error-only log files for better debugging and monitoring.

The logger configuration includes:
- Console output with colored formatting
- Daily rotating general log files (retained for 3 days)
- Daily rotating error-only log files (retained for 3 days)
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


# Init default handler
logger.remove()

# Define default log directory
DEFAULT_LOG_DIR = ensure_dir(os.path.join(os.getcwd(), "logs"))

# Add console handler with custom format
logger.add(
    sys.stderr,
    format="<green>{time:YYYY-MM-DD HH:mm:ss}</green> | <level>{level: <8}</level> | <cyan>{name}</cyan>:<cyan>{function}</cyan>:<cyan>{line}</cyan> - <level>{message}</level>",
    level="INFO",
    colorize=True,
)

# Add file handler with rotation
logger.add(
    os.path.join(DEFAULT_LOG_DIR, "forecast_{time:YYYY-MM-DD}.log"),
    rotation="00:00",  # Rotate at midnight
    retention="3 days",  # Keep logs for 30 days
    compression="zip",  # Compress rotated logs
    format="{time:YYYY-MM-DD HH:mm:ss} | {level: <8} | {name}:{function}:{line} - {message}",
    level="INFO",  # File logs everything including DEBUG
    enqueue=True,  # Thread-safe logging
)

# Add error-specific log file
logger.add(
    os.path.join(DEFAULT_LOG_DIR, "errors_{time:YYYY-MM-DD}.log"),
    rotation="00:00",
    retention="3 days",  # Keep error logs longer
    compression="zip",
    format="{time:YYYY-MM-DD HH:mm:ss} | {level: <8} | {name}:{function}:{line} - {message}",
    level="ERROR",  # Only errors and above
    enqueue=True,
)


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
    logger.remove()  # Remove all handlers

    # Re-add console handler with new level
    logger.add(
        sys.stderr,
        format="<green>{time:YYYY-MM-DD HH:mm:ss}</green> | <level>{level: <8}</level> | <cyan>{name}</cyan>:<cyan>{function}</cyan>:<cyan>{line}</cyan> - <level>{message}</level>",
        level=level.upper(),
        colorize=True,
    )

    # Re-add file handlers (keep their original levels)
    logger.add(
        os.path.join(DEFAULT_LOG_DIR, "forecast_{time:YYYY-MM-DD}.log"),
        rotation="00:00",
        retention="30 days",
        compression="zip",
        format="{time:YYYY-MM-DD HH:mm:ss} | {level: <8} | {name}:{function}:{line} - {message}",
        level="DEBUG",
        enqueue=True,
    )

    logger.add(
        os.path.join(DEFAULT_LOG_DIR, "errors_{time:YYYY-MM-DD}.log"),
        rotation="00:00",
        retention="90 days",
        compression="zip",
        format="{time:YYYY-MM-DD HH:mm:ss} | {level: <8} | {name}:{function}:{line} - {message}",
        level="ERROR",
        enqueue=True,
    )


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

    # Reconfigure with new directory
    logger.remove()

    logger.add(
        sys.stderr,
        format="<green>{time:YYYY-MM-DD HH:mm:ss}</green> | <level>{level: <8}</level> | <cyan>{name}</cyan>:<cyan>{function}</cyan>:<cyan>{line}</cyan> - <level>{message}</level>",
        level="INFO",
        colorize=True,
    )

    logger.add(
        os.path.join(DEFAULT_LOG_DIR, "forecast_{time:YYYY-MM-DD}.log"),
        rotation="00:00",
        retention="30 days",
        compression="zip",
        format="{time:YYYY-MM-DD HH:mm:ss} | {level: <8} | {name}:{function}:{line} - {message}",
        level="DEBUG",
        enqueue=True,
    )

    logger.add(
        os.path.join(DEFAULT_LOG_DIR, "errors_{time:YYYY-MM-DD}.log"),
        rotation="00:00",
        retention="90 days",
        compression="zip",
        format="{time:YYYY-MM-DD HH:mm:ss} | {level: <8} | {name}:{function}:{line} - {message}",
        level="ERROR",
        enqueue=True,
    )

    logger.info(f"Log directory changed to: {DEFAULT_LOG_DIR}")
