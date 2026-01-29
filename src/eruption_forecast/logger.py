"""
Centralized logger module using loguru for the package.
This provides a single logger instance that all modules can import and use.
"""

import sys
import os
from loguru import logger

# Init default handler
logger.remove()

# Define default log directory
DEFAULT_LOG_DIR = os.path.join(os.path.expanduser("~"), ".my_package", "logs")
os.makedirs(DEFAULT_LOG_DIR, exist_ok=True)

# Add console handler with custom format
logger.add(
    sys.stderr,
    format="<green>{time:YYYY-MM-DD HH:mm:ss}</green> | <level>{level: <8}</level> | <cyan>{name}</cyan>:<cyan>{function}</cyan>:<cyan>{line}</cyan> - <level>{message}</level>",
    level="INFO",
    colorize=True,
)

# Add file handler with rotation
logger.add(
    os.path.join(DEFAULT_LOG_DIR, "app_{time:YYYY-MM-DD}.log"),
    rotation="00:00",  # Rotate at midnight
    retention="30 days",  # Keep logs for 30 days
    compression="zip",  # Compress rotated logs
    format="{time:YYYY-MM-DD HH:mm:ss} | {level: <8} | {name}:{function}:{line} - {message}",
    level="DEBUG",  # File logs everything including DEBUG
    enqueue=True,  # Thread-safe logging
)

# Add error-specific log file
logger.add(
    os.path.join(DEFAULT_LOG_DIR, "errors_{time:YYYY-MM-DD}.log"),
    rotation="00:00",
    retention="90 days",  # Keep error logs longer
    compression="zip",
    format="{time:YYYY-MM-DD HH:mm:ss} | {level: <8} | {name}:{function}:{line} - {message}",
    level="ERROR",  # Only errors and above
    enqueue=True,
)


def get_logger():
    """
    Get the configured logger instance.

    Returns:
        loguru.Logger: The configured logger instance
    """
    return logger


def set_log_level(level: str):
    """
    Change the console log level dynamically.

    Args:
        level: Log level (DEBUG, INFO, WARNING, ERROR, CRITICAL)
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
        os.path.join(DEFAULT_LOG_DIR, "app_{time:YYYY-MM-DD}.log"),
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


def set_log_directory(log_dir: str):
    """
    Change the log directory dynamically.

    Args:
        log_dir: New directory path for log files
    """
    global DEFAULT_LOG_DIR
    DEFAULT_LOG_DIR = os.path.abspath(log_dir)
    os.makedirs(DEFAULT_LOG_DIR, exist_ok=True)

    # Reconfigure with new directory
    logger.remove()

    logger.add(
        sys.stderr,
        format="<green>{time:YYYY-MM-DD HH:mm:ss}</green> | <level>{level: <8}</level> | <cyan>{name}</cyan>:<cyan>{function}</cyan>:<cyan>{line}</cyan> - <level>{message}</level>",
        level="INFO",
        colorize=True,
    )

    logger.add(
        os.path.join(DEFAULT_LOG_DIR, "app_{time:YYYY-MM-DD}.log"),
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
