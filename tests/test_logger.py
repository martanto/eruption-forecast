"""Unit tests for the category-scoped logging feature."""

import os
import tempfile

import pytest

from eruption_forecast import logger as logger_module
from eruption_forecast.logger import (
    logger,
    set_log_directory,
    get_category_logger,
    _uncategorized_filter,
    register_error_category,
)


@pytest.fixture()
def _isolated_log_dir():
    """Redirect the loguru sinks to a temp directory for the duration of the test.

    Drops every sink before the temp directory is cleaned up so Windows can
    release the file handles loguru holds open, then restores the module-wide
    log directory so subsequent tests keep the default configuration.
    """
    original_dir = logger_module.DEFAULT_LOG_DIR
    with tempfile.TemporaryDirectory() as tmp:
        set_log_directory(tmp)
        try:
            yield tmp
        finally:
            logger.remove()
    set_log_directory(original_dir)


def test_categorized_warning_routes_to_category_file_only(_isolated_log_dir):
    """A warning tagged with ``category=telegram`` lands only in ``telegram_*.log``."""
    telegram_captured: list[str] = []
    general_captured: list[str] = []

    tg_sink = logger.add(
        lambda msg: telegram_captured.append(str(msg)),
        level="WARNING",
        filter=lambda record: record["extra"].get("category") == "telegram",
    )
    general_sink = logger.add(
        lambda msg: general_captured.append(str(msg)),
        level="WARNING",
        filter=_uncategorized_filter,
    )

    try:
        get_category_logger("telegram").warning("send failed")
    finally:
        logger.remove(tg_sink)
        logger.remove(general_sink)

    assert any("send failed" in m for m in telegram_captured)
    assert not any("send failed" in m for m in general_captured)


def test_uncategorized_warning_stays_out_of_category_sink(_isolated_log_dir):
    """A plain ``logger.warning`` bypasses the telegram sink."""
    telegram_captured: list[str] = []
    general_captured: list[str] = []

    tg_sink = logger.add(
        lambda msg: telegram_captured.append(str(msg)),
        level="WARNING",
        filter=lambda record: record["extra"].get("category") == "telegram",
    )
    general_sink = logger.add(
        lambda msg: general_captured.append(str(msg)),
        level="WARNING",
        filter=_uncategorized_filter,
    )

    try:
        logger.warning("plain warning")
    finally:
        logger.remove(tg_sink)
        logger.remove(general_sink)

    assert not any("plain warning" in m for m in telegram_captured)
    assert any("plain warning" in m for m in general_captured)


def test_general_error_sink_excludes_known_categories(_isolated_log_dir):
    """The exclusion filter used by ``errors_*.log`` rejects categorised records."""
    error_captured: list[str] = []

    error_sink = logger.add(
        lambda msg: error_captured.append(str(msg)),
        level="ERROR",
        filter=_uncategorized_filter,
    )

    try:
        get_category_logger("telegram").error("telegram-scoped error")
        logger.error("global error")
    finally:
        logger.remove(error_sink)

    assert not any("telegram-scoped error" in m for m in error_captured)
    assert any("global error" in m for m in error_captured)


def test_register_error_category_is_idempotent(_isolated_log_dir):
    """Re-registering a category rebuilds sinks without duplicating handlers."""
    register_error_category("telegram")
    baseline_handlers = dict(logger._core.handlers)  # ty:ignore[unresolved-attribute]

    register_error_category("telegram")
    after_handlers = dict(logger._core.handlers)  # ty:ignore[unresolved-attribute]

    assert len(after_handlers) == len(baseline_handlers)


def test_register_error_category_creates_file_on_write(_isolated_log_dir):
    """A newly-registered category creates its dedicated file when a record is written."""
    register_error_category("smoke", level="WARNING", retention="1 day")

    get_category_logger("smoke").warning("smoke test entry")

    files = os.listdir(_isolated_log_dir)
    assert any(f.startswith("smoke_") and f.endswith(".log") for f in files), files
