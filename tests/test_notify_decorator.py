"""Manual integration tests for the notify decorator.

Run with:
    uv run pytest tests/test_notify_decorator.py -v -s

These tests send real Telegram messages. They require TELEGRAM_BOT_TOKEN and
TELEGRAM_CHAT_ID to be set in the .env file or environment. Each test is marked
with @pytest.mark.integration so they can be skipped in CI with:
    uv run pytest -m "not integration"
"""

# Standard library imports
import time
from unittest.mock import patch

# Third party imports
import pytest

# Project imports
from eruption_forecast.decorators import notify


pytestmark = pytest.mark.integration


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


def _make_decorator(**kwargs):
    """Return a notify decorator configured from .env credentials.

    Passes any extra kwargs straight through to notify() so individual tests
    can customise on_success, on_error, name, etc.

    Args:
        **kwargs: Additional keyword arguments forwarded to notify().

    Returns:
        Callable: The configured notify decorator.
    """
    # Passing no bot_token/chat_id forces the decorator to read from .env
    return notify(**kwargs)


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


def test_success_notification():
    """Send a success Telegram message for a fast-returning function.

    Decorates a trivial function, calls it, and asserts it returns the expected
    value,� confirming the decorator does not swallow the return value.
    """

    @_make_decorator(name="test_success_notification")
    def quick_task():
        """Return a constant to verify the decorator passes return values through.

        Returns:
            int: Always 42.
        """
        return 42

    result = quick_task()
    assert result == 42


def test_error_notification():
    """Send an error Telegram message when the decorated function raises.

    Asserts that the original exception is re-raised after the notification,
    preserving the caller's error-handling contract.
    """

    @_make_decorator(name="test_error_notification")
    def failing_task():
        """Raise a ValueError to trigger the error notification path.

        Raises:
            ValueError: Always raised to exercise the on_error branch.
        """
        raise ValueError("Intentional test error � please ignore.")

    with pytest.raises(ValueError, match="Intentional test error"):
        failing_task()


def test_elapsed_time_included():
    """Send a success message that includes elapsed time.

    Sleeps briefly so the elapsed field reads > 0 seconds, making it
    visually verifiable in the Telegram message.
    """

    @_make_decorator(name="test_elapsed_time", include_elapsed=True)
    def slow_task():
        """Sleep for a short duration to produce a non-zero elapsed time.

        Returns:
            str: A simple status string.
        """
        time.sleep(2)
        return "done"

    result = slow_task()
    assert result == "done"


def test_on_success_false_no_message():
    """Confirm that on_success=False suppresses the success notification.

    The function runs normally and returns its value; no Telegram message
    should be dispatched on success.
    """

    @_make_decorator(name="test_on_success_false", on_success=False)
    def silent_task():
        """Return a value without triggering a success notification.

        Returns:
            bool: Always True.
        """
        return True

    assert silent_task() is True


def test_on_error_false_no_message():
    """Confirm that on_error=False suppresses the error notification.

    The exception must still propagate to the caller even when notification
    is disabled.
    """

    @_make_decorator(name="test_on_error_false", on_error=False)
    def silent_failing_task():
        """Raise without sending an error notification.

        Raises:
            RuntimeError: Always raised to verify suppression path.
        """
        raise RuntimeError("Silent error � no Telegram message expected.")

    with pytest.raises(RuntimeError):
        silent_failing_task()


def test_missing_credentials_disables_notifications(monkeypatch):
    """Verify that missing credentials disables notifications without raising.

    Clears the env vars so the .env fallback is also unavailable, then
    passes empty strings to confirm the decorator still wraps the function.

    Args:
        monkeypatch: pytest fixture for safely patching environment variables.
    """
    monkeypatch.delenv("TELEGRAM_BOT_TOKEN", raising=False)
    monkeypatch.delenv("TELEGRAM_CHAT_ID", raising=False)
    # Prevent load_dotenv() inside notify() from re-populating the env from .env
    with patch("eruption_forecast.decorators.notify.load_dotenv"):
        decorator = notify(bot_token="", chat_id="")
        assert callable(decorator)
