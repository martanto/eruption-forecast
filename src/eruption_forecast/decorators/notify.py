"""Notification decorator that reports function success or failure to Telegram.

The module exposes :func:`notify`, a decorator factory that wraps a target
function and dispatches a MarkdownV2-formatted status message through
:class:`~eruption_forecast.notification.telegram.TelegramNotification` on
completion or on error. All private helpers in this module — message
formatters and the dispatch adapter — exist solely to keep :func:`notify`
readable and are not part of the public API.
"""

import time
import socket
import functools
from typing import Literal
from collections.abc import Callable

from eruption_forecast.logger import get_category_logger
from eruption_forecast.notification.telegram import TelegramNotification


# Category-bound logger so Telegram-dispatch failures land in
# ``logs/telegram_*.log`` instead of the general forecast log.
# See ``eruption_forecast.logger``.
logger = get_category_logger("telegram")


def _format_elapsed(seconds: float) -> str:
    """Format elapsed seconds into a human-readable HHh MMm SSs string.

    Converts a raw elapsed time in seconds into a zero-padded string showing
    hours, minutes, and seconds for use in notification messages.

    Args:
        seconds (float): Elapsed time in seconds.

    Returns:
        str: Formatted string in the form ``00h 00m 00s``.
    """
    total = int(seconds)
    hours, remainder = divmod(total, 3600)
    minutes, secs = divmod(remainder, 60)
    return f"{hours:02d}h {minutes:02d}m {secs:02d}s"


def _escape_md(text: str) -> str:
    """Escape special characters for Telegram MarkdownV2.

    MarkdownV2 requires that all characters outside of formatting constructs
    are escaped with a preceding backslash. This function escapes the full set
    of reserved characters listed in the Telegram Bot API documentation.

    Args:
        text (str): Raw string that may contain MarkdownV2 special characters.

    Returns:
        str: String with every reserved character backslash-escaped.
    """
    # Full set of characters that must be escaped in MarkdownV2 plain text
    reserved = r"\_*[]()~`>#+-=|{}.!"
    return "".join(f"\\{ch}" if ch in reserved else ch for ch in text)


def _escape_pre(text: str) -> str:
    """Escape characters for content inside a MarkdownV2 pre or code block.

    Inside pre and code entities Telegram only requires ``\\`` and backtick to
    be escaped; every other reserved character is treated as a literal. Using
    the full ``_escape_md`` inside a code block would render literal backslashes
    for every escaped character.

    Args:
        text (str): Raw string to include inside a code block.

    Returns:
        str: String with backslashes and backticks escaped.
    """
    return text.replace("\\", "\\\\").replace("`", "\\`")


def _error_message(
    exception: Exception,
    task: str,
    timestamp: str,
    elapsed_time: float,
    message: str,
) -> str:
    """Build a MarkdownV2 error notification body.

    Args:
        exception (Exception): The exception raised by the wrapped function.
        task (str): Task label shown in the message.
        timestamp (str): Pre-formatted timestamp string.
        elapsed_time (float): Elapsed seconds since the wrapped call started.
        message (str): Free-form message body supplied to the decorator.

    Returns:
        str: MarkdownV2-formatted error message.
    """
    lines = [
        f"*Hostname*: `{_escape_md(socket.gethostname())}`",
        f"*Task*: {_escape_md(task)}",
        f"*Timestamp*: {_escape_md(timestamp)}",
        f"*Message*: {_escape_md(message)}",
        f"*Elapsed Time*: {_escape_md(_format_elapsed(elapsed_time))}",
        f"*Status*: ❌ {_escape_md(type(exception).__name__)}",
        "*Details*:",
        "```python",
        _escape_pre(str(exception)),
        "```",
    ]
    return "\n".join(lines)


def _success_message(
    task: str,
    timestamp: str,
    elapsed_time: float,
    message: str | None = None,
) -> str:
    """Build a MarkdownV2 success notification body.

    Args:
        task (str): Task label shown in the message.
        timestamp (str): Pre-formatted timestamp string.
        elapsed_time (float): Elapsed seconds since the wrapped call started.
        message (str | None): Optional free-form message body. Defaults to None.

    Returns:
        str: MarkdownV2-formatted success message.
    """
    message = message or "-"

    lines = [
        f"*Hostname*: `{_escape_md(socket.gethostname())}`",
        f"*Task*: {_escape_md(task)}",
        f"*Timestamp*: {_escape_md(timestamp)}",
        f"*Message*: {_escape_md(message)}",
        f"*Elapsed Time*: {_escape_md(_format_elapsed(elapsed_time))}",
        "*Status*: ✅ finished successfully",
    ]
    return "\n".join(lines)


def _dispatch(
    to: Literal["telegram", "email"],
    message: str,
    timeout: float,
    verbose: bool,
) -> None:
    """Send a notification via the selected backend, swallowing send errors.

    Args:
        to (Literal["telegram", "email"]): Target backend. ``"email"`` is not
            implemented yet and falls back to a warning.
        message (str): Pre-formatted MarkdownV2 message body.
        timeout (float): Socket timeout in seconds for the HTTP request.
        verbose (bool): Forwarded to ``TelegramNotification``.
    """
    if to == "telegram":
        try:
            TelegramNotification(verbose=verbose).send_message(
                message=message, timeout=timeout
            )
        except Exception as notify_exc:
            logger.warning(f"Failed to send Telegram notification: {notify_exc}")
    else:
        logger.warning("`email` currently not supported. use `telegram` instead")


def notify(
    task: str,
    message: str | None = None,
    to: Literal["telegram", "email"] = "telegram",
    on_success: bool = True,
    on_error: bool = True,
    timeout: float = 3.0,
    verbose: bool = False,
):
    """Decorator factory that sends a notification when the wrapped function finishes or fails.

    The returned decorator preserves the wrapped function's signature and
    metadata via :func:`functools.wraps`. When the wrapped call raises,
    an error notification is dispatched (if ``on_error`` is ``True``) and
    the original exception is re-raised so upstream error handling is
    preserved.

    Args:
        task (str): Task label shown in every notification message.
        message (str | None): Optional free-form message body. Defaults
            to ``None``.
        to (Literal["telegram", "email"]): Notification backend. Only
            ``"telegram"`` is currently implemented; ``"email"`` logs a
            warning and no message is sent. Defaults to ``"telegram"``.
        on_success (bool): Send a notification on successful completion.
            Defaults to ``True``.
        on_error (bool): Send a notification when an exception is raised.
            Defaults to ``True``.
        timeout (float): Socket timeout in seconds for the notification
            HTTP request. Defaults to ``3.0``.
        verbose (bool): Forwarded to ``TelegramNotification``. Defaults
            to ``False``.

    Returns:
        Callable: Decorator that wraps the target function while
            preserving its signature and re-raising any exception it
            raises.
    """

    def decorator(func: Callable) -> Callable:
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            start_time = time.perf_counter()
            timestamp = time.strftime("%Y-%m-%d %H:%M:%S")

            try:
                result = func(*args, **kwargs)
            except Exception as exc:
                elapsed_time = time.perf_counter() - start_time
                if on_error:
                    _message = _error_message(
                        exception=exc,
                        task=task,
                        timestamp=timestamp,
                        elapsed_time=elapsed_time,
                        message=message or "-",
                    )
                    _dispatch(to, _message, timeout, verbose)
                raise

            elapsed_time = time.perf_counter() - start_time
            if on_success:
                _message = _success_message(
                    task=task,
                    timestamp=timestamp,
                    elapsed_time=elapsed_time,
                    message=message or func.__name__,  # ty:ignore[unresolved-attribute]
                )
                _dispatch(to, _message, timeout, verbose)

            return result

        return wrapper

    return decorator
