import os
import json
import time
import socket
import functools
import urllib.error
import urllib.parse
import urllib.request
from typing import Any
from pathlib import Path
from datetime import datetime
from collections.abc import Callable

from dotenv import load_dotenv
from loguru import logger


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


def _truncate(text: str, max_len: int = 4096) -> str:
    """Truncate a string to a maximum length, appending a notice if cut.

    Ensures the message fits within Telegram's 4096-character limit by
    appending ``... (truncated)`` when the original text exceeds ``max_len``.

    Args:
        text (str): The string to truncate.
        max_len (int): Maximum allowed length. Defaults to 4096.

    Returns:
        str: Original string if within limit, otherwise truncated with suffix.
    """
    suffix = "... (truncated)"
    if len(text) <= max_len:
        return text
    return text[: max_len - len(suffix)] + suffix


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


def _send_telegram_message(token: str, chat_id: str | int, text: str) -> None:
    """Send a text message to a Telegram chat via the Bot API.

    Posts the given text to the specified Telegram chat using the sendMessage
    endpoint. Network errors are caught and logged via loguru instead of
    propagating, so the caller's original result or exception is unaffected.

    Args:
        token (str): Telegram bot token obtained from BotFather.
        chat_id (str | int): Telegram chat ID or username to send the message to.
        text (str): The message body to send (max 4096 characters).
    """
    url = f"https://api.telegram.org/bot{token}/sendMessage"
    payload = json.dumps(
        {"chat_id": str(chat_id), "text": text, "parse_mode": "MarkdownV2"}
    ).encode()
    req = urllib.request.Request(
        url, data=payload, headers={"Content-Type": "application/json"}, method="POST"
    )
    try:
        with urllib.request.urlopen(req, timeout=10):
            pass
    except urllib.error.URLError as exc:
        logger.warning(f"Telegram notification failed: {exc}")


def _send_telegram_file(
    token: str, chat_id: str | int, path: Path, caption: str = ""
) -> None:
    """Send a single file or photo to a Telegram chat via the Bot API.

    Uploads the file at ``path`` using ``sendDocument`` for generic files, or
    ``sendPhoto`` for PNG/JPEG images. Missing files are skipped with a warning.
    Network errors are caught and logged rather than propagated.

    Args:
        token (str): Telegram bot token.
        chat_id (str | int): Target chat ID or username.
        path (Path): Local filesystem path to the file to upload.
        caption (str): Optional caption to attach to the file. Defaults to "".
    """
    if not path.exists():
        logger.warning(f"Telegram notify: file not found, skipping: {path}")
        return

    is_photo = path.suffix.lower() in {".png", ".jpg", ".jpeg"}
    endpoint = "sendPhoto" if is_photo else "sendDocument"
    field = "photo" if is_photo else "document"
    url = f"https://api.telegram.org/bot{token}/{endpoint}"

    # Build a multipart/form-data body manually using urllib
    boundary = "----TelegramBoundary"
    file_data = path.read_bytes()
    mime = "image/png" if is_photo else "application/octet-stream"

    body_parts: list[bytes] = []

    def _field(name: str, value: str) -> bytes:
        """Encode a plain text form field for multipart upload.

        Args:
            name (str): Form field name.
            value (str): Field value as a string.

        Returns:
            bytes: Encoded form field bytes for multipart body.
        """
        return (
            f"--{boundary}\r\n"
            f'Content-Disposition: form-data; name="{name}"\r\n\r\n'
            f"{value}\r\n"
        ).encode()

    body_parts.append(_field("chat_id", str(chat_id)))
    if caption:
        body_parts.append(_field("caption", caption))
    body_parts.append(
        (
            f"--{boundary}\r\n"
            f'Content-Disposition: form-data; name="{field}"; filename="{path.name}"\r\n'
            f"Content-Type: {mime}\r\n\r\n"
        ).encode()
        + file_data
        + b"\r\n"
    )
    body_parts.append(f"--{boundary}--\r\n".encode())

    body = b"".join(body_parts)
    req = urllib.request.Request(
        url,
        data=body,
        headers={"Content-Type": f"multipart/form-data; boundary={boundary}"},
        method="POST",
    )
    try:
        with urllib.request.urlopen(req, timeout=30):
            pass
    except urllib.error.URLError as exc:
        logger.warning(f"Telegram file send failed for {path.name}: {exc}")


def notify(
    bot_token: str | None = None,
    chat_id: str | int | None = None,
    name: str | None = None,
    on_success: bool = True,
    on_error: bool = True,
    include_elapsed: bool = True,
    files: list[str | Path] | Callable[[Any], list[str | Path]] | None = None,
):
    """Decorator factory that sends Telegram notifications when a function finishes or fails.

    Wraps the decorated function so that a Telegram message is dispatched after
    execution — a success message when the function returns normally, and an
    error message when an exception is raised. The exception is always re-raised
    after notification so the caller's control flow is preserved.

    Credentials can be supplied explicitly via ``bot_token`` / ``chat_id`` or
    loaded automatically from environment variables ``TELEGRAM_BOT_TOKEN`` /
    ``TELEGRAM_CHAT_ID`` (populated via a ``.env`` file using python-dotenv).

    Args:
        bot_token (str | None): Telegram bot token. Falls back to the
            ``TELEGRAM_BOT_TOKEN`` environment variable. Defaults to None.
        chat_id (str | int | None): Telegram chat ID or username. Falls back
            to the ``TELEGRAM_CHAT_ID`` environment variable. Defaults to None.
        name (str | None): Display name used in messages. Defaults to the
            decorated function's ``__name__``.
        on_success (bool): Whether to send a notification on successful
            completion. Defaults to True.
        on_error (bool): Whether to send a notification when an exception is
            raised. Defaults to True.
        include_elapsed (bool): Whether to append elapsed time to messages.
            Defaults to True.
        files (list[str | Path] | Callable[[Any], list[str | Path]] | None):
            Files to attach after a success notification. May be a static list
            of paths or a callable that receives the function's return value
            and returns a list of paths. Defaults to None.

    Returns:
        Callable: Decorator function that wraps the target function.

    Raises:
        ValueError: If ``bot_token`` or ``chat_id`` cannot be resolved from
            arguments or environment variables at decoration time.

    Examples:
        >>> @notify(bot_token="TOKEN", chat_id="123", name="Training")
        ... def train_model():
        ...     return {"accuracy": 0.95}

        >>> @notify()  # reads TELEGRAM_BOT_TOKEN / TELEGRAM_CHAT_ID from .env
        ... def long_running_job():
        ...     pass

        >>> @notify(files=lambda result: [result["plot_path"]])
        ... def make_plot():
        ...     return {"plot_path": "output/plot.png"}
    """
    load_dotenv()

    resolved_token: str = bot_token or os.environ.get("TELEGRAM_BOT_TOKEN", "")
    resolved_chat: str | int = chat_id or os.environ.get("TELEGRAM_CHAT_ID", "")

    if not resolved_token:
        raise ValueError(
            "notify: bot_token not provided and TELEGRAM_BOT_TOKEN not set in environment."
        )
    if not resolved_chat:
        raise ValueError(
            "notify: chat_id not provided and TELEGRAM_CHAT_ID not set in environment."
        )

    hostname = socket.gethostname()

    def decorator(func: Callable) -> Callable:
        """Wrap the target function with Telegram notification logic.

        Args:
            func (Callable): The function to decorate.

        Returns:
            Callable: Wrapped function that sends Telegram notifications.
        """
        display_name = name if name else func.__name__  # ty:ignore[unresolved-attribute]

        @functools.wraps(func)
        def wrapper(*args, **kwargs) -> Any:
            """Execute the wrapped function and send a Telegram notification on finish or error.

            Returns:
                Any: The return value of the original function.

            Raises:
                Exception: Re-raises any exception raised by the original function
                    after sending an error notification.
            """
            start = time.perf_counter()
            timestamp = datetime.now().strftime("%Y\\-%m\\-%d %H:%M:%S")
            try:
                result = func(*args, **kwargs)
            except Exception as exc:
                elapsed = time.perf_counter() - start
                if on_error:
                    lines = [
                        "🖥 *Host:*",
                        f"`{_escape_md(hostname)}`",
                        "",
                        "📋 *Task:*",
                        f"`{_escape_md(display_name)}`",
                        "",
                        "🕐 *Time:*",
                        f"`{timestamp}`",
                        "",
                        "❌ *Status:*",
                        f"raised `{_escape_md(type(exc).__name__)}`",
                        "",
                        "💬 *Message:*",
                        f"`{_escape_md(str(exc))}`",
                    ]
                    if include_elapsed:
                        lines += [
                            "",
                            "⏱ *Elapsed:*",
                            f"`{_escape_md(_format_elapsed(elapsed))}`",
                        ]
                    try:
                        _send_telegram_message(
                            resolved_token, resolved_chat, _truncate("\n".join(lines))
                        )
                    except Exception as notify_exc:
                        logger.warning(
                            f"Failed to send error notification: {notify_exc}"
                        )
                raise

            elapsed = time.perf_counter() - start
            if on_success:
                lines = [
                    "🖥 *Host:*",
                    f"`{_escape_md(hostname)}`",
                    "",
                    "📋 *Task:*",
                    f"`{_escape_md(display_name)}`",
                    "",
                    "🕐 *Time:*",
                    f"`{timestamp}`",
                    "",
                    "✅ *Status:*",
                    "finished successfully\\.",
                    "",
                    "💬 *Message:*",
                    "Function completed without errors\\.",
                ]
                if include_elapsed:
                    lines += [
                        "",
                        "⏱ *Elapsed:*",
                        f"`{_escape_md(_format_elapsed(elapsed))}`",
                    ]
                try:
                    _send_telegram_message(
                        resolved_token, resolved_chat, _truncate("\n".join(lines))
                    )
                except Exception as notify_exc:
                    logger.warning(f"Failed to send success notification: {notify_exc}")

                # Send attached files if any
                if files is not None:
                    try:
                        resolved_files: list[str | Path] = (
                            files(result) if callable(files) else files
                        )
                        for fp in resolved_files:
                            _send_telegram_file(resolved_token, resolved_chat, Path(fp))
                    except Exception as file_exc:
                        logger.warning(f"Failed to send attached files: {file_exc}")

            return result

        return wrapper

    return decorator


__all__ = [
    "notify",
    "_format_elapsed",
    "_truncate",
    "_escape_md",
    "_send_telegram_message",
    "_send_telegram_file",
]
