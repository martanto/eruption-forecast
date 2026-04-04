import os
import json
import time
import socket
import functools
import urllib.error
import urllib.parse
import urllib.request
from typing import Any, cast
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


def _error_message(
    hostname: str,
    display_name: str,
    timestamp: str,
    exc: Exception,
    elapsed: float | None = None,
) -> str:
    """Build a MarkdownV2-formatted error notification message.

    Assembles the standard error notification body used when the decorated
    function raises an exception. Optionally appends an elapsed-time line
    when ``elapsed`` is provided.

    Args:
        hostname (str): Machine hostname shown in the Host field.
        display_name (str): Task label shown in the Task field.
        timestamp (str): Pre-formatted timestamp string for the Time field.
        exc (Exception): The exception that was raised.
        elapsed (float | None): Elapsed seconds to append, or None to omit. Defaults to None.

    Returns:
        str: Complete MarkdownV2 message string, truncated to 4096 characters.
    """
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
    if elapsed is not None:
        lines += [
            "",
            "⏱ *Elapsed:*",
            f"`{_escape_md(_format_elapsed(elapsed))}`",
        ]
    return _truncate("\n".join(lines))


def _success_message(
    hostname: str,
    display_name: str,
    timestamp: str,
    elapsed: float | None = None,
) -> str:
    """Build a MarkdownV2-formatted success notification message.

    Assembles the standard success notification body used when the decorated
    function returns normally. Optionally appends an elapsed-time line
    when ``elapsed`` is provided.

    Args:
        hostname (str): Machine hostname shown in the Host field.
        display_name (str): Task label shown in the Task field.
        timestamp (str): Pre-formatted timestamp string for the Time field.
        elapsed (float | None): Elapsed seconds to append, or None to omit. Defaults to None.

    Returns:
        str: Complete MarkdownV2 message string, truncated to 4096 characters.
    """
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
    if elapsed is not None:
        lines += [
            "",
            "⏱ *Elapsed:*",
            f"`{_escape_md(_format_elapsed(elapsed))}`",
        ]
    return _truncate("\n".join(lines))


def _send_telegram_message(
    token: str, chat_id: str | int, text: str, timeout: float = 10.0
) -> None:
    """Send a text message to a Telegram chat via the Bot API.

    Posts the given text to the specified Telegram chat using the sendMessage
    endpoint. Network errors are caught and logged via loguru instead of
    propagating, so the caller's original result or exception is unaffected.

    Args:
        token (str): Telegram bot token obtained from BotFather.
        chat_id (str | int): Telegram chat ID or username to send the message to.
        text (str): The message body to send (max 4096 characters).
        timeout (float): Socket timeout in seconds for the HTTP request. Defaults to 10.0.
    """
    url = f"https://api.telegram.org/bot{token}/sendMessage"
    payload = json.dumps(
        {"chat_id": str(chat_id), "text": text, "parse_mode": "MarkdownV2"}
    ).encode()
    req = urllib.request.Request(
        url, data=payload, headers={"Content-Type": "application/json"}, method="POST"
    )
    try:
        with urllib.request.urlopen(req, timeout=timeout):
            pass
    except urllib.error.URLError as exc:
        logger.warning(f"Telegram notification failed: {exc}")


def _send_telegram_file(
    token: str,
    chat_id: str | int,
    path: Path,
    caption: str = "",
    timeout: float = 30.0,
    send_as_document: bool = False,
) -> None:
    """Send a single file or photo to a Telegram chat via the Bot API.

    Uploads the file at ``path`` using ``sendPhoto`` for PNG/JPEG images and
    ``sendDocument`` for all other files. When ``send_as_document`` is True,
    image files are also sent via ``sendDocument``. Missing files are skipped
    with a warning. Network errors are caught and logged rather than propagated.

    Args:
        token (str): Telegram bot token.
        chat_id (str | int): Target chat ID or username.
        path (Path): Local filesystem path to the file to upload.
        caption (str): Optional caption to attach to the file. Defaults to "".
        timeout (float): Socket timeout in seconds for the HTTP request. Defaults to 30.0.
        send_as_document (bool): Force files (including images) to be sent via
            ``sendDocument`` instead of ``sendPhoto``. Defaults to False.
    """
    if not path.exists():
        logger.warning(f"Telegram notify: file not found, skipping: {path}")
        return

    is_photo = (
        path.suffix.lower() in {".png", ".jpg", ".jpeg"}
    ) and not send_as_document
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
        with urllib.request.urlopen(req, timeout=timeout):
            pass
    except urllib.error.URLError as exc:
        logger.warning(f"Telegram file send failed for {path.name}: {exc}")


def notify(
    name: str | None = None,
    on_success: bool = True,
    on_error: bool = True,
    timeout: float = 3.0,
    bot_token: str | None = None,
    chat_id: str | int | None = None,
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
        name (str | None): Display name used in messages. Defaults to the
            decorated function's ``__name__``.
        on_success (bool): Whether to send a notification on successful
            completion. Defaults to True.
        on_error (bool): Whether to send a notification when an exception is
            raised. Defaults to True.
        timeout (float): Socket timeout in seconds passed to the Telegram HTTP
            requests. Defaults to 3.0.
        bot_token (str | None): Telegram bot token. Falls back to the
            ``TELEGRAM_BOT_TOKEN`` environment variable. Defaults to None.
        chat_id (str | int | None): Telegram chat ID or username. Falls back
            to the ``TELEGRAM_CHAT_ID`` environment variable. Defaults to None.
        include_elapsed (bool): Whether to append elapsed time to messages.
            Defaults to True.
        files (list[str | Path] | Callable[[Any], list[str | Path]] | None):
            Files to attach after a success notification. May be a static list
            of paths or a callable that receives the function's return value
            and returns a list of paths. Defaults to None.

    Returns:
        Callable: Decorator function that wraps the target function, or a
            no-op decorator if credentials are missing or invalid.

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

    def _noop(func: Callable) -> Callable:
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            return func(*args, **kwargs)

        return wrapper

    if not resolved_token:
        logger.warning(
            "notify: bot_token not provided and TELEGRAM_BOT_TOKEN not set in environment. "
            "Notifications disabled."
        )
        return _noop
    if any(char.isspace() for char in resolved_token):
        logger.warning(
            f"notify: bot_token contains whitespace (length {len(resolved_token)}). "
            "Ensure TELEGRAM_BOT_TOKEN is set to the raw token from BotFather. "
            "Notifications disabled."
        )
        return _noop
    if not resolved_chat:
        logger.warning(
            "notify: chat_id not provided and TELEGRAM_CHAT_ID not set in environment. "
            "Notifications disabled."
        )
        return _noop
    if isinstance(resolved_chat, str) and any(char.isspace() for char in resolved_chat):
        logger.warning(
            f"notify: chat_id contains whitespace — got {resolved_chat!r}. "
            "Notifications disabled."
        )
        return _noop

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
                    msg = _error_message(
                        hostname,
                        display_name,
                        timestamp,
                        exc,
                        elapsed if include_elapsed else None,
                    )
                    try:
                        _send_telegram_message(
                            resolved_token, resolved_chat, msg, timeout
                        )
                    except Exception as notify_exc:
                        logger.warning(
                            f"Failed to send error notification: {notify_exc}"
                        )
                raise

            elapsed = time.perf_counter() - start
            if on_success:
                msg = _success_message(
                    hostname,
                    display_name,
                    timestamp,
                    elapsed if include_elapsed else None,
                )
                try:
                    _send_telegram_message(resolved_token, resolved_chat, msg, timeout)
                except Exception as notify_exc:
                    logger.warning(f"Failed to send success notification: {notify_exc}")

                # Send attached files if any
                if files is not None:
                    try:
                        if callable(files):
                            files_callable = cast(
                                Callable[[Any], list[str | Path]],
                                files,
                            )
                            resolved_files = files_callable(result)
                        else:
                            resolved_files = files
                        for fp in resolved_files:
                            _send_telegram_file(
                                resolved_token, resolved_chat, Path(fp), timeout=timeout
                            )
                    except Exception as file_exc:
                        logger.warning(f"Failed to send attached files: {file_exc}")

            return result

        return wrapper

    return decorator


def send_telegram_notification(
    message: str,
    bot_token: str | None = None,
    chat_id: str | int | None = None,
    timeout: float = 10.0,
    escape_markdown: bool = True,
    files: list[str | Path] | None = None,
    file_caption: str = "",
    send_as_document: bool = False,
) -> None:
    """Send a direct Telegram notification without using the ``notify`` decorator.

    Credentials can be supplied explicitly via ``bot_token`` / ``chat_id`` or
    loaded from ``TELEGRAM_BOT_TOKEN`` / ``TELEGRAM_CHAT_ID`` in the environment
    (including values loaded from a ``.env`` file via python-dotenv).

    Args:
        message (str): Message body to send.
        bot_token (str | None): Telegram bot token. Falls back to environment.
            Defaults to None.
        chat_id (str | int | None): Telegram chat ID. Falls back to environment.
            Defaults to None.
        timeout (float): Socket timeout in seconds for HTTP requests.
            Defaults to 10.0.
        escape_markdown (bool): Escape Telegram MarkdownV2 special characters in
            ``message`` before sending. Defaults to True.
        files (list[str | Path] | None): Optional list of file paths to attach
            after the text message is sent. Defaults to None.
        file_caption (str): Optional caption for attached files. Defaults to "".
        send_as_document (bool): Force attachments (including PNG/JPG) to be
            sent via ``sendDocument``. Defaults to False.

    Raises:
        ValueError: If ``message`` is empty, credentials are missing, or
            credentials contain whitespace.
    """
    load_dotenv()

    resolved_token: str = bot_token or os.environ.get("TELEGRAM_BOT_TOKEN", "")
    resolved_chat: str | int = chat_id or os.environ.get("TELEGRAM_CHAT_ID", "")

    if not message:
        raise ValueError("message must not be empty.")
    if not resolved_token:
        raise ValueError(
            "bot_token not provided and TELEGRAM_BOT_TOKEN not set in environment."
        )
    if any(char.isspace() for char in resolved_token):
        raise ValueError("bot_token contains whitespace.")
    if not resolved_chat:
        raise ValueError(
            "chat_id not provided and TELEGRAM_CHAT_ID not set in environment."
        )
    if isinstance(resolved_chat, str) and any(char.isspace() for char in resolved_chat):
        raise ValueError("chat_id contains whitespace.")

    text = _escape_md(message) if escape_markdown else message
    _send_telegram_message(resolved_token, resolved_chat, text, timeout)

    if files:
        for file_path in files:
            _send_telegram_file(
                resolved_token,
                resolved_chat,
                Path(file_path),
                caption=file_caption,
                timeout=timeout,
                send_as_document=send_as_document,
            )


__all__ = ["notify", "send_telegram_notification"]
