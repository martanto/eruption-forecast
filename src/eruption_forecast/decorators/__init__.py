import os
import json
import time
import socket
import inspect
import functools
import urllib.error
import urllib.parse
import urllib.request
from typing import Any, Literal
from pathlib import Path
from datetime import datetime, timedelta
from collections.abc import Callable

from dotenv import load_dotenv
from loguru import logger

from eruption_forecast.decorators.decorator_class import (
    AutoSaveDict,
    SerializationWrapper,
)


def save_parameters(
    filepath: str | None = None,
    save_as: Literal["json", "yaml"] = "json",
    include_self: bool = False,
    append_timestamp: bool = False,
):
    """Decorator factory that saves function/method parameters to a file before execution.

    Captures all parameters passed to the decorated function using introspection,
    serializes them to JSON or YAML format, and saves them to disk. Useful for
    tracking function calls, debugging, or creating audit logs.

    Args:
        filepath (str | None, optional): Path to the output file. If None, defaults
            to `{function_name}_params.{save_as}`. Defaults to None.
        save_as (Literal["json", "yaml"], optional): Serialization format for the
            output file. Defaults to "json".
        include_self (bool, optional): Whether to include `self` or `cls` parameters
            in the saved data for instance or class methods. Defaults to False.
        append_timestamp (bool, optional): Whether to append a timestamp in
            `YYYYMMDD_HHMMSS` format to the filename. Defaults to False.

    Returns:
        Callable: Decorator function that wraps the target function/method.

    Examples:
        >>> @save_parameters('my_params.json', save_as='json')
        ... def my_function(x, y, z=10):
        ...     return x + y + z
        >>> my_function(5, 3)
        8
        >>> # Creates my_params.json with parameters x=5, y=3, z=10

        >>> @save_parameters(append_timestamp=True, save_as='yaml')
        ... def process_data(data, threshold=0.5):
        ...     return len(data)
        >>> process_data([1, 2, 3], threshold=0.8)
        3
        >>> # Creates process_data_params_20250120_143025.yaml
    """

    def decorator(func: Callable) -> Callable:
        """Wrap the target function to save its parameters before execution.

        Args:
            func (Callable): The function to decorate.

        Returns:
            Callable: Wrapped function that saves parameters then calls func.
        """

        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            """Execute the wrapped function after saving its call parameters to disk.

            Returns:
                Any: The return value of the original function.
            """
            # Get function signature
            sig = inspect.signature(func)
            bound_args = sig.bind(*args, **kwargs)
            bound_args.apply_defaults()

            # Prepare data to save
            params_dict = dict(bound_args.arguments)

            # Remove 'self' or 'cls' if not requested
            if not include_self:
                params_dict.pop("self", None)
                params_dict.pop("cls", None)

            data = {
                "function": func.__name__,  # ty:ignore[unresolved-attribute]
                "timestamp": datetime.now().isoformat(),
                "parameters": params_dict,
            }

            # Determine filepath
            if filepath:
                output_path = filepath
            else:
                output_path = f"{func.__name__}_params.{format}"  # ty:ignore[unresolved-attribute]

            if append_timestamp:
                path = Path(output_path)
                timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                output_path = f"{path.stem}_{timestamp}{path.suffix}"

            # Save to file
            SerializationWrapper.save_to_file(data, output_path, save_as)

            # Execute original function
            return func(*args, **kwargs)

        return wrapper

    return decorator


def save_properties(
    filepath: str | None = None,
    save_as: Literal["json", "yaml"] = "json",
    include_private: bool = False,
    properties: list | None = None,
):
    """Class decorator that adds property persistence capabilities to a class.

    Adds `save_properties()` and `enable_auto_save()` methods to the decorated
    class, allowing instances to save their attributes to JSON or YAML files.
    Useful for configuration management, state persistence, or checkpointing.

    Args:
        filepath (str | None, optional): Default output file path. If None, defaults
            to `{ClassName}_properties.{save_as}`. Defaults to None.
        save_as (Literal["json", "yaml"], optional): Serialization format for saved
            properties. Defaults to "json".
        include_private (bool, optional): Whether to include attributes starting
            with underscore (`_`) when saving. Defaults to False.
        properties (list | None, optional): List of specific attribute names to save.
            If None, all public (or private if enabled) attributes are saved.
            Defaults to None.

    Returns:
        Callable: Decorator function that enhances the target class with save methods.

    Examples:
        >>> @save_properties('my_class.json', save_as='json')
        ... class MyClass:
        ...     def __init__(self, x, y):
        ...         self.x = x
        ...         self.y = y
        >>> obj = MyClass(10, 20)
        >>> obj.save_properties()
        'my_class.json'
        >>> # Creates my_class.json with properties x=10, y=20

        >>> @save_properties(include_private=True, properties=['x', '_internal'])
        ... class Config:
        ...     def __init__(self):
        ...         self.x = 5
        ...         self._internal = "secret"
        >>> cfg = Config()
        >>> cfg.save_properties()
        'Config_properties.json'
    """

    def decorator(cls):
        """Enhance the target class with property save methods.

        Args:
            cls: The class to decorate.

        Returns:
            type: The decorated class with save_properties and enable_auto_save methods.
        """
        original_init = cls.__init__

        @functools.wraps(original_init)
        def new_init(self, *args, **kwargs):
            """Initialize the instance and inject save-related configuration attributes.

            Args:
                *args: Positional arguments forwarded to the original __init__.
                **kwargs: Keyword arguments forwarded to the original __init__.
            """
            original_init(self, *args, **kwargs)
            self._save_format = save_as
            self._save_filepath = filepath or f"{cls.__name__}_properties.{save_as}"

        def save_properties_method(
            self,
            output_path: str | None = None,
            output_format: Literal["json", "yaml"] | None = None,
        ):
            """Save current instance properties to a file.

            Args:
                output_path (str | None, optional): Path to the output file. If None,
                    uses the default path configured in the decorator. Defaults to None.
                output_format (Literal["json", "yaml"] | None, optional): Serialization
                    format. If None, uses the format configured in the decorator.
                    Defaults to None.

            Returns:
                str: Path to the saved file.
            """
            path = output_path or self._save_filepath
            fmt = output_format or self._save_format

            # Collect properties
            props = {}
            for key, value in self.__dict__.items():
                # Skip internal save configuration
                if key.startswith("_save_"):
                    continue

                # Handle private attributes
                if key.startswith("_") and not include_private:
                    continue

                # Handle specific properties filter
                if properties is not None and key not in properties:
                    continue

                props[key] = value

            data = {
                "class": cls.__name__,
                "timestamp": datetime.now().isoformat(),
                "properties": props,
            }

            SerializationWrapper.save_to_file(data, path, fmt)
            return path

        def save_on_change(self, attr_name: str):
            """Enable auto-save when a specific attribute changes.

            Converts the attribute to a property with a setter that triggers
            `save_properties()` whenever the attribute value is modified.

            Args:
                attr_name (str): Name of the attribute to monitor for changes.

            Returns:
                None
            """
            original_value = getattr(self, attr_name)

            def setter(new_value):
                """Set the monitored attribute value and trigger auto-save.

                Args:
                    new_value: The new value to assign to the monitored attribute.
                """
                setattr(self, f"_{attr_name}_internal", new_value)
                self.save_properties()

            def getter():
                """Return the current value of the monitored attribute.

                Returns:
                    Any: The current value of the monitored attribute.
                """
                return getattr(self, f"_{attr_name}_internal", original_value)

            setattr(self, f"_{attr_name}_internal", original_value)
            setattr(cls, attr_name, property(getter, setter))  # ty:ignore[invalid-argument-type]

        # Add methods to class
        cls.__init__ = new_init
        cls.save_properties = save_properties_method
        cls.enable_auto_save = save_on_change

        return cls

    return decorator


def snapshot(filepath: str | None = None, save_as: Literal["json", "yaml"] = "json"):
    """Decorator factory that captures function parameters and return value.

    Saves both the input parameters and the function's return value to a file,
    creating a complete snapshot of the function call for debugging or auditing.

    Args:
        filepath (str | None, optional): Path to the output file. If None, defaults
            to `{function_name}_snapshot.{save_as}`. Defaults to None.
        save_as (Literal["json", "yaml"], optional): Serialization format for the
            snapshot file. Defaults to "json".

    Returns:
        Callable: Decorator function that wraps the target function.

    Examples:
        >>> @snapshot('calculation.json')
        ... def calculate(x, y):
        ...     return x * y
        >>> result = calculate(5, 3)
        >>> result
        15
        >>> # Creates calculation.json with parameters x=5, y=3, return_value=15

        >>> @snapshot(save_as='yaml')
        ... def process(data, multiplier=2):
        ...     return [x * multiplier for x in data]
        >>> process([1, 2, 3], multiplier=3)
        [3, 6, 9]
        >>> # Creates process_snapshot.yaml with full call details
    """

    def decorator(func: Callable) -> Callable:
        """Wrap the target function to capture its call as a snapshot.

        Args:
            func (Callable): The function to decorate.

        Returns:
            Callable: Wrapped function that captures parameters and return value.
        """

        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            """Execute the wrapped function and save both its parameters and return value.

            Returns:
                Any: The return value of the original function.
            """
            # Capture parameters
            sig = inspect.signature(func)
            bound_args = sig.bind(*args, **kwargs)
            bound_args.apply_defaults()
            params_dict = dict(bound_args.arguments)
            params_dict.pop("self", None)
            params_dict.pop("cls", None)

            # Execute function
            result = func(*args, **kwargs)

            # Save snapshot
            data = {
                "function": func.__name__,  # ty:ignore[unresolved-attribute]
                "timestamp": datetime.now().isoformat(),
                "parameters": params_dict,
                "return_value": result,
            }

            output_path = filepath or f"{func.__name__}_snapshot.{save_as}"  # ty:ignore[unresolved-attribute]
            SerializationWrapper.save_to_file(data, output_path, save_as)

            return result

        return wrapper

    return decorator


def timer(name: str | None = None, verbose: bool = True):
    """Decorator factory for measuring and displaying function execution time.

    Measures the wall-clock time taken by a function and optionally prints a
    formatted duration message in hours, minutes, and seconds format.

    Args:
        name (str | None, optional): Custom name to display in the timing message.
            If None, uses the function's `__name__`. Defaults to None.
        verbose (bool, optional): Whether to print the timing results to stdout.
            If False, the function executes silently. Defaults to True.

    Returns:
        Callable: Decorator function that wraps the target function.

    Examples:
        >>> @timer()
        ... def my_function():
        ...     time.sleep(1)
        >>> my_function()
        ====================================================================================================
        || my_function: took 00 hours, 00 minutes, 01 seconds
        ====================================================================================================

        >>> @timer(name="Data Processing", verbose=True)
        ... def process_large_dataset(data):
        ...     return sum(data)
        >>> process_large_dataset(range(1000000))
        ====================================================================================================
        || Data Processing: took 00 hours, 00 minutes, 00 seconds
        ====================================================================================================
        499999500000
    """

    def decorator(func: Callable) -> Callable:
        """Wrap the target function with execution timing logic.

        Args:
            func (Callable): The function to decorate.

        Returns:
            Callable: Wrapped function that measures and optionally prints elapsed time.
        """
        operation_name = name if name else func.__name__  # ty:ignore[unresolved-attribute]

        @functools.wraps(func)
        def wrapper(*args, **kwargs) -> Any:
            """Execute the wrapped function and measure its elapsed wall-clock time.

            Returns:
                Any: The return value of the original function.
            """
            start = time.perf_counter()
            result = func(*args, **kwargs)
            end = time.perf_counter()
            elapsed = timedelta(seconds=end - start).seconds
            hours, remainder = divmod(elapsed, 3600)
            minutes, seconds = divmod(remainder, 60)
            if verbose:
                print("==" * 50)
                print(
                    f"|| {operation_name}: took {hours:02d} hours, {minutes:02d} minutes, {seconds:02d} seconds"
                )
                print("==" * 50)
            return result

        return wrapper

    return decorator


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
    "save_parameters",
    "save_properties",
    "snapshot",
    "timer",
    "notify",
]
