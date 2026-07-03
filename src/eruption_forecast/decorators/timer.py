"""Timing decorator for measuring and reporting function elapsed time.

The module exposes :func:`timer`, a decorator factory that logs the wall-clock
elapsed time of the wrapped function through :mod:`loguru` and, optionally,
forwards the same message to Telegram via
:class:`~eruption_forecast.notification.telegram.TelegramNotification`.
"""

import time
import functools
from typing import Any, Literal
from datetime import timedelta
from collections.abc import Callable

from loguru import logger

from eruption_forecast.notification.telegram import TelegramNotification


def timer(
    name: str | None = None,
    send_to: Literal["telegram"] | None = None,
):
    """Decorator factory that logs the elapsed time of the wrapped function.

    The returned decorator preserves the wrapped function's signature and
    metadata via :func:`functools.wraps`. When ``send_to='telegram'`` the same
    message is dispatched through
    :class:`~eruption_forecast.notification.telegram.TelegramNotification`;
    send failures are swallowed by the client and never propagate to the
    caller.

    Args:
        name (str | None): Task label shown in the log message. Falls back to
            the wrapped function's ``__name__`` when ``None``. Defaults to
            ``None``.
        send_to (Literal["telegram"] | None): Optional notification backend.
            Only ``"telegram"`` is supported today. Defaults to ``None``.

    Returns:
        Callable: Decorator that wraps the target function while preserving
            its signature.
    """

    def decorator(func: Callable) -> Callable:
        function_name = name if name else func.__name__  # ty:ignore[unresolved-attribute]

        @functools.wraps(func)
        def wrapper(*args, **kwargs) -> Any:
            start_time = time.perf_counter()
            result = func(*args, **kwargs)
            end_time = time.perf_counter()
            elapsed_time = timedelta(seconds=end_time - start_time).seconds

            hours, remainder = divmod(elapsed_time, 3600)
            minutes, seconds = divmod(remainder, 60)
            message = f"{function_name}: Took {hours:02d} hours, {minutes:02d} minutes, {seconds:02d} seconds"

            logger.info(message)

            if send_to == "telegram":
                TelegramNotification().send_message(message=message)

            return result

        return wrapper

    return decorator
