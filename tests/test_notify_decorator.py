"""Unit tests for the ``notify`` decorator.

The current ``notify(task, message=None, to="telegram", on_success=True,
on_error=True, timeout=3.0, verbose=False)`` decorator dispatches through
``eruption_forecast.decorators.notify._dispatch`` which in turn calls
:class:`TelegramNotification`. Every test patches ``_dispatch`` so no HTTP
call is ever made — the assertions verify the wrapping contract
(return-value passthrough, exception re-raise) and that dispatch is gated
by ``on_success`` / ``on_error``.
"""

import importlib

import pytest

from eruption_forecast.decorators.notify import notify


# ``eruption_forecast.decorators.__init__`` re-exports the ``notify`` function
# under the same attribute name as the submodule, so ``import
# eruption_forecast.decorators.notify`` returns the function. Reach the actual
# module through ``sys.modules`` (populated by ``importlib.import_module``).
notify_module = importlib.import_module("eruption_forecast.decorators.notify")


@pytest.fixture()
def dispatched(monkeypatch) -> list[tuple]:
    """Capture every ``_dispatch`` call the decorator issues.

    Returns:
        list[tuple]: one entry per dispatched notification, shaped
        ``(to, message, timeout, verbose)``.
    """
    calls: list[tuple] = []

    def _fake_dispatch(to, message, timeout, verbose):
        calls.append((to, message, timeout, verbose))

    monkeypatch.setattr(notify_module, "_dispatch", _fake_dispatch)
    return calls


def test_success_dispatches_notification(dispatched: list[tuple]) -> None:
    """``on_success=True`` (default) sends a success notification and returns the value."""

    @notify(task="quick task")
    def quick_task() -> int:
        return 42

    result = quick_task()

    assert result == 42
    assert len(dispatched) == 1
    to, message, _, _ = dispatched[0]
    assert to == "telegram"
    assert "quick task" in message


def test_error_dispatches_and_reraises(dispatched: list[tuple]) -> None:
    """``on_error=True`` (default) sends the error and re-raises the original exception."""

    @notify(task="failing task")
    def failing_task() -> None:
        raise ValueError("boom")

    with pytest.raises(ValueError, match="boom"):
        failing_task()

    assert len(dispatched) == 1
    _, message, _, _ = dispatched[0]
    assert "failing task" in message


def test_on_success_false_suppresses_success(dispatched: list[tuple]) -> None:
    """``on_success=False`` skips dispatch on a successful call."""

    @notify(task="silent", on_success=False)
    def silent_task() -> bool:
        return True

    assert silent_task() is True
    assert dispatched == []


def test_on_error_false_suppresses_error_but_reraises(dispatched: list[tuple]) -> None:
    """``on_error=False`` skips dispatch on failure but still re-raises."""

    @notify(task="silent failure", on_error=False)
    def silent_failing_task() -> None:
        raise RuntimeError("nope")

    with pytest.raises(RuntimeError, match="nope"):
        silent_failing_task()

    assert dispatched == []


def test_custom_message_forwarded(dispatched: list[tuple]) -> None:
    """A ``message`` kwarg is embedded in the dispatched notification text.

    MarkdownV2 reserved characters (``-``, ``.`` etc.) are backslash-escaped
    before the body is dispatched, so we use a plain-alphanumeric substring
    that survives escaping unchanged.
    """

    @notify(task="custom", message="bodyOfTheMessage")
    def custom_task() -> None:
        return None

    custom_task()

    assert len(dispatched) == 1
    _, message, _, _ = dispatched[0]
    assert "bodyOfTheMessage" in message


def test_timeout_and_verbose_forwarded(dispatched: list[tuple]) -> None:
    """The ``timeout`` and ``verbose`` kwargs reach ``_dispatch`` unchanged."""

    @notify(task="typed", timeout=1.5, verbose=True)
    def typed_task() -> str:
        return "done"

    assert typed_task() == "done"
    assert len(dispatched) == 1
    _, _, timeout, verbose = dispatched[0]
    assert timeout == 1.5
    assert verbose is True


def test_wraps_preserves_metadata() -> None:
    """The decorator preserves the wrapped function's ``__name__`` via ``functools.wraps``."""

    @notify(task="named")
    def original_name() -> None:
        return None

    assert original_name.__name__ == "original_name"
