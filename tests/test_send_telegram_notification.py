"""Unit tests for direct Telegram notification function."""

from pathlib import Path
import importlib

import pytest

from eruption_forecast.decorators.notify import send_telegram_notification


notify_module = importlib.import_module("eruption_forecast.decorators.notify")


def test_send_telegram_notification_sends_message_and_files(monkeypatch):
    """Send one text message and all requested file attachments."""
    monkeypatch.delenv("TELEGRAM_BOT_TOKEN", raising=False)
    monkeypatch.delenv("TELEGRAM_CHAT_ID", raising=False)
    monkeypatch.setattr(notify_module, "load_dotenv", lambda: None)

    calls: list[tuple] = []

    def fake_send_message(token, chat_id, text, timeout):
        calls.append(("msg", token, chat_id, text, timeout))

    def fake_send_file(
        token,
        chat_id,
        path,
        caption="",
        timeout=30.0,
        send_as_document=False,
    ):
        calls.append(
            ("file", token, chat_id, Path(path), caption, timeout, send_as_document)
        )

    monkeypatch.setattr(notify_module, "_send_telegram_message", fake_send_message)
    monkeypatch.setattr(notify_module, "_send_telegram_file", fake_send_file)

    send_telegram_notification(
        message="Hello *team*!",
        bot_token="TOKEN",
        chat_id="123",
        timeout=7.5,
        files=["output/plot.png", "output/metrics.csv"],
        file_caption="Run artifacts",
    )

    assert calls[0] == ("msg", "TOKEN", "123", r"Hello \*team\*\!", 7.5)
    assert calls[1] == (
        "file",
        "TOKEN",
        "123",
        Path("output/plot.png"),
        "Run artifacts",
        7.5,
        False,
    )
    assert calls[2] == (
        "file",
        "TOKEN",
        "123",
        Path("output/metrics.csv"),
        "Run artifacts",
        7.5,
        False,
    )


def test_send_telegram_notification_disable_markdown_escape(monkeypatch):
    """Allow sending raw markdown by disabling escaping."""
    captured: list[str] = []

    def fake_send_message(token, chat_id, text, timeout):
        captured.append(text)

    monkeypatch.setattr(notify_module, "_send_telegram_message", fake_send_message)

    send_telegram_notification(
        message="*bold*",
        bot_token="TOKEN",
        chat_id="123",
        escape_markdown=False,
    )

    assert captured == ["*bold*"]


def test_send_telegram_notification_reads_credentials_from_env(monkeypatch):
    """Resolve bot token and chat id from environment variables."""
    monkeypatch.setenv("TELEGRAM_BOT_TOKEN", "ENV_TOKEN")
    monkeypatch.setenv("TELEGRAM_CHAT_ID", "ENV_CHAT")

    captured: list[tuple[str, str | int]] = []

    def fake_send_message(token, chat_id, text, timeout):
        captured.append((token, chat_id))

    monkeypatch.setattr(notify_module, "_send_telegram_message", fake_send_message)

    send_telegram_notification(message="hello")
    assert captured == [("ENV_TOKEN", "ENV_CHAT")]


def test_send_telegram_notification_validates_inputs(monkeypatch):
    """Raise explicit ValueError for missing/invalid inputs."""
    monkeypatch.delenv("TELEGRAM_BOT_TOKEN", raising=False)
    monkeypatch.delenv("TELEGRAM_CHAT_ID", raising=False)
    monkeypatch.setattr(notify_module, "load_dotenv", lambda: None)

    with pytest.raises(ValueError, match="message"):
        send_telegram_notification(message="", bot_token="TOKEN", chat_id="123")

    with pytest.raises(ValueError, match="bot_token"):
        send_telegram_notification(message="ok", bot_token="", chat_id="123")

    with pytest.raises(ValueError, match="chat_id"):
        send_telegram_notification(message="ok", bot_token="TOKEN", chat_id="")

    with pytest.raises(ValueError, match="whitespace"):
        send_telegram_notification(message="ok", bot_token="T O K E N", chat_id="123")

    with pytest.raises(ValueError, match="whitespace"):
        send_telegram_notification(message="ok", bot_token="TOKEN", chat_id="12 3")
