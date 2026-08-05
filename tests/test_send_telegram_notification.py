"""Unit tests for :class:`TelegramNotification`.

The legacy top-level ``send_telegram_notification`` helper has been replaced
by the fluent :class:`TelegramNotification` client
(``src/eruption_forecast/notification/telegram.py``). Every test patches
``niquests.post`` at the module level so no real HTTP request is issued.

Covers:
- constructor credential fallback to env vars
- fluent chain returns ``self``
- silent no-op when credentials are missing
- ``send_message`` posts a MarkdownV2 payload to ``sendMessage``
- ``send_document`` uploads a file via ``sendDocument``
- network failures are swallowed (never re-raised)
"""

from types import SimpleNamespace

import pytest

from eruption_forecast.notification import telegram as telegram_module
from eruption_forecast.notification.telegram import TelegramNotification


def _ok_response(text: str = "") -> SimpleNamespace:
    """Return a stub matching the ``.ok`` / ``.text`` surface ``send_*`` reads."""
    return SimpleNamespace(ok=True, text=text)


@pytest.fixture(autouse=True)
def _clear_env(monkeypatch) -> None:
    """Strip Telegram env vars so tests never accidentally pick up real creds.

    Also stubs ``load_dotenv`` so a project-level ``.env`` cannot repopulate
    the env mid-test.
    """
    monkeypatch.delenv("TELEGRAM_BOT_TOKEN", raising=False)
    monkeypatch.delenv("TELEGRAM_CHAT_ID", raising=False)
    monkeypatch.setattr(telegram_module, "load_dotenv", lambda **kwargs: None)


def test_constructor_reads_explicit_credentials():
    """Explicit ``token`` and ``chat_id`` are stored verbatim."""
    tn = TelegramNotification(token="TOKEN", chat_id="CHAT")
    assert tn.token == "TOKEN"
    assert tn.chat_id == "CHAT"
    assert tn._has_credentials is True


def test_constructor_reads_env_credentials(monkeypatch):
    """Missing kwargs fall back to ``TELEGRAM_BOT_TOKEN`` / ``TELEGRAM_CHAT_ID``."""
    monkeypatch.setenv("TELEGRAM_BOT_TOKEN", "ENV_TOKEN")
    monkeypatch.setenv("TELEGRAM_CHAT_ID", "ENV_CHAT")

    tn = TelegramNotification()
    assert tn.token == "ENV_TOKEN"
    assert tn.chat_id == "ENV_CHAT"


def test_missing_credentials_is_silent_noop(monkeypatch):
    """Without credentials, ``send_message`` returns ``self`` without posting."""
    calls: list = []

    def fake_post(**kwargs):
        calls.append(kwargs)
        return _ok_response()

    monkeypatch.setattr(telegram_module.niquests, "post", fake_post)

    tn = TelegramNotification()  # no credentials
    result = tn.send_message("hello")

    assert result is tn
    assert calls == []


def test_send_message_posts_markdown_v2(monkeypatch):
    """``send_message`` targets ``sendMessage`` with a MarkdownV2 payload."""
    captured: list[dict] = []

    def fake_post(url, data, timeout):
        captured.append({"url": url, "data": data, "timeout": timeout})
        return _ok_response()

    monkeypatch.setattr(telegram_module.niquests, "post", fake_post)

    tn = TelegramNotification(token="TOKEN", chat_id="CHAT")
    result = tn.send_message("Hello *team*!", timeout=5.0)

    assert result is tn  # fluent chain
    assert len(captured) == 1
    call = captured[0]
    assert call["url"].endswith("/botTOKEN/sendMessage")
    assert call["timeout"] == 5.0
    assert call["data"]["chat_id"] == "CHAT"
    assert call["data"]["text"] == "Hello *team*!"
    assert call["data"]["parse_mode"] == "MarkdownV2"


def test_send_message_network_failure_is_swallowed(monkeypatch):
    """A raising ``niquests.post`` is logged and never re-raised."""

    def raising_post(**kwargs):
        raise ConnectionError("network down")

    monkeypatch.setattr(telegram_module.niquests, "post", raising_post)

    tn = TelegramNotification(token="TOKEN", chat_id="CHAT")
    # Must not raise.
    assert tn.send_message("hi") is tn


def test_send_document_uploads_file(monkeypatch, tmp_path):
    """``send_document`` posts multipart form-data to ``sendDocument``."""
    file_path = tmp_path / "artifact.csv"
    file_path.write_text("col_a,col_b\n1,2\n")

    captured: list[dict] = []

    def fake_post(url, data, files, timeout):
        captured.append(
            {"url": url, "data": dict(data), "files": list(files.keys()), "timeout": timeout}
        )
        return _ok_response()

    monkeypatch.setattr(telegram_module.niquests, "post", fake_post)

    tn = TelegramNotification(token="TOKEN", chat_id="CHAT")
    result = tn.send_document(str(file_path), timeout=10.0)

    assert result is tn
    assert len(captured) == 1
    call = captured[0]
    assert call["url"].endswith("/botTOKEN/sendDocument")
    assert call["timeout"] == 10.0
    assert call["data"]["chat_id"] == "CHAT"
    assert call["files"] == ["document"]
    # Default caption is the filename when caller does not provide one.
    assert call["data"]["caption"] == file_path.name


def test_send_document_missing_file_is_swallowed(monkeypatch, tmp_path):
    """A non-existent file path is logged and skipped, not raised."""
    posts: list = []
    monkeypatch.setattr(
        telegram_module.niquests,
        "post",
        lambda **kwargs: posts.append(kwargs) or _ok_response(),
    )

    tn = TelegramNotification(token="TOKEN", chat_id="CHAT")
    ghost = tmp_path / "does_not_exist.csv"
    # Must not raise.
    assert tn.send_document(str(ghost)) is tn
    assert posts == []


def test_fluent_chain_across_calls(monkeypatch, tmp_path):
    """Both ``send_message`` and ``send_document`` return ``self`` so calls chain."""
    file_path = tmp_path / "x.txt"
    file_path.write_text("hi")

    monkeypatch.setattr(
        telegram_module.niquests, "post", lambda **kwargs: _ok_response()
    )

    tn = TelegramNotification(token="TOKEN", chat_id="CHAT")
    chained = tn.send_message("first").send_document(str(file_path))

    assert chained is tn
