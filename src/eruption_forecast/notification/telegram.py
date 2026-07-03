"""Telegram Bot API client for sending messages, files, and media albums.

The module exposes :class:`TelegramNotification`, a lightweight wrapper
around the Telegram Bot API's ``sendMessage``, ``sendDocument``,
``sendPhoto``, and ``sendMediaGroup`` endpoints. Credentials are read from
constructor arguments or the ``TELEGRAM_BOT_TOKEN`` / ``TELEGRAM_CHAT_ID``
environment variables (``.env`` supported via ``python-dotenv``).

All public methods return ``self`` for fluent chaining, and every network
failure is logged rather than raised — the client is a best-effort
notifier that never blocks the caller.
"""

import os
import json
import socket
import contextlib
from typing import Self, Literal
from pathlib import Path

import niquests
from dotenv import load_dotenv
from loguru import logger
from niquests.models import Response


TELEGRAM_API_URL = "https://api.telegram.org"
PHOTO_EXTENSIONS = frozenset({".jpg", ".jpeg", ".png", ".webp", ".gif"})


class TelegramNotification:
    """Best-effort Telegram Bot API client with fluent chaining.

    The client authenticates via a bot token and target chat, both of which
    can be supplied explicitly or picked up from the ``TELEGRAM_BOT_TOKEN``
    and ``TELEGRAM_CHAT_ID`` environment variables. Every send method
    returns ``self`` so calls can be chained; failures are logged and
    swallowed so a dead network never breaks the caller.

    Attributes:
        token (str | None): Bot token used to authenticate against the API.
        chat_id (str | int | None): Target chat identifier.
        hostname (str): Hostname of the machine sending the notification.
        response (Response | None): Response from the most recent
            successful call, or ``None`` before the first send.
        url (str): Base URL including the ``/bot{token}`` prefix.
        verbose (bool): When ``True``, successful sends emit an INFO log.
    """

    def __init__(
        self,
        token: str | None = None,
        chat_id: str | int | None = None,
        verbose: bool = False,
    ):
        """Initialize the client and resolve credentials.

        Args:
            token (str | None): Bot token. Falls back to the
                ``TELEGRAM_BOT_TOKEN`` environment variable when ``None``.
                Defaults to ``None``.
            chat_id (str | int | None): Target chat. Falls back to the
                ``TELEGRAM_CHAT_ID`` environment variable when ``None``.
                Defaults to ``None``.
            verbose (bool): When ``True``, successful sends emit an INFO
                log. Defaults to ``False``.
        """
        load_dotenv(override=True)

        self.token = token or os.environ.get("TELEGRAM_BOT_TOKEN", None)
        self.chat_id = chat_id or os.environ.get("TELEGRAM_CHAT_ID", None)
        self.hostname = socket.gethostname()
        self.response: Response | None = None
        self.url = f"{TELEGRAM_API_URL}/bot{self.token}"
        self.verbose = verbose

        if not self.token or not self.chat_id:
            missing = []
            if not self.token:
                missing.append("token (TELEGRAM_BOT_TOKEN)")
            if not self.chat_id:
                missing.append("chat_id (TELEGRAM_CHAT_ID)")
            logger.warning(
                f"Telegram credentials missing: {', '.join(missing)}. "
                f"Notifications will be skipped."
            )

    @property
    def _has_credentials(self) -> bool:
        """Return ``True`` when both token and chat_id are populated."""
        return bool(self.token) and bool(self.chat_id)

    def send_message(
        self,
        message: str,
        timeout: float = 3.0,
    ) -> Self:
        """Send a text message via ``sendMessage``.

        The message is always parsed as MarkdownV2; callers are responsible
        for escaping any reserved characters.

        Args:
            message (str): MarkdownV2-formatted message body.
            timeout (float): HTTP timeout in seconds. Defaults to ``3.0``.

        Returns:
            Self: The client instance for fluent chaining.
        """
        if not self._has_credentials:
            return self

        url = f"{self.url}/sendMessage"

        try:
            response: Response = niquests.post(
                url=url,
                data={
                    "chat_id": self.chat_id,
                    "text": message,
                    "parse_mode": "MarkdownV2",
                },
                timeout=timeout,
            )

            if response.ok:
                self.response = response
                if self.verbose:
                    logger.info("Message sent to telegram.")
            else:
                logger.warning(f"Failed to send message to telegram. {response.text}")
                logger.warning(f"Message: {message}")
            return self

        except Exception as e:
            logger.warning(f"Telegram notification failed: {e}")
            return self

    def send_document(self, file: str, timeout: float = 30.0, **kwargs) -> Self:
        """Send a single file as a document via ``sendDocument``.

        Args:
            file (str): Path to the local file to upload.
            timeout (float): HTTP timeout in seconds. Defaults to ``30.0``.
            **kwargs: Additional Telegram parameters (e.g. ``caption``,
                ``parse_mode``, ``disable_notification``). When ``caption``
                is provided without ``parse_mode``, MarkdownV2 is applied
                automatically; when omitted, the filename is used as a
                plain-text caption.

        Returns:
            Self: The client instance for fluent chaining.
        """
        if not self._has_credentials:
            return self

        url = f"{self.url}/sendDocument"
        self._send_file(url=url, file=file, field="document", timeout=timeout, **kwargs)
        return self

    def send_photo(self, file: str, timeout: float = 30.0, **kwargs) -> Self:
        """Send a single image as a photo via ``sendPhoto``.

        Files whose suffix is not in ``PHOTO_EXTENSIONS`` transparently
        fall back to :meth:`send_document`, with a warning logged.

        Args:
            file (str): Path to the local image file.
            timeout (float): HTTP timeout in seconds. Defaults to ``30.0``.
            **kwargs: Additional Telegram parameters — see
                :meth:`send_document` for caption behaviour.

        Returns:
            Self: The client instance for fluent chaining.
        """
        if not self._has_credentials:
            return self

        if Path(file).suffix.lower() not in PHOTO_EXTENSIONS:
            logger.warning(
                f"Telegram photo: {Path(file).name} is not a supported photo "
                f"format {sorted(PHOTO_EXTENSIONS)}; falling back to send_document()."
            )
            self.send_document(file=file, timeout=timeout, **kwargs)
            return self

        url = f"{self.url}/sendPhoto"
        self._send_file(url=url, file=file, field="photo", timeout=timeout, **kwargs)
        return self

    def send_media_group(
        self,
        files: str | list[str],
        kind: Literal["photo", "document"] = "photo",
        caption: str | None = None,
        timeout: float = 30.0,
        disable_notification: bool = False,
    ) -> Self:
        """Send files as one or more media-group albums.

        Telegram accepts 2-10 items per album, so larger inputs are chunked
        automatically into consecutive albums. When ``kind='photo'`` but
        any input file is not a supported photo format, the whole batch is
        downgraded to ``kind='document'`` (with a single warning) so
        Telegram's album homogeneity constraint is not violated. If a
        chunk collapses to a single surviving file after existence
        filtering, it is transparently rerouted through :meth:`send_photo`
        or :meth:`send_document`, since ``sendMediaGroup`` requires at
        least two items.

        Args:
            files (str | list[str]): One or more local file paths.
            kind (Literal["photo", "document"]): Media type applied to
                every item in the album. Defaults to ``"photo"``.
            caption (str | None): Optional MarkdownV2 caption attached to
                the first item of the first album only (Telegram
                convention). Defaults to ``None``.
            timeout (float): HTTP timeout in seconds. Defaults to ``30.0``.
            disable_notification (bool): When ``True``, delivery is
                silent. Defaults to ``False``.

        Returns:
            Self: The client instance for fluent chaining.
        """
        if not self._has_credentials:
            return self

        url = f"{self.url}/sendMediaGroup"
        files = [files] if isinstance(files, str) else files

        if kind == "photo":
            non_photo = [
                f for f in files if Path(f).suffix.lower() not in PHOTO_EXTENSIONS
            ]

            if non_photo:
                logger.warning(
                    f"Telegram media_group: {len(non_photo)} file(s) not supported "
                    f"photo format ({[Path(p).name for p in non_photo]}); "
                    f"falling back to kind='document' for the whole batch."
                )
                kind = "document"

        for chunk_start in range(0, len(files), 10):
            chunk = files[chunk_start : chunk_start + 10]
            self._send_media_chunk(
                url=url,
                files=chunk,
                kind=kind,
                caption=caption if chunk_start == 0 else None,
                timeout=timeout,
                disable_notification=disable_notification,
            )

        return self

    def _send_file(
        self,
        url: str,
        file: str,
        field: str,
        timeout: float,
        **kwargs,
    ):
        """Upload a single file via multipart form-data.

        Shared implementation for :meth:`send_document` and the photo
        path of :meth:`send_photo`. Missing files are logged and skipped
        without raising. If ``caption`` is absent from ``kwargs`` the
        filename is used as a plain-text caption; if present, MarkdownV2
        is applied automatically when no ``parse_mode`` is supplied.

        Args:
            url (str): Fully-qualified Telegram endpoint URL.
            file (str): Path to the local file to upload.
            field (str): Multipart field name (``"document"`` or
                ``"photo"``).
            timeout (float): HTTP timeout in seconds.
            **kwargs: Extra Telegram parameters forwarded into the
                request body (e.g. ``caption``, ``parse_mode``,
                ``disable_notification``).
        """
        file_path = Path(file)
        if not file_path.is_file():
            logger.warning(
                f"Telegram {field} notification failed: file not found: {file_path}"
            )
            return

        data: dict = {"chat_id": self.chat_id, **kwargs}
        if "caption" not in data:
            data["caption"] = file_path.name
        elif "parse_mode" not in data:
            data["parse_mode"] = "MarkdownV2"

        try:
            with file_path.open("rb") as fh:
                response: Response = niquests.post(
                    url=url,
                    data=data,
                    files={field: (file_path.name, fh)},
                    timeout=timeout,
                )

            if response.ok:
                self.response = response
                if self.verbose:
                    logger.info(f"{field.title()} sent to telegram: {file_path.name}")
            else:
                logger.warning(f"Failed to send {field} to telegram. {response.text}")
                logger.warning(f"File: {file_path}")

        except Exception as e:
            logger.warning(f"Telegram {field} notification failed: {e}")

    def _send_media_chunk(
        self,
        url: str,
        files: list[str],
        kind: str,
        caption: str | None,
        timeout: float,
        disable_notification: bool,
    ):
        """Upload a single ``sendMediaGroup`` album of up to 10 items.

        Missing files are logged and skipped. If only one file remains
        after filtering, the call is transparently rerouted to
        :meth:`send_photo` or :meth:`send_document`, since Telegram
        rejects single-item media groups. When ``caption`` is supplied
        it is attached to the first item with MarkdownV2; every other
        item gets its filename as a plain-text caption.

        Args:
            url (str): ``sendMediaGroup`` endpoint URL.
            files (list[str]): Local file paths forming this album.
            kind (str): ``"photo"`` or ``"document"`` — applied uniformly
                to every item in the album.
            caption (str | None): MarkdownV2 caption for the first item.
            timeout (float): HTTP timeout in seconds.
            disable_notification (bool): When ``True``, delivery is
                silent.
        """
        paths: list[Path] = []
        for file in files:
            path = Path(file)
            if not path.is_file():
                logger.warning(f"Telegram media_group: file not found: {path}")
                continue
            paths.append(path)

        if not paths:
            return

        if len(paths) == 1:
            method = self.send_photo if kind == "photo" else self.send_document
            extra: dict = {"disable_notification": disable_notification}
            if caption is not None:
                extra["caption"] = caption
            method(str(paths[0]), timeout=timeout, **extra)
            return

        media: list[dict] = []
        multipart_files: dict = {}

        try:
            with contextlib.ExitStack() as stack:
                for i, path in enumerate(paths):
                    attach_name = f"file{i}"
                    item: dict = {
                        "type": kind,
                        "media": f"attach://{attach_name}",
                    }
                    if i == 0 and caption is not None:
                        item["caption"] = caption
                        item["parse_mode"] = "MarkdownV2"
                    else:
                        item["caption"] = path.name
                    media.append(item)
                    fh = stack.enter_context(path.open("rb"))
                    multipart_files[attach_name] = (path.name, fh)

                response: Response = niquests.post(
                    url=url,
                    data={
                        "chat_id": self.chat_id,
                        "media": json.dumps(media),
                        "disable_notification": disable_notification,
                    },
                    files=multipart_files,
                    timeout=timeout,
                )

            if response.ok:
                self.response = response
                if self.verbose:
                    logger.info(
                        f"Media group sent to telegram: {len(paths)} {kind}(s)."
                    )
            else:
                logger.warning(
                    f"Failed to send media group to telegram. {response.text}"
                )

        except Exception as e:
            logger.warning(f"Telegram media_group notification failed: {e}")
