"""Native selection cancellation, containment, and failure contracts."""
import json
from subprocess import TimeoutExpired

import pytest
from fastapi.testclient import TestClient

from eruption_forecast_app.main import create_app
from eruption_forecast_app.routes import api
from eruption_forecast_app.services import directory_picker as picker
from eruption_forecast_app.settings import AppSettings


class FakeDialog:
    def __init__(self, command, **kwargs):
        self.args = command
        self.returncode = 0
        self.stdout = '{"path":null}'
        self.terminated = False

    def communicate(self, timeout=None):
        return self.stdout, ""

    def poll(self):
        return None if not self.terminated else self.returncode

    def terminate(self):
        self.terminated = True

    def kill(self):
        self.terminated = True


def test_native_selection_and_cancel(monkeypatch, tmp_path):
    monkeypatch.setattr(picker.shutil, "which", lambda name: "pwsh.exe")
    calls = []
    def execute(command, **kwargs):
        calls.append((command, kwargs))
        process = FakeDialog(command)
        process.stdout = json.dumps({"path": str(tmp_path)})
        return process
    monkeypatch.setattr(picker.subprocess, "Popen", execute)
    assert picker.pick_directory(str(tmp_path)) == str(tmp_path.resolve())
    assert calls[0][1]["env"]["ERUPTION_PICKER_INITIAL"] == str(tmp_path.resolve())
    assert str(tmp_path) not in " ".join(calls[0][0])
    monkeypatch.setattr(picker.subprocess, "Popen", FakeDialog)
    assert picker.pick_directory() is None
    assert picker._DIALOG_PROCESS is None


def test_busy_and_invalid_selection_release_lock(monkeypatch):
    monkeypatch.setattr(picker.shutil, "which", lambda name: "pwsh.exe")
    picker._DIALOG_LOCK.acquire()
    try:
        with pytest.raises(picker.DirectoryPickerBusy):
            picker.pick_directory()
    finally:
        picker._DIALOG_LOCK.release()
    def invalid(command, **kwargs):
        process = FakeDialog(command)
        process.stdout = '{"path":[]}'
        return process
    monkeypatch.setattr(picker.subprocess, "Popen", invalid)
    with pytest.raises(ValueError, match="invalid path"):
        picker.pick_directory()
    assert not picker._DIALOG_LOCK.locked()


def test_cancel_terminates_only_owned_dialog(monkeypatch):
    process = FakeDialog([])
    monkeypatch.setattr(picker, "_DIALOG_PROCESS", process)
    picker.cancel_directory_picker()
    assert process.terminated
    picker.cancel_directory_picker()


def test_timeout_kills_dialog_and_allows_retry(monkeypatch):
    monkeypatch.setattr(picker.shutil, "which", lambda name: "pwsh.exe")
    process = FakeDialog([])
    def timeout(timeout=None):
        if not process.terminated:
            raise TimeoutExpired([], 180)
        return "", ""
    process.communicate = timeout
    monkeypatch.setattr(picker.subprocess, "Popen", lambda *a, **kw: process)
    with pytest.raises(ValueError, match="timed out"):
        picker.pick_directory()
    assert process.terminated
    assert not picker._DIALOG_LOCK.locked()
    assert picker._DIALOG_PROCESS is None
    monkeypatch.setattr(picker.subprocess, "Popen", FakeDialog)
    assert picker.pick_directory() is None


def test_picker_endpoint_cancel_and_unavailable(monkeypatch, tmp_path):
    with TestClient(create_app(AppSettings(state=tmp_path / "state"))) as client:
        monkeypatch.setattr(api, "pick_directory", lambda initial: str(tmp_path))
        response = client.post("/api/v1/paths/pick-directory", json={"initial_path": None})
        assert response.status_code == 200
        assert response.json()["path"] == str(tmp_path)
        monkeypatch.setattr(api, "pick_directory", lambda initial: None)
        assert client.post("/api/v1/paths/pick-directory", json={}).json() == {"path": None}
        def unavailable(initial):
            raise picker.DirectoryPickerUnavailable("Enter an absolute path instead.")
        monkeypatch.setattr(api, "pick_directory", unavailable)
        assert client.post("/api/v1/paths/pick-directory", json={}).status_code == 503
