"""Native Windows folder selection without exposing shell interpolation."""

import os
import json
import base64
import shutil
import subprocess
from pathlib import Path
from threading import Lock, Event


_DIALOG_LOCK = Lock()
_PROCESS_LOCK = Lock()
_DIALOG_CANCELLED = Event()
_DIALOG_PROCESS: subprocess.Popen[str] | None = None
_DIALOG_SCRIPT = r"""
$ErrorActionPreference = 'Stop'
[Console]::OutputEncoding = [System.Text.UTF8Encoding]::new($false)
Add-Type -AssemblyName System.Windows.Forms
[System.Windows.Forms.Application]::EnableVisualStyles()
$dialog = [System.Windows.Forms.FolderBrowserDialog]::new()
$owner = [System.Windows.Forms.Form]::new()
try {
    $dialog.Description = 'Select your SDS archive directory'
    $dialog.ShowNewFolderButton = $false
    if ($dialog.PSObject.Properties['UseDescriptionForTitle']) {
        $dialog.UseDescriptionForTitle = $true
    }
    if ($env:ERUPTION_PICKER_INITIAL) {
        $dialog.SelectedPath = $env:ERUPTION_PICKER_INITIAL
    }
    $owner.ShowInTaskbar = $false
    $owner.Text = "SDS folder selection"
    $owner.StartPosition = [System.Windows.Forms.FormStartPosition]::CenterScreen
    $owner.Size = [System.Drawing.Size]::new(1, 1)
    $owner.TopMost = $true
    $owner.Show()
    $owner.Activate()
    $selected = $null
    if ($dialog.ShowDialog($owner) -eq [System.Windows.Forms.DialogResult]::OK) {
        $selected = $dialog.SelectedPath
    }
    @{ path = $selected } | ConvertTo-Json -Compress
} finally {
    $dialog.Dispose()
    $owner.Dispose()
}
"""


class DirectoryPickerUnavailable(RuntimeError):
    """The native picker cannot run in this environment."""


class DirectoryPickerBusy(RuntimeError):
    """Another request already owns the native folder dialog."""


def pick_directory(initial_path: str | None = None) -> str | None:
    """Show a Windows folder dialog and return the selected absolute path.

    Args:
        initial_path: Existing absolute directory to select initially.

    Returns:
        The selected directory, or None if the user cancels.

    Raises:
        DirectoryPickerUnavailable: Windows desktop selection is unavailable.
        DirectoryPickerBusy: A folder dialog is already open.
        ValueError: Selection failed, timed out, or returned an invalid directory.
    """
    if os.name != "nt":
        raise DirectoryPickerUnavailable(
            "Folder selection is available on Windows. Enter an absolute path instead."
        )
    shell = shutil.which("pwsh.exe") or shutil.which("powershell.exe")
    if shell is None:
        raise DirectoryPickerUnavailable(
            "Windows PowerShell is unavailable. Enter an absolute path instead."
        )
    if not _DIALOG_LOCK.acquire(blocking=False):
        raise DirectoryPickerBusy(
            "A folder dialog is already open. Complete or cancel it first."
        )
    try:
        initial = Path(initial_path).expanduser() if initial_path else None
        environment = {
            **os.environ,
            "ERUPTION_PICKER_INITIAL": str(initial.resolve())
            if initial and initial.is_absolute() and initial.is_dir()
            else "",
        }
        encoded = base64.b64encode(_DIALOG_SCRIPT.encode("utf-16-le")).decode("ascii")
        try:
            global _DIALOG_PROCESS
            with _PROCESS_LOCK:
                _DIALOG_CANCELLED.clear()
                process = subprocess.Popen(
                    [shell, "-NoProfile", "-STA", "-EncodedCommand", encoded],
                    stdout=subprocess.PIPE, stderr=subprocess.PIPE,
                    encoding="utf-8", errors="replace", env=environment,
                    creationflags=subprocess.CREATE_NO_WINDOW,
                )
                _DIALOG_PROCESS = process
            try:
                stdout, _ = process.communicate(timeout=180)
            except subprocess.TimeoutExpired:
                process.kill()
                process.communicate()
                raise
            finally:
                with _PROCESS_LOCK:
                    _DIALOG_PROCESS = None
            result = subprocess.CompletedProcess(process.args, process.returncode, stdout)
        except subprocess.TimeoutExpired as exc:
            raise ValueError(
                "Folder selection timed out. Try Browse again or enter an absolute path."
            ) from exc
        except OSError as exc:
            raise DirectoryPickerUnavailable(
                "The folder dialog could not open. Enter an absolute path instead."
            ) from exc
        if _DIALOG_CANCELLED.is_set():
            return None
        if result.returncode:
            raise DirectoryPickerUnavailable(
                "The folder dialog could not open on the backend computer. Enter an absolute path instead."
            )
        try:
            selected = json.loads(result.stdout.strip())["path"]
        except (ValueError, KeyError, TypeError) as exc:
            raise ValueError(
                "The folder dialog returned no usable path. Try again or enter an absolute path."
            ) from exc
        if selected is None:
            return None
        if not isinstance(selected, str):
            raise ValueError("The folder dialog returned an invalid path.")
        directory = Path(selected).expanduser()
        if not directory.is_absolute() or not directory.is_dir():
            raise ValueError("Select an existing directory using its absolute path.")
        return str(directory.resolve())
    finally:
        _DIALOG_LOCK.release()


def cancel_directory_picker() -> None:
    """Terminate only the native picker process owned by this backend.

    Cancellation releases the waiting request through its normal cleanup path.
    """
    with _PROCESS_LOCK:
        if _DIALOG_PROCESS is not None and _DIALOG_PROCESS.poll() is None:
            _DIALOG_CANCELLED.set()
            _DIALOG_PROCESS.terminate()
