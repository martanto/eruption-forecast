import { useRef, useState } from "react";
import { FolderOpen } from "lucide-react";
import { api, ApiError } from "../../api/client";

export function DirectoryInput({
  id,
  value,
  onChange,
}: {
  id: string;
  value: string;
  onChange: (value: string) => void;
}) {
  const input = useRef<HTMLInputElement>(null);
  const latestChange = useRef(onChange);
  latestChange.current = onChange;
  const [busy, setBusy] = useState(false);
  const [error, setError] = useState("");
  const [status, setStatus] = useState("");
  const [canCancel, setCanCancel] = useState(false);
  const cancelled = useRef(false);
  const cancel = async () => {
    cancelled.current = true;
    try {
      await api("/paths/pick-directory", undefined, "DELETE");
      setCanCancel(false);
      setError("");
      setStatus("Folder selection cancelled. Your path is unchanged.");
    } catch (failure) {
      setError((failure as Error).message);
    }
  };
  const choose = async () => {
    cancelled.current = false;
    setCanCancel(true);
    setBusy(true);
    setError("");
    setStatus("Choose a folder on the backend computer.");
    try {
      const result = await api<{ path: string | null }>(
        "/paths/pick-directory",
        { initial_path: value || null },
      );
      setCanCancel(false);
      if (result.path !== null) {
        latestChange.current(result.path);
        setStatus("Folder selected. Save setup to apply it.");
      } else setStatus("Folder selection cancelled. Your path is unchanged.");
    } catch (failure) {
      if (!cancelled.current) {
        setError((failure as Error).message);
        setStatus("");
        setCanCancel(failure instanceof ApiError && failure.status === 409);
      }
    } finally {
      setBusy(false);
      input.current?.focus();
    }
  };
  return (
    <>
      <div className="directory-input">
        <input
          ref={input}
          id={id}
          value={value}
          readOnly={busy}
          placeholder="Absolute path, e.g. D:\Data\OJN"
          aria-describedby={`${id}-help${error ? ` ${id}-error` : ""}`}
          aria-invalid={!!error}
          onChange={(event) => {
            onChange(event.target.value);
            setError("");
            setStatus("");
          }}
        />
        <button
          className="button secondary"
          type="button"
          disabled={busy}
          aria-label="Choose SDS directory"
          onClick={() => void choose()}
        >
          <FolderOpen size={16} />
          {busy ? "Choosing…" : "Choose folder"}
        </button>
      </div>
      <span className="field-hint" id={`${id}-help`}>
        Windows folder selection opens on the backend computer. You can also
        enter an absolute path.
      </span>
      {canCancel && (
        <button
          type="button"
          className="text-button"
          onClick={() => void cancel()}
        >
          Cancel folder selection
        </button>
      )}
      {error && (
        <span className="field-error" id={`${id}-error`} role="alert">
          {error}
        </span>
      )}
      {status && (
        <span className="field-hint" role="status">
          {status}
        </span>
      )}
    </>
  );
}
