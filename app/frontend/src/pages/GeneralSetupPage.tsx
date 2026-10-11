import { useState } from "react";
import { Save, FolderOpen, CalendarDays, ArrowRight } from "lucide-react";
import { useNavigate } from "react-router-dom";
import { api, type Job } from "../api/client";
import { DirectoryInput } from "../components/forms/DirectoryInput";
import { ParameterForm } from "../components/forms/ParameterForm";
import { useWorkspace } from "../hooks/useWorkspace";
import { type JsonObject } from "../lib/configuration";

export default function GeneralSetupPage() {
  const { configuration, setConfiguration, catalog, notice, refreshArtifacts } =
    useWorkspace();
  const [error, setError] = useState("");
  const [busy, setBusy] = useState<
    "save" | "scan" | "discover" | "browse" | null
  >(null);
  const [browsePath, setBrowsePath] = useState("");
  const [entries, setEntries] = useState<
    { name: string; path: string; directory: boolean }[]
  >([]);
  const setup = (configuration.setup as JsonObject) ?? {};
  const defaults = catalog.defaults.setup as JsonObject;
  const navigate = useNavigate();
  const action = async (kind: "save" | "scan" | "discover" | "browse") => {
    if (
      kind === "scan" &&
      (!range.start_date ||
        !range.end_date ||
        String(range.end_date) < String(range.start_date))
    ) {
      setError(
        "Choose a start date and an end date on or after it before scanning.",
      );
      return;
    }
    setBusy(kind);
    setError("");
    try {
      if (kind === "browse") {
        setEntries(
          await api(
            `/paths/browse${browsePath ? `?path=${encodeURIComponent(browsePath)}` : ""}`,
          ),
        );
        return;
      }
      const saved = await api<JsonObject>(
        "/settings",
        { ...defaults, ...setup },
        "PUT",
      );
      setConfiguration({ ...configuration, setup: saved });
      if (kind === "save") notice("General setup saved");
      if (kind === "discover") {
        await api("/artifacts/discover", { path: saved.output_dir });
        await refreshArtifacts();
        notice("Existing artifacts indexed");
      }
      if (kind === "scan") {
        const job = await api<Job>("/jobs", {
          kind: "availability",
          configuration: { ...configuration, setup: saved },
        });
        navigate(`/jobs?selected=${job.id}`);
      }
    } catch (error) {
      setError((error as Error).message);
    } finally {
      setBusy(null);
    }
  };
  const calculation = (configuration.calculation as JsonObject) ?? {};
  const range = (calculation["constructor"] as JsonObject) ?? {};
  return (
    <div className="setup-page">
      <div className="page-heading">
        <div>
          <h1>General setup</h1>
          <p>
            Connect your seismic archive and establish the defaults for every
            experiment.
          </p>
        </div>
        <button
          className="button primary"
          disabled={busy !== null}
          onClick={() => void action("save")}
        >
          <Save size={16} />
          {busy === "save" ? "Saving�" : "Save setup"}
        </button>
      </div>
      <div className="workflow-intro">
        <div>
          <h2>Your research workspace</h2>
          <p>
            Directories are on the machine running the backend. Choose the
            archive, a writable output location, and your station identity.
          </p>
        </div>
        <div className="workflow-path">
          <span>Archive</span>
          <ArrowRight size={15} />
          <span>Tremor</span>
          <ArrowRight size={15} />
          <span>Forecast</span>
        </div>
      </div>
      {error && (
        <p className="error-message" role="alert">
          {error}
        </p>
      )}
      <div className="form-panel">
        {[
          {
            title: "Directories",
            description: "Use absolute paths on the backend computer.",
            fields: ["sds_dir", "output_dir", "tremor_dir", "root_dir"],
          },
          {
            title: "Station identity",
            description:
              "Match the network, station, location, and channel in your archive.",
            fields: [
              "network",
              "station",
              "location",
              "channel",
              "channel_type",
            ],
          },
          {
            title: "Execution defaults",
            description:
              "Set the processing capacity and default analysis window.",
            fields: ["n_jobs", "window_size", "verbose"],
          },
        ].map((group) => (
          <section
            className="setup-group"
            key={group.title}
            aria-label={group.title}
          >
            <h3>{group.title}</h3>
            <p className="muted">{group.description}</p>
            <ParameterForm
              definition="Setup"
              value={setup}
              defaults={defaults}
              onChange={(value) =>
                setConfiguration({ ...configuration, setup: value })
              }
              search={false}
              include={group.fields}
              showProvenance={false}
              fieldOptions={{
                sds_dir: {
                  control: (id, value, change) => (
                    <DirectoryInput
                      id={id}
                      value={String(value ?? "")}
                      onChange={(value) => change(value || null)}
                    />
                  ),
                },
                output_dir: {
                  placeholder: "Absolute path, e.g. D:\\Forecasts\\output",
                  hint: "Generated models, tables, and figures are saved here.",
                },
                tremor_dir: {
                  placeholder: "Absolute path, e.g. D:\\Forecasts\\tremor",
                  hint: "Optional directory for existing tremor data.",
                },
                root_dir: {
                  placeholder:
                    "Absolute path, e.g. D:\\Projects\\eruption-forecast",
                  hint: "Optional project root for resolving configuration paths.",
                },
                location: {
                  placeholder: "Leave blank for an empty location code",
                },
                n_jobs: { hint: "Number of parallel processing jobs." },
                window_size: { hint: "Default analysis window in days." },
              }}
            />
          </section>
        ))}
      </div>
      <section className="availability-section">
        <div className="section-heading">
          <div>
            <h2>Seismic completeness</h2>
            <p>Inspect the raw SDS archive before calculating tremor.</p>
          </div>
          <CalendarDays size={22} />
        </div>
        <div className="availability-row">
          <label>
            Start date
            <input
              type="date"
              value={String(range.start_date ?? "")}
              onChange={(event) =>
                setConfiguration({
                  ...configuration,
                  calculation: {
                    ...calculation,
                    constructor: { ...range, start_date: event.target.value },
                  },
                })
              }
            />
          </label>
          <label>
            End date
            <input
              type="date"
              min={String(range.start_date ?? "")}
              value={String(range.end_date ?? "")}
              onChange={(event) =>
                setConfiguration({
                  ...configuration,
                  calculation: {
                    ...calculation,
                    constructor: { ...range, end_date: event.target.value },
                  },
                })
              }
            />
          </label>
          <button
            className="button secondary"
            disabled={busy !== null}
            onClick={() => void action("scan")}
          >
            {busy === "scan" ? "Scanning�" : "Scan SDS completeness"}
          </button>
        </div>
        <p className="muted">
          The scan saves a daily completeness table and calendar figure. Missing
          days remain visible in the table.
        </p>
      </section>
      <section className="directory-section">
        <div className="section-heading">
          <h2>Existing files</h2>
          <button
            className="button secondary"
            disabled={busy !== null}
            onClick={() => void action("discover")}
          >
            <FolderOpen size={15} />
            {busy === "discover" ? "Indexing�" : "Index output directory"}
          </button>
        </div>
        <div className="availability-row">
          <label>
            Browse within configured directories
            <input
              value={browsePath}
              onChange={(event) => setBrowsePath(event.target.value)}
              placeholder="Leave blank to list configured roots"
            />
          </label>
          <button
            className="button secondary"
            disabled={busy !== null}
            onClick={() => void action("browse")}
          >
            {busy === "browse" ? "Browsing�" : "Browse"}
          </button>
        </div>
        {entries.length > 0 && (
          <ul className="directory-list">
            {entries.map((entry) => (
              <li key={entry.path}>
                <button
                  className="text-button"
                  onClick={() => {
                    if (entry.directory) setBrowsePath(entry.path);
                    else
                      navigator.clipboard
                        .writeText(entry.path)
                        .then(() => notice("Path copied"));
                  }}
                >
                  {entry.name}
                  {entry.directory ? "/" : ""}
                </button>
                <span className="path-text">{entry.path}</span>
              </li>
            ))}
          </ul>
        )}
      </section>
    </div>
  );
}
