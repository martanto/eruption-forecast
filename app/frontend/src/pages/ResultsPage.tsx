import { useEffect, useState } from "react";
import { useSearchParams } from "react-router-dom";
import { FolderSearch, Search, RefreshCw } from "lucide-react";
import { api, type Artifact, type Job, type TableData } from "../api/client";
import { ArtifactViewer } from "../components/artifacts/ArtifactViewer";
import { TimeSeries } from "../components/charts/TimeSeries";
import { useWorkspace } from "../hooks/useWorkspace";
import { filename, type JsonObject } from "../lib/configuration";

export default function ResultsPage() {
  const { artifacts, refreshArtifacts, configuration, notice } = useWorkspace();
  const [params] = useSearchParams();
  const [query, setQuery] = useState("");
  const [extension, setExtension] = useState("all");
  const [stage, setStage] = useState("all");
  const [jobs, setJobs] = useState<Job[]>([]);
  const [jobId, setJobId] = useState(params.get("job") ?? "");
  const [selected, setSelected] = useState<Artifact | null>(null);
  const [error, setError] = useState("");
  const [compare, setCompare] = useState<string[]>([]);
  const [comparison, setComparison] = useState<TableData | null>(null);
  useEffect(() => {
    void refreshArtifacts();
    api<Job[]>("/jobs")
      .then(setJobs)
      .catch((error) => setError(error.message));
  }, []);
  const filtered = artifacts.filter(
    (item) =>
      (!jobId || item.job_id === jobId) &&
      (extension === "all" || item.path.endsWith(`.${extension}`)) &&
      (stage === "all" ||
        item.path.replaceAll("\\", "/").includes(`/${stage}/`)) &&
      `${item.path} ${item.job_id ?? ""}`
        .toLowerCase()
        .includes(query.toLowerCase()),
  );
  const index = async () => {
    try {
      await api("/artifacts/discover", {
        path: (configuration.setup as JsonObject)?.output_dir ?? "output",
      });
      await refreshArtifacts();
      notice("Existing outputs indexed");
      setError("");
    } catch (error) {
      setError((error as Error).message);
    }
  };
  const compareResults = async () => {
    try {
      const frames = await Promise.all(
        compare.map((id) =>
          api<TableData>(`/artifacts/${id}/preview?chart=true&limit=2000`),
        ),
      );
      const dates = new Map<string, Record<string, string | number | null>>();
      const columns = ["datetime"];
      frames.forEach((frame, index) => {
        const source = artifacts.find((item) => item.id === compare[index])!;
        const time =
          frame.columns.find((column) => /date|time|unnamed/i.test(column)) ??
          frame.columns[0];
        const probability =
          frame.columns.find((column) =>
            /consensus.*probability/.test(column),
          ) ?? frame.columns.find((column) => /probability/.test(column));
        if (!probability)
          throw new Error(
            `${filename(source.path)} has no forecast probability column. Select forecast-results CSVs.`,
          );
        const name = `${source.job_id?.slice(0, 6) ?? "Existing"} · ${source.path.replaceAll("\\", "/").split("/").slice(-4, -1).join("/")} · ${index + 1}`;
        columns.push(name);
        for (const row of frame.rows) {
          const date = String(row[time]);
          const combined = dates.get(date) ?? { datetime: date };
          combined[name] = row[probability];
          dates.set(date, combined);
        }
      });
      setComparison({
        columns,
        rows: [...dates.values()].sort((a, b) =>
          String(a.datetime).localeCompare(String(b.datetime)),
        ),
        total: null,
        sampled: true,
      });
      setError("");
    } catch (error) {
      setError((error as Error).message);
    }
  };
  return (
    <>
      <div className="page-heading">
        <div>
          <h1>Results</h1>
          <p>
            Explore scientific outputs, reuse local artifacts, and compare
            scenario forecasts.
          </p>
        </div>
        <button className="button secondary" onClick={() => void index()}>
          <FolderSearch size={16} />
          Discover existing outputs
        </button>
      </div>
      {error && (
        <p role="alert" className="error-message">
          {error}
        </p>
      )}
      <div className="result-filters">
        <label className="search-field">
          <Search size={16} />
          <input
            placeholder="Filter by station, scenario, classifier, or seed…"
            aria-label="Search artifacts"
            value={query}
            onChange={(event) => setQuery(event.target.value)}
          />
        </label>
        <label>
          Run
          <select
            value={jobId}
            onChange={(event) => setJobId(event.target.value)}
          >
            <option value="">All runs and existing files</option>
            {jobs.map((job) => (
              <option key={job.id} value={job.id}>
                {job.kind} · {job.id.slice(0, 8)} · {job.state}
              </option>
            ))}
          </select>
        </label>
        <label>
          Stage
          <select
            value={stage}
            onChange={(event) => setStage(event.target.value)}
          >
            <option value="all">All stages</option>
            {[
              "tremor",
              "training",
              "prediction",
              "evaluation",
              "explanation",
            ].map((value) => (
              <option key={value}>{value}</option>
            ))}
          </select>
        </label>
        <label>
          Format
          <select
            value={extension}
            onChange={(event) => setExtension(event.target.value)}
          >
            <option value="all">All formats</option>
            {["png", "pdf", "csv", "parquet", "pkl", "json", "yaml"].map(
              (value) => (
                <option key={value}>{value}</option>
              ),
            )}
          </select>
        </label>
        <button
          className="icon-button"
          aria-label="Refresh artifacts"
          onClick={() => void refreshArtifacts()}
        >
          <RefreshCw size={16} />
        </button>
      </div>
      {compare.length > 0 && (
        <div className="comparison-toolbar">
          <span>{compare.length} forecast files selected</span>
          <button
            className="button secondary"
            disabled={compare.length < 2}
            onClick={() => void compareResults()}
          >
            Compare forecasts
          </button>
          <button
            className="text-button"
            onClick={() => {
              setCompare([]);
              setComparison(null);
            }}
          >
            Clear comparison
          </button>
        </div>
      )}
      {comparison && (
        <section className="comparison-chart">
          <h2>Scenario forecast comparison</h2>
          <TimeSeries
            data={comparison}
            title="Scenario consensus probabilities"
          />
        </section>
      )}
      {!filtered.length ? (
        <div className="empty-state">
          <h2>No matching artifacts</h2>
          <p>
            Run a stage, index an existing output directory, or adjust the
            filters.
          </p>
        </div>
      ) : (
        <div className="results-layout">
          <div className="artifact-list">
            {filtered.map((item) => (
              <div
                className={selected?.id === item.id ? "selected" : ""}
                key={item.id}
              >
                {item.path.includes("forecast-results") &&
                  item.path.endsWith(".csv") && (
                    <input
                      aria-label={`Compare ${filename(item.path)}`}
                      type="checkbox"
                      checked={compare.includes(item.id)}
                      onChange={(event) =>
                        setCompare(
                          event.target.checked
                            ? [...compare, item.id]
                            : compare.filter((id) => id !== item.id),
                        )
                      }
                    />
                  )}
                <button
                  onClick={() => {
                    setSelected(item);
                    setComparison(null);
                  }}
                >
                  <strong>{filename(item.path)}</strong>
                  <span>
                    {item.path
                      .replaceAll("\\", "/")
                      .split("/")
                      .slice(-4, -1)
                      .join(" / ")}
                  </span>
                  <small>
                    {item.job_id
                      ? `Run ${item.job_id.slice(0, 8)}`
                      : "Existing local artifact"}
                  </small>
                </button>
              </div>
            ))}
          </div>
          {selected ? (
            <ArtifactViewer artifact={selected} />
          ) : (
            <div className="empty-state">
              <h2>Select an artifact</h2>
              <p>
                Figures, bounded data tables, and interactive time-series
                previews appear here.
              </p>
            </div>
          )}
        </div>
      )}
    </>
  );
}
