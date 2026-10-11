import { useEffect, useState } from "react";
import { Link, useSearchParams } from "react-router-dom";
import { RefreshCw, Square, ArrowRight } from "lucide-react";
import { api, type Job } from "../api/client";
import { usePolling } from "../hooks/usePolling";

export function Status({ state }: { state: string }) {
  return <span className={`status status-${state}`}>{state}</span>;
}

export default function JobsPage() {
  const {
    data: jobs,
    error: connectionError,
    refresh,
  } = usePolling<Job[]>("/jobs");
  const [params, setParams] = useSearchParams();
  const selected = params.get("selected") ?? jobs?.[0]?.id;
  const job = jobs?.find((item) => item.id === selected);
  const [logs, setLogs] = useState("");
  const [error, setError] = useState("");
  const [filter, setFilter] = useState("all");
  useEffect(() => {
    if (!selected) return;
    setLogs("");
    let cursor = 0;
    let active = true;
    const load = async () => {
      try {
        const data = await api<{ text: string; cursor: number }>(
          `/jobs/${selected}/logs?cursor=${cursor}`,
        );
        if (active) {
          cursor = data.cursor;
          setLogs((previous) => (previous + data.text).slice(-150000));
        }
      } catch (error) {
        if (active) setError((error as Error).message);
      }
    };
    void load();
    const timer = setInterval(() => void load(), 2000);
    return () => {
      active = false;
      clearInterval(timer);
    };
  }, [selected]);
  const action = async (kind: "cancel" | "retry") => {
    if (!job) return;
    try {
      const result = await api<Job>(`/jobs/${job.id}/${kind}`, {});
      if (kind === "retry") setParams({ selected: result.id });
      await refresh();
      setError("");
    } catch (error) {
      setError((error as Error).message);
    }
  };
  return (
    <>
      <div className="page-heading">
        <div>
          <h1>Jobs</h1>
          <p>
            Work continues independently of your browser. Every attempt keeps
            its configuration and logs.
          </p>
        </div>
        <button className="button secondary" onClick={() => void refresh()}>
          <RefreshCw size={15} />
          Refresh
        </button>
      </div>
      {(error || connectionError) && (
        <p className="error-message" role="alert">
          {error || connectionError}
        </p>
      )}
      <div className="list-toolbar">
        <label>
          Filter status
          <select
            value={filter}
            onChange={(event) => setFilter(event.target.value)}
          >
            <option value="all">All jobs</option>
            {[
              "queued",
              "running",
              "succeeded",
              "partial",
              "failed",
              "cancelled",
              "interrupted",
            ].map((state) => (
              <option key={state}>{state}</option>
            ))}
          </select>
        </label>
        <span className="muted">{jobs?.length ?? 0} recent attempts</span>
      </div>
      {!jobs?.length ? (
        <div className="empty-state">
          <h2>No jobs yet</h2>
          <p>Configure a stage or scenario batch, then submit it here.</p>
          <Link className="button primary" to="/setup">
            Start with general setup <ArrowRight size={15} />
          </Link>
        </div>
      ) : (
        <div className="jobs-layout">
          <div className="job-list">
            {jobs
              .filter((item) => filter === "all" || item.state === filter)
              .map((item) => (
                <button
                  key={item.id}
                  className={item.id === job?.id ? "selected" : ""}
                  onClick={() => setParams({ selected: item.id })}
                >
                  <div>
                    <strong>
                      {item.kind[0].toUpperCase() + item.kind.slice(1)}
                    </strong>
                    <Status state={item.state} />
                  </div>
                  <span>{new Date(item.created_at).toLocaleString()}</span>
                  <code>{item.id.slice(0, 12)}</code>
                </button>
              ))}
          </div>
          {job && (
            <section className="job-detail">
              <div className="section-heading">
                <h2>{job.kind[0].toUpperCase() + job.kind.slice(1)} attempt</h2>
                <Status state={job.state} />
              </div>
              <div className="job-actions">
                {["queued", "running", "cancelling"].includes(job.state) ? (
                  <button
                    className="button secondary"
                    disabled={job.state === "cancelling"}
                    onClick={() => void action("cancel")}
                  >
                    <Square size={13} />
                    Cancel job
                  </button>
                ) : (
                  <button
                    className="button secondary"
                    onClick={() => void action("retry")}
                  >
                    <RefreshCw size={14} />
                    Retry with valid caches
                  </button>
                )}
                <Link
                  className="button secondary"
                  to={`/results?job=${job.id}`}
                >
                  View artifacts <ArrowRight size={14} />
                </Link>
              </div>
              {job.error && <p className="error-message">{job.error}</p>}
              <dl className="metadata">
                <dt>Workspace</dt>
                <dd>{job.workspace}</dd>
                <dt>Last updated</dt>
                <dd>{new Date(job.updated_at).toLocaleString()}</dd>
                {job.retry_of && (
                  <>
                    <dt>Retry of</dt>
                    <dd>{job.retry_of}</dd>
                  </>
                )}
              </dl>
              <h3>Stage timeline</h3>
              {(job.events ?? []).length ? (
                <ol className="timeline">
                  {(job.events ?? []).map((event, index) => (
                    <li key={index}>
                      <span>{String(event.scenario || "Standalone")}</span>
                      <strong>{String(event.stage)}</strong>
                      <Status state={String(event.state)} />
                      {!!event.detail && <p>{String(event.detail)}</p>}
                    </li>
                  ))}
                </ol>
              ) : (
                <p className="muted">
                  Waiting for the worker to claim this job.
                </p>
              )}
              <h3>Execution log</h3>
              <pre className="log-view" tabIndex={0}>
                {logs ||
                  "No log output yet. Start the backend worker to process queued jobs."}
              </pre>
              <details>
                <summary>Resolved configuration</summary>
                <pre className="configuration-view">
                  {JSON.stringify(job.resolved, null, 2)}
                </pre>
              </details>
            </section>
          )}
        </div>
      )}
    </>
  );
}
