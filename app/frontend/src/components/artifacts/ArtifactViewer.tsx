import { useEffect, useState } from "react";
import { Download } from "lucide-react";
import { api, type Artifact, type TableData } from "../../api/client";
import { filename } from "../../lib/configuration";
import { TimeSeries } from "../charts/TimeSeries";

export function ArtifactViewer({ artifact }: { artifact: Artifact }) {
  const [table, setTable] = useState<TableData | null>(null);
  const [error, setError] = useState("");
  const [mode, setMode] = useState<"table" | "chart">("chart");
  const [offset, setOffset] = useState(0);
  const extension = artifact.path.split(".").at(-1)?.toLowerCase();
  const tabular = extension === "csv" || extension === "parquet";
  useEffect(() => {
    setOffset(0);
    setError("");
    setTable(null);
  }, [artifact.id]);
  useEffect(() => {
    if (!tabular) return;
    setError("");
    let active = true;
    api<TableData>(
      `/artifacts/${artifact.id}/preview?limit=${mode === "chart" ? 2000 : 100}&offset=${offset}&chart=${mode === "chart"}`,
    )
      .then((data) => {
        if (active) setTable(data);
      })
      .catch((error) => {
        if (active) setError(error.message);
      });
    return () => {
      active = false;
    };
  }, [artifact.id, mode, offset, tabular]);
  const url = `/api/v1/artifacts/${artifact.id}/download`;
  return (
    <section className="artifact-viewer">
      <div className="section-heading">
        <h2>{filename(artifact.path)}</h2>
        <a className="button secondary" href={url} download>
          <Download size={15} />
          Download
        </a>
      </div>
      <p className="path-text">{artifact.path}</p>
      {error && (
        <p role="alert" className="error-message">
          {error}
        </p>
      )}
      {extension === "png" ? (
        <img
          className="scientific-figure"
          src={url}
          alt={filename(artifact.path)}
        />
      ) : extension === "pdf" ? (
        <iframe
          className="pdf-viewer"
          title={filename(artifact.path)}
          src={url}
        />
      ) : tabular ? (
        <>
          <div className="tabs">
            <button
              className={mode === "chart" ? "active" : ""}
              onClick={() => setMode("chart")}
            >
              Interactive chart
            </button>
            <button
              className={mode === "table" ? "active" : ""}
              onClick={() => setMode("table")}
            >
              Data table
            </button>
          </div>
          {table ? (
            mode === "chart" ? (
              <>
                <TimeSeries data={table} title={filename(artifact.path)} />
                <p className="muted">
                  Chart preview is bounded to 2,000 points. Download for the
                  full-resolution data.
                </p>
              </>
            ) : (
              <>
                <div className="table-scroll">
                  <table>
                    <thead>
                      <tr>
                        {table.columns.map((column) => (
                          <th key={column}>{column}</th>
                        ))}
                      </tr>
                    </thead>
                    <tbody>
                      {table.rows.map((row, i) => (
                        <tr key={i}>
                          {table.columns.map((column) => (
                            <td key={column}>
                              {row[column] == null ? "—" : String(row[column])}
                            </td>
                          ))}
                        </tr>
                      ))}
                    </tbody>
                  </table>
                </div>
                <div className="pagination">
                  <button
                    className="button secondary"
                    disabled={offset === 0}
                    onClick={() => setOffset(Math.max(0, offset - 100))}
                  >
                    Previous
                  </button>
                  <span>
                    Rows {offset + 1}–{offset + table.rows.length}
                  </span>
                  <button
                    className="button secondary"
                    disabled={table.rows.length < 100}
                    onClick={() => setOffset(offset + 100)}
                  >
                    Next
                  </button>
                </div>
              </>
            )
          ) : (
            <p className="muted">Loading artifact…</p>
          )}
        </>
      ) : (
        <p className="muted">
          Download this artifact to inspect it locally. Model files are never
          deserialized for previews.
        </p>
      )}
    </section>
  );
}
