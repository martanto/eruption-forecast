import { useState } from "react";
import { ArrowUpRight, BookOpen, FileCode2, AlertCircle } from "lucide-react";
import {
  NODES,
  STAGES,
  EDGES,
  GAPS,
  SOURCE_SNAPSHOT,
} from "@/features/architecture";
import { Button } from "@/components/ui/button";
import { WikiExplanation } from "./WikiExplanation";
import { ConsensusExample } from "./ConsensusExample";
import { ForecastReplay } from "./ForecastReplay";
import historicalForecast from "@/assets/historical-forecast.png";

export function NodeInspector({
  selected,
  onSelect,
}: {
  selected: string;
  onSelect: (id: string) => void;
}) {
  const [tab, setTab] = useState<"details" | "sources">("details");
  const item = NODES.find((entry) => entry.id === selected)!;
  const stage = STAGES.find((entry) => entry.id === item.stage)!;
  const neighbors = EDGES.filter(
    (edge) => edge.source === selected || edge.target === selected,
  );
  return (
    <section
      className={`inspector tone-${stage.tone}`}
      aria-label="Node details"
    >
      <h2 className={`inspector-title tone-${stage.tone}`}>
        <span className="stage-number">
          {String(stage.number).padStart(2, "0")}
        </span>
        <span>{item.title}</span>
      </h2>
      <p className="inspector-summary">{item.summary}</p>
      <div
        className="inspector-tabs"
        role="group"
        aria-label="Inspector content"
      >
        <Button
          variant="ghost"
          aria-pressed={tab === "details"}
          onClick={() => setTab("details")}
        >
          <BookOpen size={15} />
          Details
        </Button>
        <Button
          variant="ghost"
          aria-pressed={tab === "sources"}
          onClick={() => setTab("sources")}
        >
          <FileCode2 size={15} />
          Sources ({item.sources.length})
        </Button>
      </div>
      {tab === "details" ? (
        <>
          <h3>What actually happens</h3>
          {item.behavior.map((text) => (
            <p key={text}>{text}</p>
          ))}
          {item.stage === "consensus" && <ConsensusExample />}
          {item.id === "prediction" && <ForecastReplay />}
          <WikiExplanation item={item} />
          <details className="caveats" key={item.id}>
            <summary>Implementation caveats ({GAPS.length})</summary>
            {[...GAPS]
              .sort(
                (a, b) =>
                  Number(item.gaps.includes(b.id)) -
                  Number(item.gaps.includes(a.id)),
              )
              .map((gap) => {
                const id = gap.id;
                return (
                  <div className="gap-note" key={id}>
                    <h4>
                      <AlertCircle size={16} />
                      {gap.title}
                    </h4>
                    <p>{gap.detail}</p>
                  </div>
                );
              })}
          </details>
          <h3>Parameters & settings</h3>
          <ul>
            {item.parameters.map((text) => (
              <li key={text}>{text}</li>
            ))}
          </ul>
          <h3>Artifacts</h3>
          <ul className="artifacts">
            {item.artifacts.map((text) => (
              <li key={text}>{text}</li>
            ))}
          </ul>
          <h3>Follow the connections</h3>
          <div className="connection-list">
            {neighbors.map((edge) => {
              const incoming = edge.target === selected;
              const other = NODES.find(
                (entry) => entry.id === (incoming ? edge.source : edge.target),
              )!;
              return (
                <button key={edge.id} onClick={() => onSelect(other.id)}>
                  <span>
                    <small>
                      {incoming ? "INPUT" : "OUTPUT"} · {edge.label}
                    </small>
                    {other.title}
                  </span>
                  <ArrowUpRight size={16} />
                </button>
              );
            })}
          </div>
          {item.id === "prediction" && (
            <figure className="historical-example">
              <a href={historicalForecast} target="_blank" rel="noreferrer">
                <img
                  src={historicalForecast}
                  alt="Historical forecast plot for 27 July through 22 August 2025; open full-size image"
                />
              </a>
              <figcaption>
                Historical example · 27 Jul–22 Aug 2025. Bundled from
                assets/forecast_2025-07-27_2025-08-22.png; not attributed to the
                selected scenario.
              </figcaption>
            </figure>
          )}
        </>
      ) : (
        <div className="source-list">
          <p>
            Bundled excerpts from the reviewed working tree. Readable without a
            Python server.
          </p>
          {item.sources.map((id) => {
            const source = SOURCE_SNAPSHOT.sources.find(
              (entry) => entry.id === id,
            )!;
            return (
              <details key={id}>
                <summary>
                  <FileCode2 size={15} />
                  <span>
                    {source.path}
                    <small>
                      Line {source.line} · {source.symbol}
                    </small>
                  </span>
                </summary>
                <pre>
                  <code>{source.excerpt}</code>
                </pre>
                <p className="fingerprint">SHA-256 {source.sha256}</p>
              </details>
            );
          })}
          <p className="snapshot-note">
            Reviewed {SOURCE_SNAPSHOT.reviewedAt.slice(0, 10)} · revision{" "}
            <code>{SOURCE_SNAPSHOT.revision.slice(0, 8)}</code> plus
            working-tree edits. Fingerprints detect source drift; they do not
            synchronize meaning.
          </p>
          <details>
            <summary>Working-tree snapshot</summary>
            <pre>{SOURCE_SNAPSHOT.workingTree}</pre>
          </details>
        </div>
      )}
    </section>
  );
}
