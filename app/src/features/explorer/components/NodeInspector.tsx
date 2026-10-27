import { useState } from "react";
import { Link } from "react-router-dom";
import { ArrowUpRight, BookOpen, FileCode2, AlertCircle, Tags } from "lucide-react";
import {
  NODES,
  STAGES,
  EDGES,
  GAPS,
  SOURCE_SNAPSHOT,
} from "@/features/architecture";
import { CodeSnippet } from "@/components/lab/CodeSnippet";
import { MathText } from "@/components/lab/MathText";
import { Button } from "@/components/ui/button";
import { NodeArtifacts, NodeBehavior, NodeParameters } from "./NodeSections";
import { StageNumber } from "./StageNumber";
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
        <StageNumber number={stage.number} />
        <span>{item.title}</span>
      </h2>
      <p className="inspector-summary">{item.summary}</p>
      {item.stage === "labels" && (
        <Button asChild variant="outline">
          <Link to="/labelling">
            <Tags size={16} aria-hidden="true" />
            Labelling walkthrough
          </Link>
        </Button>
      )}
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
          <NodeBehavior item={item} />
          {item.stage === "consensus" && <ConsensusExample />}
          {item.id === "prediction" && <ForecastReplay />}
          <WikiExplanation item={item} />
          <details className="caveats" key={item.id} open={item.gaps.length > 0}>
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
                    <p>
                      <MathText>{gap.detail}</MathText>
                    </p>
                  </div>
                );
              })}
          </details>
          <NodeParameters item={item} />
          <NodeArtifacts item={item} className="artifacts" />
          <h3>Connected steps</h3>
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
                      {incoming ? "Input" : "Output"}: {edge.label}
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
                <CodeSnippet
                  code={source.excerpt}
                  label={`${source.path} excerpt, scrolls sideways when wide`}
                />
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
