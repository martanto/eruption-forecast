import { useEffect, useState } from "react";
import { TraceToolbar } from "@/components/lab/TraceToolbar";
import { goToStepLabel } from "@/lib/format";
import {
  PREDICTION_EDGES,
  PREDICTION_NODES,
  PREDICTION_STEPS,
  type PredictionEdge,
  type PredictionNode,
} from "../predictionLessons";

const W = 176;
const H = 76;
const STEP_X = 206;
const STEP_Y = 160;
const TICK_MS = 1400;
const byId = new Map(PREDICTION_NODES.map((node) => [node.id, node]));

const left = (n: PredictionNode) => n.col * STEP_X;
const top = (n: PredictionNode) => 34 + n.row * STEP_Y;

function path(edge: PredictionEdge) {
  const a = byId.get(edge.from)!;
  const b = byId.get(edge.to)!;
  if (a.col === b.col) {
    const x = left(a) + W / 2;
    const up = b.row < a.row;
    // Stop short of the target so the arrowhead stays visible.
    return up
      ? `M${x} ${top(a)} L${x} ${top(b) + H + 2}`
      : `M${x} ${top(a) + H} L${x} ${top(b) - 2}`;
  }
  if (edge.elbow) {
    const x1 = left(a) + W;
    const y1 = top(a) + H * 0.3;
    const x2 = left(b) - 2;
    const y2 = top(b) + H / 2;
    const mid = x1 + (x2 - x1) / 2;
    return `M${x1} ${y1} L${mid - 10} ${y1} Q${mid} ${y1} ${mid} ${y1 - 10} L${mid} ${y2 + 10} Q${mid} ${y2} ${mid + 10} ${y2} L${x2} ${y2}`;
  }
  const shifted = a.id === "forecast" && b.id === "csv" ? H * 0.62 : H / 2;
  const y = top(a) + shifted;
  return `M${left(a) + W} ${y} L${left(b) - 2} ${top(b) + (b.id === "csv" ? shifted : H / 2)}`;
}

function labelPoint(edge: PredictionEdge) {
  const a = byId.get(edge.from)!;
  const b = byId.get(edge.to)!;
  if (a.col === b.col) return [left(a) + W / 2, (top(a) + top(b) + H) / 2];
  // Sits above the row so a wide pill never covers the neighbouring boxes.
  return [(left(a) + W + left(b)) / 2, top(a) - 14];
}

function stateOf(node: PredictionNode, step: number) {
  return node.step === step ? "current" : node.step < step ? "done" : "idle";
}

function Icon({ kind, x, y }: { kind: PredictionNode["kind"]; x: number; y: number }) {
  return kind === "code" ? (
    <path
      className="pwf-icon"
      d={`M${x + 4} ${y} l-5 5 l5 5 M${x + 10} ${y} l5 5 l-5 5`}
      fill="none"
      strokeWidth="1.6"
      strokeLinecap="round"
      strokeLinejoin="round"
    />
  ) : (
    <g className="pwf-icon" fill="none" strokeWidth="1.6" strokeLinecap="round">
      <ellipse cx={x + 8} cy={y + 2} rx="6.5" ry="2.6" />
      <path d={`M${x + 1.5} ${y + 2} v6 c0 1.5 3 2.6 6.5 2.6 s6.5 -1.1 6.5 -2.6 v-6`} />
    </g>
  );
}

export function PredictionWorkflowDiagram({
  step,
  onStepChange,
}: {
  step: number;
  onStepChange: (step: number) => void;
}) {
  const [playing, setPlaying] = useState(false);
  const last = PREDICTION_STEPS.length - 1;

  useEffect(() => {
    if (!playing) return;
    const id = window.setInterval(() => {
      onStepChange(step >= last ? 0 : step + 1);
    }, TICK_MS);
    return () => window.clearInterval(id);
  }, [playing, step, last, onStepChange]);
  useEffect(() => {
    if (playing && step === last) {
      const id = window.setTimeout(() => setPlaying(false), TICK_MS);
      return () => window.clearTimeout(id);
    }
  }, [playing, step, last]);

  function select(next: number) {
    setPlaying(false);
    onStepChange(next);
  }

  return (
    <figure className="pwf" aria-label="Prediction workflow">
      <TraceToolbar
        className="pwf-toolbar"
        playing={playing}
        idleLabel="Trace one forecast"
        onToggle={() => {
          if (!playing && step >= last) onStepChange(0);
          setPlaying(!playing);
        }}
        status={`Step ${step + 1} of ${PREDICTION_STEPS.length}: ${PREDICTION_STEPS[step]![0]}`}
      />
      <svg
        className="pwf-svg"
        viewBox="0 0 1000 444"
        role="group"
        aria-label="Prediction pipeline. Choose a box to jump to its step."
      >
        <defs>
          {["idle", "lit"].map((tone) => (
            <marker
              key={tone}
              id={`pwf-arrow-${tone}`}
              viewBox="0 0 10 10"
              refX="9"
              refY="5"
              markerWidth="9"
              markerHeight="9"
              orient="auto"
            >
              <path d="M0 0 L10 5 L0 10 z" className={`pwf-head pwf-head-${tone}`} />
            </marker>
          ))}
        </defs>
        {PREDICTION_EDGES.map((edge) => {
          const a = byId.get(edge.from)!;
          const b = byId.get(edge.to)!;
          // forecast() is what runs at step 3, so its outgoing edges light with the target node.
          const lit = Math.max(a.step, b.step) <= step;
          const [lx, ly] = labelPoint(edge);
          return (
            <g key={`${edge.from}-${edge.to}`} data-lit={lit}>
              <path
                className="pwf-edge"
                d={path(edge)}
                fill="none"
                markerEnd={`url(#pwf-arrow-${lit ? "lit" : "idle"})`}
              />
              {edge.label && (
                <g transform={`translate(${lx} ${ly})`}>
                  <rect
                    className="pwf-pill"
                    x={-edge.label.length * 3.8 - 8}
                    y="-10"
                    width={edge.label.length * 7.6 + 16}
                    height="20"
                    rx="10"
                  />
                  <text className="pwf-pill-text" textAnchor="middle" dy="4">
                    {edge.label}
                  </text>
                </g>
              )}
            </g>
          );
        })}
        {PREDICTION_NODES.map((node) => (
          <g
            key={node.id}
            className="pwf-node"
            data-kind={node.kind}
            data-state={stateOf(node, step)}
            transform={`translate(${left(node)} ${top(node)})`}
            role="button"
            tabIndex={0}
            aria-label={goToStepLabel(node)}
            aria-current={node.step === step ? "step" : undefined}
            onClick={() => select(node.step)}
            onKeyDown={(event) => {
              if (event.key === "Enter" || event.key === " ") {
                event.preventDefault();
                select(node.step);
              }
            }}
          >
            <rect className="pwf-box" width={W} height={H} rx="9" />
            <Icon kind={node.kind} x={11} y={11} />
            <text className="pwf-title" x={W / 2} y="40" textAnchor="middle">
              {node.title}
            </text>
            <text className="pwf-detail" x={W / 2} y="60" textAnchor="middle">
              {node.detail}
            </text>
          </g>
        ))}
      </svg>
      <ol className="pwf-list" aria-label="Prediction pipeline, in reading order">
        {PREDICTION_NODES.map((node) => (
          <li
            key={node.id}
            className="pwf-list-node"
            data-kind={node.kind}
            data-state={stateOf(node, step)}
          >
            <button type="button" onClick={() => select(node.step)}>
              <strong>{node.title}</strong>
              <span>{node.detail}</span>
            </button>
          </li>
        ))}
      </ol>
      <figcaption>
        Green boxes are code; purple boxes are files or objects. The model
        branch joins from above: <code>from_any</code> turns whatever you saved
        into a ClassifierEnsemble, and <code>forecast()</code> hands it the
        window features. Results fan out to the CSV, the plot and the cache.
      </figcaption>
    </figure>
  );
}
