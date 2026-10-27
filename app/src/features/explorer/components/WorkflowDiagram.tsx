import { useCallback, useLayoutEffect, useRef, useState } from "react";
import { Code2, Database, ExternalLink } from "lucide-react";
import { Legend } from "@/components/lab/Legend";
import { TraceToolbar } from "@/components/lab/TraceToolbar";
import { useTrace } from "@/hooks/useTrace";
import { goToStepLabel } from "@/lib/format";
import type {
  WorkflowEdge,
  WorkflowKind,
  WorkflowNode,
  WorkflowSide,
} from "../workflowDiagrams";

const KIND_LABEL: Record<WorkflowKind, string> = {
  backend: "Backend",
  database: "Database",
  external: "External",
};
const KIND_ICON = { backend: Code2, database: Database, external: ExternalLink };

interface Rect {
  x: number;
  y: number;
  w: number;
  h: number;
}
interface Route {
  id: string;
  d: string;
  label: string;
  lx: number;
  ly: number;
}

function anchor(rect: Rect, side: WorkflowSide) {
  if (side === "left") return { x: rect.x, y: rect.y + rect.h / 2 };
  if (side === "right") return { x: rect.x + rect.w, y: rect.y + rect.h / 2 };
  if (side === "top") return { x: rect.x + rect.w / 2, y: rect.y };
  return { x: rect.x + rect.w / 2, y: rect.y + rect.h };
}

// Straight when the sides face each other; otherwise one right-angle elbow.
function route(edge: WorkflowEdge, from: Rect, to: Rect): Route {
  const a = anchor(from, edge.fromSide);
  const b = anchor(to, edge.toSide);
  const aVertical = edge.fromSide === "top" || edge.fromSide === "bottom";
  const bVertical = edge.toSide === "top" || edge.toSide === "bottom";
  const id = `${edge.from}-${edge.to}`;
  if (aVertical !== bVertical) {
    const corner = aVertical ? { x: a.x, y: b.y } : { x: b.x, y: a.y };
    const first = Math.abs(corner.x - a.x) + Math.abs(corner.y - a.y);
    const second = Math.abs(b.x - corner.x) + Math.abs(b.y - corner.y);
    const mid =
      first >= second
        ? { x: (a.x + corner.x) / 2, y: (a.y + corner.y) / 2 }
        : { x: (corner.x + b.x) / 2, y: (corner.y + b.y) / 2 };
    return {
      id,
      d: `M${a.x},${a.y} L${corner.x},${corner.y} L${b.x},${b.y}`,
      label: edge.label,
      lx: mid.x,
      ly: mid.y,
    };
  }
  return {
    id,
    d: `M${a.x},${a.y} L${b.x},${b.y}`,
    label: edge.label,
    lx: (a.x + b.x) / 2,
    ly: (a.y + b.y) / 2,
  };
}

export function WorkflowDiagram({
  title,
  nodes,
  edges,
  step,
  onStepChange,
  caption,
}: {
  title: string;
  // Array order is the execution order used by "Trace one run".
  nodes: readonly WorkflowNode[];
  edges: readonly WorkflowEdge[];
  step: number;
  onStepChange: (step: number) => void;
  caption: string;
}) {
  const trace = useTrace(nodes, step, onStepChange);
  const { cursor } = trace;
  const [routes, setRoutes] = useState<Route[]>([]);
  const canvas = useRef<HTMLDivElement>(null);
  const refs = useRef(new Map<string, HTMLLIElement>());
  const indexOf = useCallback(
    (id: string) => nodes.findIndex((node) => node.id === id),
    [nodes],
  );

  const measure = useCallback(() => {
    const host = canvas.current;
    if (!host) return;
    const rect = (id: string): Rect | null => {
      const el = refs.current.get(id);
      return el
        ? { x: el.offsetLeft, y: el.offsetTop, w: el.offsetWidth, h: el.offsetHeight }
        : null;
    };
    setRoutes(
      edges.flatMap((edge) => {
        const from = rect(edge.from);
        const to = rect(edge.to);
        return from && to ? [route(edge, from, to)] : [];
      }),
    );
  }, [edges]);

  useLayoutEffect(() => {
    measure();
    const host = canvas.current;
    if (!host) return;
    const observer = new ResizeObserver(measure);
    observer.observe(host);
    return () => observer.disconnect();
  }, [measure]);

  const edgeLit = (edge: WorkflowEdge) => {
    const to = indexOf(edge.to);
    return cursor < 0 ? nodes[to]!.step === step : to <= cursor;
  };
  const counts = (Object.keys(KIND_LABEL) as WorkflowKind[]).map((kind) => ({
    kind,
    count: nodes.filter((node) => node.kind === kind).length,
  }));

  return (
    <figure className="wf" aria-label={title}>
      <TraceToolbar
        className="swf-toolbar"
        playing={trace.playing}
        finished={trace.finished}
        onToggle={trace.togglePlay}
        status={
          trace.current
            ? `${trace.current.title}: ${trace.current.detail}`
            : "Select a box to jump to its step, or trace the whole run."
        }
      />
      <div className="wf-scroll">
        <div className="wf-canvas" ref={canvas}>
          <svg className="wf-edges" aria-hidden="true">
            <defs>
              <marker
                id="wf-arrow"
                viewBox="0 0 10 10"
                refX="9"
                refY="5"
                markerWidth="8"
                markerHeight="8"
                orient="auto-start-reverse"
              >
                <path d="M0,0 L10,5 L0,10 z" fill="currentColor" />
              </marker>
            </defs>
            {routes.map((r, i) => {
              const edge = edges[i]!;
              return (
                <path
                  key={r.id}
                  d={r.d}
                  className="wf-edge"
                  data-dashed={edge.dashed || undefined}
                  data-lit={edgeLit(edge)}
                  markerEnd="url(#wf-arrow)"
                />
              );
            })}
          </svg>
          <ol className="wf-nodes">
            {nodes.map((node, index) => {
              const Icon = KIND_ICON[node.kind];
              const out = edges.find((edge) => edge.from === node.id);
              return (
                <li
                  key={node.id}
                  ref={(el) => {
                    if (el) refs.current.set(node.id, el);
                    else refs.current.delete(node.id);
                  }}
                  className="wf-node"
                  data-kind={node.kind}
                  data-state={trace.stateOf(index)}
                  data-out={out?.label}
                  style={{ gridColumn: node.col, gridRow: node.row }}
                >
                  <button
                    type="button"
                    onClick={() => trace.select(node.step)}
                    aria-label={goToStepLabel(node)}
                  >
                    <Icon size={14} aria-hidden="true" />
                    <strong>{node.title}</strong>
                    <span>{node.detail}</span>
                  </button>
                </li>
              );
            })}
          </ol>
          {routes.map((r, i) => (
            <span
              key={r.id}
              className="wf-pill"
              data-lit={edgeLit(edges[i]!)}
              style={{ left: r.lx, top: r.ly }}
            >
              {r.label}
            </span>
          ))}
        </div>
      </div>
      <Legend
        className="wf-legend"
        items={counts.map(({ kind, count }) => ({
          swatch: ["kind", kind],
          label: (
            <>
              {KIND_LABEL[kind]} <b>{count}</b>
            </>
          ),
        }))}
      />
      <figcaption>{caption}</figcaption>
    </figure>
  );
}
