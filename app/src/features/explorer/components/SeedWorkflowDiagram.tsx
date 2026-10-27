import { CodeSnippet } from "@/components/lab/CodeSnippet";
import { TraceToolbar } from "@/components/lab/TraceToolbar";
import { useTrace, type TraceState } from "@/hooks/useTrace";
import { goToStepLabel } from "@/lib/format";
import {
  SEED_INPUT_NODE,
  SEED_JOB_NODES,
  SEED_MERGE_NODES,
  type SeedWorkflowNode,
} from "../trainingLessons";

const ORDER = [SEED_INPUT_NODE, ...SEED_JOB_NODES, ...SEED_MERGE_NODES];

function Node({
  node,
  state,
  onSelect,
}: {
  node: SeedWorkflowNode;
  state: TraceState;
  onSelect: (step: number) => void;
}) {
  return (
    <li
      className="swf-node"
      data-kind={node.kind}
      data-state={state}
      data-edge-label={node.edgeLabel}
    >
      <button
        type="button"
        onClick={() => onSelect(node.step)}
        aria-label={goToStepLabel(node)}
      >
        <strong>{node.title}</strong>
        <span>{node.detail}</span>
      </button>
    </li>
  );
}

export function SeedWorkflowDiagram({
  step,
  onStepChange,
}: {
  step: number;
  onStepChange: (step: number) => void;
}) {
  const trace = useTrace(ORDER, step, onStepChange);
  const { cursor } = trace;
  const inJob = cursor >= 1 && cursor <= SEED_JOB_NODES.length;
  const node = (item: SeedWorkflowNode) => (
    <Node
      key={item.id}
      node={item}
      state={trace.stateOf(ORDER.indexOf(item))}
      onSelect={trace.select}
    />
  );
  return (
    <figure className="swf" aria-label="Seed ensemble workflow">
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
      <div className="swf-row">
        <ol className="swf-list swf-input">{node(SEED_INPUT_NODE)}</ol>
        <div className="swf-group" data-running={inJob}>
          <p className="swf-group-label">One job per seed (random_state)</p>
          <ol className="swf-list">{SEED_JOB_NODES.map(node)}</ol>
        </div>
      </div>
      <div className="swf-drop" aria-hidden="true" data-lit={cursor >= SEED_JOB_NODES.length} />
      <ol className="swf-list swf-merge">{SEED_MERGE_NODES.map(node)}</ol>
      <figcaption>
        Every seed repeats the boxed steps independently, so seeds can run in
        parallel (<CodeSnippet inline code="n_jobs" />). The stacked outlines behind the box are
        the other seeds. Only after all jobs finish are their models
        registered and bundled. Green boxes are code steps; the other boxes are files or artifacts.
      </figcaption>
    </figure>
  );
}
