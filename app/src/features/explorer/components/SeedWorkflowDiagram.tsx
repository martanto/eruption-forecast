import { useEffect, useState } from "react";
import { Pause, Play } from "lucide-react";
import { Button } from "@/components/ui/button";
import {
  SEED_INPUT_NODE,
  SEED_JOB_NODES,
  SEED_MERGE_NODES,
  type SeedWorkflowNode,
} from "../trainingLessons";

const ORDER = [SEED_INPUT_NODE, ...SEED_JOB_NODES, ...SEED_MERGE_NODES];
const TICK_MS = 900;

function Node({
  node,
  step,
  cursor,
  onSelect,
}: {
  node: SeedWorkflowNode;
  step: number;
  cursor: number;
  onSelect: (step: number) => void;
}) {
  const index = ORDER.indexOf(node);
  // While a trace is running the cursor owns the highlight; otherwise the
  // walkthrough step does.
  const state =
    cursor < 0
      ? node.step === step
        ? "current"
        : "idle"
      : index < cursor
        ? "done"
        : index === cursor
          ? "current"
          : "idle";
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
        aria-label={`${node.title}: ${node.detail}. Go to step ${node.step + 1}`}
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
  const [cursor, setCursor] = useState(-1);
  const [playing, setPlaying] = useState(false);

  useEffect(() => {
    if (!playing) return;
    const id = window.setInterval(() => {
      setCursor((current) => {
        if (current + 1 >= ORDER.length) {
          setPlaying(false);
          return current;
        }
        return current + 1;
      });
    }, TICK_MS);
    return () => window.clearInterval(id);
  }, [playing]);

  useEffect(() => {
    if (cursor >= 0) onStepChange(ORDER[cursor]!.step);
  }, [cursor, onStepChange]);

  function togglePlay() {
    if (playing) {
      setPlaying(false);
      return;
    }
    // Replay from the start once a run has finished.
    setCursor((current) => (current + 1 >= ORDER.length ? 0 : current < 0 ? 0 : current));
    setPlaying(true);
  }

  function select(next: number) {
    setPlaying(false);
    setCursor(-1);
    onStepChange(next);
  }

  const inJob = cursor >= 1 && cursor <= SEED_JOB_NODES.length;
  return (
    <figure className="swf" aria-label="Seed ensemble workflow">
      <div className="swf-toolbar">
        <Button variant="outline" size="small" onClick={togglePlay}>
          {playing ? <Pause size={15} /> : <Play size={15} />}
          {playing ? "Pause" : cursor >= ORDER.length - 1 ? "Replay" : "Trace one run"}
        </Button>
        <span aria-live="polite">
          {cursor < 0
            ? "Select a box to jump to its step, or trace the whole run."
            : `${ORDER[cursor]!.title}: ${ORDER[cursor]!.detail}`}
        </span>
      </div>
      <div className="swf-row">
        <ol className="swf-list swf-input">
          <Node node={SEED_INPUT_NODE} step={step} cursor={cursor} onSelect={select} />
        </ol>
        <div className="swf-group" data-running={inJob}>
          <p className="swf-group-label">One job per seed (random_state)</p>
          <ol className="swf-list">
            {SEED_JOB_NODES.map((node) => (
              <Node key={node.id} node={node} step={step} cursor={cursor} onSelect={select} />
            ))}
          </ol>
        </div>
      </div>
      <div className="swf-drop" aria-hidden="true" data-lit={cursor >= SEED_JOB_NODES.length} />
      <ol className="swf-list swf-merge">
        {SEED_MERGE_NODES.map((node) => (
          <Node key={node.id} node={node} step={step} cursor={cursor} onSelect={select} />
        ))}
      </ol>
      <figcaption>
        Every seed repeats the boxed steps independently, so seeds can run in
        parallel (<code>n_jobs</code>). The stacked outlines behind the box are
        the other seeds. Only after all jobs finish are their models
        registered and bundled. Green boxes are code steps; the other boxes are files or artifacts.
      </figcaption>
    </figure>
  );
}
