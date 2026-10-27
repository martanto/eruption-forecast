import { useEffect, useState } from "react";

export type TraceState = "idle" | "current" | "done";

/**
 * Cursor-driven "trace one run" for a diagram whose nodes are listed in
 * execution order. While a trace runs the cursor owns the highlight and drives
 * the walkthrough step; otherwise the walkthrough step does.
 */
export function useTrace<T extends { step: number }>(
  order: readonly T[],
  step: number,
  onStepChange: (step: number) => void,
  tickMs = 900,
) {
  const [cursor, setCursor] = useState(-1);
  const [playing, setPlaying] = useState(false);

  useEffect(() => {
    if (!playing) return;
    const id = window.setInterval(() => {
      setCursor((current) => {
        if (current + 1 >= order.length) {
          setPlaying(false);
          return current;
        }
        return current + 1;
      });
    }, tickMs);
    return () => window.clearInterval(id);
  }, [playing, order.length, tickMs]);

  useEffect(() => {
    if (cursor >= 0) onStepChange(order[cursor]!.step);
  }, [cursor, order, onStepChange]);

  function togglePlay() {
    if (playing) {
      setPlaying(false);
      return;
    }
    // Replay from the start once a run has finished.
    setCursor((current) =>
      current + 1 >= order.length || current < 0 ? 0 : current,
    );
    setPlaying(true);
  }

  function select(next: number) {
    setPlaying(false);
    setCursor(-1);
    onStepChange(next);
  }

  function stateOf(index: number): TraceState {
    if (cursor < 0) return order[index]!.step === step ? "current" : "idle";
    return index < cursor ? "done" : index === cursor ? "current" : "idle";
  }

  return {
    cursor,
    playing,
    finished: cursor >= order.length - 1,
    current: cursor >= 0 ? order[cursor]! : null,
    togglePlay,
    select,
    stateOf,
  };
}
