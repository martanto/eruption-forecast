import type { ReactNode } from "react";
import { Pause, Play } from "lucide-react";
import { Button } from "@/components/ui/button";

/** Play/pause button plus a live status line for a diagram's guided trace. */
export function TraceToolbar({
  className,
  playing,
  finished = false,
  idleLabel = "Trace one run",
  onToggle,
  status,
}: {
  className: string;
  playing: boolean;
  /** Shows "Replay" instead of the idle label once a trace has reached the end. */
  finished?: boolean;
  idleLabel?: string;
  onToggle: () => void;
  status: ReactNode;
}) {
  return (
    <div className={className}>
      <Button variant="outline" size="small" onClick={onToggle}>
        {playing ? <Pause size={15} /> : <Play size={15} />}
        {playing ? "Pause" : finished ? "Replay" : idleLabel}
      </Button>
      <span aria-live="polite">{status}</span>
    </div>
  );
}
