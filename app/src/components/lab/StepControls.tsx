import { ArrowRight, RotateCcw } from "lucide-react";
import { Button } from "@/components/ui/button";

/** Numbered buttons, one per item; `labels` are the accessible names. */
export function StepButtons({
  labels,
  current,
  onSelect,
  groupLabel,
}: {
  labels: readonly string[];
  current: number;
  onSelect: (index: number) => void;
  groupLabel: string;
}) {
  return (
    <div className="lab-step-nav" role="group" aria-label={groupLabel}>
      {labels.map((label, index) => (
        <button
          key={label}
          aria-label={label}
          aria-pressed={current === index}
          onClick={() => onSelect(index)}
        >
          {index + 1}
        </button>
      ))}
    </div>
  );
}

/** Step picker, the current step's explanation, and previous/next/restart actions. */
export function StepControls({
  steps,
  step,
  onStepChange,
  groupLabel,
  restartLabel = "Restart walkthrough",
  onRestart = () => onStepChange(0),
}: {
  /** One `[title, explanation]` pair per step. */
  steps: readonly (readonly [string, string])[];
  step: number;
  onStepChange: (step: number) => void;
  groupLabel: string;
  restartLabel?: string;
  onRestart?: () => void;
}) {
  const [heading, copy] = steps[step]!;
  return (
    <section className="lab-operation" aria-label="Walkthrough controls">
      <StepButtons
        labels={steps.map(([title], index) => `Step ${index + 1}: ${title}`)}
        current={step}
        onSelect={onStepChange}
        groupLabel={groupLabel}
      />
      <div className="lab-operation-copy" aria-live="polite">
        <h2>{heading}</h2>
        <p>{copy}</p>
      </div>
      <div className="lab-step-actions">
        <Button
          variant="ghost"
          disabled={step === 0}
          onClick={() => onStepChange(step - 1)}
        >
          Previous
        </Button>
        <Button
          variant="outline"
          disabled={step === steps.length - 1}
          onClick={() => onStepChange(step + 1)}
        >
          Next step <ArrowRight size={15} />
        </Button>
        <Button
          variant="ghost"
          size="icon"
          aria-label={restartLabel}
          onClick={onRestart}
        >
          <RotateCcw size={16} />
        </Button>
      </div>
    </section>
  );
}
