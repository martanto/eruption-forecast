import { ArrowLeft, ArrowRight, Play, X } from "lucide-react";
import { TOUR } from "@/features/architecture";
import { MathText } from "@/components/lab/MathText";
import { Button } from "@/components/ui/button";
import { useGuidedTour } from "../useGuidedTour";

export function GuidedTour({ onSelect }: { onSelect: (id: string) => void }) {
  const { step, go, close } = useGuidedTour(onSelect);
  if (step === null)
    return (
      <Button variant="outline" onClick={() => go(0)}>
        <Play size={15} />
        Take the guided tour
      </Button>
    );
  const current = TOUR[step]!;
  return (
    <section className="guided-tour" aria-label="Guided tour">
      <div aria-live="polite">
        <span>
          {step + 1} / {TOUR.length}
        </span>
        <h3>{current.title}</h3>
        <p>
          <MathText>{current.text}</MathText>
        </p>
      </div>
      <div className="tour-actions">
        <Button
          variant="outline"
          size="icon"
          aria-label="Previous tour step"
          disabled={step === 0}
          onClick={() => go(step - 1)}
        >
          <ArrowLeft size={16} />
        </Button>
        <Button
          variant="outline"
          onClick={() => (step === TOUR.length - 1 ? close() : go(step + 1))}
        >
          {step === TOUR.length - 1 ? "Finish tour" : "Next"}
          <ArrowRight size={16} />
        </Button>
        <Button
          variant="ghost"
          size="icon"
          aria-label="Close tour"
          onClick={close}
        >
          <X size={16} />
        </Button>
      </div>
    </section>
  );
}
