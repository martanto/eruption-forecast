import { useState } from "react";
import { TOUR } from "@/features/architecture";
export function useGuidedTour(onSelect: (id: string) => void) {
  const [step, setStep] = useState<number | null>(null);
  function go(index: number) {
    const next = Math.min(Math.max(index, 0), TOUR.length - 1);
    setStep(next);
    onSelect(TOUR[next]!.target);
  }
  return { step, go, close: () => setStep(null) };
}
