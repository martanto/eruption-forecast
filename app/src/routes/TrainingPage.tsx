import { TrainingWalkthrough, type TrainingLesson } from "@/features/explorer";

export function TrainingPage({ topic }: { topic: TrainingLesson }) {
  return <TrainingWalkthrough key={topic} topic={topic} />;
}
