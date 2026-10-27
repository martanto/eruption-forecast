import { useState } from "react";
import { Link } from "react-router-dom";
import { ParameterStrip } from "@/components/lab/ParameterStrip";
import { StepControls } from "@/components/lab/StepControls";
import {
  ImplementationNotes,
  WhySection,
} from "@/components/lab/WalkthroughNotes";
import { TRAINING_LESSONS, type TrainingLesson } from "../trainingLessons";
import { ClassifierEnsembleLesson } from "./ClassifierEnsembleLesson";
import { FeatureSelectionLesson } from "./FeatureSelectionLesson";
import { FeaturesMatrixLesson } from "./FeaturesMatrixLesson";
import { GridCvLesson } from "./GridCvLesson";
import { SeedEnsembleLesson } from "./SeedEnsembleLesson";
import { SignificanceExplainer } from "./SignificanceExplainer";
import {
  SamplingRatioExplainer,
  UnderSamplerLesson,
} from "./UnderSamplerLesson";
import { WalkthroughHeader } from "./WalkthroughHeader";
import {
  WalkthroughFooter,
  WalkthroughMenu,
  WalkthroughPager,
} from "./WalkthroughNav";
import "@/styles/labelling.css";
import "@/styles/training.css";

const LESSON_KEYS = Object.keys(TRAINING_LESSONS) as TrainingLesson[];

function LessonVisual({
  topic,
  step,
  onStepChange,
  seed,
  setSeed,
}: {
  topic: TrainingLesson;
  step: number;
  onStepChange: (step: number) => void;
  seed: number;
  setSeed: (seed: number) => void;
}) {
  switch (topic) {
    case "features-matrix":
      return <FeaturesMatrixLesson step={step} onStepChange={onStepChange} />;
    case "feature-selection":
      return (
        <FeatureSelectionLesson step={step} seed={seed} setSeed={setSeed} />
      );
    case "grid-cv":
      return <GridCvLesson step={step} seed={seed} setSeed={setSeed} />;
    case "random-under-sampler":
      return <UnderSamplerLesson step={step} seed={seed} setSeed={setSeed} />;
    case "seed-ensemble":
      return (
        <SeedEnsembleLesson
          step={step}
          onStepChange={onStepChange}
          seed={seed}
          setSeed={setSeed}
        />
      );
    default:
      return <ClassifierEnsembleLesson step={step} />;
  }
}

export function TrainingWalkthrough({ topic }: { topic: TrainingLesson }) {
  const lesson = TRAINING_LESSONS[topic];
  const [step, setStep] = useState(0);
  const [seed, setSeed] = useState(0);
  // Bumping the key remounts the lesson, so its own controls restart too.
  const [run, setRun] = useState(0);
  function restart() {
    setStep(0);
    setSeed(0);
    setRun(run + 1);
  }
  return (
    <main className="label-lab training-lab">
      <WalkthroughHeader title={lesson.title} intro={lesson.intro} />
      <WalkthroughMenu current="training" />
      <nav
        className="training-links training-lessons"
        aria-label="Training lessons"
      >
        {LESSON_KEYS.map((key) => (
          <Link
            key={key}
            to={`/${key}`}
            aria-current={topic === key ? "page" : undefined}
          >
            {TRAINING_LESSONS[key].title}
          </Link>
        ))}
      </nav>
      <WalkthroughPager
        previous="labelling"
        next="prediction"
        label="Previous and next walkthrough"
      />
      <WhySection>
        <p>{lesson.why}</p>
      </WhySection>
      <ParameterStrip
        items={[
          ["Teaching data", "Synthetic windows and probabilities"],
          ["Training label", "1 = positive forecast window · 0 = negative"],
          ["Pipeline", "Matrix → resample → select → grid CV → bundle"],
        ]}
      />
      <StepControls
        steps={lesson.steps}
        step={step}
        onStepChange={setStep}
        groupLabel="Build steps"
        onRestart={restart}
      />
      <section className="lab-visual" aria-label="Interactive training example">
        <LessonVisual
          key={run}
          topic={topic}
          step={step}
          onStepChange={setStep}
          seed={seed}
          setSeed={setSeed}
        />
      </section>
      <ImplementationNotes
        caveat={lesson.caveat}
        implementation={lesson.implementation}
        sources={lesson.sources}
      />
      {topic === "feature-selection" ? <SignificanceExplainer /> : null}
      {topic === "random-under-sampler" ? <SamplingRatioExplainer /> : null}
      <WalkthroughFooter
        previous="labelling"
        next="prediction"
        prompt="Continue through the training pipeline"
        links={LESSON_KEYS.filter((key) => key !== topic).map((key) => ({
          to: `/${key}`,
          label: TRAINING_LESSONS[key].title,
        }))}
      />
    </main>
  );
}
