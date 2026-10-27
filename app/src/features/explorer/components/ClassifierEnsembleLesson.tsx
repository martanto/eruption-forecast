import { useState } from "react";
import { LabSectionTitle } from "@/components/lab/LabSectionTitle";
import { MathExpr } from "@/components/lab/MathExpr";
import { ResultOutput } from "@/components/lab/ResultOutput";
import { fixed, percent } from "@/lib/format";
import { FAMILIES, summarize } from "../trainingLessons";

const FUTURE_WINDOW_IDS = [55, 56, 57, 58];

export function ClassifierEnsembleLesson({ step }: { step: number }) {
  const [enabled, setEnabled] = useState(FAMILIES.map(() => true));
  const activeFamilies = FAMILIES.filter((_, i) => enabled[i]);
  const consensus = summarize(
    activeFamilies.map((item) => summarize(item.probabilities).mean),
  );
  return (
    <>
      <LabSectionTitle
        title={
          step < 2
            ? "A container of classifier families"
            : "The same future window, evaluated by each classifier family"
        }
        aside="3 seeds per family · equal family weight"
      />
      <p>
        Each family keeps its fitted seed models and selected feature lists.
        Enable or disable a family to observe its contribution to the shared
        container.
      </p>
      <div className="training-family-list">
        {FAMILIES.map((item, i) => (
          <label key={item.name} className="training-family">
            <span>
              <input
                type="checkbox"
                aria-label={`${item.name} SeedEnsemble`}
                checked={enabled[i]}
                disabled={enabled[i] && activeFamilies.length === 1}
                onChange={() =>
                  setEnabled(enabled.map((on, j) => (j === i ? !on : on)))
                }
              />{" "}
              {item.name} SeedEnsemble
            </span>
            <span>
              {step === 0
                ? "3 fitted seed models"
                : step === 1
                  ? "Bundled with its feature lists"
                  : `Mean P(positive) = ${percent(summarize(item.probabilities).mean)}`}
            </span>
          </label>
        ))}
      </div>
      <div className="seed-merge" aria-label="Bundled classifier families">
        <strong>ClassifierEnsemble</strong>
        <span>
          {activeFamilies.map((item) => item.name).join(" + ")} /{" "}
          {activeFamilies.length * 3} fitted seed models
        </span>
        <span>
          {step === 0
            ? "Collect the family objects before bundling."
            : "One container, with each family identity preserved. No additional model is fitted."}
        </span>
      </div>
      <section
        className="classifier-prediction"
        aria-labelledby="future-prediction-heading"
      >
        <h3 id="future-prediction-heading">Model prediction</h3>
        <p>
          Future windows stay outside the training sample. Extract their
          features, then apply the fitted ClassifierEnsemble to each window.
        </p>
        <ol className="classifier-prediction-stages">
          <li>
            <strong>Future feature rows</strong>
            <span>One row per forecast window</span>
            <div
              className="classifier-window-rows"
              aria-label="Example future window ids"
            >
              {FUTURE_WINDOW_IDS.map((id) => (
                <span key={id}>#{id}</span>
              ))}
            </div>
          </li>
          <li>
            <strong>Apply ClassifierEnsemble</strong>
            <span>
              {step < 2
                ? "Ready for future features"
                : "Each seed reads its own feature columns"}
            </span>
            {step >= 2 && (
              <ul>
                {activeFamilies.map((item) => (
                  <li key={item.name}>
                    {item.name}:{" "}
                    {item.probabilities.map((p) => percent(p, 0)).join(" + ")} /
                    3 = {percent(summarize(item.probabilities).mean)}
                  </li>
                ))}
              </ul>
            )}
          </li>
          <li>
            <strong>Forecast results</strong>
            <span>
              {step < 3
                ? "Advance to step 4 to view the consensus for window #55."
                : `Window #55  /  probability ${percent(consensus.mean)}`}
            </span>
            {step === 3 && (
              <span>
                Prediction vote fraction:{" "}
                {percent(
                  summarize(
                    activeFamilies.map(
                      (item) =>
                        item.probabilities.filter((p) => p >= 0.5).length /
                        item.probabilities.length,
                    ),
                  ).mean,
                )}
              </span>
            )}
            <span>Keep per-family results alongside consensus.</span>
          </li>
        </ol>
        <p>
          Probabilities above are synthetic examples for window #55. The
          prediction vote fraction averages seed votes within each family, then
          averages across families, using a 0.5 vote threshold here.
        </p>
      </section>
      <p>
        At least one family stays enabled. Selection changes only this synthetic
        calculation.
      </p>
      <ResultOutput>
        {step < 3
          ? `${activeFamilies.length} seed ensembles → 1 ClassifierEnsemble`
          : `Consensus probability = ${percent(consensus.mean)} · inter-family standard deviation (SD) = ${fixed(consensus.std * 100, 1)} percentage points`}
      </ResultOutput>
      {step === 3 && (
        <p className="training-equation">
          <MathExpr
            tex={String.raw`\bar{P} = \frac{${activeFamilies
              .map((item) =>
                percent(summarize(item.probabilities).mean).replace("%", String.raw`\%`),
              )
              .join(" + ")}}{${activeFamilies.length}}`}
          />{" "}
          (mean over {activeFamilies.length} families).{" "}
          {activeFamilies.length === 1 &&
            "With one family, the inter-family SD is zero; this does not imply certainty."}
        </p>
      )}
    </>
  );
}
