import { useState } from "react";
import { LabSectionTitle } from "@/components/lab/LabSectionTitle";
import { ResultOutput } from "@/components/lab/ResultOutput";
import { SeedSelect, SelectField } from "@/components/lab/SelectField";
import {
  FDR_ILLUSTRATIVE_CUTOFF,
  FEATURE_CANDIDATES,
  scoreFeatures,
} from "../trainingFixtures";
import { retainedNegatives } from "../trainingLessons";

type Method = "tsfresh" | "random_forest";
const METHODS: readonly { value: Method; label: string }[] = [
  { value: "tsfresh", label: "tsfresh (BY-corrected p-values)" },
  { value: "random_forest", label: "random_forest (permutation importance)" },
];

export function FeatureSelectionLesson({
  step,
  seed,
  setSeed,
}: {
  step: number;
  seed: number;
  setSeed: (seed: number) => void;
}) {
  const [method, setMethod] = useState<Method>("tsfresh");
  const [topN, setTopN] = useState(4);
  const ranked = scoreFeatures(seed, method);
  const byName = new Map(ranked.map((feature) => [feature.name, feature]));
  // Before ranking, show the matrix order; afterwards the sorted order.
  const rows =
    step < 2 ? FEATURE_CANDIDATES.map((f) => byName.get(f.name)!) : ranked;
  const kept = ranked.filter((feature) => feature.rank <= topN);
  return (
    <>
      <LabSectionTitle
        title={
          step === 0
            ? "Candidate features from the resampled rows"
            : step === 1
              ? "Score each feature against the label"
              : step === 2
                ? `Keep the top ${topN}`
                : "This seed's saved feature list"
        }
        aside="8 teaching features · a real matrix has thousands"
      />
      <div className="lab-options">
        <SeedSelect value={seed} onChange={setSeed} />
        <SelectField
          label="Method"
          value={method}
          options={METHODS}
          onChange={setMethod}
        />
        <label>
          top_n{" "}
          <input
            type="range"
            min={1}
            max={8}
            value={topN}
            onChange={(e) => setTopN(Number(e.target.value))}
          />{" "}
          {topN}
        </label>
      </div>
      <p className="fs-sample" aria-live="polite">
        <b>Seed {seed} training sample:</b> all 6 positive windows plus
        non-eruptive windows{" "}
        {retainedNegatives(seed)
          .map((id) => `#${id}`)
          .join(", ")}{" "}
        (8 of 48). A different undersampled draw changes which features
        separate the classes, so it changes the scores below.
      </p>
      <ol className="fs-list" data-scored={step >= 1} aria-label="Feature ranking example">
        {rows.map((feature) => {
          const isKept = feature.rank <= topN;
          const width =
            method === "tsfresh"
              ? Math.min(100, (feature.score / 7) * 100)
              : Math.min(100, (feature.score / 0.3) * 100);
          return (
            <li
              key={feature.name}
              className="fs-row"
              data-kept={step >= 2 ? isKept : undefined}
            >
              <span className="fs-rank">{step >= 2 ? feature.rank : "·"}</span>
              <code>{feature.name}</code>
              <span className="fs-bar" aria-hidden="true">
                <i style={{ width: step >= 1 ? `${width}%` : "0%" }} />
              </span>
              <span className="fs-value">
                {step < 1
                  ? "—"
                  : method === "tsfresh"
                    ? `p = ${feature.p.toExponential(1)}${feature.p <= FDR_ILLUSTRATIVE_CUTOFF ? "" : " (> 0.05)"}`
                    : feature.importance.toFixed(3)}
              </span>
              <span className="fs-status">
                {step >= 2 ? (isKept ? "kept" : "dropped") : ""}
              </span>
            </li>
          );
        })}
      </ol>
      {step === 3 && (
        <p className="training-equation">
          seed/{String(seed).padStart(5, "0")}.csv →{" "}
          {kept.map((feature) => feature.name).join(", ")}
        </p>
      )}
      <ResultOutput>
        {step < 2
          ? "8 candidates → no list yet"
          : `${kept.length} of 8 features kept for seed ${seed}`}
      </ResultOutput>
      <p>
        {method === "tsfresh"
          ? "Illustrative p-values; “(> 0.05)” marks features above the 0.05 significance level. tsfresh applies the Benjamini–Yekutieli correction, which is stricter than this uncorrected cut."
          : "Illustrative permutation importances: the decrease in the probe classifier’s score when that feature’s values are shuffled."}{" "}
        The ranking changes with the seed because each seed selects features
        on a different undersampled training set. The values here are set by
        hand and exaggerated so the difference is visible; a strong feature in
        real data usually moves less.
      </p>
    </>
  );
}
