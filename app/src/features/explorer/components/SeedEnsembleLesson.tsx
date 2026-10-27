import { useState } from "react";
import { ArrowRight } from "lucide-react";
import { DataTable } from "@/components/lab/DataTable";
import { LabSectionTitle } from "@/components/lab/LabSectionTitle";
import { MathExpr } from "@/components/lab/MathExpr";
import { ResultOutput } from "@/components/lab/ResultOutput";
import { SelectField } from "@/components/lab/SelectField";
import { fixed, percent } from "@/lib/format";
import { FAMILIES, retainedNegatives, summarize } from "../trainingLessons";
import { SeedWorkflowDiagram } from "./SeedWorkflowDiagram";

const POSITIVE_IDS = [49, 50, 51, 52, 53, 54];
const SEED_FEATURES = [
  ["RSAM mean", "DSAR variance"],
  ["RSAM mean", "entropy mean"],
  ["DSAR variance", "entropy mean"],
];

export function SeedEnsembleLesson({
  step,
  onStepChange,
  seed,
  setSeed,
}: {
  step: number;
  onStepChange: (step: number) => void;
  seed: number;
  setSeed: (seed: number) => void;
}) {
  const [family, setFamily] = useState(0);
  const selectedFamily = FAMILIES[family]!;
  const selectedIds = retainedNegatives(seed);
  const { mean, std } = summarize(selectedFamily.probabilities);
  return (
    <>
      <LabSectionTitle
        title={
          step === 0
            ? "One training record, three random seeds"
            : step < 3
              ? "Trace each seed to its fitted model"
              : "One classifier family, three seed models"
        }
        aside="Same classifier family throughout"
      />
      <div className="lab-options">
        <SelectField
          label="Classifier family"
          value={family}
          options={FAMILIES.map((item, i) => ({ value: i, label: item.name }))}
          onChange={setFamily}
        />
      </div>
      <SeedWorkflowDiagram step={step} onStepChange={onStepChange} />
      <div className="seed-flow" aria-label="Shared history to seed ensemble">
        <div className="seed-history">
          <h3>Shared training history</h3>
          <p>
            48 negative + 6 positive windows · one feature matrix. Select a seed
            below to trace its draw.
          </p>
          <div className="seed-history-cells" aria-hidden="true">
            {Array.from({ length: 54 }, (_, i) => (
              <span
                key={i}
                className={`${i >= 48 ? "sample-positive" : ""} ${step >= 1 && i < 48 && !selectedIds.includes(i + 1) ? "seed-omitted" : ""}`}
                title={`Window ${i + 1}`}
              >
                {i >= 48 ? 1 : 0}
              </span>
            ))}
          </div>
        </div>
        <p className="seed-flow-caption">
          Same history → a different reproducible draw for each seed
        </p>
        <div className="seed-lanes">
          {selectedFamily.probabilities.map((p, i) => (
            <section
              className="seed-lane"
              data-selected={seed === i}
              key={i}
              aria-label={`Seed ${i} training path`}
            >
              <button
                className="seed-select"
                aria-pressed={seed === i}
                onClick={() => setSeed(i)}
              >
                Trace seed {i}
              </button>
              <p>
                {step === 0
                  ? "Starts with all 54 windows"
                  : "8 negatives + all 6 positives"}
              </p>
              <div
                className="seed-draw"
                aria-label={
                  step === 0 ? "Sampling pending" : "Retained window ids"
                }
              >
                {step === 0 ? (
                  <span>Shared rows and labels</span>
                ) : (
                  [...retainedNegatives(i), ...POSITIVE_IDS].map((id) => (
                    <span
                      key={id}
                      className={id >= 49 ? "sample-positive" : ""}
                      title={`Window ${id} · label ${id >= 49 ? 1 : 0}`}
                    >
                      #{id}
                    </span>
                  ))
                )}
              </div>
              <div className="seed-model">
                <ArrowRight size={18} aria-hidden="true" />
                <strong>
                  {step >= 2
                    ? `${selectedFamily.name} · fitted model ${i}`
                    : step === 1
                      ? "Select features → fit"
                      : "Awaiting sampling"}
                </strong>
              </div>
              {step === 3 && (
                <div className="seed-probability">
                  <span>P(positive)</span>
                  <strong>{percent(p, 0)}</strong>
                  <meter
                    min={0}
                    max={1}
                    value={p}
                    aria-label={`Seed ${i} positive probability`}
                  />
                </div>
              )}
            </section>
          ))}
        </div>
        <div className="seed-merge">
          <strong>{selectedFamily.name} SeedEnsemble</strong>
          <span>
            {step === 3
              ? `Same future window → mean ${percent(mean)}`
              : "Bundle three estimators with their own feature lists"}
          </span>
        </div>
      </div>
      <details className="seed-details">
        <summary>Inspect retained ids and selected features</summary>
        <DataTable
          label="Seed model details"
          caption="Synthetic per-seed example · 6 shared positive windows"
          headers={[
            "Seed",
            "Negative ids retained",
            "Selected features",
            step < 3 ? "Model" : "P(positive)",
          ]}
        >
          {selectedFamily.probabilities.map((p, i) => (
            <tr key={i}>
              <th>{i}</th>
              <td>
                {step >= 1 ? retainedNegatives(i).join(", ") : "48 available"}
              </td>
              <td>
                {step >= 1
                  ? SEED_FEATURES[i]!.join(" + ")
                  : "Shared feature matrix"}
              </td>
              <td>
                {step >= 3
                  ? percent(p, 0)
                  : step === 2
                    ? `${selectedFamily.name} · fitted seed ${i}`
                    : "Awaiting fit"}
              </td>
            </tr>
          ))}
        </DataTable>
      </details>
      <ResultOutput>
        {step < 3
          ? `${selectedFamily.name}: 3 seed models → 1 SeedEnsemble`
          : `${selectedFamily.name} mean = ${percent(mean)} · inter-seed standard deviation (SD) = ${fixed(std * 100, 1)} percentage points`}
      </ResultOutput>
      {step === 3 && (
        <p>
          For {selectedFamily.name}:{" "}
          <MathExpr
            tex={String.raw`\bar{P} = \frac{${selectedFamily.probabilities
              .map((p) => percent(p, 0).replace("%", String.raw`\%`))
              .join(" + ")}}{3}`}
          />
          . These example probabilities describe the same future window, after
          model fitting.
        </p>
      )}
    </>
  );
}
