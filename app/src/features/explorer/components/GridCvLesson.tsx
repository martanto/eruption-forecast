import { useState } from "react";
import { CodeSnippet } from "@/components/lab/CodeSnippet";
import { DataTable } from "@/components/lab/DataTable";
import { LabSectionTitle } from "@/components/lab/LabSectionTitle";
import { ResultOutput } from "@/components/lab/ResultOutput";
import { SeedSelect, SelectField } from "@/components/lab/SelectField";
import { StepButtons } from "@/components/lab/StepControls";
import {
  CV_ROWS,
  CV_STRATEGIES,
  GRID,
  gridScores,
  type CvStrategy,
} from "../trainingFixtures";
import { SampleCell } from "./SampleCell";

export function GridCvLesson({
  step,
  seed,
  setSeed,
}: {
  step: number;
  seed: number;
  setSeed: (seed: number) => void;
}) {
  const [strategy, setStrategy] = useState<CvStrategy>("shuffle-stratified");
  const [fold, setFold] = useState(0);
  const cv = CV_STRATEGIES[strategy];
  const test: readonly number[] = cv.folds[fold]!;
  const testPositives = test.filter((id) => id > 8).length;
  const scores = gridScores(seed);
  const best = scores.indexOf(Math.max(...scores));
  return (
    <>
      <LabSectionTitle
        title={
          step === 0
            ? "Fourteen retained rows, six candidate hyperparameter combinations"
            : step === 1
              ? "Hold out a subset of rows in each fold"
              : step === 2
                ? "Score each combination across the folds"
                : "Refit the best combination on all retained rows"
        }
        aside="8 negative + 6 positive rows · 3 folds · illustrative scores"
      />
      <div className="lab-options">
        <SeedSelect value={seed} onChange={setSeed} />
        <SelectField
          label="CV strategy"
          value={strategy}
          options={(Object.keys(CV_STRATEGIES) as CvStrategy[]).map((key) => ({
            value: key,
            label: key,
          }))}
          onChange={setStrategy}
        />
      </div>
      <div className="cv-cells" aria-label="Retained rows and fold membership">
        {Array.from({ length: CV_ROWS }, (_, i) => {
          const id = i + 1;
          const positive = id > 8;
          const role =
            step === 1 || step === 2 ? (test.includes(id) ? "test" : "train") : undefined;
          return (
            <SampleCell
              key={id}
              id={id}
              positive={positive}
              role={role}
              title={`Row ${id} · label ${positive ? 1 : 0}${role ? ` · ${role}` : ""}`}
            />
          );
        })}
      </div>
      {(step === 1 || step === 2) && (
        <div className="cv-folds">
          <StepButtons
            labels={cv.folds.map((_, i) => `Fold ${i + 1}`)}
            current={fold}
            onSelect={setFold}
            groupLabel="CV fold"
          />
          <p>
            <strong>{cv.name}</strong> · fold {fold + 1}: {test.length} held-out
            rows, {testPositives} positive. {cv.note}
          </p>
        </div>
      )}
      {step >= 2 && (
        <DataTable
          label="Grid search scores"
          className="cv-table"
          caption="Mean balanced_accuracy over 3 folds · RandomForest grid reduced to two dimensions"
          headers={["n_estimators", "max_depth", "Mean score"]}
        >
          {GRID.map((combo, i) => (
            <tr key={i} data-best={step === 3 && i === best ? true : undefined}>
              <td>{combo.n_estimators}</td>
              <td>{combo.max_depth}</td>
              <td>
                {scores[i]!.toFixed(3)}
                {step === 3 && i === best ? " ← best" : ""}
              </td>
            </tr>
          ))}
        </DataTable>
      )}
      <ResultOutput>
        {step === 0
          ? `${GRID.length} settings × 3 folds = ${GRID.length * 3} fits for seed ${seed}`
          : step === 1
            ? `Fold ${fold + 1}: ${CV_ROWS - test.length} train · ${test.length} test`
            : step === 2
              ? `${GRID.length * 3} fits scored → ${GRID.length} mean scores`
              : `Best: n_estimators=${GRID[best]!.n_estimators}, max_depth=${GRID[best]!.max_depth} → models/${String(seed).padStart(5, "0")}.pkl`}
      </ResultOutput>
      {step === 3 && (
        <p>
          GridSearchCV refits that setting on all {CV_ROWS} rows. The fitted{" "}
          <CodeSnippet inline code="best_estimator_" /> is this seed&apos;s model; the other five
          settings are discarded.
        </p>
      )}
    </>
  );
}
