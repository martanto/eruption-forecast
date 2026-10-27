import { useState } from "react";
import {
  CV_ROWS,
  CV_STRATEGIES,
  GRID,
  gridScores,
  type CvStrategy,
} from "../trainingFixtures";

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
      <div className="lab-section-title">
        <h2>
          {step === 0
            ? "Fourteen retained rows, six candidate settings"
            : step === 1
              ? "Hold some rows out in each fold"
              : step === 2
                ? "Score each setting across the folds"
                : "Refit the winner on every retained row"}
        </h2>
        <span>8 negative + 6 positive rows · 3 folds · illustrative scores</span>
      </div>
      <div className="lab-options">
        <label>
          Seed{" "}
          <select
            value={seed}
            onChange={(e) => setSeed(Number(e.target.value))}
          >
            {[0, 1, 2].map((i) => (
              <option key={i} value={i}>
                Seed {i}
              </option>
            ))}
          </select>
        </label>
        <label>
          CV strategy{" "}
          <select
            value={strategy}
            onChange={(e) => setStrategy(e.target.value as CvStrategy)}
          >
            {(Object.keys(CV_STRATEGIES) as CvStrategy[]).map((key) => (
              <option key={key} value={key}>
                {key}
              </option>
            ))}
          </select>
        </label>
      </div>
      <div className="cv-cells" aria-label="Retained rows and fold membership">
        {Array.from({ length: CV_ROWS }, (_, i) => {
          const id = i + 1;
          const positive = id > 8;
          const role =
            step === 1 || step === 2 ? (test.includes(id) ? "test" : "train") : "";
          return (
            <span
              key={id}
              className={positive ? "sample-positive" : ""}
              data-role={role || undefined}
              title={`Row ${id} · label ${positive ? 1 : 0}${role ? ` · ${role}` : ""}`}
            >
              <small>#{id}</small>
              <strong>{positive ? 1 : 0}</strong>
            </span>
          );
        })}
      </div>
      {(step === 1 || step === 2) && (
        <div className="cv-folds">
          <div className="lab-step-nav" role="group" aria-label="CV fold">
            {cv.folds.map((_, i) => (
              <button
                key={i}
                aria-label={`Fold ${i + 1}`}
                aria-pressed={fold === i}
                onClick={() => setFold(i)}
              >
                {i + 1}
              </button>
            ))}
          </div>
          <p>
            <strong>{cv.name}</strong> · fold {fold + 1}: {test.length} test
            rows, {testPositives} positive. {cv.note}
          </p>
        </div>
      )}
      {step >= 2 && (
        <div className="training-table-scroll">
          <table className="training-table cv-table">
            <caption>
              Mean balanced_accuracy over 3 folds · RandomForest grid reduced to
              two dimensions
            </caption>
            <thead>
              <tr>
                <th>n_estimators</th>
                <th>max_depth</th>
                <th>Mean score</th>
              </tr>
            </thead>
            <tbody>
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
            </tbody>
          </table>
        </div>
      )}
      <output className="training-result" aria-live="polite">
        {step === 0
          ? `${GRID.length} settings × 3 folds = ${GRID.length * 3} fits for seed ${seed}`
          : step === 1
            ? `Fold ${fold + 1}: ${CV_ROWS - test.length} train · ${test.length} test`
            : step === 2
              ? `${GRID.length * 3} fits scored → ${GRID.length} mean scores`
              : `Best: n_estimators=${GRID[best]!.n_estimators}, max_depth=${GRID[best]!.max_depth} → models/${String(seed).padStart(5, "0")}.pkl`}
      </output>
      {step === 3 && (
        <p>
          GridSearchCV refits that setting on all {CV_ROWS} rows. The fitted{" "}
          <code>best_estimator_</code> is this seed&apos;s model; the other five
          settings are discarded.
        </p>
      )}
    </>
  );
}
