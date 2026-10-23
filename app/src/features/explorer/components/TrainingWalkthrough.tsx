import { useState } from "react";
import { Link } from "react-router-dom";
import { ArrowLeft, ArrowRight, RotateCcw } from "lucide-react";
import { Button } from "@/components/ui/button";
import { ThemeToggle } from "@/components/common/ThemeToggle";
import {
  FAMILIES,
  retainedNegatives,
  summarize,
  TRAINING_LESSONS,
  type TrainingLesson,
} from "../trainingLessons";
import seedReference from "../../../../../assets/seed-ensemble.jpg";
import classifierReference from "../../../../../assets/classifier-ensemble.jpg";
import "@/styles/labelling.css";
import "@/styles/training.css";
const references = {
  "seed-ensemble": seedReference,
  "classifier-ensemble": classifierReference,
};
export function TrainingWalkthrough({ topic }: { topic: TrainingLesson }) {
  const lesson = TRAINING_LESSONS[topic];
  const [step, setStep] = useState(0);
  const [seed, setSeed] = useState(0);
  const [family, setFamily] = useState(0);
  const [enabled, setEnabled] = useState(FAMILIES.map(() => true));
  const selectedFamily = FAMILIES[family]!;
  const selectedIds = retainedNegatives(seed);
  const activeFamilies = FAMILIES.filter((_, i) => enabled[i]);
  const consensus = summarize(
    activeFamilies.map((item) => summarize(item.probabilities).mean),
  );
  const [heading, copy] = lesson.steps[step]!;
  function restart() {
    setStep(0);
    setSeed(0);
    setFamily(0);
    setEnabled(FAMILIES.map(() => true));
  }
  return (
    <main className="label-lab training-lab">
      <header className="lab-heading">
        <div>
          <Link className="lab-back" to="/">
            <ArrowLeft size={16} /> Back to Pipeline Atlas
          </Link>
          <h1>{lesson.title}</h1>
          <p>{lesson.intro}</p>
        </div>
        <ThemeToggle />
      </header>
      <nav className="training-links" aria-label="Pipeline walkthroughs">
        <Link to="/labelling">Labelling</Link>
        {(Object.keys(TRAINING_LESSONS) as TrainingLesson[]).map((key) => (
          <Link
            key={key}
            to={`/${key}`}
            aria-current={topic === key ? "page" : undefined}
          >
            {TRAINING_LESSONS[key].title}
          </Link>
        ))}
      </nav>
      <section className="training-why">
        <h2>Why use it?</h2>
        <p>{lesson.why}</p>
      </section>
      <div className="lab-parameters">
        <span>
          Teaching data<strong>Synthetic windows and probabilities</strong>
        </span>
        <span>
          Training label
          <strong>1 = positive forecast window · 0 = negative</strong>
        </span>
        <span>
          Pipeline<strong>Resample → fit seeds → bundle families</strong>
        </span>
      </div>
      <section className="lab-operation" aria-label="Walkthrough controls">
        <div className="lab-step-nav" role="group" aria-label="Build steps">
          {lesson.steps.map(([title], i) => (
            <button
              key={title}
              aria-label={`Step ${i + 1}: ${title}`}
              aria-pressed={step === i}
              onClick={() => setStep(i)}
            >
              {i + 1}
            </button>
          ))}
        </div>
        <div className="lab-operation-copy" aria-live="polite">
          <h2>{heading}</h2>
          <p>{copy}</p>
        </div>
        <div className="lab-step-actions">
          <Button
            variant="ghost"
            disabled={step === 0}
            onClick={() => setStep(step - 1)}
          >
            Previous
          </Button>
          <Button
            variant="outline"
            disabled={step === 3}
            onClick={() => setStep(step + 1)}
          >
            Next step <ArrowRight size={15} />
          </Button>
          <Button
            variant="ghost"
            size="icon"
            aria-label="Restart walkthrough"
            onClick={restart}
          >
            <RotateCcw size={16} />
          </Button>
        </div>
      </section>
      <section className="lab-visual" aria-label="Interactive training example">
        {topic === "random-under-sampler" ? (
          <>
            <div className="lab-section-title">
              <h2>
                {step < 2 ? "History before sampling" : "One reproducible draw"}
              </h2>
              <span>54 original windows · 11.1% positive</span>
            </div>
            <div className="lab-options">
              <label>
                Sampling seed{" "}
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
            </div>
            <p>
              Each numbered cell is one feature row. Filled 1 cells are
              positive; outlined 0 cells are negative. Dashed cells are omitted
              from this seed’s training sample.
            </p>
            <div className="training-sample-grid">
              {Array.from({ length: 54 }, (_, i) => {
                const positive = i >= 48;
                const retained = positive || selectedIds.includes(i + 1);
                return (
                  <span
                    key={i}
                    className={`${positive ? "sample-positive" : ""} ${step >= 2 && !retained ? "sample-discarded" : ""}`}
                    title={`id ${i + 1} · label ${positive ? 1 : 0} · ${step >= 2 && !retained ? "omitted" : "retained"}`}
                  >
                    <small>#{i + 1}</small>
                    <strong>{positive ? 1 : 0}</strong>
                  </span>
                );
              })}
            </div>
            <output className="training-result" aria-live="polite">
              {step < 2
                ? "48 negative + 6 positive = 54 windows"
                : "8 negative + 6 positive = 14 retained windows · 40 negatives omitted"}
            </output>
            {step >= 1 && (
              <p className="training-equation">
                Target majority = 6 ÷ 0.75 = 8 · Positive share after RUS = 6 ÷
                14 = 42.9%
              </p>
            )}
            {step >= 2 && (
              <p>
                Retained negative ids: {selectedIds.join(", ")}. All positive
                ids 49–54 remain.
              </p>
            )}
          </>
        ) : topic === "seed-ensemble" ? (
          <>
            <div className="lab-section-title">
              <h2>
                {step === 0
                  ? "One history, three possible training runs"
                  : step < 3
                    ? "Follow each seed into its model"
                    : "One family, three voices"}
              </h2>
              <span>Same classifier family throughout</span>
            </div>
            <div className="lab-options">
              <label>
                Classifier family{" "}
                <select
                  value={family}
                  onChange={(e) => setFamily(Number(e.target.value))}
                >
                  {FAMILIES.map((item, i) => (
                    <option key={item.name} value={i}>
                      {item.name}
                    </option>
                  ))}
                </select>
              </label>
            </div>
            <p>
              Scroll the table horizontally on smaller screens to compare
              selected features and model outputs.
            </p>
            <div
              className="training-table-scroll"
              tabIndex={0}
              role="region"
              aria-label="Seed model details"
            >
              <table className="training-table">
                <caption>
                  Synthetic per-seed example · 6 shared positive windows
                </caption>
                <thead>
                  <tr>
                    <th>Seed</th>
                    <th>Negative ids retained</th>
                    <th>Selected features</th>
                    <th>{step < 3 ? "Model" : "P(positive)"}</th>
                  </tr>
                </thead>
                <tbody>
                  {selectedFamily.probabilities.map((p, i) => (
                    <tr key={i}>
                      <th>{i}</th>
                      <td>
                        {step >= 1
                          ? retainedNegatives(i).join(", ")
                          : "48 available"}
                      </td>
                      <td>
                        {step >= 1
                          ? [
                              ["RSAM mean", "DSAR variance"],
                              ["RSAM mean", "entropy mean"],
                              ["DSAR variance", "entropy mean"],
                            ][i]!.join(" + ")
                          : "Shared feature matrix"}
                      </td>
                      <td>
                        {step >= 3
                          ? `${(p * 100).toFixed(0)}%`
                          : step === 2
                            ? `${selectedFamily.name} · fitted seed ${i}`
                            : "Awaiting fit"}
                      </td>
                    </tr>
                  ))}
                </tbody>
              </table>
            </div>
            <output className="training-result" aria-live="polite">
              {step < 3
                ? `${selectedFamily.name}: 3 seed models → 1 SeedEnsemble`
                : `${selectedFamily.name} mean = ${(summarize(selectedFamily.probabilities).mean * 100).toFixed(1)}% · seed disagreement (SD) = ${(summarize(selectedFamily.probabilities).std * 100).toFixed(1)} percentage points`}
            </output>
            {step === 3 && (
              <p>
                For {selectedFamily.name}: (
                {selectedFamily.probabilities
                  .map((p) => (p * 100).toFixed(0))
                  .join("% + ")}
                %) ÷ 3. These example probabilities describe the same future
                window, after fitting.
              </p>
            )}
          </>
        ) : (
          <>
            <div className="lab-section-title">
              <h2>
                {step < 2
                  ? "A container of classifier families"
                  : "The same future window, seen by each family"}
              </h2>
              <span>3 seeds per family · equal family weight</span>
            </div>
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
                        : `Mean P(positive) = ${(summarize(item.probabilities).mean * 100).toFixed(1)}%`}
                  </span>
                </label>
              ))}
            </div>
            <p>
              At least one family stays enabled. Selection changes only this
              synthetic calculation.
            </p>
            <output className="training-result" aria-live="polite">
              {step < 3
                ? `${activeFamilies.length} seed ensembles → 1 ClassifierEnsemble`
                : `Consensus probability = ${(consensus.mean * 100).toFixed(1)}% · family disagreement (SD) = ${(consensus.std * 100).toFixed(1)} percentage points`}
            </output>
            {step === 3 && (
              <p className="training-equation">
                (
                {activeFamilies
                  .map((item) =>
                    (summarize(item.probabilities).mean * 100).toFixed(1),
                  )
                  .join("% + ")}
                %) ÷ {activeFamilies.length} families.{" "}
                {activeFamilies.length === 1 &&
                  "With one family, cross-family SD is zero; this does not imply certainty."}
              </p>
            )}
          </>
        )}
      </section>
      <section className="training-notes">
        <div>
          <h2>What this does—and what it cannot fix</h2>
          <p>{lesson.caveat}</p>
        </div>
        <div>
          <h2>How this repository builds it</h2>
          <p>{lesson.implementation}</p>
          <ul>
            {lesson.sources.map((source) => (
              <li key={source}>
                <code>{source}</code>
              </li>
            ))}
          </ul>
        </div>
      </section>
      {topic === "random-under-sampler" ? (
        <section
          className="training-why"
          aria-labelledby="sampling-ratio-heading"
        >
          <h2 id="sampling-ratio-heading">
            What does sampling_strategy = 0.75 mean?
          </h2>
          <p>
            In imbalanced-learn’s RandomUnderSampler, the ratio is the minority
            count divided by the majority count after resampling. Keep every
            eruption-positive window and randomly select fewer non-eruption
            windows. A value of 0.75 describes the ratio between those counts;
            it does not mean a 75/25 class split or a train/test split.
          </p>
          <div className="training-table-scroll">
            <table className="training-table">
              <caption>
                Example training data · sampling_strategy = 0.75
              </caption>
              <thead>
                <tr>
                  <th>Class</th>
                  <th>Before</th>
                  <th>After</th>
                </tr>
              </thead>
              <tbody>
                <tr>
                  <th>Eruption-positive (minority)</th>
                  <td>300</td>
                  <td>All 300 retained</td>
                </tr>
                <tr>
                  <th>Non-eruption (majority)</th>
                  <td>2,000</td>
                  <td>400 randomly selected</td>
                </tr>
              </tbody>
            </table>
          </div>
          <p className="training-equation">
            Majority target = 300 ÷ 0.75 = 400 · Ratio = 300 ÷ 400 = 0.75
          </p>
          <p>
            The 700 retained windows are 42.9% positive and 57.1% negative.
            Sampling the minority too would discard already scarce positive
            examples. Different seeds can change the negative selection while
            keeping all positives. These are labelled windows, not separate
            eruption events. The interactive grid above uses the same ratio: 6
            positives and 8 negatives.
          </p>
        </section>
      ) : (
        <details className="training-reference">
          <summary>View the supplied {lesson.title} reference</summary>
          <p>
            Original project illustration. The interactive example above uses
            corrected, internally consistent synthetic counts and follows the
            implementation notes.
          </p>
          <img
            src={references[topic]}
            alt={`Original ${lesson.title} reference showing the training and ensemble stages`}
            loading="lazy"
            width="1280"
            height="720"
          />
        </details>
      )}
      <footer className="training-footer">
        <p>Continue through the training pipeline</p>
        <nav className="training-links" aria-label="Continue learning">
          <Link to="/labelling">Labelling</Link>
          {(Object.keys(TRAINING_LESSONS) as TrainingLesson[])
            .filter((key) => key !== topic)
            .map((key) => (
              <Link key={key} to={`/${key}`}>
                {TRAINING_LESSONS[key].title} <ArrowRight size={14} />
              </Link>
            ))}
        </nav>
      </footer>
    </main>
  );
}
