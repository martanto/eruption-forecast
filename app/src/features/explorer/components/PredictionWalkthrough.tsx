import { useState } from "react";
import { Link, useSearchParams } from "react-router-dom";
import { ArrowLeft, RotateCcw } from "lucide-react";
import { Button } from "@/components/ui/button";
import { ThemeToggle } from "@/components/common/ThemeToggle";
import { RECORDED_FORECAST } from "../forecastDataset";
import { TRAINING_LESSONS, type TrainingLesson } from "../trainingLessons";
import {
  CSV_FIELDS,
  DEFAULT_THRESHOLD,
  DAY_TO_FORECAST,
  daysBetween,
  eruptionsBetween,
  FEATURE_SOURCES,
  initialScenarios,
  OUTPUT_FILES,
  PREDICTION_COPY,
  PREDICTION_STEPS,
  windowEstimate,
  type FeatureSource,
  type RunScenario,
} from "../predictionLessons";
import { ForecastExplorer } from "./ForecastExplorer";
import { RocSimulation } from "./RocSimulation";
import { PredictionWorkflowDiagram } from "./PredictionWorkflowDiagram";
import { ScenarioRunner } from "./ScenarioRunner";
import { WalkthroughMenu, WalkthroughPager } from "./WalkthroughNav";
import "@/styles/labelling.css";
import "@/styles/training.css";
import "@/styles/prediction.css";

const CHAPTERS = ["One forecast", "Several scenarios"] as const;
const MODEL_INPUTS = [
  [
    "ClassifierEnsemble object",
    "A fitted instance from TrainingModel.fit(), used as it is.",
  ],
  [
    "ClassifierEnsemble.json or .pkl",
    "The bundle written to training/classifiers/ClassifierEnsemble_{CVName}.",
  ],
  [
    "SeedEnsemble_*.pkl",
    "One classifier family; it is wrapped into a single-family container.",
  ],
  [
    "trained-model__*.json",
    "A per-classifier registry of seed models and feature lists. Legacy .csv registries still load.",
  ],
] as const;

export function PredictionWalkthrough() {
  const [params, setParams] = useSearchParams();
  const [step, setStep] = useState(0);
  const [threshold, setThreshold] = useState(DEFAULT_THRESHOLD);
  const [source, setSource] = useState<FeatureSource>("all");
  const [modelInput, setModelInput] = useState(0);
  const [scenarios, setScenarios] = useState<RunScenario[]>(initialScenarios);
  const chapter = params.get("part") === "scenarios" ? 1 : 0;
  const selectedId = params.get("scenario") ?? scenarios.at(-1)!.id;
  const selected = scenarios.find((s) => s.id === selectedId) ?? scenarios[0]!;
  const [heading, copy] = PREDICTION_STEPS[step]!;
  const lessons = Object.keys(TRAINING_LESSONS) as TrainingLesson[];

  function update(next: Record<string, string>) {
    const merged = new URLSearchParams(params);
    Object.entries(next).forEach(([key, value]) => merged.set(key, value));
    setParams(merged, { replace: true });
  }

  return (
    <main className="label-lab training-lab prediction-lab">
      <header className="lab-heading">
        <div>
          <Link className="lab-back" to="/">
            <ArrowLeft size={16} /> Back to Pipeline Atlas
          </Link>
          <h1>{PREDICTION_COPY.title}</h1>
          <p>{PREDICTION_COPY.intro}</p>
        </div>
        <ThemeToggle />
      </header>
      <WalkthroughMenu current="prediction" />
      <WalkthroughPager previous="training" label="Previous walkthrough" />
      <section className="training-why">
        <h2>Why use it?</h2>
        <p>{PREDICTION_COPY.why}</p>
      </section>
      <div className="lab-parameters">
        <span>
          Window size<strong>{DAY_TO_FORECAST} days (day_to_forecast)</strong>
        </span>
        <span>
          Window step<strong>10 minutes</strong>
        </span>
        <span>
          Alert threshold<strong>0.7, drawn on the plot</strong>
        </span>
        <span>
          Recorded data<strong>OJN scenario 5.2, 1–22 Aug 2025 · UTC</strong>
        </span>
      </div>
      <nav
        className="lab-chapters"
        aria-label="Prediction walkthrough chapters"
      >
        {CHAPTERS.map((title, index) => (
          <button
            key={title}
            aria-current={chapter === index ? "step" : undefined}
            onClick={() =>
              update({ part: index === 1 ? "scenarios" : "forecast" })
            }
          >
            <span>{index + 1}</span>
            {title}
          </button>
        ))}
      </nav>

      {chapter === 0 ? (
        <>
          <section className="lab-operation" aria-label="Walkthrough controls">
            <div
              className="lab-step-nav"
              role="group"
              aria-label="Prediction steps"
            >
              {PREDICTION_STEPS.map(([title], i) => (
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
                disabled={step === PREDICTION_STEPS.length - 1}
                onClick={() => setStep(step + 1)}
              >
                Next step
              </Button>
              <Button
                variant="ghost"
                size="icon"
                aria-label="Restart walkthrough"
                onClick={() => setStep(0)}
              >
                <RotateCcw size={16} />
              </Button>
            </div>
          </section>
          <section
            className="lab-visual"
            aria-label="Interactive prediction example"
          >
            <PredictionWorkflowDiagram step={step} onStepChange={setStep} />

            {step === 0 && (
              <>
                <div className="lab-section-title">
                  <h2>A grid of windows nobody has labelled</h2>
                  <span>
                    {selected.name} · {selected.predictionStart} to{" "}
                    {selected.predictionEnd}
                  </span>
                </div>
                <div className="lab-options">
                  <label>
                    Scenario{" "}
                    <select
                      value={selected.id}
                      onChange={(event) =>
                        update({ scenario: event.target.value })
                      }
                    >
                      {scenarios.map((s) => (
                        <option key={s.id} value={s.id}>
                          {s.name}
                        </option>
                      ))}
                    </select>
                  </label>
                </div>
                <div
                  className="pw-grid"
                  role="img"
                  aria-label="Five consecutive windows, each two days long and offset by ten minutes"
                >
                  {[0, 1, 2, 3, 4].map((i) => (
                    <span key={i} style={{ marginLeft: `${i * 2.2}%` }}>
                      window {i + 1}
                    </span>
                  ))}
                  <small>
                    Each window slides 10 minutes and shares all but 10 minutes
                    with the one before. Not to scale.
                  </small>
                </div>
                <output className="training-result" aria-live="polite">
                  {daysBetween(
                    selected.predictionStart,
                    selected.predictionEnd,
                  )}{" "}
                  days × 144 per day ≈{" "}
                  {windowEstimate(selected).toLocaleString()} windows
                </output>
                <p>
                  {
                    eruptionsBetween(
                      selected.predictionStart,
                      selected.predictionEnd,
                    ).length
                  }{" "}
                  known eruption dates fall in this forecast period. Prediction
                  never reads them: they matter only later, when evaluating. The
                  count is calendar days times windows per day; the lead-in
                  before the first window is extra.
                </p>
              </>
            )}

            {step === 1 && (
              <>
                <div className="lab-section-title">
                  <h2>Same features, no verdict</h2>
                  <span>Prediction mode</span>
                </div>
                <div className="pw-compare">
                  <section>
                    <h3>Training</h3>
                    <p>
                      Windows + labels → tsfresh → keep features that separate
                      positive from negative windows.
                    </p>
                  </section>
                  <section>
                    <h3>Prediction</h3>
                    <p>
                      Windows only → tsfresh → keep every column the ensemble
                      might read. There is no label to test against.
                    </p>
                  </section>
                </div>
                <fieldset className="pw-radios">
                  <legend>use_features_from</legend>
                  {FEATURE_SOURCES.map((item) => (
                    <label key={item.value} data-on={source === item.value}>
                      <input
                        type="radio"
                        name="features-source"
                        checked={source === item.value}
                        onChange={() => setSource(item.value)}
                      />
                      <span>
                        <strong>{item.label}</strong>
                        {item.effect}
                      </span>
                    </label>
                  ))}
                </fieldset>
                <output className="training-result" aria-live="polite">
                  {FEATURE_SOURCES.find((item) => item.value === source)!.cost}
                </output>
              </>
            )}

            {step === 2 && (
              <>
                <div className="lab-section-title">
                  <h2>Whatever you saved, one container</h2>
                  <span>ClassifierEnsemble.from_any</span>
                </div>
                <fieldset className="pw-radios">
                  <legend>The model argument</legend>
                  {MODEL_INPUTS.map(([name, detail], i) => (
                    <label key={name} data-on={modelInput === i}>
                      <input
                        type="radio"
                        name="model-input"
                        checked={modelInput === i}
                        onChange={() => setModelInput(i)}
                      />
                      <span>
                        <strong>{name}</strong>
                        {detail}
                      </span>
                    </label>
                  ))}
                </fieldset>
                <output className="training-result" aria-live="polite">
                  {MODEL_INPUTS[modelInput]![0]} → ClassifierEnsemble
                </output>
                <p>
                  Each seed model keeps its own feature list, so the window
                  features are subset per seed at inference. The cache identity
                  includes the training hash: retrain, and every forecast built
                  on the old ensemble is stale.
                </p>
              </>
            )}

            {step >= 3 && (
              <>
                <div className="lab-section-title">
                  <h2>
                    {step === 3
                      ? "Read one window of the recorded forecast"
                      : "The curve forecast() leaves behind"}
                  </h2>
                  <span>OJN scenario 5.2 · eruptions 2 and 18 Aug</span>
                </div>
                <ForecastExplorer
                  dataset={RECORDED_FORECAST}
                  revealRows={step === 3}
                  threshold={threshold}
                  onThresholdChange={setThreshold}
                />
                {step === 4 && (
                  <RocSimulation
                    dataset={RECORDED_FORECAST}
                    threshold={threshold}
                    onThresholdChange={setThreshold}
                  />
                )}
              </>
            )}

            {step === 4 && (
              <div className="pw-outputs">
                <div
                  className="training-table-scroll"
                  tabIndex={0}
                  role="region"
                  aria-label="Files written by a prediction run"
                >
                  <table className="training-table">
                    <caption>Files under {"{output_dir}/{nslc}/"}</caption>
                    <thead>
                      <tr>
                        <th>Path</th>
                        <th>Holds</th>
                      </tr>
                    </thead>
                    <tbody>
                      {OUTPUT_FILES.map(([path, holds]) => (
                        <tr key={path}>
                          <th>
                            <code>{path}</code>
                          </th>
                          <td>{holds}</td>
                        </tr>
                      ))}
                    </tbody>
                  </table>
                </div>
                <div
                  className="training-table-scroll"
                  tabIndex={0}
                  role="region"
                  aria-label="Forecast CSV columns"
                >
                  <table className="training-table">
                    <caption>Columns in the forecast CSV</caption>
                    <thead>
                      <tr>
                        <th>Column</th>
                        <th>Meaning</th>
                      </tr>
                    </thead>
                    <tbody>
                      {CSV_FIELDS.map(([field, meaning]) => (
                        <tr key={field}>
                          <th>
                            <code>{field}</code>
                          </th>
                          <td>{meaning}</td>
                        </tr>
                      ))}
                    </tbody>
                  </table>
                </div>
              </div>
            )}
          </section>
        </>
      ) : (
        <section
          className="lab-visual"
          aria-label="Interactive scenario runner"
        >
          <div className="lab-section-title">
            <h2>Run the same forecast under several scenarios</h2>
            <span>ForecastModelScenario · forecast-scenario.py</span>
          </div>
          <p>
            One ForecastModel calculates tremor once. Each scenario then trains
            on its own period, predicts the next, evaluates and explains. Later
            training periods include more eruptions, so scenarios show how the
            forecast changes as history grows.
          </p>
          <ScenarioRunner
            scenarios={scenarios}
            onScenariosChange={setScenarios}
            selectedId={selected.id}
            onSelect={(id) => update({ scenario: id, part: "scenarios" })}
          />
        </section>
      )}

      <section className="training-notes">
        <div>
          <h2>What this does—and what it cannot fix</h2>
          <p>{PREDICTION_COPY.caveat}</p>
        </div>
        <div>
          <h2>How this repository builds it</h2>
          <p>{PREDICTION_COPY.implementation}</p>
          <ul>
            {PREDICTION_COPY.sources.map((item) => (
              <li key={item}>
                <code>{item}</code>
              </li>
            ))}
          </ul>
        </div>
      </section>
      <footer className="training-footer">
        <WalkthroughPager
          previous="training"
          label="Previous walkthrough, bottom"
        />
        <p>Walk the pipeline again</p>
        <nav className="training-links" aria-label="Continue learning">
          <Link to="/labelling">Labelling</Link>
          <Link to={`/${lessons[0]}`}>Training</Link>
        </nav>
      </footer>
    </main>
  );
}
