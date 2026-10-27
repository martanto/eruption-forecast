import { useState } from "react";
import { useSearchParams } from "react-router-dom";
import { ChapterNav } from "@/components/lab/ChapterNav";
import { DataTable } from "@/components/lab/DataTable";
import { LabSectionTitle } from "@/components/lab/LabSectionTitle";
import { ParameterStrip } from "@/components/lab/ParameterStrip";
import { ResultOutput } from "@/components/lab/ResultOutput";
import { SelectField } from "@/components/lab/SelectField";
import { StepControls } from "@/components/lab/StepControls";
import { ImplementationNotes, WhySection } from "@/components/lab/WalkthroughNotes";
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
import { RadioCards } from "./RadioCards";
import { ScenarioRunner } from "./ScenarioRunner";
import { WalkthroughHeader } from "./WalkthroughHeader";
import {
  WalkthroughFooter,
  WalkthroughMenu,
  WalkthroughPager,
} from "./WalkthroughNav";
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
  const lessons = Object.keys(TRAINING_LESSONS) as TrainingLesson[];

  function update(next: Record<string, string>) {
    const merged = new URLSearchParams(params);
    Object.entries(next).forEach(([key, value]) => merged.set(key, value));
    setParams(merged, { replace: true });
  }

  return (
    <main className="label-lab training-lab prediction-lab">
      <WalkthroughHeader
        title={PREDICTION_COPY.title}
        intro={PREDICTION_COPY.intro}
      />
      <WalkthroughMenu current="prediction" />
      <WalkthroughPager previous="training" label="Previous walkthrough" />
      <WhySection>
        <p>{PREDICTION_COPY.why}</p>
      </WhySection>
      <ParameterStrip
        items={[
          ["Window size", `${DAY_TO_FORECAST} days (day_to_forecast)`],
          ["Window step", "10 minutes"],
          ["Alert threshold", "0.7, drawn on the plot"],
          ["Recorded data", "OJN scenario 5.2, 1–22 Aug 2025 · UTC"],
        ]}
      />
      <ChapterNav
        titles={CHAPTERS}
        current={chapter}
        onSelect={(index) => update({ part: index === 1 ? "scenarios" : "forecast" })}
        label="Prediction walkthrough chapters"
      />

      {chapter === 0 ? (
        <>
          <StepControls
            steps={PREDICTION_STEPS}
            step={step}
            onStepChange={setStep}
            groupLabel="Prediction steps"
          />
          <section
            className="lab-visual"
            aria-label="Interactive prediction example"
          >
            <PredictionWorkflowDiagram step={step} onStepChange={setStep} />

            {step === 0 && (
              <>
                <LabSectionTitle
                  title="A grid of windows nobody has labelled"
                  aside={`${selected.name} · ${selected.predictionStart} to ${selected.predictionEnd}`}
                />
                <div className="lab-options">
                  <SelectField
                    label="Scenario"
                    value={selected.id}
                    options={scenarios.map((s) => ({ value: s.id, label: s.name }))}
                    onChange={(id) => update({ scenario: id })}
                  />
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
                <ResultOutput>
                  {daysBetween(
                    selected.predictionStart,
                    selected.predictionEnd,
                  )}{" "}
                  days × 144 per day ≈{" "}
                  {windowEstimate(selected).toLocaleString()} windows
                </ResultOutput>
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
                <LabSectionTitle
                  title="Same features, no verdict"
                  aside="Prediction mode"
                />
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
                <RadioCards
                  legend="use_features_from"
                  name="features-source"
                  options={FEATURE_SOURCES.map((item) => ({
                    value: item.value,
                    title: item.label,
                    detail: item.effect,
                  }))}
                  value={source}
                  onChange={setSource}
                />
                <ResultOutput>
                  {FEATURE_SOURCES.find((item) => item.value === source)!.cost}
                </ResultOutput>
              </>
            )}

            {step === 2 && (
              <>
                <LabSectionTitle
                  title="Whatever you saved, one container"
                  aside="ClassifierEnsemble.from_any"
                />
                <RadioCards
                  legend="The model argument"
                  name="model-input"
                  options={MODEL_INPUTS.map(([name, detail], i) => ({
                    value: i,
                    title: name,
                    detail,
                  }))}
                  value={modelInput}
                  onChange={setModelInput}
                />
                <ResultOutput>
                  {MODEL_INPUTS[modelInput]![0]} → ClassifierEnsemble
                </ResultOutput>
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
                <LabSectionTitle
                  title={
                    step === 3
                      ? "Read one window of the recorded forecast"
                      : "The curve forecast() leaves behind"
                  }
                  aside="OJN scenario 5.2 · eruptions 2 and 18 Aug"
                />
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
                <DataTable
                  label="Files written by a prediction run"
                  caption={<>Files under {"{output_dir}/{nslc}/"}</>}
                  headers={["Path", "Holds"]}
                >
                  {OUTPUT_FILES.map(([path, holds]) => (
                    <tr key={path}>
                      <th>
                        <code>{path}</code>
                      </th>
                      <td>{holds}</td>
                    </tr>
                  ))}
                </DataTable>
                <DataTable
                  label="Forecast CSV columns"
                  caption="Columns in the forecast CSV"
                  headers={["Column", "Meaning"]}
                >
                  {CSV_FIELDS.map(([field, meaning]) => (
                    <tr key={field}>
                      <th>
                        <code>{field}</code>
                      </th>
                      <td>{meaning}</td>
                    </tr>
                  ))}
                </DataTable>
              </div>
            )}
          </section>
        </>
      ) : (
        <section
          className="lab-visual"
          aria-label="Interactive scenario runner"
        >
          <LabSectionTitle
            title="Run the same forecast under several scenarios"
            aside="ForecastModelScenario · forecast-scenario.py"
          />
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

      <ImplementationNotes
        caveat={PREDICTION_COPY.caveat}
        implementation={PREDICTION_COPY.implementation}
        sources={PREDICTION_COPY.sources}
      />
      <WalkthroughFooter
        previous="training"
        prompt="Walk the pipeline again"
        links={[
          { to: "/labelling", label: "Labelling" },
          { to: `/${lessons[0]}`, label: "Training" },
        ]}
      />
    </main>
  );
}
