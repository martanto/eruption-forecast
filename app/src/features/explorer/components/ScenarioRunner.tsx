import { useEffect, useState } from "react";
import { Pause, Play, Plus, RotateCcw, StepForward } from "lucide-react";
import { CodeSnippet } from "@/components/lab/CodeSnippet";
import { DataTable } from "@/components/lab/DataTable";
import { ResultOutput } from "@/components/lab/ResultOutput";
import { SelectField } from "@/components/lab/SelectField";
import { Button } from "@/components/ui/button";
import { ERUPTIONS } from "@/features/architecture";
import { shortDate } from "@/lib/time";
import {
  coveredByFirst,
  daysBetween,
  eruptionsBetween,
  FEATURE_SOURCES,
  PHASES,
  RECORDED_SCENARIO_ID,
  scenarioDict,
  TREMOR_END,
  TREMOR_START,
  validateScenario,
  windowEstimate,
  type FeatureSource,
  type RunScenario,
} from "../predictionLessons";

const TICK_MS = 800;
const AXIS_START = Date.parse(`${TREMOR_START}T00:00:00Z`);
const AXIS_END = Date.parse("2025-08-31T00:00:00Z");
const pctOf = (date: string) =>
  ((Date.parse(`${date}T00:00:00Z`) - AXIS_START) / (AXIS_END - AXIS_START)) * 100;
const MONTHS = ["Jan", "Feb", "Mar", "Apr", "May", "Jun", "Jul", "Aug"].map(
  (label, i) => [label, `2025-0${i + 1}-01`] as const,
);

interface Props {
  scenarios: RunScenario[];
  onScenariosChange: (next: RunScenario[]) => void;
  selectedId: string;
  onSelect: (id: string) => void;
}

export function ScenarioRunner({ scenarios, onScenariosChange, selectedId, onSelect }: Props) {
  const [included, setIncluded] = useState<string[]>(() => scenarios.map((s) => s.id));
  const [featuresFrom, setFeaturesFrom] = useState<FeatureSource>("files");
  const [cursor, setCursor] = useState(-1);
  const [playing, setPlaying] = useState(false);
  const [draft, setDraft] = useState({ trainEnd: "2025-07-15", predictionStart: "2025-07-16", predictionEnd: "2025-08-22" });
  const [error, setError] = useState<string | null>(null);

  const run = scenarios.filter((s) => included.includes(s.id));
  // calculate() is cursor 0; every scenario then adds one tick per phase.
  const total = 1 + run.length * PHASES.length;
  const selected = scenarios.find((s) => s.id === selectedId) ?? scenarios[0]!;
  const first = run[0];

  // A finished run stops itself without an effect writing state back.
  const running = playing && cursor < total - 1;
  useEffect(() => {
    if (!running) return;
    const id = window.setTimeout(() => setCursor((c) => c + 1), TICK_MS);
    return () => window.clearTimeout(id);
  }, [running, cursor]);

  const offset = cursor - 1;
  const position =
    cursor < 0
      ? null
      : cursor === 0
        ? ({ calculate: true } as const)
        : {
            scenario: run[Math.floor(offset / PHASES.length)]!,
            phase: PHASES[offset % PHASES.length]!,
          };

  function phaseState(scenarioIndex: number, phaseIndex: number) {
    if (cursor < 0) return "idle";
    const at = 1 + scenarioIndex * PHASES.length + phaseIndex;
    return at < cursor ? "done" : at === cursor ? "current" : "idle";
  }
  function reset() {
    setPlaying(false);
    setCursor(-1);
  }
  function toggle(id: string) {
    reset();
    setIncluded((now) => (now.includes(id) ? now.filter((x) => x !== id) : [...now, id]));
  }
  function addScenario() {
    const problem = validateScenario(draft);
    setError(problem);
    if (problem) return;
    const n = scenarios.filter((s) => s.custom).length + 1;
    const created: RunScenario = {
      id: `custom-${n}`,
      name: `Custom ${n}`,
      trainStart: TREMOR_START,
      trainEnd: draft.trainEnd,
      predictionStart: draft.predictionStart,
      predictionEnd: draft.predictionEnd,
      predictionStepMinutes: 10,
      custom: true,
    };
    onScenariosChange([...scenarios, created]);
    setIncluded((now) => [...now, created.id]);
    onSelect(created.id);
    reset();
  }

  const status = !position
    ? "Press run to replay the loop in forecast-scenario.py."
    : "calculate" in position
      ? "calculate() runs once: tremor for the whole year is shared by every scenario."
      : `${position.scenario.name} · ${position.phase}${
          position.phase === "predict" && featuresFrom === "files"
            ? position.scenario === first
              ? " · extracts features fresh and saves the matrix"
              : " · reuses the matrix from the first scenario"
            : position.phase === "predict"
              ? " · extracts features again"
              : ""
        }`;

  return (
    <div className="psr">
      <fieldset className="psr-pick">
        <legend>Scenarios in this run</legend>
        {scenarios.map((s) => (
          <label key={s.id} className="psr-chip" data-on={included.includes(s.id)}>
            <input type="checkbox" checked={included.includes(s.id)} onChange={() => toggle(s.id)} />
            {s.name}
          </label>
        ))}
      </fieldset>

      <figure className="psr-axis" aria-label="Training and forecast periods for each scenario">
        <div className="psr-axis-head" aria-hidden="true">
          {MONTHS.map(([label, date]) => (
            <span key={label} style={{ left: `${pctOf(date)}%` }}>
              {label}
            </span>
          ))}
        </div>
        <div className="psr-axis-body">
          {MONTHS.map(([label, date]) => (
            <i key={label} className="psr-month" style={{ left: `${pctOf(date)}%` }} />
          ))}
          {ERUPTIONS.map((date) => (
            <i key={date} className="psr-eruption" style={{ left: `${pctOf(date)}%` }} title={`Eruption ${date}`} />
          ))}
          {scenarios.map((s) => (
            <div key={s.id} className="psr-track" data-off={!included.includes(s.id)} data-selected={s.id === selectedId}>
              <button type="button" className="psr-track-name" onClick={() => onSelect(s.id)} aria-pressed={s.id === selectedId}>
                {s.name}
              </button>
              <span className="psr-train" style={{ left: `${pctOf(s.trainStart)}%`, width: `${pctOf(s.trainEnd) - pctOf(s.trainStart)}%` }} />
              <span className="psr-predict" style={{ left: `${pctOf(s.predictionStart)}%`, width: `${pctOf(s.predictionEnd) - pctOf(s.predictionStart)}%` }} />
            </div>
          ))}
        </div>
        <figcaption>
          <span className="psr-key psr-key-train">Training period</span>
          <span className="psr-key psr-key-predict">Forecast period</span>
          <span className="psr-key psr-key-eruption">Eruption date</span>
          Dates are UTC. Select a scenario name to inspect it below.
        </figcaption>
      </figure>

      <div className="lab-options">
        <SelectField
          label="use_features_from"
          value={featuresFrom}
          options={FEATURE_SOURCES.map((item) => ({ value: item.value, label: item.label }))}
          onChange={(next: FeatureSource) => {
            reset();
            setFeaturesFrom(next);
          }}
        />
        <div className="psr-actions">
          <Button
            variant="outline"
            disabled={run.length === 0}
            onClick={() => {
              if (!running && cursor >= total - 1) setCursor(0);
              else if (cursor < 0) setCursor(0);
              setPlaying(!running);
            }}
          >
            {running ? <Pause size={15} /> : <Play size={15} />}
            {running ? "Pause" : cursor >= total - 1 ? "Run again" : "Run scenarios"}
          </Button>
          <Button
            variant="ghost"
            disabled={run.length === 0 || cursor >= total - 1}
            onClick={() => {
              setPlaying(false);
              setCursor((c) => c + 1);
            }}
          >
            <StepForward size={15} /> Next operation
          </Button>
          <Button variant="ghost" size="icon" aria-label="Reset scenario run" onClick={reset}>
            <RotateCcw size={16} />
          </Button>
        </div>
      </div>
      <p className="psr-note">
        {FEATURE_SOURCES.find((item) => item.value === featuresFrom)!.effect}{" "}
        {FEATURE_SOURCES.find((item) => item.value === featuresFrom)!.cost}
      </p>

      <ResultOutput className="psr-status">{status}</ResultOutput>

      <DataTable
        label="Scenario run table"
        className="psr-table"
        caption={`One ForecastModel, ${run.length} scenario${run.length === 1 ? "" : "s"}, same calculated tremor`}
        headers={["Scenario", "Train → forecast", "Eruptions", "≈ windows", ...PHASES]}
      >
        {run.map((s, i) => {
          const inTrain = eruptionsBetween(s.trainStart, s.trainEnd);
          const inForecast = eruptionsBetween(s.predictionStart, s.predictionEnd);
          const reuseProblem = first && featuresFrom === "files" && s !== first && !coveredByFirst(first, s);
          return (
            <tr key={s.id} data-selected={s.id === selectedId}>
              <th>
                <button type="button" className="psr-link" onClick={() => onSelect(s.id)}>
                  {s.name}
                </button>
              </th>
              <td>
                {shortDate(s.trainStart)}–{shortDate(s.trainEnd)} → {shortDate(s.predictionStart)}–{shortDate(s.predictionEnd)}
                {reuseProblem && <span className="psr-warn"> · outside the first scenario's feature window</span>}
              </td>
              <td>
                {inTrain.length} in training · {inForecast.length} to forecast
              </td>
              <td>{windowEstimate(s).toLocaleString()}</td>
              {PHASES.map((phase, p) => (
                <td key={phase}>
                  <span className="psr-phase" data-state={phaseState(i, p)}>
                    {phaseState(i, p) === "done" ? "done" : phaseState(i, p) === "current" ? "running" : "waiting"}
                  </span>
                </td>
              ))}
            </tr>
          );
        })}
      </DataTable>

      <div className="psr-detail">
        <section aria-label={`${selected.name} definition`}>
          <h3>{selected.name} as a Scenario</h3>
          <CodeSnippet code={scenarioDict(selected)} />
        </section>
        <section aria-label={`${selected.name} facts`}>
          <h3>What this scenario forecasts</h3>
          <ul>
            <li>
              The training period spans {daysBetween(selected.trainStart, selected.trainEnd)} days and contains{" "}
              {eruptionsBetween(selected.trainStart, selected.trainEnd).length} eruption date
              {eruptionsBetween(selected.trainStart, selected.trainEnd).length === 1 ? "" : "s"}.
            </li>
            <li>
              The forecast period spans {daysBetween(selected.predictionStart, selected.predictionEnd)} days
              (about {windowEstimate(selected).toLocaleString()} windows) and contains{" "}
              {eruptionsBetween(selected.predictionStart, selected.predictionEnd).join(", ") || "no eruption dates"}.
            </li>
            <li>
              {selected.id === RECORDED_SCENARIO_ID
                ? "Its recorded probabilities are those inspected in step 4."
                : "No recorded probabilities are bundled for this scenario; the atlas therefore shows its periods only, not a probability curve."}
            </li>
          </ul>
        </section>
      </div>

      <form
        className="psr-form"
        onSubmit={(event) => {
          event.preventDefault();
          addScenario();
        }}
      >
        <h3>Add a scenario</h3>
        <div className="lab-options">
          {(
            [
              ["trainEnd", "Training ends"],
              ["predictionStart", "Forecast starts"],
              ["predictionEnd", "Forecast ends"],
            ] as const
          ).map(([key, label]) => (
            <label key={key}>
              {label}
              <input
                type="date"
                min={TREMOR_START}
                max={TREMOR_END}
                value={draft[key]}
                onChange={(event) => setDraft({ ...draft, [key]: event.target.value })}
              />
            </label>
          ))}
          <Button type="submit" variant="outline">
            <Plus size={15} /> Add to run
          </Button>
        </div>
        {error && (
          <p role="alert" className="psr-error">
            {error}
          </p>
        )}
        <p>
          Training always starts {shortDate(TREMOR_START)}, as in the example file. A real run must also stay
          inside the dates passed to calculate().
        </p>
      </form>
    </div>
  );
}
