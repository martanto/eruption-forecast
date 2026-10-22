import { SCENARIOS, ERUPTIONS } from "@/features/architecture";
import { formatDate, shortDate } from "@/lib/time";

export function ScenarioComparison({
  selected,
  onSelect,
}: {
  selected: string;
  onSelect: (id: string) => void;
}) {
  const scenario = SCENARIOS.find((item) => item.id === selected)!;
  return (
    <section className="scenario-comparison">
      <h3>Compare example periods</h3>
      <p>
        Dates from the active script; description text is not used to infer
        eruption counts.
      </p>
      <div className="scenario-rows">
        {SCENARIOS.map((item) => (
          <button
            key={item.id}
            aria-pressed={selected === item.id}
            onClick={() => onSelect(item.id)}
          >
            <strong>{item.name}</strong>
            <span>
              Train to {shortDate(item.trainEnd)}
              <br />
              Predict from {shortDate(item.predictionStart)}
            </span>
          </button>
        ))}
      </div>
      <dl>
        <dt>Training</dt>
        <dd>
          {formatDate(scenario.trainStart)}  to  {formatDate(scenario.trainEnd)}
        </dd>
        <dt>Prediction</dt>
        <dd>
          {formatDate(scenario.predictionStart)}  to {" "}
          {formatDate(scenario.predictionEnd)}
        </dd>
        <dt>Output directory</dt>
        <dd>
          <code>output-test/VG.OJN.00.EHZ/scenarios/{scenario.id}/</code>
          <small>
            Station-directory prefix follows the resolved station configuration.
          </small>
        </dd>
      </dl>
      <details>
        <summary>Shared example settings</summary>
        <p>
          RF + Lite-RF, 2 seeds, 20 features, recall, 5 shuffle-stratified
          splits, under-sampling. Training every 6 h; prediction every 10 min.
          Two-day horizon / lookback.
        </p>
        <p>
          Calculate once for 2025. Each scenario trains and predicts, then
          evaluates and explains prediction. A second run changes evaluation and
          explanation to training reuse.
        </p>
        <p>
          First prediction extracts all features. Later scenarios reuse its
          files over their narrower prediction periods. These are script
          settings, not package defaults.
        </p>
      </details>
      <details>
        <summary>All eruption dates ({ERUPTIONS.length})</summary>
        <ul>
          {ERUPTIONS.map((date) => (
            <li key={date}>{formatDate(date)}</li>
          ))}
        </ul>
      </details>
    </section>
  );
}
