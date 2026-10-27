import { useState } from "react";
import type { ForecastDataset } from "../forecastDataset";
import { buildLabels, confusionAt } from "../rocAnalysis";
import { RocPanel, ratio, show } from "./RocPanel";
import "@/styles/prediction.css";

export interface RocSimulationProps {
  dataset: ForecastDataset;
  threshold: number;
  onThresholdChange: (value: number) => void;
}

/** ROC panels for every model in the dataset, with a threshold slider and confusion counts. */
export function RocSimulation({
  dataset,
  threshold,
  onThresholdChange,
}: RocSimulationProps) {
  const primary = dataset.models[0]![0];
  const [model, setModel] = useState(primary);
  const { positives, negatives } = buildLabels(dataset);
  const c = confusionAt(dataset, model, threshold);
  const recall = ratio(c.tp, c.tp + c.fn);
  const fpr = ratio(c.fp, c.fp + c.tn);
  const precision = ratio(c.tp, c.tp + c.fp);
  const specificity = ratio(c.tn, c.tn + c.fp);

  return (
    <section className="roc" aria-labelledby={`roc-heading-${dataset.id}`}>
      <div className="lab-section-title">
        <h2 id={`roc-heading-${dataset.id}`}>ROC curve and the threshold</h2>
        <span>
          {positives.toLocaleString()} positive · {negatives.toLocaleString()}{" "}
          negative windows
        </span>
      </div>
      <p>
        Each recorded window is labelled like a training window: positive on the{" "}
        {dataset.lookaheadDays} days before an eruption and on the eruption day
        itself ({dataset.eruptionDates.join(" and ")}), negative otherwise, one
        label per {dataset.stepMinutes}-minute window. Sweeping the threshold
        from 1 down to 0 traces the curve; the dot marks where{" "}
        <strong>{threshold.toFixed(2)}</strong> sits. AUC does not depend on the
        threshold.
      </p>
      <div className="lab-options">
        <label>
          Threshold <strong>{threshold.toFixed(2)}</strong>
          <input
            type="range"
            min={0}
            max={1}
            step={0.01}
            value={threshold}
            onChange={(event) => onThresholdChange(Number(event.target.value))}
            aria-label="Classification threshold for the ROC curves"
          />
        </label>
        <label>
          Read counts for
          <select
            value={model}
            onChange={(event) => setModel(event.target.value)}
          >
            {dataset.models.map(([value, label]) => (
              <option key={value} value={value}>
                {label}
              </option>
            ))}
          </select>
        </label>
      </div>
      <div className="roc-grid-panels">
        {dataset.models.map(([value, label]) => (
          <RocPanel
            key={value}
            dataset={dataset}
            model={value}
            title={value === primary ? label : value}
            threshold={threshold}
          />
        ))}
      </div>
      <div
        className="training-table-scroll"
        tabIndex={0}
        role="region"
        aria-label="Confusion counts at the threshold"
      >
        <table className="training-table roc-table">
          <caption>
            {dataset.models.find(([value]) => value === model)![1]} at threshold{" "}
            {threshold.toFixed(2)}
          </caption>
          <thead>
            <tr>
              <th />
              <th>Predicted positive</th>
              <th>Predicted negative</th>
            </tr>
          </thead>
          <tbody>
            <tr>
              <th>Positive window</th>
              <td>{c.tp.toLocaleString()} true positives</td>
              <td>{c.fn.toLocaleString()} missed</td>
            </tr>
            <tr>
              <th>Negative window</th>
              <td>{c.fp.toLocaleString()} false alarms</td>
              <td>{c.tn.toLocaleString()} true negatives</td>
            </tr>
          </tbody>
        </table>
      </div>
      <output className="training-result" aria-live="polite">
        TPR (recall) {show(recall)} · FPR {show(fpr)} · precision{" "}
        {show(precision)} · specificity {show(specificity)}
      </output>
      <p>
        Lowering the threshold moves the dot up and right: more positive windows
        are caught, and more quiet windows raise an alarm. Shaded bands show the
        curve when every probability is shifted down or up by its own
        uncertainty (±1 SD of seed disagreement). The recorded file keeps seed
        means and SDs, not the individual seeds, so this band is not the
        per-seed range drawn by the evaluation plots.
      </p>
      <p>
        Treat these numbers with care. Only {dataset.eruptionDates.length}{" "}
        eruptions fall in this period, neighbouring windows overlap, and the
        positive windows come in blocks, so AUC and the counts can swing with a
        single eruption. Precision depends on how rare positives are and is the
        figure most affected.
      </p>
    </section>
  );
}
