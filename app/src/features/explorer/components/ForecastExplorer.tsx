import { useMemo, useState } from "react";
import { Download } from "lucide-react";
import { DataTable } from "@/components/lab/DataTable";
import { ResultOutput } from "@/components/lab/ResultOutput";
import { SelectField } from "@/components/lab/SelectField";
import { fixed, percent } from "@/lib/format";
import { utcStamp } from "@/lib/time";
import {
  readRow,
  thresholdSummary,
  type ForecastDataset,
} from "../forecastDataset";
import { ForecastChart } from "./ForecastChart";
import "@/styles/prediction.css";

export interface ForecastExplorerProps {
  dataset: ForecastDataset;
  threshold: number;
  onThresholdChange: (value: number) => void;
  /** Show the per-classifier table for the selected window. */
  revealRows?: boolean;
}

/** Forecast chart with a window scrubber, threshold and overlay controls, and read-outs. */
export function ForecastExplorer({
  dataset,
  threshold,
  onThresholdChange,
  revealRows = true,
}: ForecastExplorerProps) {
  const { rows, models } = dataset;
  const primary = models[0]![0];
  const [index, setIndex] = useState(() => {
    // Open on the strongest primary window so the first read is informative.
    let best = 0;
    rows.forEach((row, i) => {
      if (
        (row.values[`${primary}_probability`] ?? 0) >
        (rows[best]!.values[`${primary}_probability`] ?? 0)
      )
        best = i;
    });
    return best;
  });
  const [overlay, setOverlay] = useState(primary);
  const summary = useMemo(
    () => thresholdSummary(dataset, threshold),
    [dataset, threshold],
  );
  const row = rows[index]!;
  const lookaheadHours = dataset.lookaheadDays * 24;

  return (
    <div className="pwr">
      <ForecastChart
        dataset={dataset}
        threshold={threshold}
        cursorIndex={index}
        onCursorChange={setIndex}
        overlayModel={overlay}
      />
      <label className="pwr-scrub">
        <span>
          Window <strong>{utcStamp(row.time)} UTC</strong> · row {index + 1} of{" "}
          {rows.length}
        </span>
        <input
          type="range"
          min={0}
          max={rows.length - 1}
          step={1}
          value={index}
          onChange={(event) => setIndex(Number(event.target.value))}
        />
      </label>
      <div className="lab-options">
        <label>
          Threshold <strong>{threshold.toFixed(2)}</strong>
          <input
            type="range"
            min={0.3}
            max={0.95}
            step={0.05}
            value={threshold}
            onChange={(event) => onThresholdChange(Number(event.target.value))}
            aria-label="Alert threshold"
          />
        </label>
        <SelectField
          label="Overlay"
          value={overlay}
          options={models.map(([value, label]) => ({ value, label }))}
          onChange={setOverlay}
        />
      </div>
      {revealRows && (
        <DataTable
          label="Values for the selected window"
          className="pwr-table"
          caption={`Recorded row at ${utcStamp(row.time)} UTC · every classifier and the consensus`}
          headers={["Model", "probability", "uncertainty (SD)", "confidence", "prediction"]}
        >
          {models.map(([key, label]) => {
            const value = readRow(dataset, index, key);
            return (
              <tr
                key={key}
                data-consensus={key === primary}
                data-above={value.probability >= threshold}
              >
                <th>{label}</th>
                <td>{percent(value.probability)}</td>
                <td>{fixed(value.uncertainty)}</td>
                <td>{fixed(value.confidence)}</td>
                <td>{fixed(value.prediction)}</td>
              </tr>
            );
          })}
        </DataTable>
      )}
      <ResultOutput>
        {summary.above.toLocaleString()} of {summary.total.toLocaleString()}{" "}
        windows ({percent(summary.above / summary.total)}) reach{" "}
        {threshold.toFixed(2)}
      </ResultOutput>
      <ul className="pwr-eruptions">
        {summary.eruptions.map((item) => (
          <li key={item.date}>
            <strong>{item.date}</strong>
            {item.crossings === 0
              ? ` · no window reached ${threshold.toFixed(2)} in the ${lookaheadHours} h before`
              : ` · ${item.crossings} windows at or above, first ${item.leadHours!.toFixed(1)} h before`}
          </li>
        ))}
      </ul>
      <p>
        Recorded {dataset.label} predictions, sampled every{" "}
        {dataset.stepMinutes} minutes. The threshold only changes what is
        counted here, not the saved file. Green shading marks probability below
        the threshold and ember shading the part above it. The pale vertical
        bands span each eruption day, 00:00 to 23:59 UTC. Counts describe this
        one run and are not a skill score.{" "}
        {dataset.downloadUrl && (
          <a href={dataset.downloadUrl} download>
            <Download size={14} aria-hidden="true" /> Download the CSV
          </a>
        )}
      </p>
    </div>
  );
}
