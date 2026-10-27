import { useState } from "react";
import { LabSectionTitle } from "@/components/lab/LabSectionTitle";
import { ResultOutput } from "@/components/lab/ResultOutput";
import { ScrollRegion } from "@/components/lab/ScrollRegion";
import { SelectField } from "@/components/lab/SelectField";
import {
  FEATURES_MATRIX_EDGES,
  FEATURES_MATRIX_NODES,
} from "../workflowDiagrams";
import { WorkflowDiagram } from "./WorkflowDiagram";
import {
  CALCULATORS,
  calculate,
  RAW_WINDOWS,
  TREMOR_COLUMNS,
  WINDOW_LABELS,
} from "../trainingFixtures";

type Column = (typeof TREMOR_COLUMNS)[number];

export function FeaturesMatrixLesson({
  step,
  onStepChange,
}: {
  step: number;
  onStepChange: (step: number) => void;
}) {
  const [column, setColumn] = useState<Column>(TREMOR_COLUMNS[0]);
  const shown: readonly Column[] = step === 3 ? TREMOR_COLUMNS : [column];
  const featureCount = shown.length * CALCULATORS.length;
  return (
    <>
      <WorkflowDiagram
        title="Features matrix builder workflow"
        nodes={FEATURES_MATRIX_NODES}
        edges={FEATURES_MATRIX_EDGES}
        step={step}
        onStepChange={onStepChange}
        caption="FeaturesBuilder.extract_features validates the tremor matrix, picks training or prediction mode, then extracts each tremor column separately. A cache hit reuses the per-column Parquet file instead of calling tsfresh, and all columns are concatenated side by side. Dashed lines are optional or cached paths."
      />
      <LabSectionTitle
        title={
          step === 0
            ? "Raw tremor samples inside each window"
            : step === 3
              ? "One row per window, one column per feature"
              : `Features computed from ${column}`
        }
        aside="3 example windows · 4 samples each (real windows hold hundreds)"
      />
      {step > 0 && step < 3 && (
        <div className="lab-options">
          <SelectField
            label="Tremor column"
            value={column}
            options={TREMOR_COLUMNS.map((name) => ({ value: name, label: name }))}
            onChange={setColumn}
          />
        </div>
      )}
      <ScrollRegion label="Features matrix example">
        <table className="training-table fm-table" data-step={step}>
          <caption>
            {step === 0
              ? "Tremor matrix · window id + samples per tremor column"
              : `Features matrix · ${3} rows × ${featureCount} columns shown`}
          </caption>
          <thead>
            <tr>
              <th>id</th>
              {step === 0
                ? TREMOR_COLUMNS.map((name) => <th key={name}>{name}</th>)
                : shown.flatMap((name) =>
                    CALCULATORS.map((calc) => (
                      <th key={`${name}-${calc}`} className="fm-new">
                        {step === 1 ? calc : `${name}__${calc}`}
                      </th>
                    )),
                  )}
              {step === 3 && <th>is_erupted</th>}
            </tr>
          </thead>
          <tbody>
            {[0, 1, 2].map((w) => (
              <tr key={w}>
                <th>{w + 1}</th>
                {step === 0
                  ? TREMOR_COLUMNS.map((name) => (
                      <td key={name}>{RAW_WINDOWS[name][w]!.map((v) => v.toFixed(1)).join(", ")}</td>
                    ))
                  : shown.flatMap((name) =>
                      CALCULATORS.map((calc) => (
                        <td key={`${name}-${calc}`} className="fm-new">
                          {calculate(calc, RAW_WINDOWS[name][w]!).toFixed(2)}
                        </td>
                      )),
                    )}
                {step === 3 && <td>{WINDOW_LABELS[w]}</td>}
              </tr>
            ))}
          </tbody>
        </table>
      </ScrollRegion>
      <ResultOutput>
        {step === 0
          ? "3 windows × 3 tremor columns of raw samples"
          : step === 1
            ? `${column}: raw samples → ${CALCULATORS.length} statistics per window`
            : step === 2
              ? `${column}__mean = average of that window's ${column} samples`
              : `3 windows × ${featureCount} features, joined on id`}
      </ResultOutput>
      {step === 3 && (
        <p>
          The real matrix has the same shape with hundreds of calculators per
          tremor column. Labels stay in a separate file and join on the window
          id.
        </p>
      )}
    </>
  );
}
