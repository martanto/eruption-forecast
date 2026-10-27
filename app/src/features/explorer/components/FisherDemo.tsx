import { useState } from "react";
import { ChoiceButtons } from "@/components/lab/ChoiceButtons";
import { ChartFrame } from "@/components/lab/ChartFrame";
import { DataTable } from "@/components/lab/DataTable";
import { Legend } from "@/components/lab/Legend";
import { integer, probabilityLabel } from "@/lib/format";
import { DUMMY_FEATURES, PRE_COUNT, QUIET_COUNT } from "../dummyDataset";
import { fisherRows, fisherTable, formatP, verdict } from "../significance";
import { CombinationsLesson } from "./CombinationsLesson";

const BINARY = DUMMY_FEATURES.filter((feature) => feature.kind === "binary");
const MEANING: Record<string, string> = {
  "rsam_f2__has_duplicate_max": "Indicator: 1 if the window's maximum amplitude occurs more than once, else 0.",
  "rsam_f2__symmetry_looking__r_0.05": "Indicator: 1 if the amplitude distribution in the window is symmetric about its mean, else 0.",
  "rsam_f2__large_standard_deviation__r_0.25": "Indicator: 1 if the window's standard deviation is large relative to its range, else 0.",
};
export function FisherDemo() {
  const [name, setName] = useState("rsam_f2__large_standard_deviation__r_0.25");
  const feature = BINARY.find((item) => item.name === name)!;
  const table = fisherTable(feature);
  const fisher = fisherRows(table);
  const { yes, total, ways, rows, p } = fisher;
  const v = verdict(p);
  const peak = Math.max(...rows.map((row) => row.probability));
  const observed = rows.find((row) => row.x === table.preYes)!;

  return (
    <div className="sig-demo">
      <ChoiceButtons
        label="Pick a yes/no feature"
        options={BINARY.map((item) => item.name)}
        value={name}
        onChange={setName}
      />
      <div className="sig-grid sig-grid-even">
        <div className="sig-card">
          <h3>3a · Tabulate the windows in a 2×2 contingency table</h3>
          <p>{MEANING[name]}</p>
          <DataTable
            label="Contingency table"
            className="sig-table"
            headers={[null, "feature = 1", "feature = 0", "total"]}
          >
            <tr>
              <th data-group="pre">label 1 · pre-eruptive</th>
              <td className="sig-focus">{table.preYes}</td>
              <td>{table.preNo}</td>
              <td>{PRE_COUNT}</td>
            </tr>
            <tr>
              <th data-group="quiet">label 0 · non-eruptive</th>
              <td>{table.quietYes}</td>
              <td>{table.quietNo}</td>
              <td>{QUIET_COUNT}</td>
            </tr>
            <tr>
              <th>total</th>
              <td>{yes}</td>
              <td>{total - yes}</td>
              <td>{total}</td>
            </tr>
          </DataTable>
          <p className="sig-caption">
            <b>Table 2.</b> The 24 synthetic windows cross-tabulated by class
            label and feature value. The highlighted cell drives the test:{" "}
            {table.preYes} of {PRE_COUNT} pre-eruptive windows have feature = 1,
            against {table.quietYes} of {QUIET_COUNT} non-eruptive windows.
          </p>
          <div className="sig-result" data-tone={v.tone}>
            <dl>
              <div>
                <dt>p-value · sum of the orange bars</dt>
                <dd>{formatP(p)}</dd>
              </div>
            </dl>
            <p className="sig-chip">{v.label}</p>
          </div>
        </div>
        <div className="sig-card">
          <h3>3b · Null distribution and p-value</h3>
          <p>
            Condition on the margins: {PRE_COUNT} pre-eruptive windows and {yes}{" "}
            windows with feature = 1. Under the null hypothesis of independence,
            the {yes} windows with feature = 1 are a random draw from all{" "}
            {total}. The number x that fall among the {PRE_COUNT} pre-eruptive
            windows then follows a hypergeometric distribution. Each bar is
            P(X = x).
          </p>
          <ChartFrame
            width={rows.length * 52 + 20}
            height={190}
            className="sig-bars"
            label={`Chance of each possible count. Observed count ${table.preYes}; p = ${formatP(p)}.`}
          >
            {rows.map((row, i) => {
              const height = Math.max(2, (row.probability / peak) * 120);
              const x = 10 + i * 52;
              return (
                <g key={row.x}>
                  <rect
                    x={x}
                    y={140 - height}
                    width={40}
                    height={height}
                    className="sig-bar"
                    data-in={row.inP}
                    data-observed={row.x === table.preYes || undefined}
                  />
                  <text x={x + 20} y={134 - height} textAnchor="middle" className="chart-axis">
                    {probabilityLabel(row.probability)}
                  </text>
                  <text x={x + 20} y={160} textAnchor="middle" className="chart-axis">
                    {row.x}
                  </text>
                  {row.x === table.preYes && (
                    <text x={x + 20} y={176} textAnchor="middle" className="sig-ours">observed</text>
                  )}
                </g>
              );
            })}
          </ChartFrame>
          <p className="sig-axis-caption">
            x = number of pre-eruptive windows with feature = 1, under the null
            hypothesis
          </p>
          <Legend
            items={[
              { swatch: ["bar", "in"], label: "as extreme as observed, or more" },
              { swatch: ["bar", "out"], label: "less extreme" },
            ]}
          />
          <p className="sig-caption">
            <b>Figure 3.</b> The null distribution of the count x. The bar
            outlined in black is the observed count. The two-sided p-value sums
            the probabilities of all outcomes no more probable than the
            observed one (orange).
          </p>
          <div className="sig-formula">
            <code>
              P(x) = C({PRE_COUNT}, x) × C({QUIET_COUNT}, {yes} − x) / C({total}, {yes})
            </code>
            <p>
              For the observed table: P({table.preYes}) = {integer(observed.top)} ×{" "}
              {integer(observed.bottom)} / {integer(ways)} ={" "}
              {observed.probability.toFixed(5)}. Two-sided p-value: the sum of
              P(x) over every outcome with P(x) ≤ P(observed), in both tails:{" "}
              <b>{formatP(p)}</b>.
            </p>
          </div>
        </div>
      </div>

      <CombinationsLesson table={table} fisher={fisher} />
      <p className="sig-note">
        Same two-sided p-value in scipy:{" "}
        <code>
          _, p = fisher_exact([[{table.preYes}, {table.preNo}], [{table.quietYes}, {table.quietNo}]])
        </code>
      </p>
    </div>
  );
}
