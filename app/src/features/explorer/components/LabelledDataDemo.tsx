import { useMemo, useState } from "react";
import { ChoiceButtons } from "@/components/lab/ChoiceButtons";
import { DataTable } from "@/components/lab/DataTable";
import { Legend } from "@/components/lab/Legend";
import { ChartScroll } from "@/components/lab/ScrollRegion";
import {
  DUMMY_FEATURES,
  PICK_FEATURES,
  PRE_COUNT,
  QUIET_COUNT,
} from "../dummyDataset";
import { split } from "../significance";

const PICKS = PICK_FEATURES;
// Time order: 5 non-eruptive, 5 pre-eruption, eruption, 6 non-eruptive, 5 pre-eruption, eruption, 3 non-eruptive.
const PATTERN = [
  0, 0, 0, 0, 0, 1, 1, 1, 1, 1, "E", 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, "E", 0, 0, 0,
] as const;
const SLOT = 28;
const GAP = 30;
const LEFT = 20;
const BASE = 214;
const MAX_BAR = 150;

interface Win {
  id: number;
  label: 0 | 1;
  x: number;
  values: Record<string, number>;
}

// Lay the frozen dummy values out along a clock, so the figure reads like real windows.
function buildTimeline() {
  const parts = PICKS.map((name) => {
    const { quiet, pre } = split(DUMMY_FEATURES.find((f) => f.name === name)!);
    return { name, quiet: [...quiet], pre: [...pre] };
  });
  const windows: Win[] = [];
  const eruptions: number[] = [];
  let x = LEFT;
  let id = 0;
  for (const item of PATTERN) {
    if (item === "E") {
      eruptions.push(x + GAP / 2 - (SLOT - 22) / 2);
      x += GAP;
      continue;
    }
    id += 1;
    const values: Record<string, number> = {};
    for (const part of parts) {
      values[part.name] = (item === 1 ? part.pre : part.quiet).shift()!;
    }
    windows.push({ id, label: item, x, values });
    x += SLOT;
  }
  return { windows, eruptions, width: x + 8 };
}
const TIMELINE = buildTimeline();

export function LabelledDataDemo() {
  const [name, setName] = useState<string>(PICKS[0]);
  const { windows, eruptions, width } = TIMELINE;
  const max = useMemo(
    () => Math.max(...windows.map((w) => w.values[name]!)),
    [windows, name],
  );
  const rows = windows.filter((w) => w.id >= 4 && w.id <= 8);
  const H = 266;

  return (
    <div className="sig-demo">
      <ChoiceButtons
        label="Pick a feature"
        options={PICKS}
        value={name}
        onChange={setName}
      />
      <figure className="sig-card sig-figure">
        <Legend
          items={[
            { swatch: ["group", "quiet"], label: "label 0 · non-eruptive" },
            { swatch: ["group", "pre"], label: "label 1 · pre-eruptive" },
          ]}
        />
        <ChartScroll>
        <svg
          viewBox={`0 0 ${width} ${H}`}
          role="img"
          aria-label={`Bar chart of ${name} for 24 windows in time order. Windows right before each of the two eruptions are labelled 1.`}
          style={{ minWidth: 640 }}
        >
          <line x1={LEFT - 6} x2={width - 4} y1={BASE} y2={BASE} className="sig-grid-line" />
          {eruptions.map((ex) => (
            <g key={ex}>
              <line x1={ex} x2={ex} y1={46} y2={BASE} className="sig-sweep" />
              <path d={`M${ex},22 l-7,12 h14 z`} className="sig-eruption-mark" />
              <text x={ex} y={14} textAnchor="middle" className="sig-axis">
                ERUPTION
              </text>
            </g>
          ))}
          {windows.map((w) => {
            const value = w.values[name]!;
            const height = (value / max) * MAX_BAR;
            return (
              <g key={w.id}>
                <rect
                  x={w.x}
                  y={BASE - height}
                  width={22}
                  height={height}
                  rx={2}
                  className="sig-bar-data"
                  data-group={w.label === 1 ? "pre" : "quiet"}
                />
                <text x={w.x + 11} y={BASE - height - 5} textAnchor="middle" className="sig-axis sig-tiny">
                  {Math.round(value)}
                </text>
                <text
                  x={w.x + 11}
                  y={BASE + 26}
                  textAnchor="middle"
                  className="sig-labelnum"
                  data-group={w.label === 1 ? "pre" : "quiet"}
                >
                  {w.label}
                </text>
                <text x={w.x + 11} y={BASE + 44} textAnchor="middle" className="sig-axis sig-tiny">
                  {w.id}
                </text>
              </g>
            );
          })}
        </svg>
        </ChartScroll>
        <figcaption>
          <b>Figure 1.</b> Value of <code>{name}</code> in each of 24 consecutive
          time windows. Upper row of digits: the class label. Lower row: the
          window index. The 5 windows immediately preceding each eruption onset
          are labelled 1 (pre-eruptive, the positive class); all others are
          labelled 0 (non-eruptive). The result is {PRE_COUNT} positive and{" "}
          {QUIET_COUNT} negative windows, an imbalanced two-class problem. The
          values are synthetic, not measured seismic tremor.
        </figcaption>
      </figure>

      <div className="sig-grid sig-grid-even">
        <figure className="sig-card sig-figure">
          <h3>Feature matrix: one row per time window</h3>
          <DataTable
            label="Labelled feature rows"
            className="sig-table sig-mini"
            headers={["window", ...PICKS.map((pick) => pick.split("__")[1]), "is_erupted"]}
          >
            {rows.map((w) => (
              <tr key={w.id}>
                <th>{w.id}</th>
                {PICKS.map((pick) => (
                  <td key={pick}>{w.values[pick]!.toFixed(1)}</td>
                ))}
                <td data-group={w.label === 1 ? "pre" : "quiet"}>
                  <b>{w.label}</b>
                </td>
              </tr>
            ))}
          </DataTable>
          <figcaption>
            <b>Table 1.</b> Windows 4–8 of the synthetic feature matrix: 24 rows
            (samples) and {DUMMY_FEATURES.length} feature columns, three shown,
            plus the target <code>is_erupted</code>. Real runs have hundreds of
            feature columns per tremor channel.
          </figcaption>
        </figure>
        <div className="sig-card sig-why">
          <h3>The hypothesis test, applied to every feature</h3>
          <p>
            <b>Null hypothesis: the feature has the same distribution in
            pre-eruptive and non-eruptive windows.</b>
          </p>
          <p>
            If the data are consistent with it, the feature carries no
            precursory signal and is dropped. If they are not, it is a candidate
            predictor.
          </p>
          <p>
            <b>Steps 2 and 3</b> test one feature at a time and return a
            p-value: the KS test for continuous features, Fisher’s exact test
            for binary features.
          </p>
          <p>
            <b>Step 4</b> applies the BY correction, because testing hundreds of
            features at once inflates the number of false positives.
          </p>
        </div>
      </div>
    </div>
  );
}
