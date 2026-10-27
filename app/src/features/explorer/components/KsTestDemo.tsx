import { useMemo, useState } from "react";
import { ChoiceButtons } from "@/components/lab/ChoiceButtons";
import { Legend } from "@/components/lab/Legend";
import { ChartScroll } from "@/components/lab/ScrollRegion";
import { Button } from "@/components/ui/button";
import {
  DUMMY_FEATURES,
  PICK_FEATURES,
  PRE_COUNT,
  QUIET_COUNT,
} from "../dummyDataset";
import { ecdf, formatP, ksGap, split, verdict } from "../significance";

const REAL = DUMMY_FEATURES.filter((feature) => feature.kind === "real");
const W = 640;
const H = 380;
const LEFT = 44;
const RIGHT = 16;
const STRIP = { quiet: 28, pre: 58 };
const PLOT = { top: 112, bottom: 340 };

// Staircase path: climbs 1/n each time it passes a sorted value.
function staircase(
  values: readonly number[],
  sx: (x: number) => number,
  sy: (y: number) => number,
  lo: number,
  hi: number,
) {
  const sorted = [...values].sort((a, b) => a - b);
  let d = `M${sx(lo)},${sy(0)}`;
  sorted.forEach((v, i) => {
    d += ` H${sx(v)} V${sy((i + 1) / sorted.length)}`;
  });
  return `${d} H${sx(hi)}`;
}

export function KsTestDemo() {
  const [name, setName] = useState<string>(PICK_FEATURES[0]);
  const feature = REAL.find((item) => item.name === name)!;
  const { quiet, pre } = useMemo(() => split(feature), [feature]);
  const gap = useMemo(() => ksGap(quiet, pre), [quiet, pre]);
  const all = [...quiet, ...pre];
  const lo = Math.floor(Math.min(...all) - 6);
  const hi = Math.ceil(Math.max(...all) + 6);
  const [sweepAt, setSweepAt] = useState<number | null>(null);
  const sweep = sweepAt ?? gap.at;
  const sx = (x: number) => LEFT + ((x - lo) / (hi - lo)) * (W - LEFT - RIGHT);
  const sy = (y: number) => PLOT.bottom - y * (PLOT.bottom - PLOT.top);
  const f0 = ecdf(quiet, sweep);
  const f1 = ecdf(pre, sweep);
  const v = verdict(feature.p);
  const odds = Math.max(1, Math.round(1 / feature.p));

  return (
    <div className="sig-demo">
      <ChoiceButtons
        label="Pick a feature"
        options={PICK_FEATURES}
        value={name}
        onChange={(pick) => {
          setName(pick);
          setSweepAt(null);
        }}
      />
      <div className="sig-grid">
        <div className="sig-card sig-chart">
          <Legend
            items={[
              { swatch: ["group", "quiet"], label: `label 0 · non-eruptive (n = ${QUIET_COUNT})` },
              { swatch: ["group", "pre"], label: `label 1 · pre-eruptive (n = ${PRE_COUNT})` },
            ]}
          />
          <ChartScroll>
          <svg
            viewBox={`0 0 ${W} ${H}`}
            role="img"
            aria-label={`Empirical CDFs of ${feature.name} for the two classes. KS statistic D = ${gap.d.toFixed(2)}.`}
          >
            <text x={4} y={STRIP.quiet + 4} className="sig-axis">0</text>
            <text x={4} y={STRIP.pre + 4} className="sig-axis">1</text>
            <line x1={LEFT} x2={W - RIGHT} y1={STRIP.pre + 18} y2={STRIP.pre + 18} className="sig-grid-line" />
            {quiet.map((value, i) => (
              <circle key={i} cx={sx(value)} cy={STRIP.quiet} r={5} className="sig-dot" data-group="quiet" />
            ))}
            {pre.map((value, i) => (
              <rect
                key={i}
                x={sx(value) - 4.5}
                y={STRIP.pre - 4.5}
                width={9}
                height={9}
                transform={`rotate(45 ${sx(value)} ${STRIP.pre})`}
                className="sig-dot"
                data-group="pre"
              />
            ))}
            <text x={LEFT} y={PLOT.top - 12} className="sig-axis">ECDF: share of windows with value ≤ x</text>
            {[0, 0.5, 1].map((tick) => (
              <g key={tick}>
                <line x1={LEFT} x2={W - RIGHT} y1={sy(tick)} y2={sy(tick)} className="sig-grid-line" />
                <text x={LEFT - 8} y={sy(tick) + 4} textAnchor="end" className="sig-axis">{tick}</text>
              </g>
            ))}
            {[0, 0.25, 0.5, 0.75, 1].map((t) => {
              const value = Math.round(lo + t * (hi - lo));
              return (
                <text key={t} x={sx(value)} y={PLOT.bottom + 20} textAnchor="middle" className="sig-axis">
                  {value}
                </text>
              );
            })}
            <path d={staircase(quiet, sx, sy, lo, hi)} className="sig-step" data-group="quiet" />
            <path d={staircase(pre, sx, sy, lo, hi)} className="sig-step" data-group="pre" />
            <line x1={sx(sweep)} x2={sx(sweep)} y1={STRIP.quiet - 14} y2={PLOT.bottom} className="sig-sweep" />
            <line x1={sx(sweep)} x2={sx(sweep)} y1={sy(f0)} y2={sy(f1)} className="sig-gap" />
            <circle cx={sx(sweep)} cy={sy(f0)} r={4} className="sig-marker" data-group="quiet" />
            <circle cx={sx(sweep)} cy={sy(f1)} r={4} className="sig-marker" data-group="pre" />
            <text
              x={sx(sweep) + 8}
              y={(sy(f0) + sy(f1)) / 2 + 4}
              className="sig-gap-label"
            >
              gap = {Math.abs(f1 - f0).toFixed(2)}
            </text>
          </svg>
          </ChartScroll>
          <div className="sig-sweep-control">
            <label>
              Move x along the feature axis
              <input
                type="range"
                min={lo}
                max={hi}
                step={0.5}
                value={sweep}
                onChange={(e) => setSweepAt(Number(e.target.value))}
                aria-valuetext={`x = ${sweep}: ECDF ${Math.round(f0 * 100)}% for non-eruptive, ${Math.round(f1 * 100)}% for pre-eruptive`}
              />
            </label>
            <Button variant="outline" size="small" onClick={() => setSweepAt(gap.at)}>
              Jump to the maximum distance
            </Button>
          </div>
          <p className="sig-readout" aria-live="polite">
            At x = {sweep}: the ECDF is {Math.round(f0 * 100)}% for
            non-eruptive windows and {Math.round(f1 * 100)}% for pre-eruptive
            windows, a distance of {Math.abs(f1 - f0).toFixed(2)}.
          </p>
          <p className="sig-caption">
            <b>Figure 2.</b> Top: each window plotted on the feature axis by
            class (circles are non-eruptive, diamonds are pre-eruptive). Bottom:
            the two ECDFs, the proportion of each class with value ≤ x. The
            solid black bar marks D, the maximum vertical distance between them.
          </p>
        </div>
        <div className="sig-side">
          <div className="sig-result" data-tone={v.tone}>
            <dl>
              <div>
                <dt>D · KS statistic</dt>
                <dd>{gap.d.toFixed(2)}</dd>
              </div>
              <div>
                <dt>p-value</dt>
                <dd>{formatP(feature.p)}</dd>
              </div>
            </dl>
            <p className="sig-chip">{v.label}</p>
            <p>
              {feature.p >= 0.05
                ? `The two samples overlap. Under the null hypothesis, a statistic this large occurs about ${Math.round(feature.p * 100)} times in 100. The feature does not discriminate the classes.`
                : `Under the null hypothesis (feature independent of label), a statistic this large occurs about 1 time in ${odds.toLocaleString("en-US")}.`}
            </p>
          </div>
          <ol className="sig-steps">
            <li>Dots: each window’s feature value, one row per class.</li>
            <li>
              ECDF: moving left to right, each curve rises by 1/n at every
              observation (1/14 for non-eruptive, 1/10 for pre-eruptive).
            </li>
            <li>
              D = max |F₁(x) − F₀(x)|. D = 0 means identical samples; D = 1
              means no overlap.
            </li>
            <li>
              The p-value is the probability, under the null hypothesis, of a D
              at least this large.
            </li>
          </ol>
        </div>
      </div>
      <p className="sig-note">
        tsfresh offers the KS test as its Smirnov option. Its default for
        continuous features and a binary label is the Mann–Whitney U test,
        which also yields one p-value per feature, so the later steps are the
        same for either test.
      </p>
    </div>
  );
}
