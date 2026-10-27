import { useState } from "react";
import { correct } from "../significance";
import { formatP } from "./KsTestDemo";

const W = 760;
const H = 360;
const LEFT = 64;
const TOP = 16;
const BOTTOM = 40;
const LOG_MIN = -7;
// Minor gridlines at 2..9 x 10^e make the logarithmic spacing visible.
const MINOR = [-6, -5, -4, -3, -2, -1].flatMap((e) =>
  [2, 3, 4, 5, 6, 7, 8, 9].map((d) => d * 10 ** e),
);

// p-values span many orders of magnitude, so the chart is logarithmic.
const sy = (p: number) => {
  const t = (Math.log10(Math.max(p, 10 ** LOG_MIN)) - LOG_MIN) / -LOG_MIN;
  return H - BOTTOM - t * (H - BOTTOM - TOP);
};

export function ByCorrectionDemo() {
  const [q, setQ] = useState(0.05);
  const { rows, m, c } = correct(q);
  const sx = (rank: number) => LEFT + ((rank - 0.5) / m) * (W - LEFT - 12);
  const counts = {
    raw: rows.filter((row) => row.keptRaw).length,
    bh: rows.filter((row) => row.keptBh).length,
    by: rows.filter((row) => row.keptBy).length,
  };
  const line = (key: "bh" | "by") =>
    rows.map((row, i) => `${i === 0 ? "M" : "L"}${sx(row.rank)},${sy(row[key])}`).join(" ");
  const lastBy = rows.filter((row) => row.keptBy).length;
  const rank5 = rows[4]!;

  return (
    <div className="sig-demo">
      <p>
        Testing all {m} synthetic features gives {m} p-values, one per null
        hypothesis. The Benjamini–Yekutieli (BY) step-up procedure selects the
        features to retain so that the expected proportion of false discoveries
        among them (the false discovery rate, FDR) stays at or below q. Unlike
        Benjamini–Hochberg (BH), BY remains valid under arbitrary dependence
        between tests.
      </p>
      <div className="sig-cards">
        <div className="sig-card sig-why">
          <h3>Why correct for multiple testing?</h3>
          <p>
            At a significance level of 0.05, a feature with no signal still
            passes with probability 0.05 (a Type I error). Among {m} null
            features, expect about {(m * 0.05).toFixed(1)} false positives.
            Among 700 null features, expect about 35 to pass by chance.
          </p>
        </div>
        <div className="sig-card sig-count">
          <span>Uncorrected, p ≤ {q}</span>
          <b>{counts.raw}</b> / {m} kept
        </div>
        <div className="sig-card sig-count">
          <span>Benjamini–Hochberg (BH)</span>
          <b>{counts.bh}</b> / {m} kept
        </div>
        <div className="sig-card sig-count" data-focus="true">
          <span>BY, what tsfresh uses</span>
          <b>{counts.by}</b> / {m} kept
        </div>
      </div>
      <p className="sig-caption">
        <b>Figure 5.</b> Number of the {m} synthetic features each rule retains
        at the FDR level q set below. Stricter control retains fewer features.
        The outlined card is the rule tsfresh uses (BY).
      </p>
      <div className="lab-options">
        <label>
          FDR level q{" "}
          <input
            type="range"
            min={0.01}
            max={0.2}
            step={0.01}
            value={q}
            onChange={(e) => setQ(Number(e.target.value))}
          />{" "}
          <b>{q.toFixed(2)}</b> <span className="sig-hint">(tsfresh default 0.05)</span>
        </label>
      </div>
      <div className="sig-stack">
        <div className="sig-card">
          <h3>Ordered p-values against three rejection thresholds</h3>
          <div className="sig-by-chart">
          <div>
          <ul className="sig-legend" aria-label="Legend">
            <li data-line="raw"><i aria-hidden="true" /> uncorrected level q</li>
            <li data-line="bh"><i aria-hidden="true" /> BH threshold</li>
            <li data-line="by"><i aria-hidden="true" /> BY threshold</li>
            <li data-dot="kept"><i aria-hidden="true" /> kept by BY</li>
            <li data-dot="dropped"><i aria-hidden="true" /> dropped</li>
          </ul>
          <div className="sig-scroll" tabIndex={0} role="region" aria-label="Chart, scrolls sideways on narrow screens">
          <svg viewBox={`0 0 ${W} ${H}`} role="img" aria-label={`BY keeps the first ${lastBy} of ${m} features at q = ${q}.`}>
            {lastBy > 0 && (
              <rect
                x={LEFT}
                y={TOP}
                width={sx(lastBy) + (sx(2) - sx(1)) / 2 - LEFT}
                height={H - BOTTOM - TOP}
                className="sig-keep-zone"
              />
            )}
            {MINOR.map((tick) => (
              <line key={tick} x1={LEFT} x2={W - 12} y1={sy(tick)} y2={sy(tick)} className="sig-grid-minor" />
            ))}
            {[1, 1e-1, 1e-2, 1e-3, 1e-4, 1e-5, 1e-6].map((tick) => (
              <g key={tick}>
                <line x1={LEFT} x2={W - 12} y1={sy(tick)} y2={sy(tick)} className="sig-grid-line" />
                <text x={LEFT - 6} y={sy(tick) + 4} textAnchor="end" className="sig-axis">
                  {tick === 1 ? "1" : tick.toExponential(0).replace("e-", "e−")}
                </text>
              </g>
            ))}
            <line x1={LEFT} x2={W - 12} y1={sy(q)} y2={sy(q)} className="sig-line" data-line="raw" />
            <path d={line("bh")} className="sig-line" data-line="bh" />
            <path d={line("by")} className="sig-line" data-line="by" />
            {rows.map((row) => (
              <g key={row.name}>
                <circle
                  cx={sx(row.rank)}
                  cy={sy(row.p)}
                  r={5.5}
                  className="sig-point"
                  data-kept={row.keptBy}
                />
                <text x={sx(row.rank)} y={H - BOTTOM + 18} textAnchor="middle" className="sig-axis">
                  {row.rank}
                </text>
              </g>
            ))}
            <text x={(LEFT + W) / 2} y={H - 6} textAnchor="middle" className="sig-axis">
              rank k of the ordered p-values (1 = smallest)
            </text>
            <text
              transform={`rotate(-90 14 ${(TOP + H - BOTTOM) / 2})`}
              x={14}
              y={(TOP + H - BOTTOM) / 2}
              textAnchor="middle"
              className="sig-axis"
            >
              p-value (log scale)
            </text>
            {lastBy > 0 && (
              <text x={LEFT + 8} y={TOP + 16} className="sig-zone-label">
                BY keeps ranks 1–{lastBy}
              </text>
            )}
          </svg>
          </div>
          </div>
          <div>
          <div className="sig-formula">
            <code>BY threshold for rank k = k × q / (m × c(m))</code>
            <p>
              m = {m} tests; c(m) = 1 + 1/2 + … + 1/{m} = {c.toFixed(2)}, the
              m-th harmonic number. Seismic features are correlated:{" "}
              <code>rsam_f2__mean</code> and <code>rsam_f3__maximum</code>, for
              example, rise and fall together. BH assumes independent or
              positively dependent tests. Dividing by c(m) keeps BY valid under
              any dependence structure.
            </p>
          </div>
          <p className="sig-caption">
            <b>Figure 6.</b> The {m} ordered p-values on a logarithmic y-axis
            (each labelled tick is 10 times the one below it). BY
            finds the largest rank k with p(k) below the red BY threshold, and
            rejects the null hypothesis for that feature and every feature of
            smaller rank (filled dots). The dashed line is the uncorrected cut
            p ≤ {q}; the grey line is the less strict BH threshold.
          </p>
          </div>
          </div>
        </div>
        <div className="sig-card">
          <h3>The step-up procedure, row by row</h3>
          <p>
            Sort the p-values in ascending order. Rank k gets a threshold that
            grows with k. Find the largest k whose p-value is at or below its
            threshold, then retain that feature and every feature ranked above
            it.
          </p>
          <div className="training-table-scroll" tabIndex={0} role="region" aria-label="BY procedure table">
            <table className="training-table sig-table sig-mini">
              <thead>
                <tr>
                  <th>k</th>
                  <th>feature</th>
                  <th>test</th>
                  <th>p</th>
                  <th>BH thr.</th>
                  <th>BY thr.</th>
                  <th>BY</th>
                </tr>
              </thead>
              <tbody>
                {rows.map((row) => (
                  <tr key={row.name} data-kept={row.keptBy}>
                    <th>{row.rank}</th>
                    <td><code>{row.name}</code></td>
                    <td>{row.test}</td>
                    <td>{formatP(row.p)}</td>
                    <td>{row.bh.toFixed(4)}</td>
                    <td>{row.by.toFixed(4)}</td>
                    <td><b>{row.keptBy ? "keep" : "drop"}</b></td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
          <p className="sig-caption">
            <b>Table 5.</b> The same procedure in numbers: each p-value beside
            its BH and BY thresholds (thr.). Shaded rows are retained by BY.
          </p>
          <p className="sig-note">
            Take <code>{rank5.name}</code>. On its own, p = {formatP(rank5.p)}{" "}
            {rank5.p <= q ? "is below the uncorrected level" : "already exceeds the uncorrected level"}.
            Its BY threshold at rank {rank5.rank} is {rank5.by.toFixed(4)}, so
            BY <b>{rank5.keptBy ? "retains" : "rejects"}</b> it.
          </p>
        </div>
      </div>
      <p className="sig-note">
        After the BY filter, tsfresh returns every p-value in ascending order.
        This repository retains the first top_n features (20 by default). If
        fewer than top_n pass BY, it falls back to the top_n smallest p-values
        so that a feature list is always produced.
      </p>
    </div>
  );
}
