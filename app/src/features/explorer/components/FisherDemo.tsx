import { useState } from "react";
import { DUMMY_FEATURES, PRE_COUNT, QUIET_COUNT } from "../dummyDataset";
import { comb, fisherRows, fisherTable } from "../significance";
import { formatP, verdict } from "./KsTestDemo";

const BINARY = DUMMY_FEATURES.filter((feature) => feature.kind === "binary");
const MEANING: Record<string, string> = {
  "rsam_f2__has_duplicate_max": "Indicator: 1 if the window's maximum amplitude occurs more than once, else 0.",
  "rsam_f2__symmetry_looking__r_0.05": "Indicator: 1 if the amplitude distribution in the window is symmetric about its mean, else 0.",
  "rsam_f2__large_standard_deviation__r_0.25": "Indicator: 1 if the window's standard deviation is large relative to its range, else 0.",
};
const LETTERS = "ABCDE";

function percent(probability: number) {
  const pct = probability * 100;
  if (pct < 0.01) return "<0.01%";
  if (pct < 1) return `${pct.toFixed(2)}%`;
  if (pct < 10) return `${pct.toFixed(1)}%`;
  return `${pct.toFixed(0)}%`;
}

function pairs(n: number, r: number) {
  const out: string[] = [];
  const walk = (start: number, picked: string[]) => {
    if (picked.length === r) {
      out.push(picked.join(" "));
      return;
    }
    for (let i = start; i < n; i++) walk(i + 1, [...picked, LETTERS[i]!]);
  };
  walk(0, []);
  return out;
}

export function FisherDemo() {
  const [name, setName] = useState("rsam_f2__large_standard_deviation__r_0.25");
  const [n, setN] = useState(4);
  const [r, setR] = useState(2);
  const feature = BINARY.find((item) => item.name === name)!;
  const table = fisherTable(feature);
  const { yes, total, ways, rows, waysInP, p } = fisherRows(table);
  const v = verdict(p);
  const peak = Math.max(...rows.map((row) => row.probability));
  const observed = rows.find((row) => row.x === table.preYes)!;
  const listed = pairs(n, r);
  const topShort = (size: number, pick: number) =>
    `(${size} × ${size - 1} × … × ${size - pick + 1}) / (${pick} × ${Math.max(pick - 1, 1)} × … × 1)`;

  return (
    <div className="sig-demo">
      <div className="lab-options" role="group" aria-label="Pick a yes/no feature">
        {BINARY.map((item) => (
          <button
            key={item.name}
            className="seed-select"
            aria-pressed={item.name === name}
            onClick={() => setName(item.name)}
          >
            <code>{item.name}</code>
          </button>
        ))}
      </div>
      <div className="sig-grid sig-grid-even">
        <div className="sig-card">
          <h3>3a · Tabulate the windows in a 2×2 contingency table</h3>
          <p>{MEANING[name]}</p>
          <div className="training-table-scroll">
            <table className="training-table sig-table">
              <thead>
                <tr>
                  <th />
                  <th>feature = 1</th>
                  <th>feature = 0</th>
                  <th>total</th>
                </tr>
              </thead>
              <tbody>
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
              </tbody>
            </table>
          </div>
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
          <div className="sig-scroll" tabIndex={0} role="region" aria-label="Chart, scrolls sideways on narrow screens">
          <svg
            className="sig-bars"
            viewBox={`0 0 ${rows.length * 52 + 20} 190`}
            role="img"
            aria-label={`Chance of each possible count. Observed count ${table.preYes}; p = ${formatP(p)}.`}
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
                  <text x={x + 20} y={134 - height} textAnchor="middle" className="sig-axis">
                    {percent(row.probability)}
                  </text>
                  <text x={x + 20} y={160} textAnchor="middle" className="sig-axis">
                    {row.x}
                  </text>
                  {row.x === table.preYes && (
                    <text x={x + 20} y={176} textAnchor="middle" className="sig-ours">observed</text>
                  )}
                </g>
              );
            })}
          </svg>
          </div>
          <p className="sig-axis-caption">
            x = number of pre-eruptive windows with feature = 1, under the null
            hypothesis
          </p>
          <ul className="sig-legend" aria-label="Legend">
            <li data-bar="in"><i aria-hidden="true" /> as extreme as observed, or more</li>
            <li data-bar="out"><i aria-hidden="true" /> less extreme</li>
          </ul>
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
              For the observed table: P({table.preYes}) = {observed.top.toLocaleString("en-US")} ×{" "}
              {observed.bottom.toLocaleString("en-US")} / {ways.toLocaleString("en-US")} ={" "}
              {observed.probability.toFixed(5)}. Two-sided p-value: the sum of
              P(x) over every outcome with P(x) ≤ P(observed), in both tails:{" "}
              <b>{formatP(p)}</b>.
            </p>
          </div>
        </div>
      </div>

      <section className="sig-card sig-comb" aria-labelledby="comb-heading">
        <h3 id="comb-heading">3c · How C(n, r) works</h3>
        <p>
          C stands for <b>combinations</b>. C(n, r), the binomial coefficient,
          counts the unordered subsets of r windows that can be drawn from n
          windows without replacement. It is read “n choose r”.
        </p>
        <div className="sig-comb-grid">
          <div>
            <h4>First, a tiny example</h4>
            <div className="lab-options">
              <label>
                n (windows){" "}
                <select
                  value={n}
                  onChange={(e) => {
                    const next = Number(e.target.value);
                    setN(next);
                    setR(Math.min(r, next));
                  }}
                >
                  {[3, 4, 5].map((i) => (
                    <option key={i}>{i}</option>
                  ))}
                </select>
              </label>
              <label>
                r (to pick){" "}
                <select value={r} onChange={(e) => setR(Number(e.target.value))}>
                  {Array.from({ length: n }, (_, i) => i + 1).map((i) => (
                    <option key={i}>{i}</option>
                  ))}
                </select>
              </label>
            </div>
            <div className="sig-letters" aria-label={`${n} windows`}>
              {LETTERS.slice(0, n)
                .split("")
                .map((letter) => (
                  <span key={letter}>{letter}</span>
                ))}
            </div>
            <p>All combinations of {r} windows:</p>
            <div className="sig-picks" aria-live="polite">
              {listed.map((pick) => (
                <span key={pick}>{pick}</span>
              ))}
            </div>
            <p className="sig-caption">
              <b>Figure 4.</b> {n} windows (grey) and every combination of {r}
              of them (orange). There are {listed.length}, so C({n}, {r}) ={" "}
              {listed.length}. Order is ignored: “A B” and “B A” are the same
              combination.
            </p>
            <div className="sig-formula">
              <code>C(n, r) = n! / (r! × (n − r)!)</code>
              <p>
                Shortcut: multiply r factors counting down from n, then divide
                by r! = r × (r − 1) × … × 1.
              </p>
              <code>
                C({n}, {r}) = {comb(n, r)}
              </code>
              <p>In Python: <code>math.comb({n}, {r})</code></p>
            </div>
          </div>
          <div>
            <h4>Next, the three binomial coefficients for this table</h4>
            <ul className="sig-threec">
              <li>
                <b>C({PRE_COUNT}, {table.preYes})</b>
                <strong data-group="pre">{comb(PRE_COUNT, table.preYes).toLocaleString("en-US")}</strong>
                <span>Combinations in which {table.preYes} of the {PRE_COUNT} pre-eruptive windows have feature = 1.</span>
                <small>{topShort(PRE_COUNT, Math.min(table.preYes, PRE_COUNT - table.preYes))}</small>
              </li>
              <li>
                <b>C({QUIET_COUNT}, {table.quietYes})</b>
                <strong data-group="quiet">{comb(QUIET_COUNT, table.quietYes).toLocaleString("en-US")}</strong>
                <span>Combinations in which {table.quietYes} of the {QUIET_COUNT} non-eruptive windows have feature = 1.</span>
                <small>{topShort(QUIET_COUNT, Math.min(table.quietYes, QUIET_COUNT - table.quietYes))}</small>
              </li>
              <li>
                <b>C({total}, {yes})</b>
                <strong>{ways.toLocaleString("en-US")}</strong>
                <span>Combinations of {yes} windows with feature = 1 among all {total}, ignoring class labels. This is the size of the sample space.</span>
              </li>
            </ul>
            <p className="sig-caption">
              <b>Table 3.</b> The three binomial coefficients for the observed
              table. By symmetry, C(n, r) = C(n, n − r): choosing the{" "}
              {table.preYes} pre-eruptive windows with feature = 1 is the same
              as choosing the {PRE_COUNT - table.preYes} without it.
            </p>
          </div>
          <div>
            <h4>Finally, multiply, divide, add up</h4>
            <p>
              <b>Multiply</b> by the multiplication principle: each combination
              of pre-eruptive windows pairs with each combination of
              non-eruptive windows. <b>Divide</b> by C({total}, {yes}): under
              the null hypothesis every arrangement is equally likely, so
              probability = favorable count / total count. <b>Add up</b> the
              outcomes as extreme as observed.
            </p>
            <div className="sig-multiply">
            <div>
            <div className="training-table-scroll" tabIndex={0} role="region" aria-label="Fisher probability table">
              <table className="training-table sig-table sig-mini">
                <thead>
                  <tr>
                    <th>x</th>
                    <th>C({PRE_COUNT}, x) × C({QUIET_COUNT}, {yes} − x)</th>
                    <th>P(x)</th>
                    <th>in p?</th>
                  </tr>
                </thead>
                <tbody>
                  {rows.map((row) => (
                    <tr key={row.x} data-observed={row.x === table.preYes || undefined}>
                      <th>{row.x}</th>
                      <td>
                        {row.top.toLocaleString("en-US")} × {row.bottom.toLocaleString("en-US")} ={" "}
                        {row.count.toLocaleString("en-US")}
                      </td>
                      <td>{row.probability < 0.001 ? row.probability.toExponential(1) : row.probability.toFixed(4)}</td>
                      <td>{row.inP ? <b>yes</b> : "no"}</td>
                    </tr>
                  ))}
                </tbody>
              </table>
            </div>
            <p className="sig-caption">
              <b>Table 4.</b> Each possible count x, the number of arrangements
              that produce it, and its probability P(x). Rows marked “yes” are
              as extreme as the observed count or more; their arrangements are
              summed for the p-value.
            </p>
            </div>
            <div className="sig-result" data-tone="plain">
              <p>
                Arrangements in the “yes” rows: {waysInP.toLocaleString("en-US")}
              </p>
              <p className="sig-total">
                p = {waysInP.toLocaleString("en-US")} / {ways.toLocaleString("en-US")} = {formatP(p)}
              </p>
              <p>
                Check: all rows together add up to {rows.reduce((s, r2) => s + r2.count, 0).toLocaleString("en-US")} = C({total}, {yes}), so the probabilities sum to 1.
              </p>
            </div>
            </div>
          </div>
        </div>
      </section>
      <p className="sig-note">
        Same two-sided p-value in scipy:{" "}
        <code>
          _, p = fisher_exact([[{table.preYes}, {table.preNo}], [{table.quietYes}, {table.quietNo}]])
        </code>
      </p>
    </div>
  );
}
