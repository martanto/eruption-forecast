import { useState } from "react";
import { DataTable } from "@/components/lab/DataTable";
import { MathExpr } from "@/components/lab/MathExpr";
import { SelectField } from "@/components/lab/SelectField";
import { integer } from "@/lib/format";
import { PRE_COUNT, QUIET_COUNT } from "../dummyDataset";
import { comb, fisherRows, fisherTable, formatP } from "../significance";

const LETTERS = "ABCDE";

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


/** The "how C(n, r) works" lesson: a tiny enumerated example, then the three coefficients for the live table. */
export function CombinationsLesson({
  table,
  fisher,
}: {
  table: ReturnType<typeof fisherTable>;
  fisher: ReturnType<typeof fisherRows>;
}) {
  const [n, setN] = useState(4);
  const [r, setR] = useState(2);
  const { yes, total, ways, rows, waysInP, p } = fisher;
  const listed = pairs(n, r);
  const topShort = (size: number, pick: number) =>
    `(${size} × ${size - 1} × … × ${size - pick + 1}) / (${pick} × ${Math.max(pick - 1, 1)} × … × 1)`;

  return (
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
              <SelectField
                label="n (windows)"
                value={n}
                options={[3, 4, 5].map((i) => ({ value: i, label: i }))}
                onChange={(next) => {
                  setN(next);
                  setR(Math.min(r, next));
                }}
              />
              <SelectField
                label="r (to select)"
                value={r}
                options={Array.from({ length: n }, (_, i) => ({ value: i + 1, label: i + 1 }))}
                onChange={setR}
              />
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
              <MathExpr
                display
                tex={String.raw`\binom{n}{r} = \frac{n!}{r!\,(n-r)!}`}
              />
              <p>
                Shortcut: multiply r factors counting down from n, then divide
                by <MathExpr tex={String.raw`r! = r(r-1)\cdots 1`} />.
              </p>
              <MathExpr
                display
                tex={String.raw`\binom{${n}}{${r}} = ${comb(n, r)}`}
              />
              <p>In Python: <code>math.comb({n}, {r})</code></p>
            </div>
          </div>
          <div>
            <h4>Next, the three binomial coefficients for this table</h4>
            <ul className="sig-threec">
              <li>
                <b>C({PRE_COUNT}, {table.preYes})</b>
                <strong data-group="pre">{integer(comb(PRE_COUNT, table.preYes))}</strong>
                <span>Combinations in which {table.preYes} of the {PRE_COUNT} pre-eruptive windows have feature = 1.</span>
                <small>{topShort(PRE_COUNT, Math.min(table.preYes, PRE_COUNT - table.preYes))}</small>
              </li>
              <li>
                <b>C({QUIET_COUNT}, {table.quietYes})</b>
                <strong data-group="quiet">{integer(comb(QUIET_COUNT, table.quietYes))}</strong>
                <span>Combinations in which {table.quietYes} of the {QUIET_COUNT} non-eruptive windows have feature = 1.</span>
                <small>{topShort(QUIET_COUNT, Math.min(table.quietYes, QUIET_COUNT - table.quietYes))}</small>
              </li>
              <li>
                <b>C({total}, {yes})</b>
                <strong>{integer(ways)}</strong>
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
            <h4>Finally, multiply, divide and sum</h4>
            <p>
              <b>Multiply</b> by the multiplication principle: each combination
              of pre-eruptive windows pairs with each combination of
              non-eruptive windows. <b>Divide</b> by C({total}, {yes}): under
              the null hypothesis every arrangement is equally likely, so
              probability = number of favourable arrangements / total number of
              arrangements. <b>Sum</b> the probabilities of all outcomes at
              least as extreme as the one observed.
            </p>
            <div className="sig-multiply">
            <div>
            <DataTable
              label="Fisher probability table"
              className="sig-table sig-mini"
              headers={[
                "x",
                `C(${PRE_COUNT}, x) × C(${QUIET_COUNT}, ${yes} − x)`,
                "P(x)",
                "in p?",
              ]}
            >
              {rows.map((row) => (
                <tr key={row.x} data-observed={row.x === table.preYes || undefined}>
                  <th>{row.x}</th>
                  <td>
                    {integer(row.top)} × {integer(row.bottom)} ={" "}
                    {integer(row.count)}
                  </td>
                  <td>{row.probability < 0.001 ? row.probability.toExponential(1) : row.probability.toFixed(4)}</td>
                  <td>{row.inP ? <b>yes</b> : "no"}</td>
                </tr>
              ))}
            </DataTable>
            <p className="sig-caption">
              <b>Table 4.</b> Each possible count x, the number of arrangements
              that produce it, and its probability P(x). Rows marked “yes” are
              as extreme as the observed count or more; their arrangements are
              summed for the p-value.
            </p>
            </div>
            <div className="sig-result" data-tone="plain">
              <p>
                Arrangements in the “yes” rows: {integer(waysInP)}
              </p>
              <p className="sig-total">
                p = {integer(waysInP)} / {integer(ways)} = {formatP(p)}
              </p>
              <p>
                Check: all rows together add up to {integer(rows.reduce((sum, row) => sum + row.count, 0))} = C({total}, {yes}), so the probabilities sum to 1.
              </p>
            </div>
            </div>
          </div>
        </div>
      </section>
  );
}
