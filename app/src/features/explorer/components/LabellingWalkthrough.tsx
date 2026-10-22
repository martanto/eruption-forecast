import { useEffect, useRef, useState } from "react";
import { Link } from "react-router-dom";
import { ArrowLeft, ArrowRight, RotateCcw } from "lucide-react";
import { Button } from "@/components/ui/button";
import { ThemeToggle } from "@/components/common/ThemeToggle";
import {
  buildLabels,
  DAY,
  START,
  END,
  ERUPTION_DATES,
  eruptionWindows,
  LESSONS,
  matrixRows,
  stamp,
  tremorValue,
  type LabelMode,
} from "../labelling";
import "@/styles/labelling.css";

export function LabellingWalkthrough() {
  const [chapter, setChapter] = useState(0);
  const [step, setStep] = useState(0);
  const [daysBefore, setDaysBefore] = useState(7);
  const [source, setSource] = useState<LabelMode>("continuous");
  const [selectedTime, setSelectedTime] = useState(
    ERUPTION_DATES[0]! - 2 * DAY,
  );
  const [missing, setMissing] = useState(false);
  const ruler = useRef<HTMLDivElement>(null);
  const mode =
    chapter === 1 ? "dynamic" : chapter === 0 ? "continuous" : source;
  const labels = buildLabels(mode, daysBefore);
  const selected =
    labels.find((row) => row.time === selectedTime) ?? labels[0]!;
  const selectedIndex = labels.indexOf(selected);
  const lesson = LESSONS[chapter]!;
  const operation = lesson.steps[step]!;
  const labelled = chapter === 0 ? step >= 1 : chapter === 1 ? step >= 2 : true;
  const showSlice =
    chapter === 2 ? step >= 1 : chapter === 0 ? step >= 2 : step >= 3;
  const samples = matrixRows(selected, missing);
  const rejected = chapter === 2 && step >= 2 && missing;
  const gridStart = Math.min(START, ...labels.map((row) => row.time)) - 2 * DAY;
  const gridEnd = END;
  const x = (time: number) =>
    85 + ((time - gridStart) / (gridEnd - gridStart)) * 930;
  const positive = labels.filter((row) => row.positive).length;
  const eventWindows = eruptionWindows(daysBefore);
  const rawCount = eventWindows.flat().length;
  const beforeMerge = chapter === 1 && step === 0;
  const rulerPosition = x(selected.time);
  useEffect(() => {
    const host = ruler.current;
    if (!host) return;
    const center = () => {
      if (host.scrollWidth > host.clientWidth)
        host.scrollLeft = rulerPosition - host.clientWidth * 0.45;
    };
    center();
    const observer = new ResizeObserver(center);
    observer.observe(host);
    return () => observer.disconnect();
  }, [rulerPosition, chapter, step]);
  const visibleRows = [0, 1, 2, samples.length - 2, samples.length - 1].map(
    (index) => samples[index]!,
  );
  const line = Array.from({ length: 230 }, (_, index) => {
    const time = gridStart + (index / 229) * (gridEnd - gridStart);
    return `${x(time)},${185 - tremorValue(time) * 2.3}`;
  }).join(" ");
  function changeChapter(index: number) {
    setChapter(index);
    setStep(0);
  }
  return (
    <main className="label-lab">
      <header className="lab-heading">
        <div>
          <Link className="lab-back" to="/">
            <ArrowLeft size={16} /> Back to Pipeline Atlas
          </Link>
          <h1>From eruption dates to learning windows</h1>
          <p>
            Follow the clock, mark the answers, then collect the tremor behind
            each answer.
          </p>
        </div>
        <ThemeToggle />
      </header>
      <nav className="lab-chapters" aria-label="Labelling walkthrough chapters">
        {LESSONS.map((item, index) => (
          <button
            key={item.title}
            aria-current={chapter === index ? "step" : undefined}
            onClick={() => changeChapter(index)}
          >
            <span>{index + 1}</span>
            {item.title}
          </button>
        ))}
      </nav>
      <div className="lab-parameters">
        <span>
          Look-back <strong>2 days</strong>
        </span>
        <span>
          Slide <strong>6 hours</strong>
        </span>
        <span>
          Forecast <strong>3 calendar days, including eruption day</strong>
        </span>
        <span>
          Tremor <strong>10-minute samples</strong>
        </span>
        <span>
          Time <strong>UTC</strong>
        </span>
      </div>
      <section className="lab-operation" aria-label="Walkthrough controls">
        <div className="lab-step-nav" role="group" aria-label="Build steps">
          {lesson.steps.map((item, index) => (
            <button
              key={item[0]}
              aria-label={`Step ${index + 1}: ${item[0]}`}
              aria-pressed={step === index}
              onClick={() => setStep(index)}
            >
              {index + 1}
            </button>
          ))}
        </div>
        <div className="lab-operation-copy" aria-live="polite">
          <h2>{operation[0]}</h2>
          <p>{operation[1]}</p>
        </div>
        <div className="lab-step-actions">
          <Button
            variant="ghost"
            disabled={step === 0}
            onClick={() => setStep(step - 1)}
          >
            Previous
          </Button>
          <Button
            variant="outline"
            disabled={step === 3}
            onClick={() => setStep(step + 1)}
          >
            Next step <ArrowRight size={15} />
          </Button>
          <Button
            variant="ghost"
            size="icon"
            aria-label="Restart chapter"
            onClick={() => setStep(0)}
          >
            <RotateCcw size={16} />
          </Button>
        </div>
      </section>
      <div className="lab-options">
        {chapter === 2 && (
          <label>
            Label source{" "}
            <select
              value={source}
              onChange={(event) => setSource(event.target.value as LabelMode)}
            >
              <option value="continuous">Continuous</option>
              <option value="dynamic">Dynamic</option>
            </select>
          </label>
        )}
        {mode === "dynamic" && (
          <label>
            History before each eruption{" "}
            <select
              value={daysBefore}
              onChange={(event) => setDaysBefore(Number(event.target.value))}
            >
              <option value={3}>3 days · separated slices</option>
              <option value={7}>7 days · overlapping slices</option>
              <option value={10}>10 days · longer overlap</option>
            </select>
          </label>
        )}
        {chapter === 2 && step >= 2 && (
          <label className="lab-checkbox">
            <input
              type="checkbox"
              checked={missing}
              onChange={(event) => setMissing(event.target.checked)}
            />{" "}
            Remove every fourth sample from selected window
          </label>
        )}
      </div>
      <section className="lab-visual" aria-label="Shared time ruler">
        <div className="lab-section-title">
          <h2>
            {chapter === 2
              ? "Read the history behind one label"
              : "One clock, two ways to choose timestamps"}
          </h2>
          <span>Illustrative dates · March 2025</span>
        </div>
        <p className="lab-ruler-hint" id="ruler-help">
          The ruler follows your selected timestamp. Scroll horizontally to
          explore other dates; focus the ruler and use Left/Right arrow keys.
        </p>
        <div
          className="lab-ruler-scroll"
          ref={ruler}
          role="region"
          tabIndex={0}
          aria-label="Scrollable time ruler"
          aria-describedby="ruler-help"
        >
          <svg
            className="lab-ruler"
            viewBox="0 0 1060 244"
            role="img"
            aria-label={`Eruptions on March 20 and 25. ${showSlice ? `Selected window from ${stamp(selected.time - 2 * DAY)} to ${stamp(selected.time)}, right edge excluded.` : "Labels occur every six hours."}`}
          >
            {Array.from(
              { length: Math.ceil((gridEnd - gridStart) / DAY) },
              (_, index) => gridStart + index * DAY,
            ).map((time) => (
              <g key={time}>
                <line
                  x1={x(time)}
                  x2={x(time)}
                  y1={34}
                  y2={204}
                  className="lab-grid-line"
                />
                {indexLabel(time, gridStart) && (
                  <text x={x(time)} y={229} textAnchor="middle">
                    {new Date(time).getUTCDate()} Mar
                  </text>
                )}
              </g>
            ))}
            {labelled &&
              ERUPTION_DATES.map((time) => (
                <rect
                  key={time}
                  x={x(time - 2 * DAY)}
                  y={36}
                  width={x(time + DAY) - x(time - 2 * DAY)}
                  height={168}
                  className="lab-positive-zone"
                />
              ))}
            {mode === "dynamic" &&
              eventWindows.map((times, index) => (
                <g key={index}>
                  <text x={12} y={62 + index * 28}>
                    Event {index + 1}
                  </text>
                  <rect
                    x={x(times[0]!)}
                    y={48 + index * 28}
                    width={x(times.at(-1)! + 6 * 3_600_000) - x(times[0]!)}
                    height={17}
                    className="lab-event-range"
                  />
                </g>
              ))}
            {beforeMerge &&
              eventWindows.map((times, event) =>
                times.map((time) => (
                  <circle
                    key={`${event}-${time}`}
                    cx={x(time)}
                    cy={56 + event * 28}
                    r={2.5}
                    className="lab-negative-dot"
                  />
                )),
              )}
            <text x={12} y={118}>
              {beforeMerge ? "Not merged" : "Labels"}
            </text>
            {!beforeMerge &&
              labels.map((row) => (
                <circle
                  key={row.time}
                  cx={x(row.time)}
                  cy={114}
                  r={row.time === selected.time ? 5 : 2.5}
                  className={
                    labelled && row.positive
                      ? "lab-positive-dot"
                      : "lab-negative-dot"
                  }
                />
              ))}
            {showSlice && (
              <rect
                x={x(selected.time - 2 * DAY)}
                y={134}
                width={x(selected.time) - x(selected.time - 2 * DAY)}
                height={65}
                className="lab-history-zone"
              />
            )}
            <text x={12} y={177}>
              Tremor
            </text>
            <polyline points={line} className="lab-signal" />
            {ERUPTION_DATES.map((time) => (
              <g key={time}>
                <line
                  x1={x(time)}
                  x2={x(time)}
                  y1={30}
                  y2={204}
                  className="lab-eruption-line"
                />
                <text
                  x={x(time)}
                  y={20}
                  textAnchor="middle"
                  className="lab-eruption-text"
                >
                  Eruption · {new Date(time).getUTCDate()} Mar
                </text>
              </g>
            ))}
            <line
              x1={x(selected.time)}
              x2={x(selected.time)}
              y1={102}
              y2={204}
              className="lab-selected-line"
            />
          </svg>
        </div>
        <div className="lab-legend">
          <span>
            <i className="lab-key-zero" />0 · outside positive days
          </span>
          <span>
            <i className="lab-key-one" />1 · positive days
          </span>
          <span>
            <i className="lab-key-history" />
            2-day tremor slice
          </span>
          {mode === "dynamic" && <span>Gaps have no label</span>}
        </div>
      </section>
      <section className="lab-selection" aria-label="Select a label timestamp">
        <div className="lab-section-title">
          <h2>
            {beforeMerge
              ? "Two event grids before merging"
              : "Select a label timestamp"}
          </h2>
          <span>
            {beforeMerge ? `${rawCount} event rows` : `${labels.length} labels`}
            {labelled
              ? ` · ${positive} positive · ${labels.length - positive} negative`
              : " · all initialised to 0"}
          </span>
        </div>
        <p>
          {beforeMerge
            ? "Each event starts its own ids at 0. Shared dates appear in both grids; the next step keeps each date once."
            : "Each cell is a six-hour step. Select a cell, or use the window controls."}
        </p>
        {beforeMerge ? (
          eventWindows.map((times, event) => (
            <div key={event}>
              <h3>
                Event {event + 1} · eruption{" "}
                {stamp(ERUPTION_DATES[event]!).slice(0, 5)}
              </h3>
              <div className="lab-label-grid">
                {times.map((time, localId) => (
                  <button
                    key={time}
                    aria-label={`Event ${event + 1}, local id ${localId}, ${stamp(time)} UTC, 0`}
                    aria-pressed={time === selected.time}
                    onClick={() => setSelectedTime(time)}
                    title={`${stamp(time)} UTC · event ${event + 1} · local id ${localId}`}
                  >
                    0
                  </button>
                ))}
              </div>
            </div>
          ))
        ) : (
          <div className="lab-label-grid">
            {labels.map((row) => (
              <button
                key={row.time}
                className={labelled && row.positive ? "lab-label-positive" : ""}
                aria-pressed={row.time === selected.time}
                aria-label={`Label ${row.id}, ${stamp(row.time)} UTC, ${labelled ? row.positive : 0}`}
                title={`${stamp(row.time)} UTC · id ${row.id}`}
                onClick={() => setSelectedTime(row.time)}
              >
                {labelled ? row.positive : 0}
              </button>
            ))}
          </div>
        )}
        <div className="lab-window-controls">
          <Button
            variant="outline"
            disabled={selectedIndex === 0}
            onClick={() => setSelectedTime(labels[selectedIndex - 1]!.time)}
          >
            Previous window
          </Button>
          <output aria-live="polite">
            {beforeMerge ? "Unmerged date" : `id ${selected.id}`} ·{" "}
            {stamp(selected.time)} UTC · is_erupted ={" "}
            {labelled ? selected.positive : 0}
          </output>
          <Button
            variant="outline"
            disabled={selectedIndex === labels.length - 1}
            onClick={() => setSelectedTime(labels[selectedIndex + 1]!.time)}
          >
            Next window
          </Button>
        </div>
      </section>
      <section className="lab-output" aria-label="Current operation output">
        <div>
          <h2>
            {chapter === 2
              ? "From a slice to long rows"
              : mode === "dynamic" && step < 2
                ? "Event grids → unique timestamps"
                : "The label table"}
          </h2>
          {mode === "dynamic" && (
            <p>
              {rawCount} event-grid rows − {rawCount - labels.length} duplicate
              timestamps = <strong>{labels.length} unique labels</strong>.{" "}
              {step === 0
                ? "The two event ranges above show the grids before merging."
                : "Ids are reassigned after sorting the merged timestamps."}
            </p>
          )}
          {chapter === 2 ? (
            <>
              <p>
                <strong>
                  [{stamp(selected.time - 2 * DAY)}, {stamp(selected.time)}) UTC
                </strong>
                <br />
                The opening bracket includes the start. The closing parenthesis
                excludes the label timestamp.
              </p>
              <p>
                {step >= 2
                  ? `${samples.length} / 288 samples = ${((samples.length / 288) * 100).toFixed(0)}% completeness. ${rejected ? "Skipped: below the 100% requirement." : "Accepted: meets the 100% requirement."}`
                  : "A complete window contains 288 ten-minute samples."}
              </p>
              {step === 3 && (
                <p>
                  {labels.length - (missing ? 1 : 0)} accepted windows × 288 ={" "}
                  <strong>
                    {(labels.length - (missing ? 1 : 0)) * 288} matrix rows
                  </strong>
                  , assuming complete tremor for all other windows. Adjacent
                  windows share 252 samples, copied under different ids.
                </p>
              )}
              <p>
                The values are synthetic. The matrix contains tremor metrics,
                while the target remains in the separate label table.
              </p>
            </>
          ) : (
            <>
              <p>
                The label timestamp is the right edge of its preceding tremor
                slice. It is not the start of a two-day block.
              </p>
              <p>
                <code>day_to_forecast=3</code>
                <br />
                <code>include_eruption_date=True</code>
                <br />
                <code>window_step=6, window_step_unit="hours"</code>
              </p>
              <p>
                For 20 March: positives run from 18 March 00:00 through 20 March
                18:00 on this grid. On 21 March 00:00 the label returns to 0.
              </p>
            </>
          )}
        </div>
        <div className="lab-table-scroll">
          {chapter === 2 ? (
            <table>
              <caption>
                {rejected
                  ? "Rejected slice · preview only, no rows appended"
                  : step < 3
                    ? "Selected tremor slice · preview before stacking"
                    : "Selected id in the stacked tremor matrix"}
              </caption>
              <thead>
                <tr>
                  {step === 3 && <th>id</th>}
                  <th>datetime (UTC)</th>
                  <th>rsam_f0</th>
                  <th>entropy</th>
                </tr>
              </thead>
              <tbody>
                {visibleRows.map((row, index) => (
                  <tr key={row.time}>
                    {step === 3 && <td>{row.id}</td>}
                    <td>
                      {index === 3 && (
                        <span className="lab-ellipsis">
                          …<br />
                        </span>
                      )}
                      {stamp(row.time)}
                    </td>
                    <td>{row.rsam.toFixed(3)}</td>
                    <td>{row.entropy.toFixed(3)}</td>
                  </tr>
                ))}
              </tbody>
            </table>
          ) : beforeMerge ? (
            <table>
              <caption>
                Unmerged event rows near selected date · shared timestamps
                repeat
              </caption>
              <thead>
                <tr>
                  <th>event</th>
                  <th>local id</th>
                  <th>datetime (UTC)</th>
                  <th>is_erupted</th>
                </tr>
              </thead>
              <tbody>
                {eventWindows
                  .flatMap((times, event) =>
                    times.map((time, localId) => ({ time, localId, event })),
                  )
                  .filter(
                    (row) =>
                      Math.abs(row.time - selected.time) <= 6 * 3_600_000,
                  )
                  .map((row) => (
                    <tr key={`${row.event}-${row.time}`}>
                      <td>{row.event + 1}</td>
                      <td>{row.localId}</td>
                      <td>{stamp(row.time)}</td>
                      <td>0</td>
                    </tr>
                  ))}
              </tbody>
            </table>
          ) : (
            <table>
              <caption>
                Rows around the selected timestamp
                {!labelled ? " · before positive marking" : ""}
              </caption>
              <thead>
                <tr>
                  <th>datetime (UTC)</th>
                  <th>id</th>
                  <th>is_erupted</th>
                </tr>
              </thead>
              <tbody>
                {labels
                  .slice(
                    Math.max(0, selectedIndex - 2),
                    Math.min(labels.length, selectedIndex + 3),
                  )
                  .map((row) => (
                    <tr key={row.id} aria-selected={row.id === selected.id}>
                      <td>{stamp(row.time)}</td>
                      <td>{row.id}</td>
                      <td>{labelled ? row.positive : 0}</td>
                    </tr>
                  ))}
              </tbody>
            </table>
          )}
        </div>
      </section>
      <footer className="lab-footer">
        <p>
          Teaching fixture with synthetic tremor. This walkthrough reproduces
          the builders’ calendar-day rules; it does not run Python or train a
          forecast model.
        </p>
        <details>
          <summary>Reference and implementation</summary>
          <p>
            Reference:{" "}
            <a
              href={
                new URL(
                  "../../../../../assets/How Eruption Forecasting Works.pdf",
                  import.meta.url,
                ).href
              }
            >
              How Eruption Forecasting Works.pdf
            </a>
            , labelling and tremor-matrix sections. The PDF’s two-day horizon
            plus an additional eruption day produces the same three positive
            dates as the requested three-day horizon with eruption day included.
          </p>
          <p>
            Implementation: <code>label/label_builder.py</code> →{" "}
            <code>update_df_eruptions</code>;{" "}
            <code>label/dynamic_label_builder.py</code> → <code>build</code>;{" "}
            <code>utils/window.py</code> → <code>construct_windows</code>;{" "}
            <code>features/tremor_matrix_builder.py</code> →{" "}
            <code>_build_tremor_matrices</code>. Relative to{" "}
            <code>src/eruption_forecast/</code>. The example includes two days
            of tremor before the first label.
          </p>
        </details>
        <div className="lab-step-actions">
          {chapter < 2 && (
            <Button onClick={() => changeChapter(chapter + 1)}>
              Continue to {LESSONS[chapter + 1]!.title} <ArrowRight size={16} />
            </Button>
          )}
        </div>
      </footer>
    </main>
  );
}

function indexLabel(time: number, start: number) {
  return Math.round((time - start) / DAY) % 2 === 0;
}
