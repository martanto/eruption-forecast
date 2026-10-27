import { useEffect, useRef, useState } from "react";
import { ArrowRight } from "lucide-react";
import { ChapterNav } from "@/components/lab/ChapterNav";
import { CodeSnippet } from "@/components/lab/CodeSnippet";
import { LabSectionTitle } from "@/components/lab/LabSectionTitle";
import { ParameterStrip } from "@/components/lab/ParameterStrip";
import { SelectField } from "@/components/lab/SelectField";
import { StepControls } from "@/components/lab/StepControls";
import { Button } from "@/components/ui/button";
import { utcStamp } from "@/lib/time";
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
import { WalkthroughHeader } from "./WalkthroughHeader";
import { WalkthroughMenu, WalkthroughPager } from "./WalkthroughNav";
import "@/styles/labelling.css";
import "@/styles/training.css";

const PARAMETERS = [
  ["Look-back", "2 days"],
  ["Window step", "6 hours"],
  ["Forecast", "3 calendar days, including eruption day"],
  ["Tremor", "10-minute samples"],
  ["Time", "UTC"],
] as const;
const LABEL_SOURCES = [
  { value: "continuous", label: "Continuous" },
  { value: "dynamic", label: "Dynamic" },
] as const;
const HISTORY_OPTIONS = [
  { value: 3, label: "3 days · separated slices" },
  { value: 7, label: "7 days · overlapping slices" },
  { value: 10, label: "10 days · longer overlap" },
] as const;

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
      <WalkthroughHeader
        title="From eruption dates to learning windows"
        intro="Generate timestamps, assign the binary labels, then extract the tremor window preceding each label."
      />
      <WalkthroughMenu current="labelling" />
      <WalkthroughPager next="training" label="Next walkthrough" />
      <ChapterNav
        titles={LESSONS.map((item) => item.title)}
        current={chapter}
        onSelect={changeChapter}
        label="Labelling walkthrough chapters"
      />
      <ParameterStrip items={PARAMETERS} />
      <StepControls
        steps={lesson.steps}
        step={step}
        onStepChange={setStep}
        groupLabel="Build steps"
        restartLabel="Restart chapter"
      />
      <div className="lab-options">
        {chapter === 2 && (
          <SelectField
            label="Label source"
            value={source}
            options={LABEL_SOURCES}
            onChange={setSource}
          />
        )}
        {mode === "dynamic" && (
          <SelectField
            label="History before each eruption"
            value={daysBefore}
            options={HISTORY_OPTIONS}
            onChange={setDaysBefore}
          />
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
        <LabSectionTitle
          title={
            chapter === 2
              ? "Inspect the tremor window preceding one label"
              : "One time axis, two ways to generate timestamps"
          }
          aside="Illustrative dates · March 2025"
        />
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
            <i className="lab-key-zero" />0 · negative class
          </span>
          <span>
            <i className="lab-key-one" />1 · positive class (pre-eruptive)
          </span>
          <span>
            <i className="lab-key-history" />
            2-day tremor slice
          </span>
          {mode === "dynamic" && <span>Gaps have no label</span>}
        </div>
      </section>
      <section className="lab-selection" aria-label="Select a label timestamp">
        <LabSectionTitle
          title={
            beforeMerge
              ? "Two event grids before merging"
              : "Select a label timestamp"
          }
          aside={`${beforeMerge ? `${rawCount} event rows` : `${labels.length} labels`}${
            labelled
              ? ` · ${positive} positive · ${labels.length - positive} negative`
              : " · all initialised to 0"
          }`}
        />
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
              <CodeSnippet
                code={`day_to_forecast=3
include_eruption_date=True
window_step=6, window_step_unit="hours"`}
              />
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
      {chapter === 2 && (
        <section
          className="lab-selection"
          aria-label="Stacked tremor matrix example"
        >
          <h2>Stack the windows by label</h2>
          <p>
            Each row below summarises one 288-row window in the long-format
            tremor matrix. Move the selected label above to compare consecutive
            windows. All timestamps are UTC.
          </p>
          <div className="lab-table-scroll">
            <table>
              <caption>
                Window summary · complete ten-minute tremor data assumed
              </caption>
              <thead>
                <tr>
                  <th>Label timestamp</th>
                  <th>Tremor data · first → last measurement</th>
                  <th>Eruption target</th>
                </tr>
              </thead>
              <tbody>
                {labels.slice(selectedIndex, selectedIndex + 4).map((row) => (
                  <tr key={row.id}>
                    <th scope="row">
                      {utcStamp(row.time)}
                      <br />
                      <small>id {row.id}</small>
                    </th>
                    <td>
                      {utcStamp(row.time - 2 * DAY)}
                      {" → "}
                      {utcStamp(row.time - 600_000)}
                    </td>
                    <td>{row.positive}</td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
          <p>
            The target is shown alongside each window for explanation; it stays
            in the separate label table used for training. Consecutive labels
            six hours apart share 252 measurements (42 hours, or 87.5%).
          </p>
        </section>
      )}
      {chapter === 2 && (
        <section
          className="lab-selection"
          aria-label="Construction of the tremor matrix"
        >
          <h2>Construction of the tremor matrix</h2>
          <h3>1. Generate labels</h3>
          <p>
            Generate a timestamp every 6 hours across the study period: for
            example, 2026-04-20 00:00:00 followed by 2026-04-20 06:00:00 UTC.
            Each timestamp identifies one sample. In the implementation, a
            numeric id links that timestamp to its tremor rows and binary
            target. The target is 1 for the pre-eruptive class and 0 otherwise.
          </p>
          <p>
            This walkthrough uses three inclusive calendar dates: the two dates
            before an eruption and the eruption date itself. This matches the
            builders’ calendar-day rule, rather than a rolling 72-hour interval
            before an exact onset time. The eruption class represents the period
            the model is trained to recognise.
          </p>
          <h3>2. Extract the preceding tremor window</h3>
          <p>
            For each label, extract the 2 days immediately before its timestamp
            at 10-minute resolution: 2 × 24 × 6 = 288 measurements per tremor
            column. For label 2026-04-20 00:00:00, the first measurement is
            2026-04-18 00:00:00 and the last is 2026-04-19 23:50:00 UTC. The
            label timestamp is excluded, so the sample contains only earlier
            measurements.
          </p>
          <p>
            The next label advances by 6 hours while the window stays 48 hours
            long. Its measurements run from 2026-04-18 06:00:00 through
            2026-04-20 05:50:00 UTC. Consecutive windows therefore overlap by 42
            hours: 252 measurements, or 87.5% of each window. These densely
            spaced samples track how tremor evolves toward an eruption.
          </p>
          <h3>3. Stack windows into the tremor matrix</h3>
          <p>
            Stack accepted windows vertically into a long-format table. Each row
            contains the sample id, measurement timestamp and tremor values;
            each complete id occupies 288 rows. Shared measurements repeat under
            different ids. A separate target vector links each id to its binary
            eruption class. Time-series feature extraction, such as tsfresh,
            computes features for each id; those features are then paired with
            the target to train the classifier.
          </p>
          <h3>Overlap and validation</h3>
          <p>
            Adjacent samples share most of their measurements and are strongly
            correlated. Randomly splitting these windows can place nearly
            identical samples in training and validation sets and inflate
            performance. For independent evaluation, use a chronological holdout
            or hold out eruption episodes, and keep overlapping windows across
            the split apart. Earlier-only inputs prevent future measurements
            entering a sample; they do not by themselves prevent leakage between
            datasets.
          </p>
          <p>
            The package currently uses randomized internal cross-validation
            strategies, including the default shuffle-stratified strategy. A
            separate future-period or eruption-episode evaluation is therefore
            needed to assess generalisation beyond those internal scores.
          </p>
        </section>
      )}
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
            <CodeSnippet inline code="update_df_eruptions()" />;{" "}
            <code>label/dynamic_label_builder.py</code> →{" "}
            <CodeSnippet inline code="build()" />;{" "}
            <code>utils/window.py</code> →{" "}
            <CodeSnippet inline code="construct_windows()" />;{" "}
            <code>features/tremor_matrix_builder.py</code> →{" "}
            <CodeSnippet inline code="_build_tremor_matrices()" />. Relative to{" "}
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
