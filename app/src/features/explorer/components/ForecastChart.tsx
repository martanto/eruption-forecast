import { useId, useMemo } from "react";
import { XAxis, YAxis } from "@/components/lab/ChartAxes";
import { useElementWidth } from "@/hooks/useElementWidth";
import { percent } from "@/lib/format";
import { linearScale, linePoints } from "@/lib/scale";
import { shortDate, utcStamp } from "@/lib/time";
import {
  DAY_MS,
  primarySeries,
  readRow,
  type ForecastDataset,
} from "../forecastDataset";
import "@/styles/prediction.css";

const HEIGHT = 230;
const PAD = { left: 40, right: 12, top: 14, bottom: 28 };

export interface ForecastChartProps {
  dataset: ForecastDataset;
  threshold: number;
  /** Row under the cursor. Omit both cursor props for a read-only chart. */
  cursorIndex?: number;
  onCursorChange?: (index: number) => void;
  /** Model key drawn as a second line, for example `RandomForestClassifier`. */
  overlayModel?: string;
}

/** Primary probability curve with threshold, eruption days and a scrub cursor. */
export function ForecastChart({
  dataset,
  threshold,
  cursorIndex,
  onCursorChange,
  overlayModel,
}: ForecastChartProps) {
  const [host, width] = useElementWidth();
  // Unique per chart, so several charts can share a page without clipping each other.
  const uid = useId().replace(/:/g, "");
  const { rows } = dataset;
  const first = rows[0]!.time;
  const lastTime = rows.at(-1)!.time;
  const plotW = width - PAD.left - PAD.right;
  const plotH = HEIGHT - PAD.top - PAD.bottom;
  const x = linearScale([first, lastTime], [PAD.left, width - PAD.right]);
  const y = linearScale([0, 1], [HEIGHT - PAD.bottom, PAD.top]);

  const series = useMemo(
    () => primarySeries(dataset, Math.min(480, Math.floor(plotW))),
    [dataset, plotW],
  );
  const curve = series.map((p) => [x(p.time), y(p.value)] as const);
  const line = linePoints(curve);
  const area = linePoints([
    [x(series[0]!.time), y(0)],
    ...curve,
    [x(series.at(-1)!.time), y(0)],
  ]);
  const overlay =
    overlayModel && overlayModel !== dataset.models[0]![0]
      ? linePoints(
          series.map(
            (p) =>
              [
                x(p.time),
                y(readRow(dataset, p.index, overlayModel).probability || 0),
              ] as const,
          ),
        )
      : null;
  const dates = dataset.eruptionDates.filter((date) => {
    const at = Date.parse(`${date}T00:00:00Z`);
    return at >= first && at <= lastTime + DAY_MS;
  });
  const days = Array.from(
    { length: Math.ceil((lastTime - first) / DAY_MS) + 1 },
    (_, i) => first + i * DAY_MS,
  ).filter((_, i) => i % (width < 520 ? 7 : 3) === 0);
  const hasCursor = cursorIndex !== undefined;
  const row = hasCursor ? rows[cursorIndex]! : null;
  const current = hasCursor
    ? readRow(dataset, cursorIndex, dataset.models[0]![0])
    : null;

  function scrub(event: React.PointerEvent<SVGSVGElement>) {
    if (!onCursorChange) return;
    const box = event.currentTarget.getBoundingClientRect();
    const ratio = Math.min(
      1,
      Math.max(0, (event.clientX - box.left - PAD.left) / plotW),
    );
    onCursorChange(Math.round(ratio * (rows.length - 1)));
  }

  return (
    <div className="pwr-chart" ref={host}>
      <svg
        width={width}
        height={HEIGHT}
        role="img"
        data-scrub={onCursorChange ? "true" : undefined}
        aria-label={`${dataset.label} ${dataset.models[0]![1].toLowerCase()} probability, ${shortDate(first)} to ${shortDate(lastTime)}.${
          row && current
            ? ` Selected window ${utcStamp(row.time)} UTC, ${percent(current.probability)}.`
            : ""
        }`}
        onPointerDown={(event) => {
          if (!onCursorChange) return;
          event.currentTarget.setPointerCapture(event.pointerId);
          scrub(event);
        }}
        onPointerMove={(event) => {
          if (event.buttons) scrub(event);
        }}
      >
        <YAxis
          ticks={[0, 0.25, 0.5, 0.75, 1]}
          scale={y}
          left={PAD.left}
          right={width - PAD.right}
        />
        <XAxis ticks={days} scale={x} labelY={HEIGHT - 8} format={shortDate} />
        {dates.map((date) => {
          const px = x(Date.parse(`${date}T00:00:00Z`));
          return (
            <g key={date}>
              <line
                className="pwr-eruption"
                x1={px}
                x2={px}
                y1={PAD.top}
                y2={PAD.top + plotH}
              />
              <text className="pwr-eruption-text" x={px + 4} y={PAD.top + 11}>
                {shortDate(date)}
              </text>
            </g>
          );
        })}
        <line
          className="pwr-threshold"
          x1={PAD.left}
          x2={width - PAD.right}
          y1={y(threshold)}
          y2={y(threshold)}
        />
        <text
          className="pwr-threshold-text"
          x={width - PAD.right - 4}
          y={y(threshold) - 5}
          textAnchor="end"
        >
          threshold {threshold.toFixed(2)}
        </text>
        {overlay && <polyline className="pwr-focus" points={overlay} />}
        {dates.map((date) => {
          const start = Date.parse(`${date}T00:00:00Z`);
          // 23:59 UTC, so the band covers exactly the calendar day of the eruption.
          const end = start + DAY_MS - 60_000;
          return (
            <rect
              key={`day-${date}`}
              className="pwr-eruption-day"
              x={x(Math.max(start, first))}
              y={PAD.top}
              width={Math.max(
                0,
                x(Math.min(end, lastTime)) - x(Math.max(start, first)),
              )}
              height={plotH}
            >
              <title>{`Eruption day ${date}, 00:00 to 23:59 UTC`}</title>
            </rect>
          );
        })}
        <defs>
          <clipPath id={`${uid}-hot`}>
            <rect
              x={PAD.left}
              y={PAD.top}
              width={plotW}
              height={Math.max(0, y(threshold) - PAD.top)}
            />
          </clipPath>
          <clipPath id={`${uid}-calm`}>
            <rect
              x={PAD.left}
              y={y(threshold)}
              width={plotW}
              height={Math.max(0, PAD.top + plotH - y(threshold))}
            />
          </clipPath>
        </defs>
        <polygon
          className="pwr-area"
          points={area}
          clipPath={`url(#${uid}-calm)`}
        />
        <polygon
          className="pwr-area-hot"
          points={area}
          clipPath={`url(#${uid}-hot)`}
        />
        <polyline className="pwr-line" points={line} />
        {row && current && (
          <>
            <line
              className="pwr-cursor"
              x1={x(row.time)}
              x2={x(row.time)}
              y1={PAD.top}
              y2={PAD.top + plotH}
            />
            <circle
              className="pwr-dot"
              cx={x(row.time)}
              cy={y(current.probability || 0)}
              r="5"
            />
          </>
        )}
      </svg>
    </div>
  );
}
