import { confusionAt, rocFor } from "../rocAnalysis";
import type { ForecastDataset } from "../forecastDataset";
import "@/styles/prediction.css";

const W = 300;
const H = 270;
const PAD = { left: 40, right: 12, top: 26, bottom: 38 };
const PALETTE = [
  "var(--color-amber)",
  "var(--color-burgundy)",
  "var(--color-success)",
  "var(--color-rose)",
  "var(--color-terracotta)",
];
/** The primary (first) model is drawn in ink; the others cycle through the palette. */
export const rocColor = (dataset: ForecastDataset, model: string) => {
  const index = dataset.models.findIndex(([key]) => key === model);
  return index <= 0
    ? "var(--color-ink)"
    : PALETTE[(index - 1) % PALETTE.length]!;
};

const x = (value: number) => PAD.left + value * (W - PAD.left - PAD.right);
const y = (value: number) => PAD.top + (1 - value) * (H - PAD.top - PAD.bottom);
export const ratio = (num: number, den: number) =>
  den === 0 ? Number.NaN : num / den;
export const show = (value: number) =>
  Number.isFinite(value) ? value.toFixed(3) : "n/a";

export interface RocPanelProps {
  dataset: ForecastDataset;
  /** Model key, for example `RandomForestClassifier` or `consensus`. */
  model: string;
  title: string;
  threshold: number;
}

/** One ROC curve with its AUC, an uncertainty band and a dot at the chosen threshold. */
export function RocPanel({ dataset, model, title, threshold }: RocPanelProps) {
  const roc = rocFor(dataset, model);
  const counts = confusionAt(dataset, model, threshold);
  const fpr = ratio(counts.fp, counts.fp + counts.tn);
  const tpr = ratio(counts.tp, counts.tp + counts.fn);
  const color = rocColor(dataset, model);
  // Thin the curve for drawing; every distinct score still counts toward the AUC.
  const stride = Math.max(1, Math.floor(roc.points.length / 400));
  const line = roc.points
    .filter((_, i) => i % stride === 0 || i === roc.points.length - 1)
    .map((p) => `${x(p.fpr).toFixed(1)},${y(p.tpr).toFixed(1)}`)
    .join(" ");
  const lowerReversed = [...roc.lower].reverse();
  const band = [
    ...roc.grid.map(
      (g, i) => `${x(g).toFixed(1)},${y(roc.upper[i]!).toFixed(1)}`,
    ),
    ...[...roc.grid]
      .reverse()
      .map((g, i) => `${x(g).toFixed(1)},${y(lowerReversed[i]!).toFixed(1)}`),
  ].join(" ");
  return (
    <figure className="roc-panel">
      <svg
        viewBox={`0 0 ${W} ${H}`}
        role="img"
        aria-label={`${title} ROC curve, AUC ${roc.auc.toFixed(3)}. At threshold ${threshold.toFixed(2)}: false positive rate ${show(fpr)}, true positive rate ${show(tpr)}.`}
      >
        <text className="roc-title" x={W / 2} y="14" textAnchor="middle">
          {title}
        </text>
        {[0, 0.2, 0.4, 0.6, 0.8, 1].map((tick) => (
          <g key={tick}>
            <line
              className="roc-grid"
              x1={x(0)}
              x2={x(1)}
              y1={y(tick)}
              y2={y(tick)}
            />
            <line
              className="roc-grid"
              x1={x(tick)}
              x2={x(tick)}
              y1={y(0)}
              y2={y(1)}
            />
            <text
              className="roc-axis"
              x={x(0) - 5}
              y={y(tick) + 4}
              textAnchor="end"
            >
              {tick.toFixed(1)}
            </text>
            <text
              className="roc-axis"
              x={x(tick)}
              y={y(0) + 14}
              textAnchor="middle"
            >
              {tick.toFixed(1)}
            </text>
          </g>
        ))}
        <text
          className="roc-axis-label"
          x={x(0.5)}
          y={H - 6}
          textAnchor="middle"
        >
          False positive rate
        </text>
        <text
          className="roc-axis-label"
          transform={`translate(10 ${y(0.5)}) rotate(-90)`}
          textAnchor="middle"
        >
          True positive rate
        </text>
        <line className="roc-chance" x1={x(0)} y1={y(0)} x2={x(1)} y2={y(1)} />
        <polygon points={band} fill={color} fillOpacity="0.16" />
        <polyline
          points={line}
          fill="none"
          stroke={color}
          strokeWidth="2"
          strokeLinejoin="round"
        />
        {Number.isFinite(fpr) && Number.isFinite(tpr) && (
          <>
            <line
              className="roc-guide"
              x1={x(fpr)}
              x2={x(fpr)}
              y1={y(0)}
              y2={y(tpr)}
            />
            <line
              className="roc-guide"
              x1={x(0)}
              x2={x(fpr)}
              y1={y(tpr)}
              y2={y(tpr)}
            />
            <circle
              className="roc-dot"
              cx={x(fpr)}
              cy={y(tpr)}
              r="5"
              fill={color}
            />
          </>
        )}
        <text
          className="roc-legend"
          x={x(1) - 4}
          y={y(0) - 8}
          textAnchor="end"
          fill={color}
        >
          AUC = {roc.auc.toFixed(3)}
        </text>
      </svg>
    </figure>
  );
}
