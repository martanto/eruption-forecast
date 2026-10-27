import { AxisTitle, XAxis, YAxis } from "@/components/lab/ChartAxes";
import { fixed } from "@/lib/format";
import { linearScale, linePoints } from "@/lib/scale";
import { confusionAt, ratio, rocFor } from "../rocAnalysis";
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

const x = linearScale([0, 1], [PAD.left, W - PAD.right]);
const y = linearScale([0, 1], [H - PAD.bottom, PAD.top]);
const TICKS = [0, 0.2, 0.4, 0.6, 0.8, 1];
const tickLabel = (tick: number) => tick.toFixed(1);

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
  // Thin the curve for drawing; every distinct probability still counts toward the AUC.
  const stride = Math.max(1, Math.floor(roc.points.length / 400));
  const line = linePoints(
    roc.points
      .filter((_, i) => i % stride === 0 || i === roc.points.length - 1)
      .map((p) => [x(p.fpr), y(p.tpr)] as const),
  );
  const lowerReversed = [...roc.lower].reverse();
  const band = linePoints([
    ...roc.grid.map((g, i) => [x(g), y(roc.upper[i]!)] as const),
    ...[...roc.grid]
      .reverse()
      .map((g, i) => [x(g), y(lowerReversed[i]!)] as const),
  ]);
  return (
    <figure className="roc-panel">
      <svg
        viewBox={`0 0 ${W} ${H}`}
        role="img"
        aria-label={`${title} ROC curve, AUC ${roc.auc.toFixed(3)}. At threshold ${threshold.toFixed(2)}: false positive rate ${fixed(fpr)}, true positive rate ${fixed(tpr)}.`}
      >
        <text className="roc-title" x={W / 2} y="14" textAnchor="middle">
          {title}
        </text>
        <YAxis
          ticks={TICKS}
          scale={y}
          left={x(0)}
          right={x(1)}
          format={tickLabel}
          grid="dashed"
          labelGap={5}
          size="small"
        />
        <XAxis
          ticks={TICKS}
          scale={x}
          labelY={y(0) + 14}
          format={tickLabel}
          gridFrom={y(0)}
          gridTo={y(1)}
          grid="dashed"
          size="small"
        />
        <AxisTitle x={x(0.5)} y={H - 6}>
          False positive rate
        </AxisTitle>
        <AxisTitle x={10} y={y(0.5)} rotate>
          True positive rate
        </AxisTitle>
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
