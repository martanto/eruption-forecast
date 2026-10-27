import { AxisTitle, XAxis, YAxis } from "@/components/lab/ChartAxes";
import { ChartFrame } from "@/components/lab/ChartFrame";
import { linearScale, linePoints } from "@/lib/scale";
import { GallerySection } from "./GallerySection";

const W = 520;
const H = 220;
const PAD = { left: 48, right: 16, top: 12, bottom: 44 };
const SAMPLE = [0.08, 0.12, 0.1, 0.22, 0.35, 0.3, 0.52, 0.71, 0.66, 0.9];

const x = linearScale([0, SAMPLE.length - 1], [PAD.left, W - PAD.right]);
const y = linearScale([0, 1], [H - PAD.bottom, PAD.top]);
const LINE = linePoints(SAMPLE.map((value, i) => [x(i), y(value)] as const));

const SOURCE = "src/components/lab";

export function ChartGallery() {
  return (
    <>
      <GallerySection
        name="ChartFrame"
        source={`${SOURCE}/ChartFrame.tsx`}
        purpose="Shell for a fixed-size SVG chart: optional legend, a scrolling svg with role img and a required text label. The sample below also uses YAxis, XAxis and AxisTitle."
      >
        <ChartFrame
          width={W}
          height={H}
          label="Sample probability rising from 0.08 to 0.9 over ten windows."
          legend={[{ swatch: ["line", "by"], label: "Eruption probability" }]}
        >
          <YAxis
            ticks={[0, 0.25, 0.5, 0.75, 1]}
            scale={y}
            left={PAD.left}
            right={W - PAD.right}
          />
          <XAxis
            ticks={SAMPLE.map((_, i) => i)}
            scale={x}
            labelY={H - PAD.bottom + 18}
            format={(tick) => tick + 1}
          />
          <AxisTitle x={(PAD.left + W - PAD.right) / 2} y={H - 6}>
            Window
          </AxisTitle>
          <AxisTitle x={12} y={(PAD.top + H - PAD.bottom) / 2} rotate>
            Probability
          </AxisTitle>
          <polyline
            points={LINE}
            fill="none"
            stroke="var(--color-terracotta)"
            strokeWidth={2}
            strokeLinejoin="round"
          />
        </ChartFrame>
      </GallerySection>

      <GallerySection
        name="YAxis / XAxis / AxisTitle"
        source={`${SOURCE}/ChartAxes.tsx`}
        purpose="Gridlines with tick labels. Grid variants: solid, dashed, minor (hairline, no labels). Label size: default or small. Scales come from lib/scale.ts: linearScale, logScale, linePoints, linePath."
      >
        <ChartFrame width={W} height={150} label="Gridline variants: dashed and minor.">
          <YAxis
            ticks={[0, 0.5, 1]}
            scale={linearScale([0, 1], [130, 20])}
            left={PAD.left}
            right={W - PAD.right}
            grid="dashed"
            size="small"
          />
          <YAxis
            ticks={[0.1, 0.2, 0.3, 0.4, 0.6, 0.7, 0.8, 0.9]}
            scale={linearScale([0, 1], [130, 20])}
            left={PAD.left}
            right={W - PAD.right}
            grid="minor"
            labels={false}
          />
        </ChartFrame>
      </GallerySection>
    </>
  );
}
