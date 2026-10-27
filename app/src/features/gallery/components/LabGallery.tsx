import { useState } from "react";
import { ChapterNav } from "@/components/lab/ChapterNav";
import { ChoiceButtons } from "@/components/lab/ChoiceButtons";
import { DataTable } from "@/components/lab/DataTable";
import { LabSectionTitle } from "@/components/lab/LabSectionTitle";
import { Legend } from "@/components/lab/Legend";
import { ParameterStrip } from "@/components/lab/ParameterStrip";
import { ResultOutput } from "@/components/lab/ResultOutput";
import { ScrollRegion } from "@/components/lab/ScrollRegion";
import { SelectField, SeedSelect } from "@/components/lab/SelectField";
import { StepButtons, StepControls } from "@/components/lab/StepControls";
import { TraceToolbar } from "@/components/lab/TraceToolbar";
import {
  ImplementationNotes,
  WhySection,
} from "@/components/lab/WalkthroughNotes";
import { GallerySection } from "./GallerySection";

const CHAPTERS = ["Continuous", "Dynamic", "Matrix"] as const;
const FEATURES = ["rsam_f2", "rsam_f3", "dsar_f3-f4", "entropy"] as const;
const WINDOWS = [
  { value: 3, label: "3 days" },
  { value: 7, label: "7 days" },
  { value: 10, label: "10 days" },
] as const;
const STEPS = [
  ["Slide the window", "Each window moves forward by the step size."],
  ["Mark positives", "Windows inside the forecast horizon become label 1."],
  ["Deduplicate", "Overlapping timestamps are kept once."],
] as const;
const LEGEND = [
  { swatch: ["group", "quiet"], label: "Quiet windows" },
  { swatch: ["group", "pre"], label: "Pre-eruption windows" },
] as const;

const SOURCE = "src/components/lab";

export function LabGallery() {
  const [chapter, setChapter] = useState(0);
  const [feature, setFeature] = useState<string>(FEATURES[0]);
  const [days, setDays] = useState<number>(7);
  const [seed, setSeed] = useState(0);
  const [step, setStep] = useState(0);
  const [playing, setPlaying] = useState(false);

  return (
    <>
      <GallerySection
        name="ChapterNav"
        source={`${SOURCE}/ChapterNav.tsx`}
        purpose="Numbered chapter tabs at the top of a multi-part walkthrough."
      >
        <ChapterNav
          titles={CHAPTERS}
          current={chapter}
          onSelect={setChapter}
          label="Gallery chapters"
        />
      </GallerySection>

      <GallerySection
        name="ChoiceButtons"
        source={`${SOURCE}/ChoiceButtons.tsx`}
        purpose="Toggle row for picking one named item, such as a tsfresh feature."
      >
        <ChoiceButtons
          label="Feature"
          options={FEATURES}
          value={feature}
          onChange={setFeature}
        />
      </GallerySection>

      <GallerySection
        name="SelectField"
        source={`${SOURCE}/SelectField.tsx`}
        purpose="Labelled native select that returns the typed option value. SeedSelect is the preset for the three teaching seeds."
      >
        <div className="lab-options">
          <SelectField
            label="History"
            value={days}
            options={WINDOWS}
            onChange={setDays}
          />
          <SeedSelect value={seed} onChange={setSeed} />
        </div>
      </GallerySection>

      <GallerySection
        name="StepControls"
        source={`${SOURCE}/StepControls.tsx`}
        purpose="Step picker, current explanation and previous / next / restart actions. StepButtons is the numbered picker alone."
      >
        <StepControls
          steps={STEPS}
          step={step}
          onStepChange={setStep}
          groupLabel="Gallery steps"
        />
        <StepButtons
          labels={STEPS.map(([title]) => title)}
          current={step}
          onSelect={setStep}
          groupLabel="Gallery step buttons"
        />
      </GallerySection>

      <GallerySection
        name="TraceToolbar"
        source={`${SOURCE}/TraceToolbar.tsx`}
        purpose="Play / pause button with a live status line for a guided trace. Shows Replay once finished."
      >
        <TraceToolbar
          className="lab-options"
          playing={playing}
          onToggle={() => setPlaying((value) => !value)}
          status={playing ? "Tracing run…" : "Idle"}
        />
        <TraceToolbar
          className="lab-options"
          playing={false}
          finished
          onToggle={() => undefined}
          status="Finished"
        />
      </GallerySection>

      <GallerySection
        name="LabSectionTitle"
        source={`${SOURCE}/LabSectionTitle.tsx`}
        purpose="Heading row of a lab panel, with an optional qualifier on the right."
      >
        <LabSectionTitle title="Label timeline" aside="UTC, 6-hour step" />
        <LabSectionTitle title="Without an aside" />
      </GallerySection>

      <GallerySection
        name="ParameterStrip"
        source={`${SOURCE}/ParameterStrip.tsx`}
        purpose="Fixed settings shown as label / value pairs under a heading."
      >
        <ParameterStrip
          items={[
            ["Look-back", "2 days"],
            ["Slide", "6 hours"],
            ["Time", "UTC"],
          ]}
        />
      </GallerySection>

      <GallerySection
        name="Legend"
        source={`${SOURCE}/Legend.tsx`}
        purpose="Colour key for a chart. Each swatch is styled from its data attribute."
      >
        <Legend items={LEGEND} />
      </GallerySection>

      <GallerySection
        name="ResultOutput"
        source={`${SOURCE}/ResultOutput.tsx`}
        purpose="Live-region read-out summarising the current state of a demo."
      >
        <ResultOutput>
          Feature {feature}, {days}-day history, seed {seed}.
        </ResultOutput>
      </GallerySection>

      <GallerySection
        name="DataTable"
        source={`${SOURCE}/DataTable.tsx`}
        purpose="Scrollable table; headers fill the head row and children are the body rows."
      >
        <DataTable
          label="Gallery table"
          caption="Window counts per class"
          headers={["Class", "Windows", "Share"]}
        >
          <tr>
            <td>Quiet</td>
            <td>48</td>
            <td>0.89</td>
          </tr>
          <tr>
            <td>Pre-eruption</td>
            <td>6</td>
            <td>0.11</td>
          </tr>
        </DataTable>
      </GallerySection>

      <GallerySection
        name="ScrollRegion"
        source={`${SOURCE}/ScrollRegion.tsx`}
        purpose="Keyboard-focusable scroll container for wide content. ChartScroll wraps an SVG chart for narrow screens."
      >
        <ScrollRegion label="Gallery scroll region">
          <p style={{ minWidth: 900, margin: 0 }}>
            Wide content stays reachable by keyboard: focus this region and
            scroll sideways with the arrow keys.
          </p>
        </ScrollRegion>
      </GallerySection>

      <GallerySection
        name="WhySection"
        source={`${SOURCE}/WalkthroughNotes.tsx`}
        purpose="Short framing card, labelled by its heading when an id is passed."
      >
        <WhySection id="gallery-why">
          <p>Resampling keeps the minority class from being ignored.</p>
        </WhySection>
      </GallerySection>

      <GallerySection
        name="ImplementationNotes"
        source={`${SOURCE}/WalkthroughNotes.tsx`}
        purpose="Closing two-column note: the method's limits and where this repository implements it."
      >
        <ImplementationNotes
          caveat="Undersampling discards majority rows, so it cannot add information."
          implementation="Applied per seed, before the cross-validation split."
          sources={["src/eruption_forecast/utils/ml.py"]}
        />
      </GallerySection>
    </>
  );
}
