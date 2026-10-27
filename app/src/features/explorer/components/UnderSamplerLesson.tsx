import { DataTable } from "@/components/lab/DataTable";
import { LabSectionTitle } from "@/components/lab/LabSectionTitle";
import { ResultOutput } from "@/components/lab/ResultOutput";
import { SeedSelect } from "@/components/lab/SelectField";
import { WhySection } from "@/components/lab/WalkthroughNotes";
import { retainedNegatives } from "../trainingLessons";
import { SampleCell } from "./SampleCell";

export function UnderSamplerLesson({
  step,
  seed,
  setSeed,
}: {
  step: number;
  seed: number;
  setSeed: (seed: number) => void;
}) {
  const selectedIds = retainedNegatives(seed);
  return (
    <>
      <LabSectionTitle
        title={step < 2 ? "History before sampling" : "One reproducible draw"}
        aside="54 original windows · 11.1% positive"
      />
      <div className="lab-options">
        <SeedSelect label="Sampling seed" value={seed} onChange={setSeed} />
      </div>
      <p>
        Each numbered cell is one feature row. Filled 1 cells are positive;
        outlined 0 cells are negative. Dashed cells are omitted from this seed’s
        training sample.
      </p>
      <div className="training-sample-grid">
        {Array.from({ length: 54 }, (_, i) => {
          const positive = i >= 48;
          const retained = positive || selectedIds.includes(i + 1);
          const discarded = step >= 2 && !retained;
          return (
            <SampleCell
              key={i}
              id={i + 1}
              positive={positive}
              discarded={discarded}
              title={`id ${i + 1} · label ${positive ? 1 : 0} · ${discarded ? "omitted" : "retained"}`}
            />
          );
        })}
      </div>
      <ResultOutput>
        {step < 2
          ? "48 negative + 6 positive = 54 windows"
          : "8 negative + 6 positive = 14 retained windows · 40 negatives omitted"}
      </ResultOutput>
      {step >= 1 && (
        <p className="training-equation">
          Target majority = 6 ÷ 0.75 = 8 · Positive share after RUS = 6 ÷ 14 =
          42.9%
        </p>
      )}
      {step >= 2 && (
        <p>
          Retained negative ids: {selectedIds.join(", ")}. All positive ids
          49–54 remain.
        </p>
      )}
    </>
  );
}

/** Worked example of what `sampling_strategy = 0.75` means, shown below the lesson notes. */
export function SamplingRatioExplainer() {
  return (
    <WhySection
      id="sampling-ratio-heading"
      title="What does sampling_strategy = 0.75 mean?"
    >
      <p>
        In imbalanced-learn’s RandomUnderSampler, the ratio is the minority
        count divided by the majority count after resampling. Keep every
        eruption-positive window and randomly select fewer non-eruption windows.
        A value of 0.75 describes the ratio between those counts; it does not
        mean a 75/25 class split or a train/test split.
      </p>
      <DataTable
        label="Sampling ratio example"
        caption="Example training data · sampling_strategy = 0.75"
        headers={["Class", "Before", "After"]}
      >
        <tr>
          <th>Eruption-positive (minority)</th>
          <td>300</td>
          <td>All 300 retained</td>
        </tr>
        <tr>
          <th>Non-eruption (majority)</th>
          <td>2,000</td>
          <td>400 randomly selected</td>
        </tr>
      </DataTable>
      <p className="training-equation">
        Majority target = 300 ÷ 0.75 = 400 · Ratio = 300 ÷ 400 = 0.75
      </p>
      <p>
        The 700 retained windows are 42.9% positive and 57.1% negative. Sampling
        the minority too would discard already scarce positive examples.
        Different seeds can change the negative selection while keeping all
        positives. These are labelled windows, not separate eruption events. The
        interactive grid above uses the same ratio: 6 positives and 8 negatives.
      </p>
    </WhySection>
  );
}
