import { WhySection } from "@/components/lab/WalkthroughNotes";
import { ByCorrectionDemo } from "./ByCorrectionDemo";
import { FisherDemo } from "./FisherDemo";
import { KsTestDemo } from "./KsTestDemo";
import { LabelledDataDemo } from "./LabelledDataDemo";
import { DUMMY_FEATURES, PRE_COUNT, QUIET_COUNT } from "../dummyDataset";
import "@/styles/significance.css";

const STEPS = [
  { id: "sig-data", label: "Labelled data" },
  { id: "sig-ks", label: "KS test" },
  { id: "sig-fisher", label: "Fisher’s exact test" },
  { id: "sig-by", label: "BY correction" },
] as const;

function StepHeading({ id, n, children }: { id: string; n: number; children: string }) {
  return (
    <h2 id={id} className="sig-step-heading">
      <span className="sig-step-no" aria-hidden="true">{n}</span>
      <span className="sr-only">Step {n}: </span>
      {children}
    </h2>
  );
}

export function SignificanceExplainer() {
  return (
    <div className="sig">
      <WhySection
        id="sig-intro"
        className="sig-intro"
        title="Where do the p-values come from?"
      >
        <p>
          The ranking above starts with one p-value per feature. Under the null
          hypothesis that a feature is independent of the eruption label, the
          p-value is the probability of a test statistic at least as extreme as
          the one observed. A small p-value is evidence against the null
          hypothesis. This repository passes the feature matrix to tsfresh. It
          tests each feature against the class label, then applies the
          Benjamini–Yekutieli (BY) procedure to control the false discovery
          rate (FDR) at 0.05.
        </p>
        <p>
          The four steps below use a synthetic dataset of{" "}
          {QUIET_COUNT + PRE_COUNT} time windows ({QUIET_COUNT} non-eruptive,
          label 0; {PRE_COUNT} pre-eruptive, label 1) and{" "}
          {DUMMY_FEATURES.length} features. The p-values are exact scipy
          results on these synthetic values, not measured seismic tremor.
        </p>
        <nav className="sig-index" aria-label="Explainer steps">
          {STEPS.map((step, i) => (
            <a key={step.id} href={`#${step.id}`}>
              <span aria-hidden="true">{i + 1}</span>
              {step.label}
            </a>
          ))}
        </nav>
      </WhySection>

      <section className="sig-section" aria-labelledby="sig-data">
        <StepHeading id="sig-data" n={1}>
          Labelled data: which features discriminate pre-eruptive windows?
        </StepHeading>
        <p>
          tsfresh summarizes each time window of the seismic tremor record
          (for example RSAM, the real-time seismic amplitude measurement) as
          hundreds of statistical features. Each window also carries a class
          label. Before training a classifier, we discard features whose
          distribution is the same in pre-eruptive and non-eruptive windows,
          because they carry no precursory signal.
        </p>
        <LabelledDataDemo />
      </section>

      <section className="sig-section" aria-labelledby="sig-ks">
        <StepHeading id="sig-ks" n={2}>
          Continuous features: do the two samples share one distribution?
        </StepHeading>
        <p>
          Split one feature’s values by class label. The two-sample
          Kolmogorov–Smirnov (KS) test compares the two empirical cumulative
          distribution functions (ECDFs) and returns the statistic D, their
          largest vertical distance, and a p-value.
        </p>
        <KsTestDemo />
      </section>

      <section className="sig-section" aria-labelledby="sig-fisher">
        <StepHeading id="sig-fisher" n={3}>
          Binary features: Fisher’s exact test on a contingency table
        </StepHeading>
        <p>
          Some tsfresh features take only the values 0 and 1, so an ECDF
          carries no information. tsfresh uses Fisher’s exact test instead:
          tabulate the windows in a 2×2 contingency table and compute the exact
          probability, under the hypergeometric distribution, of a table at
          least as extreme as the one observed.
        </p>
        <FisherDemo />
      </section>

      <section className="sig-section" aria-labelledby="sig-by">
        <StepHeading id="sig-by" n={4}>
          Multiple testing: controlling the false discovery rate
        </StepHeading>
        <ByCorrectionDemo />
      </section>
    </div>
  );
}
