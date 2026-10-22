import sources from "../../../../wiki/Data-Sources.md?raw";
import training from "../../../../wiki/Training-Workflow.md?raw";
import prediction from "../../../../wiki/Prediction-Workflow.md?raw";
import evaluation from "../../../../wiki/Evaluation-Workflow.md?raw";
import explanation from "../../../../wiki/Explanation-Workflow.md?raw";

// Bundle the actual wiki text so Details remains available offline and follows
// documentation edits at build time. Select sections rather than duplicating prose.
function section(markdown: string, heading: string) {
  const lines = markdown.split(/\r?\n/);
  const start = lines.findIndex((line) => line.startsWith(`## ${heading}`));
  const end = lines.findIndex(
    (line, index) => index > start && line.startsWith("## "),
  );
  return lines.slice(start + 1, end < 0 ? undefined : end).join("\n");
}

export const WIKI_GUIDES: Record<
  string,
  { title: string; path: string; markdown: string }[]
> = {
  tremor: [
    {
      title: "Reading seismic archives",
      path: "Data-Sources.md",
      markdown:
        section(sources, "SDS - Local SeisComP Archive").split(
          "### Use from",
        )[0]! +
        "\n\n" +
        section(sources, "FDSN - Web Service with Local Cache").split(
          "### Use from",
        )[0]!,
    },
  ],
  labels: [
    {
      title: "How eruption windows are labelled",
      path: "Training-Workflow.md",
      markdown: section(training, "Label Building"),
    },
  ],
  features: [
    {
      title: "From windows to feature vectors",
      path: "Training-Workflow.md",
      markdown: section(training, "Feature Pipeline"),
    },
  ],
  training: [
    {
      title: "Fitting each seed",
      path: "Training-Workflow.md",
      markdown:
        section(training, "Classifier Fitting").split("### Parallelism")[0]! +
        "\n\n### Scoring\n\n" +
        section(training, "Classifier Fitting").split("### Scoring")[1]!,
    },
  ],
  prediction: [
    {
      title: "Building the forecast grid",
      path: "Prediction-Workflow.md",
      markdown: section(prediction, "Build Forecast Grid"),
    },
    {
      title: "Keeping features aligned",
      path: "Prediction-Workflow.md",
      markdown: section(prediction, "Extract Features"),
    },
  ],
  consensus: [
    {
      title: "Evaluating ensemble outputs",
      path: "Evaluation-Workflow.md",
      markdown: section(evaluation, "What `evaluate()` Does").split(
        "### Method signature",
      )[0]!,
    },
  ],
  interpretation: [
    {
      title: "Evaluating without refitting",
      path: "Evaluation-Workflow.md",
      markdown: evaluation
        .split("## Two Operating Modes")[0]!
        .replace(/^# Evaluation Workflow/, ""),
    },
    {
      title: "Which models SHAP can explain",
      path: "Explanation-Workflow.md",
      markdown: section(explanation, "TreeExplainer constraint"),
    },
  ],
};
