import { readFileSync, writeFileSync, mkdirSync } from "node:fs";
import { createHash } from "node:crypto";
import { execFileSync } from "node:child_process";
import { fileURLToPath } from "node:url";
import { resolve, dirname } from "node:path";

const root = fileURLToPath(new URL("../../", import.meta.url));
const destination = resolve(root, "app/src/features/architecture/sources.json");
const specs = [
  ["tremor", "tremor/calculate_tremor.py", "    def run_job(", 93],
  ["sds", "sources/sds.py", "class ", 65],
  ["fdsn", "sources/fdsn.py", "class ", 65],
  ["rsam", "tremor/rsam.py", "def ", 100],
  ["dsar", "tremor/dsar.py", "def ", 100],
  ["entropy", "tremor/shannon_entropy.py", "def ", 85],
  ["labels", "label/label_builder.py", "class ", 100],
  ["dynamic", "label/dynamic_label_builder.py", "class ", 90],
  [
    "windows",
    "features/tremor_matrix_builder.py",
    "    def _build_tremor_matrices(",
    130,
  ],
  [
    "features",
    "features/features_builder.py",
    "        # Extract features",
    43,
  ],
  [
    "curated",
    "features/features_builder.py",
    "        # Prediction mode forces",
    55,
  ],
  ["selection", "utils/ml.py", "    selector = FeatureSelector(", 50],
  [
    "ranking",
    "model/training_model.py",
    "        # Reduced features/columns",
    45,
  ],
  [
    "resampling",
    "model/training_model.py",
    "        features_resampled, labels_resampled = resample(",
    33,
  ],
  ["training", "model/training_model.py", "    def _train(", 100],
  ["grids", "model/constants.py", "DEFAULT_GRID_PARAMS:", 125],
  ["cv", "utils/ml.py", "def grid_search_cv(", 55],
  ["cv-splitters", "model/classifier_model.py", "    def get_cv_splitter(", 67],
  [
    "seed",
    "ensemble/seed_ensemble.py",
    "    def compute_probabilities_and_predictions(",
    70,
  ],
  [
    "classifier",
    "ensemble/classifier_ensemble.py",
    "        # Cross-classifier consensus",
    34,
  ],
  ["aggregation", "utils/array.py", "def compute_model_probabilities(", 51],
  ["prediction", "model/prediction_model.py", "    def forecast(", 110],
  ["evaluation", "model/evaluation_model.py", "class EvaluationModel", 85],
  ["explanation", "model/explanation_model.py", "class ExplanationModel", 85],
  ["shap", "ensemble/explainer_ensemble.py", "    def explain(", 120],
  ["cache", "model/base_model.py", "    def save(", 100],
  [
    "orchestration",
    "model/forecast_model_scenario.py",
    "            output_dir = os.path.join(self.scenarios_dir",
    90,
  ],
  ["scenario-script", "../../forecast-scenario.py", "def main(", 310],
  ["ancestry", "../../README.md", "#", 45],
  ["architecture-doc", "../../wiki/Architecture.md", "# Architecture", 40],
  ["walkthrough", "../../wiki/Pipeline-Walkthrough.md", "#", 40],
];
const normalize = (text) => text.replace(/\r\n/g, "\n");
const fingerprint = (text) => createHash("sha256").update(text).digest("hex");
const asset = {
  path: "assets/forecast_2025-07-27_2025-08-22.png",
  bundledPath: "app/src/assets/historical-forecast.png",
};
function captureAsset() {
  const sha256 = fingerprint(readFileSync(resolve(root, asset.path)));
  if (sha256 !== fingerprint(readFileSync(resolve(root, asset.bundledPath))))
    throw new Error(
      "The bundled historical image differs from its source asset. Review and copy the intended source image.",
    );
  return { ...asset, sha256 };
}
function capture() {
  return specs.map(([id, relative, anchor, count]) => {
    const path = resolve(root, "src/eruption_forecast", relative);
    const text = normalize(readFileSync(path, "utf8"));
    const lines = text.split("\n");
    const start = lines.findIndex((line) => line.includes(anchor));
    if (start < 0) throw new Error(`Missing anchor ${id}: ${anchor}`);
    return {
      id,
      path: path.slice(root.length).replaceAll("\\", "/"),
      symbol: anchor.trim(),
      line: start + 1,
      excerpt: lines.slice(start, start + count).join("\n"),
      sha256: fingerprint(text),
    };
  });
}
if (process.argv.includes("--write")) {
  const sources = capture();
  const revision = execFileSync("git", ["rev-parse", "HEAD"], {
    cwd: root,
    encoding: "utf8",
  }).trim();
  const workingTree = execFileSync(
    "git",
    ["status", "--porcelain", "--untracked-files=normal"],
    { cwd: root, encoding: "utf8" },
  );
  mkdirSync(dirname(destination), { recursive: true });
  writeFileSync(
    destination,
    JSON.stringify(
      {
        reviewedAt: new Date().toISOString(),
        revision,
        workingTree,
        sources,
        asset: captureAsset(),
      },
      null,
      2,
    ) + "\n",
  );
  console.log(
    `Captured ${sources.length} source excerpts. Review content before accepting this snapshot.`,
  );
} else {
  const saved = JSON.parse(readFileSync(destination, "utf8"));
  const current = capture();
  const stale = current.filter(
    (source) =>
      JSON.stringify(source) !==
      JSON.stringify(saved.sources.find((item) => item.id === source.id)),
  );
  if (
    stale.length ||
    saved.sources.length !== current.length ||
    JSON.stringify(saved.asset) !== JSON.stringify(captureAsset())
  ) {
    console.error(
      "Stale source references: " +
        stale.map((source) => source.path).join(", "),
    );
    console.error(
      "Review semantic changes and update content before npm run snapshot:sources.",
    );
    process.exitCode = 1;
  } else
    console.log(
      `${current.length} source excerpts and SHA-256 fingerprints match. This is a drift check, not semantic synchronization.`,
    );
}
