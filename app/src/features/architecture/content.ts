import type {
  ArchitectureNode,
  ArchitectureEdge,
  Gap,
  Scenario,
  Stage,
  TourStep,
} from "./architecture.types";

export const STAGES: Stage[] = [
  {
    id: "tremor",
    number: 1,
    title: "Seismic data → tremor",
    summary: "Turn waveforms into measurements",
    tone: "sage",
  },
  {
    id: "labels",
    number: 2,
    title: "Labels & windows",
    summary: "Connect history to an eruption horizon",
    tone: "sage",
  },
  {
    id: "features",
    number: 3,
    title: "Feature extraction",
    summary: "Describe each window numerically",
    tone: "ochre",
  },
  {
    id: "training",
    number: 4,
    title: "Seed preparation & training",
    summary: "Select features and fit each estimator",
    tone: "ochre",
  },
  {
    id: "prediction",
    number: 5,
    title: "Ensembles & prediction",
    summary: "Apply fitted models to future windows",
    tone: "blue",
  },
  {
    id: "consensus",
    number: 6,
    title: "Consensus",
    summary: "Combine probabilities and votes",
    tone: "blue",
  },
  {
    id: "interpretation",
    number: 7,
    title: "Evaluation & SHAP",
    summary: "Measure performance and explain models",
    tone: "plum",
  },
];

export const GAPS: Gap[] = [
  {
    id: "cv-leakage",
    title: "Randomized CV is not future validation",
    detail:
      "Historical training dates and internal CV are different boundaries. Overlapping windows, feature selection and resampling before CV can share information across folds. This path does not establish leakage-free validation.",
  },
  {
    id: "reduction",
    title: "No dedicated reduction stage found",
    detail:
      "Calculator exclusions and imputation are implemented. A dedicated correlation, zero-value or linear-value feature-reduction stage was not found in the inspected training path.",
  },
  {
    id: "fdr",
    title: "Top 20 is not an FDR guarantee",
    detail:
      "P-values are ranked across all features. When the FDR-filtered set has fewer than top_n columns, get_significant_features falls back to the top p-values. Training subsequently takes the head of the ranking; the saved top-N list is not a guarantee that each feature passed FDR.",
  },
  {
    id: "ancestry",
    title: "Ancestry does not imply identical parameters",
    detail:
      "The README cites Whakaari / puia ancestry. Lite-RF here is a RandomForestClassifier with the repository’s own grid; identical parameters to the ancestral project have not been verified.",
  },
  {
    id: "fitted-evaluation",
    title: "Training reuse measures fitted-model performance",
    detail:
      "Training-reuse evaluation predicts on the fitted training feature matrix. It is not independent future validation and must be distinguished from prediction-reuse evaluation.",
  },
  {
    id: "consensus-shap",
    title: "SHAP explains constituent classifiers",
    detail:
      "TreeExplainer runs per seed of qualifying classifiers. Aggregate plots combine their explanations; no separate consensus-level SHAP explanation is implemented here.",
  },
  {
    id: "confidence",
    title: "“Confidence” is a vote-based calculation",
    detail:
      "The implemented value is 1.96 × sqrt(v × (1 − v) / n), using mean votes v. It is CI-like, not a calibrated confidence probability or a guarantee of coverage.",
  },
];

const node = (
  id: string,
  stage: string,
  title: string,
  summary: string,
  behavior: string[],
  parameters: string[],
  artifacts: string[],
  sources: string[],
  gaps: string[] = [],
  overview = false,
): ArchitectureNode => ({
  id,
  stage,
  title,
  summary,
  behavior,
  parameters,
  artifacts,
  sources,
  gaps,
  overview,
});
export const NODES: ArchitectureNode[] = [
  node(
    "tremor",
    "tremor",
    "Seismic data → tremor",
    "Waveforms become a regular series of seismic measurements.",
    [
      "CalculateTremor reads SDS or FDSN data, checks completeness and computes measurements across frequency bands.",
      "ForecastModel.calculate extends the requested start backward by day_to_forecast so the first feature window has historical coverage.",
    ],
    [
      "Example: VG.OJN.00.EHZ; 10-minute measurements",
      "Bands: 0.01–0.1, 0.1–2, 2–5, 4.5–8, 8–16 Hz",
    ],
    ["Tremor CSV; optional daily plots"],
    ["tremor", "rsam", "dsar", "entropy"],
    [],
    true,
  ),
  node(
    "sources",
    "tremor",
    "SDS / FDSN sources",
    "Load one day of waveform data for a station.",
    [
      "SDS reads a local SeisComP archive. FDSN retrieves remote waveforms and can cache them in SDS format.",
      "Station identity is network, station, location and channel.",
    ],
    ['source="sds" or "fdsn"', "Example SDS directory: D:/Data/OJN"],
    ["Daily seismic Stream / Trace; optional SDS cache"],
    ["sds", "fdsn"],
  ),
  node(
    "preprocessing",
    "tremor",
    "Preprocess & check completeness",
    "Only sufficiently complete signals enter the measurement path.",
    [
      "Anomaly removal and interpolation are configurable. Days below minimum_completion_ratio after anomaly removal are skipped.",
      "Band-specific processing and outlier handling precede aggregation. Missing data handling changes which windows remain available.",
    ],
    [
      "Example minimum_completion_ratio=0.3",
      "Example interpolate=True; remove_tremor_anomalies=False",
    ],
    ["Cleaned daily signal; completeness decisions"],
    ["tremor", "rsam"],
  ),
  node(
    "metrics",
    "tremor",
    "RSAM · DSAR · entropy",
    "Measure amplitude, frequency relationships and complexity.",
    [
      "RSAM summarizes amplitude in each frequency band.",
      "DSAR compares displacement amplitudes in consecutive bands. Shannon entropy supplies one broadband signal-complexity column.",
    ],
    ["rsam_f0…f4; dsar_f0-f1…f3-f4; entropy"],
    ["Time-indexed tremor columns"],
    ["rsam", "dsar", "entropy"],
  ),
  node(
    "labels",
    "labels",
    "Labels & windows",
    "A window ID joins seismic history to its target label.",
    [
      "Standard LabelBuilder makes a sliding time grid and binary eruption labels. DynamicLabelBuilder assembles per-eruption frames, concatenates, deduplicates and marks positives.",
      "Historical feature lookback and future label horizon describe different intervals.",
    ],
    [
      "Example horizon / lookback: 2 days",
      "Example training stride: 6 hours; prediction stride: 10 minutes",
    ],
    ["labels/*.csv; label.config.yaml; datetime ↔ id mapping"],
    ["labels", "dynamic", "windows"],
    [],
    true,
  ),
  node(
    "lookahead",
    "labels",
    "Eruption dates & lookahead",
    "Use known eruption dates to construct supervised targets.",
    [
      "The include_eruption_date flag is counterintuitive: False adds the eruption day beyond day_to_forecast pre-eruption days.",
      "LabelBuilder defaults this flag to True; TrainingModel defaults it to False. Inspect the saved configuration when interpreting labels.",
    ],
    [
      "day_to_forecast=2 in the example",
      "Training include_eruption_date=False",
    ],
    ["is_erupted indexed by window id"],
    ["labels", "training"],
  ),
  node(
    "windows",
    "labels",
    "Historical windows & IDs",
    "Slice the preceding seismic measurements for each grid point.",
    [
      "TremorMatrixBuilder aligns each label datetime with a historical window and assigns its id.",
      "Windows below the required sample completeness are skipped. Training labels are restricted to IDs actually present in the tremor matrix.",
    ],
    [
      "window_size; window_step; window_step_unit",
      "minimum_completion controls sample acceptance",
    ],
    ["Unified matrix: id, datetime, tremor columns"],
    ["windows", "features"],
  ),
  node(
    "features",
    "features",
    "Feature extraction",
    "Turn each tremor window into a vector for fitting or inference.",
    [
      "FeaturesBuilder runs tsfresh independently per tremor column, then merges the matrices by window id.",
      "Training may use extraction-time relevance filtering; prediction forces relevance filtering off because it has no target labels.",
    ],
    [
      "select_tremor_columns",
      "use_relevant_features; exclude_features; select_features",
    ],
    ["Per-column and merged Snappy Parquet matrices; aligned labels CSV"],
    ["features", "curated"],
    ["reduction"],
    true,
  ),
  node(
    "calculators",
    "features",
    "Calculators & imputation",
    "Control which numerical descriptions are computed.",
    [
      "Calculator exclusions remove named tsfresh calculators. The full extraction branch passes tsfresh impute explicitly.",
      "The optional relevance branch delegates to extract_relevant_features. Exclusions are not a general correlation/zero/linear-value reduction pass.",
    ],
    [
      "Example excludes agg_linear_trend, linear_trend_timewise, length, has_duplicate_max, has_duplicate_min, has_duplicate",
    ],
    ["Extracted features per tremor column"],
    ["features", "curated"],
    ["reduction"],
  ),
  node(
    "curated",
    "features",
    "Curated lists & persistence",
    "Reuse a known feature space or a saved matrix.",
    [
      "select_features accepts a list or CSV; it restricts tsfresh calculators before extraction and disables repeat relevance filtering.",
      "Saved matrices preserve window IDs. Loading files skips extraction and still requires aligned label/window metadata.",
    ],
    ["select_features: CSV or list[str]"],
    ["features/{cv}/extracted/*.parquet; features-matrix_*.parquet"],
    ["curated", "features"],
  ),
  node(
    "training",
    "training",
    "Seed preparation & training",
    "Each seed gets data, a feature list and fitted estimators.",
    [
      "TrainingModel prepares seeds, resamples and ranks features before fitting each configured classifier with GridSearchCV.",
      "Historical training dates bound the available data; randomized CV takes place inside that period.",
    ],
    [
      "Example: 2 seeds; top_n_features=20; recall scoring",
      "Example: 5 shuffle-stratified splits; n_jobs=2; n_grids=4",
    ],
    [
      "Per-seed feature ranking, resampled label IDs, estimator pickle and registry JSON",
    ],
    ["resampling", "ranking", "training", "cv"],
    ["cv-leakage", "fdr"],
    true,
  ),
  node(
    "resampling",
    "training",
    "Imbalance & resampling",
    "Prepare the class balance independently for each seed.",
    [
      "auto measures minority share and selects undersampling below the configured threshold; otherwise it skips resampling.",
      "The script explicitly selects under. Resampling happens before GridSearchCV, not inside a fold-local pipeline.",
    ],
    [
      "Package fit default: auto; minority_threshold=0.15",
      'Example: resample_method="under"',
    ],
    ["Resampled id + is_erupted CSV per seed"],
    ["resampling", "training"],
    ["cv-leakage"],
  ),
  node(
    "ranking",
    "training",
    "P-value ranking & top-N",
    "Choose the features that each seed will use.",
    [
      "The tsfresh selector returns an FDR-filtered matrix and an all-feature p-value ranking.",
      "If fewer than top_n survive, the helper falls back to the most significant p-values. Training takes the head of the saved ranking. FeatureSelector.fit does not forward top_n to its tsfresh helper, whose fallback defaults to 20.",
    ],
    [
      "Example top_n_features=20; FDR default 0.05",
      "Alternative method: random_forest permutation importance",
    ],
    ["Seed-specific top-N feature names and scores"],
    ["selection", "ranking"],
    ["fdr"],
  ),
  node(
    "classifiers",
    "training",
    "RF · Lite-RF · XGB · GB",
    "Search the implemented estimator grids.",
    [
      "RF averages randomized decision trees; its grid uses 50/100 trees, depths 3/5/7, gini/entropy, sqrt/log2/None, split 2/5 and leaf 1/2.",
      "Lite-RF is a RandomForestClassifier with 10/30/100 trees, depths 3/5/7, gini/entropy and sqrt/log2/None. It omits the RF split/leaf grid dimensions.",
      "Gradient Boost fits additive trees; XGBoost is gradient-boosted trees with its own sampling and regularization controls. Both grids use 100/200 trees and learning rates 0.01/0.1.",
      "Secondary supported classifiers: SVM, logistic regression, neural network, decision tree, k-nearest neighbors, naive Bayes and voting.",
    ],
    ["Active example: lite-rf and rf", "ClassifierModel exposes model + grid"],
    ["GridSearchCV best estimator per classifier and seed"],
    ["grids", "ancestry"],
    ["ancestry"],
  ),
  node(
    "cv",
    "training",
    "Cross-validation & fitting",
    "Search hyperparameters inside the historical training period.",
    [
      "TrainingModel supports shuffle, stratified and shuffle-stratified. GridSearchCV chooses a fitted estimator per seed/classifier. ClassifierModel also contains a TimeSeriesSplit fallback, but TrainingModel does not expose it in its supported strategy surface.",
      "Overlapping windows can be near duplicates across folds. Pre-CV resampling and selection also cross the fold boundary.",
    ],
    [
      'Example cv_splits=5; scoring="recall"',
      "No time-series CV in the current TrainingModel path",
    ],
    ["Fitted model pickle; trained-model__*.json registry"],
    ["cv", "cv-splitters", "training"],
    ["cv-leakage"],
  ),
  node(
    "prediction",
    "prediction",
    "Ensembles & prediction",
    "Reuse trained models over the prediction grid.",
    [
      "Each SeedEnsemble groups seeds of one classifier. ClassifierEnsemble groups classifier families.",
      "PredictionModel builds a future window grid, extracts or loads features, and runs each fitted seed against its own stored feature names.",
    ],
    [
      "use_features_from: all / training / files",
      "Example prediction stride: 10 minutes",
    ],
    ["forecast-results_*.csv; per-seed results; forecast figure"],
    ["seed", "classifier", "prediction"],
    [],
    true,
  ),
  node(
    "ensembles",
    "prediction",
    "Seed & classifier ensembles",
    "Preserve the hierarchy of fitted models.",
    [
      "SeedEnsemble stores fitted models and their feature lists; its features property is the union needed by all seeds.",
      "ClassifierEnsemble bundles these groups. Classifier averages are equally weighted even when their seed counts differ.",
    ],
    [
      "Factories accept current JSON and pickle artifacts; legacy registry CSV is also supported",
    ],
    ["SeedEnsemble_*.pkl; ClassifierEnsemble_*.pkl and .json"],
    ["seed", "classifier"],
  ),
  node(
    "reuse",
    "prediction",
    "Extract once, reuse files",
    "Share prediction features across the scenario sweep.",
    [
      "ForecastModelScenario calculates tremor once. The first scenario is forced to extract all prediction features.",
      "With files selected, later scenarios load the first prediction matrix and labels. The first grid must cover later periods; file reuse forces ForecastModel prediction caching off.",
    ],
    [
      "Example widest prediction period: 1 Apr–22 Aug 2025",
      "Shared settings; separate train → predict → evaluate → explain per scenario",
    ],
    [
      "scenarios/scenario-1-2/prediction/features/…",
      "Subsequent scenario output directories remain distinct",
    ],
    ["orchestration", "scenario-script", "prediction"],
  ),
  node(
    "cache",
    "prediction",
    "Outputs & content-addressed cache",
    "Persist stages with an auditable identity.",
    [
      "BaseModel writes a hash-named pickle and the exact identity as a params.json sidecar.",
      "Prediction identity includes the training hash. Evaluation and explanation use separate training/prediction namespaces. These paths are templates, not artifacts from a run in this app.",
    ],
    ["use_cache; save_model; overwrite; prefix_config"],
    [
      "{scenario}/training/{hash}.TrainingModel.pkl",
      "{scenario}/prediction/{hash}.PredictionModel.pkl",
      "{scenario}/evaluation/{kind}/; explanation/{kind}/",
    ],
    ["cache", "orchestration"],
  ),
  node(
    "consensus",
    "consensus",
    "Consensus",
    "Keep mean probabilities, votes and disagreement separate.",
    [
      "Mean probability averages seed probabilities within each classifier, then averages those classifier means.",
      "Mean binary votes are averaged separately; consensus_prediction can be fractional. Uncertainty is population standard deviation: over seeds within a classifier, over classifier means at consensus level.",
      "Confidence uses vote fraction v: 1.96 × sqrt(v × (1 − v) / n). At consensus level n is the number of classifiers.",
    ],
    [
      "Example below is illustrative, not a model run",
      "Plot threshold 0.7 in the script is distinct from these aggregate calculations",
    ],
    [
      "consensus_probability; consensus_prediction; consensus_uncertainty; consensus_confidence",
    ],
    ["aggregation", "classifier", "seed"],
    ["confidence"],
    true,
  ),
  node(
    "probabilities",
    "consensus",
    "Probability versus vote",
    "A probability of 0.7 and a vote fraction of 0.7 mean different things.",
    [
      "Seeds supply predict_proba and their estimator’s predict result. If only decision_function exists, the fallback applies a sigmoid and thresholds it at 0.5.",
      "ClassifierEnsemble.predict thresholds the mean probability at 0.5; the forecast statistics path separately reports mean binary votes.",
    ],
    ["Equal weighting across classifier groups"],
    ["Per-seed probability and prediction matrices"],
    ["seed", "classifier", "aggregation"],
  ),
  node(
    "interpretation",
    "interpretation",
    "Evaluation & SHAP",
    "Assess outputs and inspect why individual models respond.",
    [
      "EvaluationModel reuses fitted training or prediction models without refitting. MetricsEnsemble computes per-seed metrics; ClassifierComparator ranks families.",
      "ExplanationModel uses per-seed TreeExplainer for RF, Lite-RF, GB and XGB. Other classifiers are skipped with a warning.",
    ],
    [
      "Example runs prediction evaluation/explanation, then a second run with training evaluation/explanation",
    ],
    ["Metrics, comparison plots, per-seed and aggregate SHAP plots"],
    ["evaluation", "explanation", "shap", "scenario-script"],
    ["fitted-evaluation", "consensus-shap"],
    true,
  ),
  node(
    "evaluation",
    "interpretation",
    "Ground truth & metrics",
    "Align evaluation labels to the model’s window IDs.",
    [
      "Training reuse takes TrainingModel.labels. Prediction reuse requires eruption_dates and rebuilds LabelBuilder labels joined onto prediction window IDs.",
      "Per-seed metrics and aggregate ROC, PR, threshold, g-mean and MCC plots support comparison. Reusing the fitted training matrix is fitted-model performance.",
    ],
    [
      'model="training" or "prediction"',
      "plot_per_seed; plot_aggregate; compare_classifiers",
    ],
    [
      "evaluation/{kind}/classifiers/{clf}/predictions/{y_proba,y_pred}.csv",
      "evaluation/{kind}/comparison/; prediction labels/y_true.csv",
    ],
    ["evaluation"],
    ["fitted-evaluation"],
  ),
  node(
    "shap",
    "interpretation",
    "Per-seed SHAP & waterfalls",
    "Explain constituent tree models, feature by feature.",
    [
      "SHAP is computed per qualifying classifier seed, with that seed’s selected features. Aggregate plots use a NaN-padded union feature space.",
      "Waterfalls pick highest-probability windows around each eruption. Cached probabilities are aligned to the explanation grid first.",
      "There is no separate consensus-level explanation. Training window_ids and prediction labels supply the temporal mapping.",
    ],
    [
      "save_per_seed; check_additivity; max_display",
      "Example max_display=20; dpi=150",
    ],
    [
      "explanation/{kind}/classifiers/{clf}/shap_values/*.pkl",
      "figures/{bar,beeswarm}/; figures/aggregate/; eruptions/{date}/",
    ],
    ["shap", "explanation"],
    ["consensus-shap"],
  ),
];

export const EDGES: ArchitectureEdge[] = [
  ...STAGES.slice(0, -1).map((stage, i) => ({
    id: `${stage.id}-next`,
    source: stage.id,
    target: STAGES[i + 1]!.id,
    label: [
      "tremor series",
      "window IDs",
      "feature matrix",
      "fitted models",
      "seed outputs",
      "results + models",
    ][i]!,
  })),
  ...NODES.filter((item) => !item.overview).map((item) => ({
    id: `${item.stage}-${item.id}`,
    source: item.stage,
    target: item.id,
    label: "inspect step",
  })),
];
export const SCENARIOS: Scenario[] = [
  "03-31",
  "04-30",
  "05-31",
  "06-30",
  "07-31",
].map((end, i) => ({
  id: `scenario-${i + 1}-2`,
  name: `Scenario ${i + 1}.2`,
  trainStart: "2025-01-01",
  trainEnd: `2025-${end}`,
  predictionStart: `2025-0${i + 4}-01`,
  predictionEnd: "2025-08-22",
  predictionStepMinutes: 10,
}));
export const ERUPTIONS = [
  "2025-03-20",
  "2025-04-10",
  "2025-04-22",
  "2025-05-18",
  "2025-06-17",
  "2025-07-07",
  "2025-08-02",
  "2025-08-18",
];
export const TOUR: TourStep[] = STAGES.map((stage) => ({
  target: stage.id,
  title: stage.title,
  text:
    NODES.find((item) => item.id === stage.id)!.summary +
    " " +
    NODES.find((item) => item.id === stage.id)!.behavior[0],
}));
