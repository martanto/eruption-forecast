export const TRAINING_LESSONS = {
  "seed-ensemble": {
    title: "Seed Ensemble",
    intro: "Train one classifier several ways, then listen to the group.",
    why: "One majority draw or random initialization can be unlucky. Repeating the same classifier family with different seeds reduces dependence on a single run and can reduce variance when the models make different errors. Averaging does not automatically remove systematic bias.",
    steps: [
      [
        "Start from shared history",
        "All seeds start with the same extracted feature matrix and labels. TrainingModel uses seeds 0 through N−1. Future prediction windows stay outside this training history.",
      ],
      [
        "Resample and select per seed",
        "Each seed controls its majority draw and feature selection. Save its retained ids and top feature names. The same per-seed sample and feature list are reused across classifier families, rather than drawing again for every classifier.",
      ],
      [
        "Fit one model per seed",
        "For one family, run GridSearchCV for each seed using that seed’s selected features and resampled rows. The seed also configures the classifier and CV splitter where supported. A seed is not a new independent eruption record.",
      ],
      [
        "Bundle and average",
        "SeedEnsemble stores the fitted estimators and their feature lists. Each estimator sees its own columns at inference. Mean probability summarizes the group; population standard deviation records disagreement across its seeds.",
      ],
    ],
    caveat:
      "Seeds share data and may make correlated errors. More seeds do not fix label errors, leakage, missing eruption types or an unsuitable classifier. The displayed standard deviation is model disagreement, not a calibrated forecast interval.",
    implementation:
      "TrainingModel.fit builds the per-classifier registry, then build_seed_ensemble loads its fitted models through SeedEnsemble.from_any. The union feature list supports inference; individual estimators still use their own subsets.",
    sources: [
      "model/training_model.py · fit, build_seed_ensemble",
      "ensemble/seed_ensemble.py · compute_probabilities_and_predictions",
      "utils/array.py · compute_model_probabilities",
    ],
  },
  "features-matrix": {
    title: "Features Matrix Builder",
    intro: "Turn each window of tremor into one wide row of numbers.",
    why: "A classifier needs one fixed-length row per example, but every window holds hundreds of tremor samples. tsfresh summarizes each window with many statistical calculators, so a row describes the shape of that window rather than its raw values.",
    steps: [
      [
        "Start from windowed tremor",
        "The tremor matrix stacks each label window's tremor samples with its window id, datetime and tremor columns (rsam_*, dsar_*, entropy). The id is the join key back to the label.",
      ],
      [
        "Extract one tremor column at a time",
        "FeaturesBuilder runs tsfresh on each tremor column independently and writes one Parquet file per column. Every window id becomes one row; the calculators become columns.",
      ],
      [
        "Name every calculator",
        "tsfresh names each output {column}__{calculator}, such as rsam_f2__mean. One tremor column yields hundreds of such features; a calculator is a statistic like the mean, variance or maximum.",
      ],
      [
        "Merge by window id",
        "The per-column tables are joined side by side on id into features-matrix_*.parquet. Labels are saved separately, aligned to the same ids. Rows are windows; columns are every column × calculator pair.",
      ],
    ],
    caveat:
      "More columns are not more information. Neighbouring windows overlap, many calculators are strongly correlated, and the matrix has far more columns than rows. This step makes no judgement about relevance, which is why selection follows. Use select_tremor_columns, exclude_features or select_features to restrict extraction.",
    implementation:
      "FeaturesBuilder.extract_features loops over tremor columns with tsfresh's ComprehensiveFCParameters, caches each column's Parquet, then merges them into features-matrix_*.parquet. TrainingModel.load_features can reuse a saved matrix and skip tsfresh.",
    sources: [
      "features/features_builder.py · extract_features",
      "features/tremor_matrix_builder.py · TremorMatrixBuilder",
      "utils/feature_utils.py · merge_features_matrix",
    ],
  },
  "random-under-sampler": {
    title: "Random Under Sampler",
    intro: "Give rare positive windows a stronger voice in training.",
    why: "Most historical windows are outside the pre-eruption period. A model can appear accurate by always predicting 0. Random under-sampling keeps the minority class and draws fewer majority rows, changing the balance the classifier learns from.",
    steps: [
      [
        "Count the classes",
        "Our synthetic history has 48 negative windows and 6 positive windows. Always predicting 0 gives 88.9% accuracy, but detects none of the positives. A positive window is a forecast label, not a separate eruption.",
      ],
      [
        "Set the target ratio",
        "sampling_strategy = minority / majority after resampling. With 6 positives and a ratio of 0.75, keep 6 / 0.75 = 8 negatives. This is a 6:8 sample, not a 75% positive share.",
      ],
      [
        "Draw majority rows",
        "Keep all 6 positive rows. Select 8 of the 48 negatives without replacement, using a reproducible seed. Feature rows and labels keep the same ids; discarded rows remain in the original history.",
      ],
      [
        "Pass the sample to training",
        "The 14 retained rows feed per-seed feature selection, then classifier fitting with GridSearchCV. Future forecast windows never enter this draw. Change the seed to see another selection.",
      ],
    ],
    caveat:
      "Under-sampling discards useful majority examples and changes the training class prior. It does not create new eruption evidence or guarantee calibrated probabilities. Here auto uses RUS only when the minority share is below 0.15; explicit under always requests it.",
    implementation:
      "TrainingModel.fit → resample → per-seed feature selection → GridSearchCV. Current code resamples and selects features before CV, so internal CV is not an untouched holdout and can be optimistic. Evaluate separately on later, unresampled data.",
    sources: [
      "model/training_model.py · fit, _run_feature_selection_job",
      "utils/ml.py · resample",
    ],
  },
  "feature-selection": {
    title: "Features Selection",
    intro: "Keep the few features that discriminate pre-eruptive from non-eruptive windows.",
    why: "The feature matrix has far more columns than training samples, so a flexible classifier can overfit noise. Univariate feature selection keeps a short ranked list per seed (top_n_features, 20 by default) of the features most associated with the class label, and discards the rest.",
    steps: [
      [
        "Select on the training set only",
        "Each seed starts from its own resampled training set. Selection sees only those feature vectors and class labels; the unseen forecast windows never reach it. A different seed gives a different resample, so a different ranking.",
      ],
      [
        "Score every feature",
        "The default tsfresh method tests each feature against the class label: a two-sample test (Mann-Whitney U, or KS) for continuous features and Fisher's exact test for binary ones. It then controls the false discovery rate at 0.05 with the Benjamini-Yekutieli procedure. The random_forest method instead fits a probe random forest and ranks permutation importance. Smaller p-values and larger importances rank higher.",
      ],
      [
        "Rank and keep the top N",
        "Sort by score and keep the first top_n features. If fewer than top_n features pass the FDR filter, the tsfresh helper falls back to the top_n smallest p-values, so a list is always produced.",
      ],
      [
        "Save the per-seed list",
        "The retained feature names are written to the seed's CSV and used to subset columns before GridSearchCV. Across seeds, top_features.csv records how often each feature was selected (frequency), with its mean score and standard deviation.",
      ],
    ],
    caveat:
      "Selection runs before cross-validation on the same samples, so CV scores are optimistically biased (selection leakage). Seeds can select different features; selection frequency measures stability, not mechanism. A small p-value shows a statistical association in this sample. It does not show that the feature causes, or is a physical precursor of, an eruption.",
    implementation:
      "TrainingModel._run_feature_selection_job applies FeatureSelector per seed. get_significant_features wraps tsfresh with fdr_level=0.05 and top_n. Ranked outputs are top_features.csv and top_{N}_features.csv.",
    sources: [
      "model/training_model.py · _run_feature_selection_job",
      "features/feature_selector.py · FeatureSelector",
      "utils/ml.py · get_significant_features",
    ],
  },
  "grid-cv": {
    title: "Grid CV",
    intro: "Try every hyperparameter combination, then keep the best for this seed.",
    why: "A forest's depth or tree count changes how it fits. Instead of guessing, GridSearchCV scores every combination on held-out folds and refits the winner on all retained rows.",
    steps: [
      [
        "Define the grid and splitter",
        "ClassifierModel supplies an estimator, a parameter grid and a CV splitter. The seed reseeds the classifier. Only this seed's selected columns and resampled rows enter the search.",
      ],
      [
        "Split into folds",
        "shuffle, stratified or shuffle-stratified (the default) divide the rows into training and test folds. Stratified variants keep positives in every test fold; plain shuffle can leave a fold with none.",
      ],
      [
        "Score every combination",
        "Each combination is fitted on each training fold and scored on its test fold, using balanced_accuracy by default. Fold scores are averaged per combination. n_grids workers run these fits in parallel.",
      ],
      [
        "Refit the best",
        "The combination with the highest mean score wins and is refitted on all retained rows. This best_estimator_ is the seed model, pickled to models/{seed}.pkl and listed in the registry JSON.",
      ],
    ],
    caveat:
      "Fold scores from a small sample are noisy, and resampling and selection already used these rows, so they flatter the model. Grid search picks the best of the combinations offered, not the best possible model. Overlapping windows can leak across folds. Judge on later, unresampled data.",
    implementation:
      "utils/ml.py grid_search_cv builds GridSearchCV(estimator, param_grid, cv=get_cv_splitter(), scoring, n_jobs=n_grids), fits it on features[top_n_features] and returns best_estimator_. TrainingModel.fit runs one such job per seed.",
    sources: [
      "utils/ml.py · grid_search_cv",
      "model/classifier_model.py · get_cv_splitter",
      "model/training_model.py · fit",
    ],
  },
  "classifier-ensemble": {
    title: "Classifier Ensemble",
    intro: "Bring the seed ensembles together without losing their identities.",
    why: "Different classifier families can respond differently to the same features. ClassifierEnsemble bundles one SeedEnsemble per family, preserves each result for comparison, and provides a shared prediction interface. Consensus can soften dependence on one family, but improvement must be measured.",
    steps: [
      [
        "Collect the seed ensembles",
        "Train each configured classifier on the shared per-seed samples and feature lists. Each family becomes its own SeedEnsemble. This synthetic example uses RF, GB and XGB with three seeds each.",
      ],
      [
        "Bundle the families",
        "ClassifierEnsemble holds the family objects and their union of feature names. No extra meta-model is fitted during this merge. Adding families adds perspectives, not independent observations.",
      ],
      [
        "Predict the same future window",
        "Build features for a future window. Within each family, every seed reads its own feature subset, predicts, and contributes to that family’s mean probability. Enable or disable a family to explore its influence.",
      ],
      [
        "Read the consensus",
        "Average the family mean probabilities with equal weight per family. Consensus uncertainty is the standard deviation of family means. The prediction field separately averages the family vote fractions; it is not the probability thresholded into one binary answer.",
      ],
    ],
    caveat:
      "A family with more seeds does not receive more probability weight. Families may still share the same blind spots. Keep per-family outputs and evaluate discrimination, calibration and false alarms on later data before interpreting consensus as a stronger forecast.",
    implementation:
      "TrainingModel.build_classifier_ensemble bundles the fitted SeedEnsemble objects. PredictionModel.forecast calls ClassifierEnsemble.predict_with_uncertainty. This container is distinct from the voting classifier option trained inside TrainingModel.",
    sources: [
      "model/training_model.py · build_classifier_ensemble",
      "ensemble/classifier_ensemble.py · predict_with_uncertainty",
      "model/prediction_model.py · forecast",
      "utils/array.py · compute_model_probabilities",
    ],
  },
} as const;

export type TrainingLesson = keyof typeof TRAINING_LESSONS;
export const FAMILIES = [
  { name: "RF", probabilities: [0.62, 0.74, 0.68] },
  { name: "GB", probabilities: [0.48, 0.6, 0.54] },
  { name: "XGB", probabilities: [0.77, 0.65, 0.71] },
];
export function summarize(values: readonly number[]) {
  const mean = values.reduce((sum, value) => sum + value, 0) / values.length;
  return {
    mean,
    std: Math.sqrt(
      values.reduce((sum, value) => sum + (value - mean) ** 2, 0) /
        values.length,
    ),
  };
}

export interface SeedWorkflowNode {
  id: string;
  title: string;
  detail: string;
  kind: "code" | "data";
  // Walkthrough step (0-3) that this node belongs to.
  step: number;
  // Label drawn on the arrow leaving this node.
  edgeLabel?: string;
}

export const SEED_INPUT_NODE: SeedWorkflowNode = {
  id: "matrix",
  title: "Features Matrix",
  detail: "tsfresh, all windows",
  kind: "data",
  step: 0,
};

// Nodes inside the "one job per seed" boundary, in execution order.
export const SEED_JOB_NODES: readonly SeedWorkflowNode[] = [
  { id: "sampler", title: "Under Sampler", detail: "auto: under if <15%", kind: "code", step: 1 },
  { id: "selector", title: "Top-20 Selector", detail: "FDR or RF probe", kind: "code", step: 1, edgeLabel: "top_n" },
  { id: "top-csv", title: "Top-20 CSV", detail: "per-seed features", kind: "data", step: 1 },
  { id: "grid", title: "GridSearchCV", detail: "CV splitter + grid", kind: "code", step: 2 },
  { id: "seed-model", title: "Seed Model", detail: "best estimator pkl", kind: "data", step: 2 },
];

// Runs once after every per-seed job has finished, in execution order.
export const SEED_MERGE_NODES: readonly SeedWorkflowNode[] = [
  { id: "registry", title: "Seed Registry", detail: "trained-model JSON", kind: "data", step: 3 },
  { id: "seed-ensemble", title: "SeedEnsemble", detail: "all seeds, 1 clf", kind: "code", step: 3 },
  { id: "mean-probability", title: "Mean Probability", detail: "avg across seeds", kind: "code", step: 3 },
];

// Deterministic teaching fixture, not a reproduction of NumPy's random stream.
export function retainedNegatives(seed: number) {
  let state = seed + 1;
  const ids = Array.from({ length: 48 }, (_, i) => i + 1);
  for (let i = ids.length - 1; i > 0; i--) {
    state = (Math.imul(state, 1664525) + 1013904223) >>> 0;
    const j = state % (i + 1);
    [ids[i], ids[j]] = [ids[j]!, ids[i]!];
  }
  return ids.slice(0, 8).sort((a, b) => a - b);
}
