export const TRAINING_LESSONS = {
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
