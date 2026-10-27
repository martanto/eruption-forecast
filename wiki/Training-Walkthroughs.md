# Training walkthroughs

Pipeline Atlas provides interactive, static teaching pages after Labelling, grouped as the **Training Guideline**. Open **Training walkthroughs** from the Atlas toolbar, or continue from Labelling. These routes use synthetic data and do not execute Python.

| Route | What to explore |
|---|---|
| `/features-matrix` | Features Matrix Builder. A workflow diagram traces `FeaturesBuilder` (validate, mode dispatch, cache check, per-column tsfresh, concat). See raw tremor windows become tsfresh features, named `{column}__{calculator}`, then merge by window id. |
| `/random-under-sampler` | Count classes, interpret the sampling ratio, change a seeded majority draw, and follow retained rows into training. |
| `/seed-ensemble` | Opens the guideline. A workflow diagram boxes the per-seed job ("One job per seed") and a trace animates it. Follow one classifier family through per-seed resampling, feature selection, fitting and probability aggregation. |
| `/feature-selection` | Score features with tsfresh FDR p-values or random-forest permutation importance, change the seed and `top_n`, and see the saved list. Below it, a four-step dummy-data explainer shows where the p-values come from: (1) the labelled data, 24 windows in time order with two eruptions; (2) the KS staircase test; (3) Fisher's exact test with a worked C(n, r) calculation; and (4) the Benjamini–Yekutieli correction at FDR 0.05 (compared with BH and no correction). tsfresh's own default for number-valued features is Mann–Whitney U; KS is its alternative option. |
| `/grid-cv` | Compare CV splitters, fold membership and a hyperparameter grid, then refit the best setting as the seed model. |
| `/classifier-ensemble` | Bundle seed ensembles, predict a common future window and inspect equal-weight family consensus. |

## Random under-sampling, step by step

Historical windows outside positive forecast days usually outnumber positive windows. Predicting only 0 can yield high accuracy while missing every positive. RUS retains minority rows and selects fewer majority rows without replacement. A positive label describes a forecast window; several windows can belong to one eruption.

1. Count the classes: 48 negative and 6 positive windows in the fixture, a minority share of 11.1%.
2. Set `sampling_strategy=0.75`, the desired minority-to-majority ratio after resampling. Six positives require eight negatives: `6 / 0.75 = 8`. The positive share becomes 42.9%, not 75%.
3. Retain all six positives and draw eight negatives. Change the seed to inspect another draw. The fixture is deterministic but does not reproduce NumPy's random stream.
4. Pass retained feature rows and aligned labels into per-seed feature selection and fitting. The original history remains available to other seeds. Future prediction windows are excluded from training.

`TrainingModel.fit` with `resample_method="auto"` chooses under-sampling when the minority share is strictly below `minority_threshold=0.15`; otherwise it leaves the sample unchanged. Explicit `"under"` requests under-sampling.

The RUS page replaces its reference image with a worked example: retain all 300 positive windows and randomly select 400 of 2,000 negative windows. Thus `300 / 400 = 0.75`, giving 42.9% positive and 57.1% negative windows, rather than a 75/25 split.

Current implementation order is **resample → feature selection → GridSearchCV**. Internal CV is not an untouched holdout: pre-CV selection and overlapping windows can produce optimistic scores. Evaluate later, unresampled data separately. Under-sampling discards majority information and changes the training prior; it does not guarantee calibrated probabilities.

## Seed ensembles, step by step

1. Start with one historical feature matrix and labels. `seeds=N` produces indices `0..N-1`.
2. Resample and select features once per seed. Persist retained ids and labels plus chosen feature names. Reuse each seed's sample and feature list across classifier families.
3. Fit each classifier for each seed through GridSearchCV. Seed-controlled sampling, initialization and CV splitting, where supported, can produce different models.
4. Build one `SeedEnsemble` per classifier from the fitted registry. At inference each model reads its own feature columns. Aggregate probabilities with a mean and population standard deviation.

The synthetic RF probabilities 62%, 74% and 68% produce a 68% mean and 4.9 percentage-point SD. Multiple seeds can reduce dependence on an unlucky draw and reduce variance when errors differ. Shared data makes errors correlated; averaging does not automatically remove bias, leakage or missing event types. SD measures disagreement, not a calibrated prediction interval.

The route opens with `SeedWorkflowDiagram`: Features Matrix, then a dashed "One job per seed (random_state)" boundary around Under Sampler, Top-20 Selector, Top-20 CSV, GridSearchCV and Seed Model, then Seed Registry, SeedEnsemble and Mean Probability. Boxes jump to their step; "Trace one run" animates the flow (disabled transitions under `prefers-reduced-motion`). Data lives in `trainingLessons.ts`.

Reference: [Seed Ensemble illustration](../assets/seed-ensemble.jpg). Implementation: `TrainingModel.fit`, `build_seed_ensemble`, `SeedEnsemble.compute_probabilities_and_predictions` and `utils/array.py:compute_model_probabilities`.

## Classifier ensembles, step by step

1. Collect fitted `SeedEnsemble` objects, one per configured family.
2. Bundle them into `ClassifierEnsemble`, including their union of feature names. This merge fits no meta-model and is distinct from the `voting` classifier option.
3. Predict each future row through all seeds, then aggregate within each family. Models still receive their own feature subsets.
4. Average family mean probabilities equally. RF=68%, GB=54%, XGB=71% gives `(68 + 54 + 71) / 3 = 64.3%`. Disabling GB gives 69.5%. A family with more seeds receives no extra probability weight.

`consensus_uncertainty` is the population SD of family means. `consensus_prediction` separately averages family vote fractions; it is continuous, not the consensus probability thresholded into a binary answer. The confidence output uses the existing CI-like expression based on votes; it is not a validated forecast interval. One enabled family gives zero cross-family SD without establishing certainty.

Reference: [Classifier Ensemble illustration](../assets/classifier-ensemble.jpg). Implementation: `TrainingModel.build_classifier_ensemble`, `ClassifierEnsemble.predict_with_uncertainty`, `PredictionModel.forecast` and `utils/array.py:compute_model_probabilities`. The supplied `assets/How Eruption Forecasting Works.pdf` provides introductory context; its example counts are not mandatory settings.

## Maintenance

The fixture and copy live in `app/src/features/explorer/trainingLessons.ts`, rendered by `TrainingWalkthrough.tsx`. Vite bundles the seed and classifier ensemble JPEGs directly from repository `assets/`; keep these files available when building. `app/tests/training-lessons.test.ts` checks sampling and aggregation. `app/tests/browser/training-walkthroughs.spec.ts` verifies routes, controls, image loading, reset and mobile overflow. Screenshots stay under `app/tests/review/training/`. See [Architecture Explorer](Architecture-Explorer.md) for launch commands.
