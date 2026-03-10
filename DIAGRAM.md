# eruption-forecast Pipeline Diagrams

This document provides Mermaid diagrams for the full eruption-forecast pipeline and each
individual stage. Each diagram is preceded by a brief description of the stage's purpose
and followed by notes on key design decisions.

---

## 1. Full Pipeline Overview

The complete pipeline orchestrated by `ForecastModel` processes raw seismic data into
eruption probability forecasts through seven sequential stages. Each stage produces a
structured artifact consumed by the next.

```mermaid
flowchart TD
    SRC["Data Source\nSDS archive or FDSN web service"]

    subgraph S1["Stage 1 - CalculateTremor"]
        CT["CalculateTremor\ncalculate_tremor.py"]
    end

    subgraph S2["Stage 2 - LabelBuilder"]
        LB["LabelBuilder / DynamicLabelBuilder\nlabel_builder.py"]
    end

    subgraph S3["Stage 3 - TremorMatrixBuilder"]
        TMB["TremorMatrixBuilder\ntremor_matrix_builder.py"]
    end

    subgraph S4["Stage 4 - FeaturesBuilder"]
        FB["FeaturesBuilder + FeatureSelector\nfeatures_builder.py"]
    end

    subgraph S5["Stage 5 - ModelTrainer"]
        MT["ModelTrainer\nmodel_trainer.py"]
    end

    subgraph S6["Stage 6 - Evaluation"]
        ME["ModelEvaluator\nMultiModelEvaluator\nClassifierComparator"]
    end

    subgraph S7["Stage 7 - ModelPredictor"]
        MP["ModelPredictor\nmodel_predictor.py"]
    end

    OUT_T["Tremor CSV\nrsam_f0, dsar_f0-f1, entropy\n10-min samples"]
    OUT_L["Label CSV\nid, is_erupted\nsliding windows"]
    OUT_M["Tremor Matrix\nid, datetime, tremor columns"]
    OUT_F["Features CSV\n~5000 tsfresh features\nselected features list"]
    OUT_PKL["Model artifacts\n.pkl per seed, metrics JSON\nregistry CSV"]
    OUT_EVL["Evaluation reports\nmetrics, plots, SHAP"]
    OUT_FC["Forecast CSV\nper-classifier + consensus\nprobability columns"]

    SRC --> S1 --> OUT_T
    OUT_T --> S2
    OUT_T --> S3
    S2 --> OUT_L --> S3
    S3 --> OUT_M --> S4
    S4 --> OUT_F --> S5
    S5 --> OUT_PKL --> S6
    OUT_PKL --> S7
    S6 --> OUT_EVL
    S7 --> OUT_FC
```

---

## 2. Stage 1 - CalculateTremor

`CalculateTremor` reads raw seismic waveforms from a local SDS archive or a remote FDSN
web service, computes three complementary tremor metrics across multiple frequency bands,
and writes the result to a merged CSV file with 10-minute sampling intervals. Days are
processed in parallel via `joblib` with `n_jobs` workers.

```mermaid
flowchart TD
    CFG["Configuration\nstation, channel, start_date, end_date\nfrequency_bands, n_jobs"]

    subgraph SOURCE["Data Source Selection"]
        SRC_DECISION{{"from_sds() or from_fdsn()?"}}
        SDS_SRC["SDS archive\nSeisComP Data Structure\nsds.py"]
        FDSN_SRC["FDSN web service\nwith local SDS cache\nfdsn.py"]
    end

    subgraph PARALLEL["Parallel Processing - joblib n_jobs workers"]
        DAY["One day of waveforms\nObsPy Stream / Trace"]

        subgraph METRICS["Metric Computation per Frequency Band"]
            RSAM["RSAM\nMean amplitude per band\nrsam.py"]
            DSAR["DSAR\nRatio between consecutive bands\ndsar.py"]
            ENTROPY["ShannonEntropy\nSignal complexity 1-16 Hz\n.filter().calculate()\nshannon_entropy.py"]
        end

        DAILY_CSV["Daily CSV\ntemporary output per day"]
    end

    MERGE["Merge all daily CSVs\ninto single DataFrame"]
    OUT["Tremor CSV\nDateTime index, 10-min intervals\nrsam_f0, rsam_f1\ndsar_f0-f1, dsar_f1-f2\nentropy"]

    CFG --> SRC_DECISION
    SRC_DECISION -->|"from_sds()"| SDS_SRC
    SRC_DECISION -->|"from_fdsn()"| FDSN_SRC
    SDS_SRC --> DAY
    FDSN_SRC --> DAY
    DAY --> RSAM
    DAY --> DSAR
    DAY --> ENTROPY
    RSAM --> DAILY_CSV
    DSAR --> DAILY_CSV
    ENTROPY --> DAILY_CSV
    DAILY_CSV --> MERGE
    MERGE --> OUT
```

**Notes:**
- Frequency bands default to `(0.01-0.1), (0.1-2), (2-5), (4.5-8), (8-16) Hz`, aliased as `f0..f4`.
- `FDSN` caches downloaded data as SDS miniSEED so subsequent runs skip the network.
- Daily CSV files are optionally cleaned up after merging (`cleanup_daily_dir=True`).

---

## 3. Stage 2 - LabelBuilder

`LabelBuilder` divides a date range into overlapping sliding windows and assigns a binary
label to each window: `1` if the window falls within `day_to_forecast` days before a known
eruption date, `0` otherwise. `DynamicLabelBuilder` is a subclass that instead creates one
isolated window per eruption event spanning `days_before_eruption` days, then concatenates
all windows with unique IDs.

```mermaid
flowchart TD
    CFG["Configuration\nstart_date, end_date\neruption_dates\nwindow_step, window_step_unit\nday_to_forecast, volcano_id"]

    VARIANT{{"LabelBuilder variant?"}}

    subgraph STATIC["LabelBuilder - global sliding windows"]
        SW["Create sliding windows\nover full date range\nstep = window_step in minutes/hours"]
        INIT["Initialize all window labels to 0"]
        MARK["For each eruption_date:\nmark windows in\n[eruption_date - day_to_forecast,\n eruption_date]\nas is_erupted = 1"]
    end

    subgraph DYNAMIC["DynamicLabelBuilder - per-eruption windows"]
        FOREACH["For each eruption_date"]
        WIN_PER["Create windows spanning\ndays_before_eruption days\nbefore eruption_date"]
        MARK_D["Label windows relative\nto that eruption"]
        CONCAT["Concatenate all per-eruption\nwindows with unique IDs"]
    end

    OUT["Label CSV\nDateTime index\nid (int), is_erupted (0 or 1)\nFilename encodes all parameters"]

    CFG --> VARIANT
    VARIANT -->|"LabelBuilder"| SW
    SW --> INIT --> MARK --> OUT
    VARIANT -->|"DynamicLabelBuilder"| FOREACH
    FOREACH --> WIN_PER --> MARK_D --> CONCAT --> OUT
```

**Notes:**
- Label filename convention: `label_YYYY-MM-DD_YYYY-MM-DD_ws-X_step-X-unit_dtf-X.csv`.
- `LabelData` parses all parameters back from the filename via `cached_property`.

---

## 4. Stage 3 - TremorMatrixBuilder

`TremorMatrixBuilder` aligns the continuous tremor time-series with the discrete label
windows. For every window defined in the label DataFrame it slices the tremor DataFrame,
validates that the expected number of samples is present, prepends the window `id`, and
stacks all slices into a single matrix ready for feature extraction.

```mermaid
flowchart TD
    IN_T["Tremor DataFrame\nDateTime index, tremor columns\n10-min sampling"]
    IN_L["Label DataFrame\nid, is_erupted\nwindow start/end times"]

    ITER["Iterate over label windows"]

    subgraph WINDOW_PROC["Per-Window Processing"]
        SLICE["Slice tremor DataFrame\nto window time range"]
        VALIDATE["Validate sample count\nagainst expected window_size"]
        VALID{{"Sample count valid?"}}
        SKIP["Skip window\nlog warning"]
        PREPEND["Prepend window id column"]
    end

    CONCAT["Concatenate all valid window slices"]
    OUT["Tremor matrix DataFrame\nid, datetime\nrsam_f0, rsam_f1\ndsar_f0-f1\nentropy"]

    IN_T --> ITER
    IN_L --> ITER
    ITER --> SLICE
    SLICE --> VALIDATE
    VALIDATE --> VALID
    VALID -->|"No"| SKIP
    VALID -->|"Yes"| PREPEND
    PREPEND --> CONCAT
    CONCAT --> OUT
```

---

## 5. Stage 4 - FeaturesBuilder and FeatureSelector

`FeaturesBuilder` runs tsfresh automated feature extraction on the tremor matrix. It
operates in two distinct modes depending on whether labels are provided. After extraction,
`FeatureSelector` applies a two-stage selection pipeline to reduce the ~5000 tsfresh
features to a manageable, statistically relevant subset.

```mermaid
flowchart TD
    IN_M["Tremor matrix\nid, datetime, tremor columns"]
    IN_L{{"Label DataFrame provided?"}}

    subgraph TRAIN_MODE["Training Mode - labels provided"]
        FILTER_W["Filter tremor matrix windows\nto match label IDs"]
        ALIGN_L["Save aligned label CSV\nto features output dir"]
        EXTRACT_T["tsfresh extraction\nper tremor column\nwith relevance filter\nFDR-based feature filtering"]
    end

    subgraph PRED_MODE["Prediction Mode - no labels"]
        EXTRACT_P["tsfresh extraction\nper tremor column\nall features, no relevance filter"]
    end

    subgraph SELECTOR["FeatureSelector - 2-stage selection"]
        STAGE1["Stage 1 - tsfresh FDR\nStatistical relevance filtering\nBenjamini-Hochberg correction"]
        STAGE2{{"Method?"}}
        TSFRESH_ONLY["tsfresh only\nmethod: tsfresh"]
        RF_SEL["Stage 2 - RandomForest importance\nTop N significant features\nmethod: random_forest"]
        BOTH["Combined:\ntsfresh FDR then RandomForest\nmethod: combined"]
    end

    OUT_F["Features CSV\n~5000 raw or filtered features\none row per window"]
    OUT_S["Selected features list\nsignificant_features.txt"]

    IN_M --> IN_L
    IN_L -->|"Yes"| FILTER_W
    IN_L -->|"No"| EXTRACT_P
    FILTER_W --> ALIGN_L
    FILTER_W --> EXTRACT_T
    EXTRACT_T --> SELECTOR
    EXTRACT_P --> OUT_F
    SELECTOR --> STAGE1
    STAGE1 --> STAGE2
    STAGE2 -->|"tsfresh"| TSFRESH_ONLY
    STAGE2 -->|"random_forest"| RF_SEL
    STAGE2 -->|"combined"| BOTH
    TSFRESH_ONLY --> OUT_S
    RF_SEL --> OUT_S
    BOTH --> OUT_S
    ALIGN_L --> OUT_S
```

---

## 6. Stage 5 - ModelTrainer

`ModelTrainer` trains one or more classifiers across multiple random seeds in parallel.
The `fit()` method dispatches to one of two training modes controlled by the
`with_evaluation` flag. All seed runs for a given classifier are executed via
`joblib.Parallel` with the `loky` backend for safe nested parallelism.

```mermaid
flowchart TD
    IN_F["Features CSV\nselected features list"]
    IN_LABEL["Label CSV\naligned to features"]
    CFG["Configuration\nclassifier, cv_strategy\nrandom_state, total_seed\nn_jobs, grid_search_n_jobs\nnumber_of_significant_features"]

    FIT{{"fit(with_evaluation=?)"}}

    subgraph EVAL_MODE["train_and_evaluate() - with_evaluation=True"]
        SPLIT["80/20 train/test split\nsklearn train_test_split"]
        RESAMPLE_E["RandomUnderSampler\non training split only"]
        FEAT_SEL_E["FeatureSelector\non training data only"]
        GS_E["GridSearchCV\ncv_strategy: shuffle / stratified /\nshuffle-stratified / timeseries"]
        EVAL["ModelEvaluator\nmetrics on held-out test set\naccuracy, F1, ROC-AUC\nconfusion matrix, plots"]
        SAVE_E["Save model .pkl\nmetrics JSON\nregistry CSV entry"]
    end

    subgraph TRAIN_MODE["train() - with_evaluation=False"]
        RESAMPLE_T["RandomUnderSampler\non full dataset"]
        FEAT_SEL_T["FeatureSelector\non full resampled data"]
        GS_T["GridSearchCV\nacross CV folds"]
        SAVE_T["Save model .pkl\nno metrics\nregistry CSV entry"]
    end

    subgraph PARALLEL["joblib.Parallel - loky backend"]
        SEED_LOOP["One worker per random seed\nn_jobs outer workers\ngrid_search_n_jobs inner workers\nconstraint: n_jobs x grid_search_n_jobs <= cpu_count"]
    end

    CLASSIFIERS["10 supported classifiers\nrf, gb, xgb, svm, lr\nnn, dt, knn, nb, voting"]

    OUT["Per-seed artifacts\ntrained_model_{seed}.pkl\nmetrics_{seed}.json\nregistry.csv"]
    MERGE["merge_models() / merge_classifier_models()\nSeedEnsemble or ClassifierEnsemble"]
    OUT_MERGED["Merged model .pkl\nSeedEnsemble per classifier\nClassifierEnsemble across classifiers"]

    IN_F --> FIT
    IN_LABEL --> FIT
    CFG --> FIT
    CLASSIFIERS --> FIT
    FIT -->|"True"| EVAL_MODE
    FIT -->|"False"| TRAIN_MODE
    EVAL_MODE --> PARALLEL
    TRAIN_MODE --> PARALLEL
    PARALLEL --> SEED_LOOP
    SEED_LOOP --> OUT
    OUT --> MERGE
    MERGE --> OUT_MERGED

    SPLIT --> RESAMPLE_E --> FEAT_SEL_E --> GS_E --> EVAL --> SAVE_E
    RESAMPLE_T --> FEAT_SEL_T --> GS_T --> SAVE_T
```

**Notes:**
- Resampling and feature selection always occur inside the train split to prevent data leakage.
- `cv_strategy` options: `shuffle`, `stratified`, `shuffle-stratified`, `timeseries`.

---

## 7. Stage 6 - Evaluation

Three evaluator classes are provided at increasing levels of aggregation. `ModelEvaluator`
handles one seed, `MultiModelEvaluator` aggregates across all seeds for one classifier, and
`ClassifierComparator` compares multiple classifiers side by side.

```mermaid
flowchart TD
    IN_PKL["Per-seed model .pkl\nmetrics JSON files\nregistry CSV"]

    subgraph SINGLE["ModelEvaluator - per-seed evaluation"]
        ME_INIT["Load model and test data\n__init__() or from_files()"]
        ME_METRICS["Compute metrics\naccuracy, balanced_accuracy\nF1, precision, recall\nROC-AUC, threshold analysis"]
        ME_PLOTS["Generate plots\nROC curve, calibration\nconfusion matrix\nlearning curve\nprediction distribution"]
        ME_OUT["Per-seed metrics dict\nplot files"]
    end

    subgraph MULTI["MultiModelEvaluator - aggregate across seeds"]
        MME_LOAD["Load all per-seed JSON metrics\nor read registry CSV"]
        MME_AGG["Aggregate statistics\nmean, std across seeds"]
        MME_PLOTS["Aggregate plots\nSHAP summary\nROC envelope\naggregate calibration\nconfusion matrix heatmap\nlearning curve with variance\nseed stability"]
        MME_OUT["Ensemble-level metrics\naggregate plot files"]
    end

    subgraph COMPARE["ClassifierComparator - cross-classifier comparison"]
        CC_MAP["Accept mapping:\nclassifier_name -> registry CSV path"]
        CC_BUILD["Build one MultiModelEvaluator\nper classifier"]
        CC_TABLE["Side-by-side comparison table\nmean metrics per classifier"]
        CC_PLOTS["Comparison plots\nplot_classifier_comparison()"]
        CC_OUT["Comparison table CSV\ncomparison plot files"]
    end

    IN_PKL --> SINGLE
    ME_INIT --> ME_METRICS --> ME_PLOTS --> ME_OUT
    ME_OUT --> MULTI
    MME_LOAD --> MME_AGG --> MME_PLOTS --> MME_OUT

    IN_PKL --> COMPARE
    CC_MAP --> CC_BUILD --> CC_TABLE --> CC_PLOTS --> CC_OUT
    MME_OUT --> CC_BUILD
```

---

## 8. Stage 7 - ModelPredictor

`ModelPredictor` runs inference using models produced by `ModelTrainer`. It supports two
operating modes: evaluation mode (requires labels) and forecast mode (unlabelled). In
multi-model consensus mode it aggregates eruption probability across all seeds of each
classifier (`SeedEnsemble`) and optionally across classifiers (`ClassifierEnsemble`).

```mermaid
flowchart TD
    IN_MODELS["Trained model(s)\nSeedEnsemble .pkl per classifier\nor ClassifierEnsemble .pkl"]
    IN_TREMOR["New tremor CSV\nDatetime index, tremor columns"]

    MODE{{"Operating mode?"}}

    subgraph EVAL_MODE["Evaluation Mode - future_labels_csv provided"]
        PREP_E["Load tremor data\nTremorData"]
        MATRIX_E["TremorMatrixBuilder\nslice into labeled windows"]
        FEAT_E["FeaturesBuilder\nextract tsfresh features\nprediction mode"]
        ALIGN_LABELS["Align features to label windows"]
        PREDICT["predict() or predict_best()\nrequires labels"]
        METRICS_E["Compute per-classifier metrics\naccuracy, F1, recall\nROC-AUC, balanced accuracy"]
        OUT_E["Evaluation results\nper-seed and per-classifier metrics\nplots via ModelEvaluator"]
    end

    subgraph FORECAST_MODE["Forecast Mode - no labels"]
        PREP_F["Load tremor data\nTremorData"]
        MATRIX_F["TremorMatrixBuilder\nslice into unlabeled windows"]
        FEAT_F["FeaturesBuilder\nextract tsfresh features\nprediction mode"]
        PROBA["predict_proba()\nno labels required"]
        AGG_SEED["SeedEnsemble\naggregate across seeds per classifier\nmean probability + uncertainty"]
        AGG_CLF{{"Multi-classifier?"}}
        CLF_ENS["ClassifierEnsemble\naggregate across classifiers\nconsensus probability + uncertainty"]
        SINGLE_OUT["Single-classifier output\nprobability, uncertainty\nconfidence, prediction columns"]
        MULTI_OUT["Multi-classifier output\nper-classifier dashed columns\nconsensus solid column\nshaded uncertainty band"]
        OUT_F["Forecast CSV\nDatetime index\n{name}_eruption_probability\n{name}_uncertainty\n{name}_confidence\n{name}_prediction\nconsensus_* columns"]
        PLOT_F["Forecast plot\nplot_forecast() or\nplot_forecast_with_events()"]
    end

    IN_MODELS --> MODE
    IN_TREMOR --> MODE
    MODE -->|"labels provided"| EVAL_MODE
    MODE -->|"no labels"| FORECAST_MODE

    PREP_E --> MATRIX_E --> FEAT_E --> ALIGN_LABELS --> PREDICT --> METRICS_E --> OUT_E
    PREP_F --> MATRIX_F --> FEAT_F --> PROBA --> AGG_SEED --> AGG_CLF
    AGG_CLF -->|"No"| SINGLE_OUT --> OUT_F
    AGG_CLF -->|"Yes"| CLF_ENS --> MULTI_OUT --> OUT_F
    OUT_F --> PLOT_F
```

---

## 9. Ensemble Model Architecture

The ensemble objects produced by `merge_models()` and used by `ModelPredictor` form a
two-level hierarchy. Both levels extend `BaseEnsemble` for consistent serialisation
via joblib.

```mermaid
flowchart TD
    BASE["BaseEnsemble\nbase_ensemble.py\nsave(path) / load(path) mixin"]

    subgraph SEED_ENS["SeedEnsemble - one classifier, all seeds\nseed_ensemble.py"]
        SE_LOAD["from_registry(registry_csv)\nload all seed .pkl files"]
        SE_PRED["predict_proba(X)\naggregate across seeds\nreturn mean probability + std"]
        SE_UNC["predict_with_uncertainty(X)\nreturn mean, std, confidence, prediction"]
        SE_SAVE["save(path) / load(path)\nBaseEnsemble mixin\njoblib serialisation"]
    end

    subgraph CLF_ENS["ClassifierEnsemble - multiple classifiers\nclassifier_ensemble.py"]
        CE_LOAD["from_seed_ensembles(dict)\nor from_registry_dict(dict)"]
        CE_PRED["predict_proba(X)\naggregate across classifiers\nper-classifier dict output"]
        CE_UNC["predict_with_uncertainty(X)\nmean, std, conf, pred\nper_clf_dict"]
        CE_PROP["classifiers property\n__getitem__, __len__"]
        CE_SAVE["save(path) / load(path)\nBaseEnsemble mixin"]
    end

    PKL_SEEDS["Per-seed .pkl files\nGridSearchCV fitted estimators"]
    REG_CSV["registry.csv\nmaps seed -> model path"]
    MERGED_PKL["SeedEnsemble .pkl\nor ClassifierEnsemble .pkl"]

    PKL_SEEDS --> SE_LOAD
    REG_CSV --> SE_LOAD
    SE_LOAD --> SE_PRED
    SE_LOAD --> SE_UNC
    SE_LOAD --> SE_SAVE

    SEED_ENS --> CE_LOAD
    CE_LOAD --> CE_PRED
    CE_LOAD --> CE_UNC
    CE_LOAD --> CE_PROP
    CE_LOAD --> CE_SAVE

    BASE --> SEED_ENS
    BASE --> CLF_ENS

    SE_SAVE --> MERGED_PKL
    CE_SAVE --> MERGED_PKL
```

---

## 10. Data Artifacts and Output Directory Structure

This diagram shows all file artifacts produced at each stage and where they are written
relative to `root_dir`.

```mermaid
flowchart TD
    ROOT["root_dir/output/\n{network}.{station}.{location}.{channel}/"]

    subgraph TREMOR_DIR["tremor/"]
        T_DAILY["daily/\n{date}_tremor.csv\ntemporary, cleaned up optionally"]
        T_FIGS["figures/\n{date}_tremor.png\ncreated if plot_daily=True"]
        T_MERGED["{nslc}_{start}_{end}.csv\nfinal merged tremor CSV"]
    end

    subgraph LABEL_DIR["label/"]
        L_CSV["label_YYYY-MM-DD_YYYY-MM-DD\n_ws-X_step-X-unit_dtf-X.csv"]
    end

    subgraph FEAT_DIR["features/"]
        F_CSV["features_{start}_{end}.csv\n~5000 tsfresh features"]
        F_LABEL["aligned_labels.csv\nmatched to feature rows"]
        F_SIG["significant_features.txt\nselected feature names"]
    end

    subgraph TRAIN_DIR["trainings/{classifier-slug}/{cv-slug}/"]
        TR_PKL["trained_model_{seed}.pkl\nfitted GridSearchCV estimator"]
        TR_JSON["metrics_{seed}.json\nper-seed evaluation metrics"]
        TR_REG["registry.csv\nmaps seed to model path + metadata"]
        TR_PLOTS["plots/\nROC, calibration, confusion matrix\nlearning curve, SHAP"]
    end

    subgraph EVAL_DIR["evaluations/{classifier-slug}/{cv-slug}/"]
        EV_PLOTS["aggregate plots\nensemble ROC, SHAP summary\nstability, comparison"]
        EV_MERGED["merged_registry.csv\naggregated seed statistics"]
    end

    subgraph FORECAST_DIR["forecast/"]
        FC_CSV["forecast_{start}_{end}.csv\nprobability, uncertainty\nconsensus columns"]
        FC_PLOT["forecast_{start}_{end}.png\ntime-series probability plot"]
    end

    CONFIG["config.yaml\nfull pipeline configuration\nreplayable via from_config()"]

    ROOT --> TREMOR_DIR
    ROOT --> LABEL_DIR
    ROOT --> FEAT_DIR
    ROOT --> TRAIN_DIR
    ROOT --> EVAL_DIR
    ROOT --> FORECAST_DIR
    ROOT --> CONFIG
```

---

## 11. ForecastModel Method Chaining API

`ForecastModel` exposes every pipeline stage as a chainable method. This diagram shows
the full call sequence and the optional paths available to users.

```mermaid
flowchart TD
    INIT["ForecastModel(\n  root_dir, station, channel\n  start_date, end_date\n  window_size, volcano_id, n_jobs\n)"]

    DATA_CHOICE{{"Data acquisition choice"}}

    CALC["calculate(\n  source='sds' or 'fdsn'\n  sds_dir or fdsn_url\n)"]
    LOAD_T["load_tremor_data(\n  tremor_csv\n)\nalternative to calculate()"]

    BUILD_LABEL["build_label(\n  start_date, end_date\n  eruption_dates\n  day_to_forecast\n  window_step, window_step_unit\n)"]

    EXTRACT["extract_features(\n  select_tremor_columns\n)"]

    SET_FS["set_feature_selection_method(\n  using='tsfresh' or\n  'random_forest' or 'combined'\n)\noptional before train()"]

    TRAIN["train(\n  classifier, cv_strategy\n  random_state, total_seed\n  with_evaluation=True or False\n  number_of_significant_features\n)"]

    FORECAST["forecast(\n  start_date, end_date\n  window_size, window_step\n  window_step_unit\n)"]

    SAVE_CFG["save_config(path, fmt)\npersist pipeline configuration"]
    LOAD_CFG["ForecastModel.from_config(path)\nrestore and replay configuration"]
    SAVE_MODEL["save_model(path)\nserialise full ForecastModel\nvia joblib"]
    LOAD_MODEL["ForecastModel.load_model(path)\nrestore full model object"]
    RUN["run()\nreplay all stages\nfrom loaded config"]

    INIT --> DATA_CHOICE
    DATA_CHOICE -->|"new data"| CALC
    DATA_CHOICE -->|"existing CSV"| LOAD_T
    CALC --> BUILD_LABEL
    LOAD_T --> BUILD_LABEL
    BUILD_LABEL --> EXTRACT
    EXTRACT --> SET_FS
    SET_FS --> TRAIN
    EXTRACT --> TRAIN
    TRAIN --> FORECAST

    TRAIN --> SAVE_CFG
    SAVE_CFG -.->|"later session"| LOAD_CFG
    LOAD_CFG --> RUN
    TRAIN --> SAVE_MODEL
    SAVE_MODEL -.->|"later session"| LOAD_MODEL
```

---

*Author: martanto*
*Last updated: 2026-03-10*
