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
    SRC["Data Source<br/>SDS archive or FDSN web service"]

    subgraph S1["Stage 1 - CalculateTremor"]
        CT["CalculateTremor<br/>calculate_tremor.py"]
    end

    subgraph S2["Stage 2 - LabelBuilder"]
        LB["LabelBuilder / DynamicLabelBuilder<br/>label_builder.py"]
    end

    subgraph S3["Stage 3 - TremorMatrixBuilder"]
        TMB["TremorMatrixBuilder<br/>tremor_matrix_builder.py"]
    end

    subgraph S4["Stage 4 - FeaturesBuilder"]
        FB["FeaturesBuilder + FeatureSelector<br/>features_builder.py"]
    end

    subgraph S5["Stage 5 - ModelTrainer"]
        MT["ModelTrainer<br/>model_trainer.py"]
    end

    subgraph S6["Stage 6 - Evaluation"]
        ME["ModelEvaluator<br/>MultiModelEvaluator<br/>ClassifierComparator"]
    end

    subgraph S7["Stage 7 - ModelPredictor"]
        MP["ModelPredictor<br/>model_predictor.py"]
    end

    OUT_T["Tremor CSV<br/>rsam_f0, dsar_f0-f1, entropy<br/>10-min samples"]
    OUT_L["Label CSV<br/>id, is_erupted<br/>sliding windows"]
    OUT_M["Tremor Matrix<br/>id, datetime, tremor columns"]
    OUT_F["Features CSV<br/>~5000 tsfresh features<br/>selected features list"]
    OUT_PKL["Model artifacts<br/>.pkl per seed, metrics JSON<br/>registry CSV"]
    OUT_EVL["Evaluation reports<br/>metrics, plots, SHAP"]
    OUT_FC["Forecast CSV<br/>per-classifier + consensus<br/>probability columns"]

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
    CFG["Configuration<br/>station, channel, start_date, end_date<br/>frequency_bands, n_jobs"]

    subgraph SOURCE["Data Source Selection"]
        SRC_DECISION{{"from_sds() or from_fdsn()?"}}
        SDS_SRC["SDS archive<br/>SeisComP Data Structure<br/>sds.py"]
        FDSN_SRC["FDSN web service<br/>with local SDS cache<br/>fdsn.py"]
    end

    subgraph PARALLEL["Parallel Processing - joblib n_jobs workers"]
        DAY["One day of waveforms<br/>ObsPy Stream / Trace"]

        subgraph METRICS["Metric Computation per Frequency Band"]
            RSAM["RSAM<br/>Mean amplitude per band<br/>rsam.py"]
            DSAR["DSAR<br/>Ratio between consecutive bands<br/>dsar.py"]
            ENTROPY["ShannonEntropy<br/>Signal complexity 1-16 Hz<br/>.filter().calculate()<br/>shannon_entropy.py"]
        end

        DAILY_CSV["Daily CSV<br/>temporary output per day"]
    end

    MERGE["Merge all daily CSVs<br/>into single DataFrame"]
    OUT["Tremor CSV<br/>DateTime index, 10-min intervals<br/>rsam_f0, rsam_f1<br/>dsar_f0-f1, dsar_f1-f2<br/>entropy"]

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
    CFG["Configuration<br/>start_date, end_date<br/>eruption_dates<br/>window_step, window_step_unit<br/>day_to_forecast, volcano_id"]

    VARIANT{{"LabelBuilder variant?"}}

    subgraph STATIC["LabelBuilder - global sliding windows"]
        SW["Create sliding windows<br/>over full date range<br/>step = window_step in minutes/hours"]
        INIT["Initialize all window labels to 0"]
        MARK["For each eruption_date:<br/>mark windows in<br/>[eruption_date - day_to_forecast,<br/> eruption_date]<br/>as is_erupted = 1"]
    end

    subgraph DYNAMIC["DynamicLabelBuilder - per-eruption windows"]
        FOREACH["For each eruption_date"]
        WIN_PER["Create windows spanning<br/>days_before_eruption days<br/>before eruption_date"]
        MARK_D["Label windows relative<br/>to that eruption"]
        CONCAT["Concatenate all per-eruption<br/>windows with unique IDs"]
    end

    OUT["Label CSV<br/>DateTime index<br/>id (int), is_erupted (0 or 1)<br/>Filename encodes all parameters"]

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
    IN_T["Tremor DataFrame<br/>DateTime index, tremor columns<br/>10-min sampling"]
    IN_L["Label DataFrame<br/>id, is_erupted<br/>window start/end times"]

    ITER["Iterate over label windows"]

    subgraph WINDOW_PROC["Per-Window Processing"]
        SLICE["Slice tremor DataFrame<br/>to window time range"]
        VALIDATE["Validate sample count<br/>against expected window_size"]
        VALID{{"Sample count valid?"}}
        SKIP["Skip window<br/>log warning"]
        PREPEND["Prepend window id column"]
    end

    CONCAT["Concatenate all valid window slices"]
    OUT["Tremor matrix DataFrame<br/>id, datetime<br/>rsam_f0, rsam_f1<br/>dsar_f0-f1<br/>entropy"]

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
    IN_M["Tremor matrix<br/>id, datetime, tremor columns"]
    IN_L{{"Label DataFrame provided?"}}

    subgraph TRAIN_MODE["Training Mode - labels provided"]
        FILTER_W["Filter tremor matrix windows<br/>to match label IDs"]
        ALIGN_L["Save aligned label CSV<br/>to features output dir"]
        EXTRACT_T["tsfresh extraction<br/>per tremor column<br/>with relevance filter<br/>FDR-based feature filtering"]
    end

    subgraph PRED_MODE["Prediction Mode - no labels"]
        EXTRACT_P["tsfresh extraction<br/>per tremor column<br/>all features, no relevance filter"]
    end

    subgraph SELECTOR["FeatureSelector - 2-stage selection"]
        STAGE1["Stage 1 - tsfresh FDR<br/>Statistical relevance filtering<br/>Benjamini-Hochberg correction"]
        STAGE2{{"Method?"}}
        TSFRESH_ONLY["tsfresh only<br/>method: tsfresh"]
        RF_SEL["Stage 2 - RandomForest importance<br/>Top N significant features<br/>method: random_forest"]
        BOTH["Combined:<br/>tsfresh FDR then RandomForest<br/>method: combined"]
    end

    OUT_F["Features CSV<br/>~5000 raw or filtered features<br/>one row per window"]
    OUT_S["Selected features list<br/>significant_features.txt"]

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
    IN_F["Features CSV<br/>selected features list"]
    IN_LABEL["Label CSV<br/>aligned to features"]
    CFG["Configuration<br/>classifier, cv_strategy<br/>random_state, total_seed<br/>n_jobs, grid_search_n_jobs<br/>number_of_significant_features"]

    FIT{{"fit(with_evaluation=?)"}}

    subgraph EVAL_MODE["train_and_evaluate() - with_evaluation=True"]
        SPLIT["80/20 train/test split<br/>sklearn train_test_split"]
        RESAMPLE_E["RandomUnderSampler<br/>on training split only"]
        FEAT_SEL_E["FeatureSelector<br/>on training data only"]
        GS_E["GridSearchCV<br/>cv_strategy: shuffle / stratified /<br/>shuffle-stratified / timeseries"]
        EVAL["ModelEvaluator<br/>metrics on held-out test set<br/>accuracy, F1, ROC-AUC<br/>confusion matrix, plots"]
        SAVE_E["Save model .pkl<br/>metrics JSON<br/>registry CSV entry"]
    end

    subgraph TRAIN_MODE["train() - with_evaluation=False"]
        RESAMPLE_T["RandomUnderSampler<br/>on full dataset"]
        FEAT_SEL_T["FeatureSelector<br/>on full resampled data"]
        GS_T["GridSearchCV<br/>across CV folds"]
        SAVE_T["Save model .pkl<br/>no metrics<br/>registry CSV entry"]
    end

    subgraph PARALLEL["joblib.Parallel - loky backend"]
        SEED_LOOP["One worker per random seed<br/>n_jobs outer workers<br/>grid_search_n_jobs inner workers<br/>constraint: n_jobs x grid_search_n_jobs <= cpu_count"]
    end

    CLASSIFIERS["10 supported classifiers<br/>rf, gb, xgb, svm, lr<br/>nn, dt, knn, nb, voting"]

    OUT["Per-seed artifacts<br/>trained_model_{seed}.pkl<br/>metrics_{seed}.json<br/>registry.csv"]
    MERGE["merge_models() / merge_classifier_models()<br/>SeedEnsemble or ClassifierEnsemble"]
    OUT_MERGED["Merged model .pkl<br/>SeedEnsemble per classifier<br/>ClassifierEnsemble across classifiers"]

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
    IN_PKL["Per-seed model .pkl<br/>metrics JSON files<br/>registry CSV"]

    subgraph SINGLE["ModelEvaluator - per-seed evaluation"]
        ME_INIT["Load model and test data<br/>__init__() or from_files()"]
        ME_METRICS["Compute metrics<br/>accuracy, balanced_accuracy<br/>F1, precision, recall<br/>ROC-AUC, threshold analysis"]
        ME_PLOTS["Generate plots<br/>ROC curve, calibration<br/>confusion matrix<br/>learning curve<br/>prediction distribution"]
        ME_OUT["Per-seed metrics dict<br/>plot files"]
    end

    subgraph MULTI["MultiModelEvaluator - aggregate across seeds"]
        MME_LOAD["Load all per-seed JSON metrics<br/>or read registry CSV"]
        MME_AGG["Aggregate statistics<br/>mean, std across seeds"]
        MME_PLOTS["Aggregate plots<br/>SHAP summary<br/>ROC envelope<br/>aggregate calibration<br/>confusion matrix heatmap<br/>learning curve with variance<br/>seed stability"]
        MME_OUT["Ensemble-level metrics<br/>aggregate plot files"]
    end

    subgraph COMPARE["ClassifierComparator - cross-classifier comparison"]
        CC_MAP["Accept mapping:<br/>classifier_name -> registry CSV path"]
        CC_BUILD["Build one MultiModelEvaluator<br/>per classifier"]
        CC_TABLE["Side-by-side comparison table<br/>mean metrics per classifier"]
        CC_PLOTS["Comparison plots<br/>plot_classifier_comparison()"]
        CC_OUT["Comparison table CSV<br/>comparison plot files"]
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
    IN_MODELS["Trained model(s)<br/>SeedEnsemble .pkl per classifier<br/>or ClassifierEnsemble .pkl"]
    IN_TREMOR["New tremor CSV<br/>Datetime index, tremor columns"]

    MODE{{"Operating mode?"}}

    subgraph EVAL_MODE["Evaluation Mode - future_labels_csv provided"]
        PREP_E["Load tremor data<br/>TremorData"]
        MATRIX_E["TremorMatrixBuilder<br/>slice into labeled windows"]
        FEAT_E["FeaturesBuilder<br/>extract tsfresh features<br/>prediction mode"]
        ALIGN_LABELS["Align features to label windows"]
        PREDICT["predict() or predict_best()<br/>requires labels"]
        METRICS_E["Compute per-classifier metrics<br/>accuracy, F1, recall<br/>ROC-AUC, balanced accuracy"]
        OUT_E["Evaluation results<br/>per-seed and per-classifier metrics<br/>plots via ModelEvaluator"]
    end

    subgraph FORECAST_MODE["Forecast Mode - no labels"]
        PREP_F["Load tremor data<br/>TremorData"]
        MATRIX_F["TremorMatrixBuilder<br/>slice into unlabeled windows"]
        FEAT_F["FeaturesBuilder<br/>extract tsfresh features<br/>prediction mode"]
        PROBA["predict_proba()<br/>no labels required"]
        AGG_SEED["SeedEnsemble<br/>aggregate across seeds per classifier<br/>mean probability + uncertainty"]
        AGG_CLF{{"Multi-classifier?"}}
        CLF_ENS["ClassifierEnsemble<br/>aggregate across classifiers<br/>consensus probability + uncertainty"]
        SINGLE_OUT["Single-classifier output<br/>probability, uncertainty<br/>confidence, prediction columns"]
        MULTI_OUT["Multi-classifier output<br/>per-classifier dashed columns<br/>consensus solid column<br/>shaded uncertainty band"]
        OUT_F["Forecast CSV<br/>Datetime index<br/>{name}_eruption_probability<br/>{name}_uncertainty<br/>{name}_confidence<br/>{name}_prediction<br/>consensus_* columns"]
        PLOT_F["Forecast plot<br/>plot_forecast() or<br/>plot_forecast_with_events()"]
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
    BASE["BaseEnsemble<br/>base_ensemble.py<br/>save(path) / load(path) mixin"]

    subgraph SEED_ENS["SeedEnsemble - one classifier, all seeds<br/>seed_ensemble.py"]
        SE_LOAD["from_registry(registry_csv)<br/>load all seed .pkl files"]
        SE_PRED["predict_proba(X)<br/>aggregate across seeds<br/>return mean probability + std"]
        SE_UNC["predict_with_uncertainty(X)<br/>return mean, std, confidence, prediction"]
        SE_SAVE["save(path) / load(path)<br/>BaseEnsemble mixin<br/>joblib serialisation"]
    end

    subgraph CLF_ENS["ClassifierEnsemble - multiple classifiers<br/>classifier_ensemble.py"]
        CE_LOAD["from_seed_ensembles(dict)<br/>or from_registry_dict(dict)"]
        CE_PRED["predict_proba(X)<br/>aggregate across classifiers<br/>per-classifier dict output"]
        CE_UNC["predict_with_uncertainty(X)<br/>mean, std, conf, pred<br/>per_clf_dict"]
        CE_PROP["classifiers property<br/>__getitem__, __len__"]
        CE_SAVE["save(path) / load(path)<br/>BaseEnsemble mixin"]
    end

    PKL_SEEDS["Per-seed .pkl files<br/>GridSearchCV fitted estimators"]
    REG_CSV["registry.csv<br/>maps seed -> model path"]
    MERGED_PKL["SeedEnsemble .pkl<br/>or ClassifierEnsemble .pkl"]

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
    ROOT["root_dir/output/<br/>{network}.{station}.{location}.{channel}/"]

    subgraph TREMOR_DIR["tremor/"]
        T_DAILY["daily/<br/>{date}_tremor.csv<br/>temporary, cleaned up optionally"]
        T_FIGS["figures/<br/>{date}_tremor.png<br/>created if plot_daily=True"]
        T_MERGED["{nslc}_{start}_{end}.csv<br/>final merged tremor CSV"]
    end

    subgraph LABEL_DIR["label/"]
        L_CSV["label_YYYY-MM-DD_YYYY-MM-DD<br/>_ws-X_step-X-unit_dtf-X.csv"]
    end

    subgraph FEAT_DIR["features/"]
        F_CSV["features_{start}_{end}.csv<br/>~5000 tsfresh features"]
        F_LABEL["aligned_labels.csv<br/>matched to feature rows"]
        F_SIG["significant_features.txt<br/>selected feature names"]
    end

    subgraph TRAIN_DIR["trainings/{classifier-slug}/{cv-slug}/"]
        TR_PKL["trained_model_{seed}.pkl<br/>fitted GridSearchCV estimator"]
        TR_JSON["metrics_{seed}.json<br/>per-seed evaluation metrics"]
        TR_REG["registry.csv<br/>maps seed to model path + metadata"]
        TR_PLOTS["plots/<br/>ROC, calibration, confusion matrix<br/>learning curve, SHAP"]
    end

    subgraph EVAL_DIR["evaluations/{classifier-slug}/{cv-slug}/"]
        EV_PLOTS["aggregate plots<br/>ensemble ROC, SHAP summary<br/>stability, comparison"]
        EV_MERGED["merged_registry.csv<br/>aggregated seed statistics"]
    end

    subgraph FORECAST_DIR["forecast/"]
        FC_CSV["forecast_{start}_{end}.csv<br/>probability, uncertainty<br/>consensus columns"]
        FC_PLOT["forecast_{start}_{end}.png<br/>time-series probability plot"]
    end

    CONFIG["config.yaml<br/>full pipeline configuration<br/>replayable via from_config()"]

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
    INIT["ForecastModel(<br/>  root_dir, station, channel<br/>  start_date, end_date<br/>  window_size, volcano_id, n_jobs<br/>)"]

    DATA_CHOICE{{"Data acquisition choice"}}

    CALC["calculate(<br/>  source='sds' or 'fdsn'<br/>  sds_dir or fdsn_url<br/>)"]
    LOAD_T["load_tremor_data(<br/>  tremor_csv<br/>)<br/>alternative to calculate()"]

    BUILD_LABEL["build_label(<br/>  start_date, end_date<br/>  eruption_dates<br/>  day_to_forecast<br/>  window_step, window_step_unit<br/>)"]

    EXTRACT["extract_features(<br/>  select_tremor_columns<br/>)"]

    SET_FS["set_feature_selection_method(<br/>  using='tsfresh' or<br/>  'random_forest' or 'combined'<br/>)<br/>optional before train()"]

    TRAIN["train(<br/>  classifier, cv_strategy<br/>  random_state, total_seed<br/>  with_evaluation=True or False<br/>  number_of_significant_features<br/>)"]

    FORECAST["forecast(<br/>  start_date, end_date<br/>  window_size, window_step<br/>  window_step_unit<br/>)"]

    SAVE_CFG["save_config(path, fmt)<br/>persist pipeline configuration"]
    LOAD_CFG["ForecastModel.from_config(path)<br/>restore and replay configuration"]
    SAVE_MODEL["save_model(path)<br/>serialise full ForecastModel<br/>via joblib"]
    LOAD_MODEL["ForecastModel.load_model(path)<br/>restore full model object"]
    RUN["run()<br/>replay all stages<br/>from loaded config"]

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
