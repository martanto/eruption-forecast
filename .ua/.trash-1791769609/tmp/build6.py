import json
R={r['path']:r for r in json.load(open('tmp/ua-file-extract-results-6.json'))['results']}
imp=json.load(open('tmp/ua-file-analyzer-input-6.json'))['batchImportData']
C='src/eruption_forecast/config/'; P='src/eruption_forecast/utils/pathutils.py'
nodes=[];edges=[]
def E(s,t,ty,w): edges.append(dict(source=s,target=t,type=ty,direction='forward',weight=w))
files={
C+'__init__.py':("Package barrel for the config subpackage, re-exporting constants, BaseConfig, LabelConfig, the ForecastConfig family and the standalone per-stage configs (training, prediction, evaluation, explanation).",["entry-point","barrel","configuration","exports"],"simple"),
C+'base_config.py':("Defines the BaseConfig dataclass mixin providing shared version/saved_at stamping plus dict round-tripping (to_dict, to_init_kwargs, from_dict) for every config dataclass.",["data-model","configuration","serialization","base-class"],"moderate"),
C+'evaluation_config.py':("EvaluationConfig dataclass snapshotting EvaluationModel constructor and evaluate() arguments, with YAML/JSON save and load for per-stage config files.",["data-model","configuration","serialization","evaluation"],"moderate"),
C+'explanation_config.py':("ExplanationConfig dataclass mirroring ExplanationModel settings (SHAP explain/plot options) with YAML/JSON persistence for explanation config files.",["data-model","configuration","serialization","explainability"],"moderate"),
C+'forecast_config.py':("Hierarchical ForecastConfig for the ForecastModel orchestrator: base station settings plus per-stage calculate/train/predict/evaluate/explain sub-configs, serialised to YAML/JSON so pipelines can be replayed via from_config() and run().",["data-model","configuration","serialization","pipeline","orchestration"],"complex"),
C+'label_config.py':("LabelConfig dataclass capturing LabelBuilder init arguments (window size/step, day_to_forecast, eruption dates) and saving them as label.config.yaml.",["data-model","configuration","serialization","labeling"],"moderate"),
C+'prediction_config.py':("PredictionConfig dataclass mirroring PredictionModel constructor parameters with YAML/JSON save/load used by forecast() to write prediction config files.",["data-model","configuration","serialization","prediction"],"moderate"),
C+'training_config.py':("TrainingConfig dataclass mirroring TrainingModel.__init__ fields and defaults (classifiers, CV, seeds, resampling, features) with YAML/JSON persistence written at the end of fit().",["data-model","configuration","serialization","training"],"moderate"),
P:("Filesystem and I/O helpers used throughout the package: output-directory resolution, NSLC directory setup, figure/PDF saving with metadata, generic data saving (CSV/Parquet), and JSON/pickle loading.",["utility","file-io","persistence","plotting"],"complex"),
}
for p,(s,t,c) in files.items():
    nodes.append(dict(id='file:'+p,type='file',name=p.split('/')[-1],filePath=p,summary=s,tags=t,complexity=c))
    for tgt in imp.get(p,[]): E('file:'+p,'file:'+tgt,'imports',0.7)
syms={
(C+'base_config.py','function','_package_version'):("Resolves the installed eruption-forecast version via importlib.metadata, falling back to 0.0.0 when not installed.",["utility","versioning","metadata"],"simple"),
(C+'base_config.py','class','BaseConfig'):("Base dataclass for all configs carrying version and saved_at stamps and providing to_dict, to_init_kwargs and from_dict helpers.",["base-class","data-model","serialization"],"moderate"),
(C+'evaluation_config.py','class','EvaluationConfig'):("Snapshot of EvaluationModel settings with to_dict/save/load supporting YAML and JSON.",["data-model","configuration","evaluation"],"moderate"),
(C+'explanation_config.py','class','ExplanationConfig'):("Snapshot of ExplanationModel SHAP settings with to_dict/save/load supporting YAML and JSON.",["data-model","configuration","explainability"],"moderate"),
(C+'label_config.py','class','LabelConfig'):("Snapshot of LabelBuilder initialisation arguments, persisted to label.config.yaml via save/load.",["data-model","configuration","labeling"],"moderate"),
(C+'prediction_config.py','class','PredictionConfig'):("Snapshot of PredictionModel constructor parameters with YAML/JSON save and load.",["data-model","configuration","prediction"],"moderate"),
(C+'training_config.py','class','TrainingConfig'):("Snapshot of TrainingModel constructor fields and defaults with YAML/JSON save and load; kept in sync with TrainingModel.__init__.",["data-model","configuration","training"],"moderate"),
(C+'forecast_config.py','class','BaseForecastConfig'):("Station-level ForecastModel constructor settings (station, channel, network, day_to_forecast, output dirs, n_jobs).",["data-model","configuration","station"],"simple"),
(C+'forecast_config.py','class','ForecastCalculateConfig'):("Captured arguments of ForecastModel.calculate(): date range, data source (SDS/FDSN), tremor methods and cleaning options.",["data-model","configuration","tremor"],"moderate"),
(C+'forecast_config.py','class','ForecastTrainConfig'):("Captured arguments of ForecastModel.train(): eruption dates, window step, classifiers, CV strategy, seeds, feature and resampling options.",["data-model","configuration","training"],"moderate"),
(C+'forecast_config.py','class','ForecastPredictConfig'):("Captured arguments of ForecastModel.predict(): forecast window, step, feature source and plotting options.",["data-model","configuration","prediction"],"moderate"),
(C+'forecast_config.py','class','ForecastEvaluateConfig'):("Captured arguments of ForecastModel.evaluate(): reuse mode, eruption dates and plotting flags.",["data-model","configuration","evaluation"],"simple"),
(C+'forecast_config.py','class','ForecastExplainConfig'):("Captured arguments of ForecastModel.explain(): reuse mode, eruption dates, per-seed saving and SHAP plotting options.",["data-model","configuration","explainability"],"moderate"),
(C+'forecast_config.py','class','ForecastConfig'):("Top-level container aggregating the base and per-stage forecast sub-configs, with to_dict/save/load in YAML or JSON for pipeline replay.",["data-model","configuration","serialization","orchestration"],"moderate"),
(P,'function','pdf_metadata'):("Builds PDF metadata (title, author, creator, version) from installed package metadata for saved figures.",["utility","metadata","pdf"],"simple"),
(P,'function','ensure_dir'):("Creates a directory (including parents) if missing and returns its path.",["utility","file-io","filesystem"],"simple"),
(P,'function','load_json'):("Loads and returns parsed JSON content from a file path with existence checks.",["utility","file-io","json"],"simple"),
(P,'function','load_pickle'):("Loads a pickled object from disk with joblib/pickle and error logging.",["utility","file-io","deserialization"],"simple"),
(P,'function','resolve_output_dir'):("Resolves an output directory from explicit, root-relative or default locations.",["utility","filesystem","path-resolution"],"simple"),
(P,'function','save_figure_as_pdf'):("Saves a matplotlib figure as a PDF with embedded package metadata.",["utility","plotting","pdf"],"simple"),
(P,'function','save_figure'):("Saves a matplotlib figure to image files, ensuring directories, handling dpi and closing figures.",["utility","plotting","file-io"],"moderate"),
(P,'function','save_data'):("Persists a DataFrame to CSV, Parquet or other formats based on extension, creating directories as needed.",["utility","file-io","persistence"],"simple"),
(P,'function','setup_nslc_directories'):("Creates and returns the per-station NSLC output directory layout used by pipeline stages.",["utility","filesystem","station"],"simple"),
(P,'function','generate_features_filepaths'):("Generates standard file paths for the features matrix and aligned label CSVs within a features directory.",["utility","path-resolution","features"],"simple"),
}
for (p,k,n),(s,t,c) in syms.items():
    src=[x for x in R[p]['functions' if k=='function' else 'classes'] if x['name']==n][0]
    nid=f'{k}:{p}:{n}'
    nodes.append(dict(id=nid,type=k,name=n,filePath=p,lineRange=[src['startLine'],src['endLine']],summary=s,tags=t,complexity=c))
    E('file:'+p,nid,'contains',1.0); E('file:'+p,nid,'exports',0.8)
    if k=='class' and n!='BaseConfig': E(nid,'class:'+C+'base_config.py:BaseConfig','inherits',0.9)
    if k=='class' and n in('EvaluationConfig','ExplanationConfig','LabelConfig','PredictionConfig','TrainingConfig','ForecastConfig'):
        E(nid,'function:'+P+':ensure_dir','calls',0.8)
ids={n['id'] for n in nodes}; assert len(ids)==len(nodes)
for e in edges:
    for x in (e['source'],e['target']):
        assert x in ids or x.startswith('file:'),x
print(len(nodes),len(edges),sum(1 for e in edges if e['type']=='imports'),sum(len(v) for v in imp.values()))
json.dump(dict(nodes=nodes,edges=edges),open('intermediate/batch-6.json','w'),indent=1)
