const fs=require('fs');const d=require('./arch-input.json');const P='src/eruption_forecast/';
const L=[
{id:'layer:entry',name:'Entry Points',description:'End-user pipeline scripts (main.py, forecast-scenario.py) and the package __init__ that re-exports the public ForecastModel/TremorData/LabelData API.',t:p=>p==='main.py'||p==='forecast-scenario.py'||p===P+'__init__.py'},
{id:'layer:model',name:'Pipeline Orchestration & Stage Models',description:'ForecastModel and ForecastModelScenario orchestrators plus the BaseModel-derived Training, Prediction, Evaluation and Explanation stage models with content-addressed caching, classifier definitions and cross-classifier comparison.',t:p=>p.startsWith(P+'model/')},
{id:'layer:ensemble',name:'Ensemble Engine',description:'Multi-seed SeedEnsemble and ClassifierEnsemble estimators plus the MetricsEnsemble and ExplainerEnsemble engines that compute per-seed metrics and SHAP explanations.',t:p=>p.startsWith(P+'ensemble/')},
{id:'layer:seismic-ingestion',name:'Seismic Ingestion & Tremor',description:'SDS and FDSN seismic data sources and the CalculateTremor pipeline that computes RSAM, DSAR and Shannon entropy across frequency bands into tremor CSVs.',t:p=>p.startsWith(P+'sources/')||p.startsWith(P+'tremor/')},
{id:'layer:labels-features',name:'Labelling & Feature Engineering',description:'Sliding-window eruption label builders and the tsfresh-based tremor matrix, feature extraction, feature selection and feature-count sweep components.',t:p=>p.startsWith(P+'label/')||p.startsWith(P+'features/')},
{id:'layer:config',name:'Configuration & Data Types',description:'ForecastConfig and per-stage config dataclasses, shared constants, and immutable data containers such as StationData, BaseDataContainer and SHAP/ensemble summary payloads.',t:p=>p.startsWith(P+'config/')||p.startsWith(P+'dataclass/')||p===P+'data_container.py'},
{id:'layer:visualization',name:'Visualization',description:'Matplotlib/SHAP plotting modules for tremor, labels, features, forecasts, evaluation metrics and explanations with shared styling.',t:p=>p.startsWith(P+'plots/')},
{id:'layer:utility',name:'Utilities & Cross-Cutting Services',description:'Focused helper modules (arrays, windows, dates, ML, validation, paths, dataframes, features, formatting) plus Loguru logging, timer/notify decorators and Telegram notification.',t:p=>p.startsWith(P+'utils/')||p.startsWith(P+'decorators/')||p.startsWith(P+'notification/')||p===P+'logger.py'},
{id:'layer:documentation',name:'Documentation',description:'README and the 15-page GitHub wiki covering getting started, pipeline workflows, configuration, output structure, architecture and API reference.',t:p=>p==='README.md'||p.startsWith('wiki/')},
];
const out=L.map(l=>({id:l.id,name:l.name,description:l.description,nodeIds:[]}));const miss=[];
for(const n of d.fileNodes){const i=L.findIndex(l=>l.t(n.filePath));if(i<0)miss.push(n.id);else out[i].nodeIds.push(n.id);}
if(miss.length){console.error('UNASSIGNED',miss);process.exit(1)}
fs.mkdirSync('../intermediate',{recursive:true});fs.writeFileSync('../intermediate/layers.json',JSON.stringify(out,null,2));
let s=0;out.forEach(l=>{s+=l.nodeIds.length;console.log(l.name,l.nodeIds.length)});console.log('total',s,'unique',new Set(out.flatMap(l=>l.nodeIds)).size);
