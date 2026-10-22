import csv from '../../../../assets/forecast-results_2025-08-01_2025-08-22_ws-2.csv?raw';

export const ERUPTIONS = ['2025-08-02', '2025-08-18'];
export const THRESHOLD = 0.7;
export const FORECAST_MODELS = [
  ['consensus', 'Consensus'],
  ['LiteRandomForestClassifier', 'Lite random forest'],
  ['RandomForestClassifier', 'Random forest'],
  ['GradientBoostingClassifier', 'Gradient boosting'],
  ['XGBClassifier', 'XGBoost'],
] as const;
const [header, ...lines] = csv.trim().split(/\r?\n/);
const columns = header!.split(',');
export const FORECAST_ROWS = lines.map((line) => {
  const cells = line.split(',');
  return {
    time: Date.parse(cells[0]!.replace(' ', 'T') + 'Z'),
    values: Object.fromEntries(columns.slice(1).map((key, i) => [key, Number(cells[i + 1])])),
  };
});

export function forecastValues(model: string, count: number) {
  const key = model === 'consensus' ? 'consensus_probability' : `${model}_probability`;
  return FORECAST_ROWS.slice(0, count).map((row) => {
    const mean = row.values[key]!;
    const std = row.values[`${model}_uncertainty`]!;
    const lower = Math.max(0, mean - std);
    return { time: row.time, mean, lower, width: Math.min(1, mean + std) - lower };
  });
}
