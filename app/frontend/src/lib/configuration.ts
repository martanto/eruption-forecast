export type JsonObject = Record<string, unknown>;

export function merge(base: JsonObject, override: JsonObject): JsonObject {
  const result = { ...base };
  for (const [key, value] of Object.entries(override)) {
    result[key] =
      value &&
      typeof value === "object" &&
      !Array.isArray(value) &&
      result[key] &&
      typeof result[key] === "object" &&
      !Array.isArray(result[key])
        ? merge(result[key] as JsonObject, value as JsonObject)
        : value;
  }
  return result;
}

export function updateAt(
  object: JsonObject,
  path: string[],
  value: unknown,
  remove = false,
): JsonObject {
  const [key, ...rest] = path;
  const result = { ...object };
  if (rest.length)
    result[key] = updateAt(
      (result[key] as JsonObject) ?? {},
      rest,
      value,
      remove,
    );
  else if (remove) delete result[key];
  else result[key] = value;
  return result;
}

export function label(value: string): string {
  const names: Record<string, string> = {
    n_jobs: "Parallel workers",
    n_grids: "Grid-search workers",
    sds_dir: "SDS directory",
    output_dir: "Output directory",
    tremor_dir: "Tremor directory",
    cv_splits: "Cross-validation splits",
    freq_bands: "Frequency bands (Hz)",
    use_cache: "Reuse valid cache",
    constructor: "Model & input",
    label: "Window & labels",
    features: "Feature extraction",
    fit: "Model fitting",
    forecast: "Forecast output",
    plot: "Figure settings",
    execute: "Computation",
    run: "Tremor figures",
    load_features: "Existing feature files",
  };
  return (
    names[value] ??
    value
      .replaceAll("_", " ")
      .replace(/^./, (character) => character.toUpperCase())
  );
}

export function filename(path: string): string {
  return path.replaceAll("\\", "/").split("/").at(-1) ?? path;
}
export function bytes(value: number): string {
  const unit = value >= 2 ** 30 ? "GB" : "MB";
  return `${(value / (unit === "GB" ? 2 ** 30 : 2 ** 20)).toFixed(1)} ${unit}`;
}
