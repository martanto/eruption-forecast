export type WorkflowKind = "backend" | "database" | "external";
export type WorkflowSide = "left" | "right" | "top" | "bottom";

export interface WorkflowNode {
  id: string;
  title: string;
  detail: string;
  kind: WorkflowKind;
  // Walkthrough step (0-3) this node belongs to.
  step: number;
  // 1-based grid cell on the desktop canvas.
  col: number;
  row: number;
}
export interface WorkflowEdge {
  from: string;
  to: string;
  fromSide: WorkflowSide;
  toSide: WorkflowSide;
  label: string;
  dashed?: boolean;
}

// Array order is the order of FeaturesBuilder.extract_features, so the trace
// reads top to bottom on small screens too.
export const FEATURES_MATRIX_NODES: readonly WorkflowNode[] = [
  { id: "tremor", title: "Tremor matrix", detail: "id, datetime, columns", kind: "external", step: 0, col: 1, row: 1 },
  { id: "validate", title: "Validate", detail: "required columns", kind: "backend", step: 0, col: 2, row: 1 },
  { id: "dispatch", title: "Mode dispatch", detail: "train or predict", kind: "backend", step: 0, col: 3, row: 1 },
  { id: "params", title: "tsfresh params", detail: "excludes, selection", kind: "backend", step: 1, col: 4, row: 1 },
  { id: "cache", title: "Cache check", detail: "merged vs per-column", kind: "backend", step: 1, col: 5, row: 1 },
  { id: "loop", title: "Per-column loop", detail: "one tremor column", kind: "backend", step: 1, col: 5, row: 2 },
  { id: "tsfresh", title: "tsfresh", detail: "extract features", kind: "external", step: 2, col: 4, row: 2 },
  { id: "files", title: "Per-column files", detail: "extracted/*.parquet", kind: "database", step: 1, col: 6, row: 2 },
  { id: "concat", title: "Concat features", detail: "axis=1 merge", kind: "backend", step: 3, col: 5, row: 3 },
  { id: "matrix", title: "Features matrix", detail: "features-matrix Parquet", kind: "database", step: 3, col: 4, row: 3 },
  { id: "labels", title: "Aligned labels", detail: "features-label CSV", kind: "database", step: 3, col: 3, row: 2 },
];

export const FEATURES_MATRIX_EDGES: readonly WorkflowEdge[] = [
  { from: "tremor", to: "validate", fromSide: "right", toSide: "left", label: "matrix" },
  { from: "validate", to: "dispatch", fromSide: "right", toSide: "left", label: "columns" },
  { from: "dispatch", to: "params", fromSide: "right", toSide: "left", label: "excludes" },
  { from: "params", to: "cache", fromSide: "right", toSide: "left", label: "kwargs" },
  { from: "cache", to: "loop", fromSide: "bottom", toSide: "top", label: "miss" },
  { from: "loop", to: "tsfresh", fromSide: "left", toSide: "right", label: "extract" },
  { from: "loop", to: "files", fromSide: "right", toSide: "left", label: "Parquet" },
  { from: "loop", to: "concat", fromSide: "bottom", toSide: "top", label: "frames" },
  { from: "files", to: "concat", fromSide: "bottom", toSide: "right", label: "cached paths", dashed: true },
  { from: "concat", to: "matrix", fromSide: "left", toSide: "right", label: "Parquet" },
  { from: "dispatch", to: "labels", fromSide: "bottom", toSide: "top", label: "training", dashed: true },
];
