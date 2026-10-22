export type ExplorerView = "pipeline" | "timeline";
export interface ExplorerLocation {
  view: ExplorerView;
  scenario: string;
  node: string;
}
