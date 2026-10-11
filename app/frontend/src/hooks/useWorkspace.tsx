import { createContext, useContext } from "react";
import type { JsonObject } from "../lib/configuration";

export interface Schema {
  $ref?: string;
  title?: string;
  type?: string;
  description?: string;
  properties?: Record<string, Schema>;
  anyOf?: Schema[];
  enum?: unknown[];
  default?: unknown;
  items?: Schema;
  prefixItems?: Schema[];
}
export interface Catalog {
  schema: Schema & { $defs: Record<string, Schema> };
  defaults: JsonObject;
  stages: JsonObject;
  docs: Record<string, string>;
}
export interface Workspace {
  configuration: JsonObject;
  setConfiguration: (value: JsonObject) => void;
  catalog: Catalog;
  artifacts: import("../api/client").Artifact[];
  refreshArtifacts: () => Promise<void>;
  notice: (message: string) => void;
}
export const WorkspaceContext = createContext<Workspace | null>(null);
export function useWorkspace() {
  const value = useContext(WorkspaceContext);
  if (!value) throw new Error("Workspace is not initialized");
  return value;
}
