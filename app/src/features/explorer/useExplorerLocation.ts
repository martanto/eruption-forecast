import { useSearchParams } from "react-router-dom";
import { NODES, SCENARIOS } from "@/features/architecture";
import type { ExplorerLocation } from "./explorer.types";

export function useExplorerLocation() {
  const [params, setParams] = useSearchParams();
  const location: ExplorerLocation = {
    view: params.get("view") === "timeline" ? "timeline" : "pipeline",
    scenario:
      SCENARIOS.find((item) => item.id === params.get("scenario"))?.id ??
      SCENARIOS[0]!.id,
    node: NODES.find((item) => item.id === params.get("node"))?.id ?? "tremor",
  };
  function navigate(patch: Partial<ExplorerLocation>) {
    setParams((previous) => {
      const next = new URLSearchParams(previous);
      Object.entries({ ...location, ...patch }).forEach(([key, value]) =>
        next.set(key, value),
      );
      return next;
    });
  }
  return { location, navigate };
}
