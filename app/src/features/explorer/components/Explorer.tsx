import { useState, type ReactNode } from "react";
import {
  Search,
  List,
  Info,
  GitBranch,
  CalendarRange,
  AlertCircle,
  X,
  Plus,
  Minus,
} from "lucide-react";
import { STAGES, NODES, GAPS, SOURCE_SNAPSHOT } from "@/features/architecture";
import { Button } from "@/components/ui/button";
import { Sheet } from "@/components/ui/sheet";
import { useInteractionStore } from "@/stores/interactionStore";
import { NodeInspector } from "./NodeInspector";
import atlasLogo from "../../../../../assets/logo.svg";
import { ExplorerCanvas } from "./ExplorerCanvas";
import type { ExplorerLocation } from "../explorer.types";

interface ExplorerProps {
  location: ExplorerLocation;
  navigate: (patch: Partial<ExplorerLocation>) => void;
  scenarioSelector: ReactNode;
  scenarioComparison: ReactNode;
  tour: (onSelect: (id: string) => void) => ReactNode;
}
export function Explorer({
  location,
  navigate,
  scenarioSelector,
  scenarioComparison,
  tour,
}: ExplorerProps) {
  const [query, setQuery] = useState("");
  const [gapsOnly, setGapsOnly] = useState(false);
  const [drawer, setDrawer] = useState<"stages" | "details" | null>(null);
  const [focusRequest, setFocusRequest] = useState(0);
  const expand = useInteractionStore((state) => state.expand);
  const expanded = useInteractionStore((state) => state.expanded);
  const toggleExpanded = useInteractionStore((state) => state.toggleExpanded);
  const selectedStage = NODES.find((item) => item.id === location.node)!.stage;
  const queryActive = query.trim().length > 0;
  const results = NODES.filter(
    (item) =>
      (!gapsOnly || item.gaps.length > 0) &&
      `${item.title} ${item.summary} ${item.behavior.join(" ")} ${item.parameters.join(" ")} ${item.artifacts.join(" ")} ${item.sources.map((id) => SOURCE_SNAPSHOT.sources.find((source) => source.id === id)?.path).join(" ")}`
        .toLowerCase()
        .includes(query.trim().toLowerCase()),
  );
  function select(id: string, focus = false) {
    const item = NODES.find((entry) => entry.id === id)!;
    if (!item.overview) expand(item.stage);
    navigate({ node: id });
    if (focus) setFocusRequest((value) => value + 1);
  }
  const navigation = (
    <nav className="stage-navigation" aria-label="Pipeline stages">
      <h2>Explore the pipeline</h2>
      {STAGES.map((stage) => (
        <div className={`stage-nav-group tone-${stage.tone}`} key={stage.id}>
          <button
            className="stage-link"
            aria-current={selectedStage === stage.id ? "step" : undefined}
            onClick={() => {
              select(stage.id, true);
              setDrawer(null);
            }}
          >
            <span className="stage-number">
              {String(stage.number).padStart(2, "0")}
            </span>
            <span>{stage.title}</span>
          </button>
          <button
            className="stage-expand"
            aria-label={`${expanded.includes(stage.id) ? "Collapse" : "Expand"} ${stage.title}`}
            aria-expanded={expanded.includes(stage.id)}
            onClick={() => toggleExpanded(stage.id)}
          >
            {expanded.includes(stage.id) ? (
              <Minus size={15} />
            ) : (
              <Plus size={15} />
            )}
          </button>
          {expanded.includes(stage.id) && (
            <div className="stage-children">
              {NODES.filter(
                (item) => item.stage === stage.id && !item.overview,
              ).map((item) => (
                <button
                  key={item.id}
                  aria-current={location.node === item.id ? "true" : undefined}
                  onClick={() => {
                    select(item.id, true);
                    setDrawer(null);
                  }}
                >
                  {item.title}
                </button>
              ))}
            </div>
          )}
        </div>
      ))}
      <div className="reading-key">
        <h3>Reading the atlas</h3>
        <p>
          <AlertCircle size={14} />
          Implementation caveat
        </p>
        <p>
          Connections show data flow. Expand a stage to reveal its internal
          steps.
        </p>
        <p>
          Tab to a node; Enter to inspect. Arrow keys pan the focused canvas.
          Drag or pinch to explore.
        </p>
      </div>
      {scenarioComparison}
    </nav>
  );
  const inspector = (
    <NodeInspector
      selected={location.node}
      onSelect={(id) => select(id, true)}
    />
  );
  return (
    <>
      <header className="explorer-heading">
        <div>
          <h1>
            <img className="atlas-logo" src={atlasLogo} alt="" />
            Pipeline Atlas<span>Eruption Forecast</span>
          </h1>
          <p>A field guide to the code, from seismic signal to explanation.</p>
        </div>
        {scenarioSelector}
      </header>
      <div className="explorer-toolbar">
        <div className="view-switch" role="group" aria-label="Canvas view">
          <Button
            variant="ghost"
            aria-pressed={location.view === "pipeline"}
            onClick={() => navigate({ view: "pipeline" })}
          >
            <GitBranch size={16} />
            Pipeline
          </Button>
          <Button
            variant="ghost"
            aria-pressed={location.view === "timeline"}
            onClick={() => navigate({ view: "timeline" })}
          >
            <CalendarRange size={16} />
            Timeline
          </Button>
        </div>
        <div className="search-box">
          <Search size={17} />
          <input
            aria-label="Search stages, parameters and sources"
            placeholder="Find a step, parameter or source…"
            value={query}
            onChange={(event) => setQuery(event.target.value)}
          />
          {query && (
            <button aria-label="Clear search" onClick={() => setQuery("")}>
              <X size={15} />
            </button>
          )}
        </div>
        <Button
          variant="ghost"
          aria-pressed={gapsOnly}
          onClick={() => setGapsOnly((value) => !value)}
        >
          <AlertCircle size={16} />
          <span>Caveats</span>
          <span className="count">{GAPS.length}</span>
        </Button>
      </div>
      {(queryActive || gapsOnly) && (
        <section className="search-results" aria-label="Search results">
          <strong>
            {results.length} matching steps{gapsOnly ? " with caveats" : ""}
          </strong>
          <div>
            {results.map((item) => (
              <button
                key={item.id}
                onClick={() => {
                  select(item.id, true);
                  setQuery("");
                  setGapsOnly(false);
                  if (window.matchMedia("(max-width: 1000px)").matches)
                    setDrawer("details");
                }}
              >
                {item.title}
              </button>
            ))}
            {!results.length && (
              <p>No matching steps. Try “SHAP”, “FDR” or “window”.</p>
            )}
          </div>
        </section>
      )}
      <div className="mobile-tools">
        <Button
          id="stages-trigger"
          variant="outline"
          size="small"
          onClick={() => setDrawer("stages")}
        >
          <List size={16} />
          Stages & scenarios
        </Button>
        <Button
          id="inspector-trigger"
          variant="outline"
          size="small"
          onClick={() => setDrawer("details")}
        >
          <Info size={16} />
          Inspect selection
        </Button>
      </div>
      <div className="explorer-workspace">
        <aside className="desktop-navigation">{navigation}</aside>
        <main className="canvas-column">
          <ExplorerCanvas
            view={location.view}
            selected={location.node}
            scenario={location.scenario}
            gapsOnly={gapsOnly}
            focusRequest={focusRequest}
            onSelect={(id) => select(id)}
          />
          <div className="tour-slot">{tour((id) => select(id, true))}</div>
        </main>
        <aside className="desktop-inspector">{inspector}</aside>
      </div>
      <footer className="explorer-footer">
        <span>
          Read-only guide · source snapshot{" "}
          {SOURCE_SNAPSHOT.revision.slice(0, 8)} + local edits
        </span>
        <span>Example settings ≠ package defaults · UTC</span>
      </footer>
      <Sheet
        open={drawer !== null}
        onOpenChange={(open) => {
          if (!open) setDrawer(null);
        }}
        returnFocusId={
          drawer === "stages" ? "stages-trigger" : "inspector-trigger"
        }
        title={drawer === "stages" ? "Stages & scenarios" : "Inspect selection"}
      >
        {drawer === "stages" ? navigation : inspector}
      </Sheet>
    </>
  );
}
