import { useCallback, useEffect, useMemo, useRef, useState } from "react";
import {
  ReactFlow,
  ReactFlowProvider,
  Background,
  MiniMap,
  Controls,
  Panel,
  useReactFlow,
  type NodeChange,
} from "@xyflow/react";
import { Focus, RotateCcw, Maximize2 } from "lucide-react";
import { NODES } from "@/features/architecture";
import { Button } from "@/components/ui/button";
import { useInteractionStore } from "@/stores/interactionStore";
import type { ExplorerView } from "../explorer.types";
import { pipelineLayout, timelineLayout } from "../layouts";
import { AtlasNode } from "./AtlasNode";
import { TimelinePeriods } from "./TimelinePeriods";

const NODE_TYPES = { atlas: AtlasNode, periods: TimelinePeriods };
interface CanvasProps {
  view: ExplorerView;
  selected: string;
  scenario: string;
  gapsOnly: boolean;
  focusRequest: number;
  onSelect: (id: string) => void;
}
function Canvas({
  view,
  selected,
  scenario,
  gapsOnly,
  focusRequest,
  onSelect,
}: CanvasProps) {
  const { fitView, getViewport, setViewport, setCenter } = useReactFlow();
  const expanded = useInteractionStore((state) => state.expanded);
  const reset = useInteractionStore((state) => state.reset);
  const [initialViewport] = useState(
    () => useInteractionStore.getState().viewports[view],
  );
  const [measurements, setMeasurements] = useState<
    Record<string, { width: number; height: number }>
  >({});
  const graph = useMemo(
    () =>
      view === "pipeline"
        ? pipelineLayout(expanded, selected, gapsOnly)
        : timelineLayout(selected, scenario),
    [view, expanded, selected, scenario, gapsOnly],
  );
  const measuredNodes = useMemo(
    () =>
      graph.nodes.map((node) => ({ ...node, measured: measurements[node.id] })),
    [graph.nodes, measurements],
  );
  const onNodesChange = useCallback((changes: NodeChange[]) => {
    const dimensions = changes.filter((change) => change.type === "dimensions");
    if (!dimensions.length) return;
    setMeasurements((previous) => {
      const next = { ...previous };
      let changed = false;
      dimensions.forEach((change) => {
        if (
          change.dimensions &&
          (next[change.id]?.width !== change.dimensions.width ||
            next[change.id]?.height !== change.dimensions.height)
        ) {
          next[change.id] = change.dimensions;
          changed = true;
        }
      });
      return changed ? next : previous;
    });
  }, []);
  const focus = useCallback(() => {
    const item = NODES.find((node) => node.id === selected)!;
    const ids =
      view === "pipeline" && expanded.includes(item.stage)
        ? graph.nodes
            .filter(
              (node) =>
                node.id === item.stage ||
                NODES.find((entry) => entry.id === node.id)?.stage ===
                  item.stage,
            )
            .map((node) => ({ id: node.id }))
        : [{ id: selected }];
    void fitView({ nodes: ids, padding: 0.3, duration: 0, maxZoom: 1.15 });
  }, [selected, expanded, graph.nodes, view, fitView]);
  const lastFocus = useRef(focusRequest);
  const previousExpanded = useRef(expanded);
  useEffect(() => {
    const added = expanded.find((id) => !previousExpanded.current.includes(id));
    previousExpanded.current = expanded;
    if (added && view === "pipeline") {
      const frame = requestAnimationFrame(
        () =>
          void fitView({
            nodes: graph.nodes.filter(
              (node) =>
                NODES.find((item) => item.id === node.id)?.stage === added,
            ),
            padding: 0.2,
            maxZoom: 1,
          }),
      );
      return () => cancelAnimationFrame(frame);
    }
  }, [expanded, graph.nodes, view, fitView]);
  useEffect(() => {
    if (lastFocus.current !== focusRequest) {
      lastFocus.current = focusRequest;
      const frame = requestAnimationFrame(focus);
      return () => cancelAnimationFrame(frame);
    }
  }, [focusRequest, focus]);
  useEffect(
    () => () => {
      useInteractionStore.getState().saveViewport(view, getViewport());
    },
    [view, getViewport],
  );
  return (
    <div
      className="canvas"
      data-testid="explorer-canvas"
      onKeyDownCapture={(event) => {
        if (event.key === "Enter" || event.key === " ") {
          const target = event.target as HTMLElement;
          const id = target
            .closest(".react-flow__node")
            ?.getAttribute("data-id");
          if (id && id !== "periods" && !target.closest("button")) {
            event.preventDefault();
            onSelect(id);
          }
        }
      }}
    >
      {view === "pipeline" && (
        <div className="canvas-caption">
          <strong>From signal to explanation</strong>
          <span>
            Select a stage to read its transformations. Fit all for the
            complete pipeline
          </span>
        </div>
      )}
      <ReactFlow
        nodes={measuredNodes}
        onNodesChange={onNodesChange}
        edges={graph.edges}
        nodeTypes={NODE_TYPES}
        fitView={!initialViewport}
        defaultViewport={initialViewport}
        fitViewOptions={{
          padding: 0.13,
          maxZoom: 1,
          ...(view === "pipeline" ? { nodes: [{ id: selected }] } : {}),
        }}
        minZoom={0.2}
        maxZoom={1.8}
        nodesDraggable={false}
        nodesConnectable={false}
        edgesFocusable={false}
        deleteKeyCode={null}
        onNodeClick={(_, node) => {
          if (node.id !== "periods") onSelect(node.id);
        }}
        onMoveEnd={(_, viewport) =>
          useInteractionStore.getState().saveViewport(view, viewport)
        }
        onInit={() => {
          if (initialViewport) void setViewport(initialViewport);
        }}
        aria-label={`${view === "pipeline" ? "Pipeline" : "Timeline"} canvas`}
      >
        <Background gap={22} size={1} />
        <Controls showInteractive={false} position="bottom-left" />
        <Panel position="bottom-center">
          <div className="canvas-actions">
            <Button
              variant="outline"
              size="small"
              onClick={() => void fitView({ padding: 0.13, duration: 0 })}
            >
              <Maximize2 size={15} />
              Fit all
            </Button>
            <Button variant="outline" size="small" onClick={focus}>
              <Focus size={15} />
              Focus stage
            </Button>
            <Button
              variant="ghost"
              size="small"
              onClick={() => {
                reset();
                requestAnimationFrame(
                  () =>
                    void fitView({
                      nodes: graph.nodes.filter(
                        (node) =>
                          NODES.find((item) => item.id === node.id)?.overview ||
                          node.id === "periods",
                      ),
                      padding: 0.13,
                    }),
                );
              }}
            >
              <RotateCcw size={15} />
              Reset
            </Button>
          </div>
        </Panel>
        <MiniMap
          pannable
          zoomable
          onClick={(_, position) =>
            void setCenter(position.x, position.y, { zoom: getViewport().zoom })
          }
          position="bottom-right"
          nodeColor="var(--color-terracotta-soft)"
          maskColor="var(--minimap-mask)"
          ariaLabel="Canvas minimap"
        />
      </ReactFlow>
    </div>
  );
}
export function ExplorerCanvas(props: CanvasProps) {
  return (
    <ReactFlowProvider key={props.view}>
      <Canvas {...props} />
    </ReactFlowProvider>
  );
}
