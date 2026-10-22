import { Handle, Position, useViewport, type NodeProps } from "@xyflow/react";
import { ChevronDown, ChevronUp, AlertCircle } from "lucide-react";
import { STAGES, type ArchitectureNode } from "@/features/architecture";
import { useInteractionStore } from "@/stores/interactionStore";

export function AtlasNode({
  data,
  selected,
  sourcePosition,
  targetPosition,
}: NodeProps) {
  const item = data.item as ArchitectureNode;
  const stage = STAGES.find((entry) => entry.id === item.stage)!;
  const toggleExpanded = useInteractionStore((state) => state.toggleExpanded);
  const { zoom } = useViewport();
  return (
    <article
      className={`atlas-node tone-${stage.tone} ${selected ? "is-selected" : ""} ${data.related ? "is-related" : ""} ${data.muted ? "is-muted" : ""} ${data.compact ? "is-compact" : ""}`}
    >
      <Handle
        type="target"
        position={targetPosition ?? Position.Left}
        isConnectable={false}
      />
      <div className="node-heading">
        <span className="stage-number">
          {String(stage.number).padStart(2, "0")}
        </span>
        <h3>{item.title}</h3>
        {item.gaps.length > 0 && (
          <AlertCircle size={16} aria-label="Has implementation caveats" />
        )}
      </div>
      {!data.compact && zoom > 0.45 && <p>{item.summary}</p>}
      {item.overview && !data.compact && (
        <button
          className="node-expand nodrag nopan"
          aria-expanded={Boolean(data.expanded)}
          onClick={(event) => {
            event.stopPropagation();
            toggleExpanded(item.stage);
          }}
        >
          {data.expanded ? <ChevronUp size={14} /> : <ChevronDown size={14} />}
          {data.expanded ? "Collapse steps" : "Explore steps"}
        </button>
      )}
      <Handle
        type="source"
        position={sourcePosition ?? Position.Right}
        isConnectable={false}
      />
    </article>
  );
}
