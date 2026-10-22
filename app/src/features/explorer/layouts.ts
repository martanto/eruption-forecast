import { Position, type Node, type Edge } from "@xyflow/react";
import { NODES, STAGES, EDGES } from "@/features/architecture";

export function pipelineLayout(
  expanded: string[],
  selected: string,
  gapsOnly: boolean,
): { nodes: Node[]; edges: Edge[] } {
  const positions = [
    { x: 0, y: 0 },
    { x: 395, y: 0 },
    { x: 790, y: 0 },
    { x: 790, y: 255 },
    { x: 395, y: 255 },
    { x: 0, y: 255 },
    { x: 0, y: 510 },
  ];
  const visible = NODES.filter(
    (item) =>
      item.overview || expanded.includes(item.stage) || item.id === selected,
  );
  const connected = new Set(
    EDGES.filter(
      (edge) => edge.source === selected || edge.target === selected,
    ).flatMap((edge) => [edge.source, edge.target]),
  );
  const nodes: Node[] = visible.map((item) => {
    const stageIndex = STAGES.findIndex((stage) => stage.id === item.stage);
    const childIndex = NODES.filter(
      (child) => child.stage === item.stage && !child.overview,
    ).findIndex((child) => child.id === item.id);
    return {
      id: item.id,
      type: "atlas",
      position: item.overview
        ? positions[stageIndex]!
        : { x: 1200 + childIndex * 395, y: stageIndex * 200 },
      sourcePosition: item.overview
        ? [
            Position.Right,
            Position.Right,
            Position.Bottom,
            Position.Left,
            Position.Left,
            Position.Bottom,
            Position.Right,
          ][stageIndex]
        : Position.Right,
      targetPosition: item.overview
        ? [
            Position.Left,
            Position.Left,
            Position.Left,
            Position.Top,
            Position.Right,
            Position.Right,
            Position.Top,
          ][stageIndex]
        : Position.Left,
      selected: item.id === selected,
      data: {
        item,
        expanded: expanded.includes(item.stage),
        related: connected.has(item.id),
        muted: gapsOnly && item.gaps.length === 0,
      },
      draggable: false,
    };
  });
  return {
    nodes,
    edges: EDGES.filter(
      (edge) =>
        visible.some((item) => item.id === edge.source) &&
        visible.some((item) => item.id === edge.target),
    ).map((edge) => ({
      ...edge,
      type: "smoothstep",
      className:
        edge.source === selected || edge.target === selected
          ? "edge-active"
          : "",
      style: {
        strokeWidth:
          edge.source === selected || edge.target === selected ? 2 : 1,
      },
      markerEnd: { type: "arrowclosed" as const },
      labelBgPadding: [7, 4] as [number, number],
      labelBgBorderRadius: 4,
    })),
  };
}

export function timelineLayout(
  selected: string,
  scenario: string,
): { nodes: Node[]; edges: Edge[] } {
  const selectedItem = NODES.find((item) => item.id === selected)!;
  return {
    nodes: [
      {
        id: "periods",
        type: "periods",
        position: { x: 0, y: 0 },
        data: { scenario },
        selectable: false,
        focusable: false,
        draggable: false,
      },
      ...STAGES.map((stage, i) => ({
        id: stage.id,
        type: "atlas",
        position: { x: i * 152, y: 775 },
        data: {
          item: NODES.find((item) => item.id === stage.id)!,
          compact: true,
        },
        selected: selected === stage.id,
        draggable: false,
      })),
      ...(!selectedItem.overview
        ? [
            {
              id: selected,
              type: "atlas",
              position: {
                x:
                  STAGES.findIndex((stage) => stage.id === selectedItem.stage) *
                  152,
                y: 970,
              },
              data: { item: selectedItem, compact: true },
              selected: true,
              draggable: false,
            },
          ]
        : []),
    ],
    edges: EDGES.filter(
      (edge) =>
        STAGES.some((stage) => stage.id === edge.source) &&
        STAGES.some((stage) => stage.id === edge.target),
    ).map((edge) => ({
      ...edge,
      label: undefined,
      type: "smoothstep",
      markerEnd: { type: "arrowclosed" as const },
    })),
  };
}
