import { MathText } from "@/components/lab/MathText";
import type { ArchitectureNode } from "@/features/architecture";

/** The node's behaviour paragraphs, shared by the inspector and the expanded dialog. */
export function NodeBehavior({ item }: { item: ArchitectureNode }) {
  return (
    <>
      <h3>Implemented behaviour</h3>
      {item.behavior.map((text) => (
        <p key={text}>
          <MathText>{text}</MathText>
        </p>
      ))}
    </>
  );
}

export function NodeParameters({ item }: { item: ArchitectureNode }) {
  return (
    <>
      <h3>Parameters & settings</h3>
      <ul>
        {item.parameters.map((text) => (
          <li key={text}>{text}</li>
        ))}
      </ul>
    </>
  );
}

export function NodeArtifacts({
  item,
  className,
}: {
  item: ArchitectureNode;
  className?: string;
}) {
  return (
    <>
      <h3>Artifacts</h3>
      <ul className={className}>
        {item.artifacts.map((text) => (
          <li key={text}>{text}</li>
        ))}
      </ul>
    </>
  );
}
