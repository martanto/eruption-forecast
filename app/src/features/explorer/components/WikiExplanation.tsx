import Markdown from "react-markdown";
import remarkGfm from "remark-gfm";
import { ExplanationDialog } from "@/components/common/ExplanationDialog";
import {
  WIKI_GUIDES,
  GAPS,
  type ArchitectureNode,
} from "@/features/architecture";
import { NodeArtifacts, NodeBehavior, NodeParameters } from "./NodeSections";

export function WikiExplanation({ item }: { item: ArchitectureNode }) {
  const guides = WIKI_GUIDES[item.stage] ?? [];
  const content = guides.map((guide) => (
    <section key={guide.title}>
      <h3>{guide.title}</h3>
      <p className="wiki-attribution">From wiki/{guide.path}</p>
      <Markdown
        remarkPlugins={[remarkGfm]}
        components={{ a: ({ children }) => <span>{children}</span> }}
      >
        {guide.markdown}
      </Markdown>
    </section>
  ));
  return (
    <div className="wiki-guide">
      <ExplanationDialog
        key={item.id}
        triggerLabel="Expand explanation"
        triggerClassName="expand-explanation"
        title={item.title}
        closeLabel="Close full explanation"
        bodyClassName="inspector wiki-guide"
      >
        <p className="inspector-summary">{item.summary}</p>
        <NodeBehavior item={item} />
        {content}
        <NodeParameters item={item} />
        <NodeArtifacts item={item} />
        {item.gaps.map((id) => {
          const gap = GAPS.find((entry) => entry.id === id)!;
          return (
            <section className="gap-note" key={id}>
              <h4>{gap.title}</h4>
              <p>{gap.detail}</p>
            </section>
          );
        })}
      </ExplanationDialog>
      {content}
    </div>
  );
}
