import Markdown, { type Components } from "react-markdown";
import remarkGfm from "remark-gfm";
import { ExplanationDialog } from "@/components/common/ExplanationDialog";
import { CodeSnippet } from "@/components/lab/CodeSnippet";
import { MathText } from "@/components/lab/MathText";
import {
  WIKI_GUIDES,
  GAPS,
  type ArchitectureNode,
} from "@/features/architecture";
import { NodeArtifacts, NodeBehavior, NodeParameters } from "./NodeSections";

const PYTHON_FENCE = /language-(python|py)\b/;
const isPythonFence = (child: unknown) =>
  PYTHON_FENCE.test(
    String(
      (child as { properties?: { className?: unknown } } | undefined)
        ?.properties?.className ?? "",
    ),
  );

/** Markdown element overrides: Python fences use CodeSnippet, links render as plain text. */
export const WIKI_MARKDOWN_COMPONENTS: Components = {
  a: ({ children }) => <span>{children}</span>,
  // CodeSnippet draws its own <pre>, so Python fences skip the default one.
  pre: ({ node, children }) =>
    isPythonFence(node?.children?.[0]) ? <>{children}</> : <pre>{children}</pre>,
  code: ({ className, children }) =>
    PYTHON_FENCE.test(className ?? "") ? (
      <CodeSnippet code={String(children)} />
    ) : (
      <code className={className}>{children}</code>
    ),
};

export function WikiExplanation({ item }: { item: ArchitectureNode }) {
  const guides = WIKI_GUIDES[item.stage] ?? [];
  const content = guides.map((guide) => (
    <section key={guide.title}>
      <h3>{guide.title}</h3>
      <p className="wiki-attribution">From wiki/{guide.path}</p>
      <Markdown
        remarkPlugins={[remarkGfm]}
        components={WIKI_MARKDOWN_COMPONENTS}
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
              <p>
                <MathText>{gap.detail}</MathText>
              </p>
            </section>
          );
        })}
      </ExplanationDialog>
      {content}
    </div>
  );
}
