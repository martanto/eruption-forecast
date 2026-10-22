import * as Dialog from "@radix-ui/react-dialog";
import Markdown from "react-markdown";
import remarkGfm from "remark-gfm";
import { Maximize2, X } from "lucide-react";
import {
  WIKI_GUIDES,
  GAPS,
  type ArchitectureNode,
} from "@/features/architecture";
import { Button } from "@/components/ui/button";

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
      <Dialog.Root key={item.id}>
        <Dialog.Trigger asChild>
          <Button variant="outline" className="expand-explanation">
            <Maximize2 size={16} /> Expand explanation
          </Button>
        </Dialog.Trigger>
        <Dialog.Portal>
          <Dialog.Overlay className="sheet-overlay" />
          <Dialog.Content
            className="explanation-dialog"
            aria-describedby={undefined}
          >
            <header className="explanation-dialog-heading">
              <Dialog.Title>{item.title}</Dialog.Title>
              <Dialog.Close asChild>
                <Button
                  variant="ghost"
                  size="icon"
                  aria-label="Close full explanation"
                >
                  <X size={20} />
                </Button>
              </Dialog.Close>
            </header>
            <div className="explanation-dialog-body inspector wiki-guide">
              <p className="inspector-summary">{item.summary}</p>
              <h3>What actually happens</h3>
              {item.behavior.map((text) => (
                <p key={text}>{text}</p>
              ))}
              {content}
              <h3>Parameters & settings</h3>
              <ul>
                {item.parameters.map((text) => (
                  <li key={text}>{text}</li>
                ))}
              </ul>
              <h3>Artifacts</h3>
              <ul>
                {item.artifacts.map((text) => (
                  <li key={text}>{text}</li>
                ))}
              </ul>
              {item.gaps.map((id) => {
                const gap = GAPS.find((entry) => entry.id === id)!;
                return (
                  <section className="gap-note" key={id}>
                    <h4>{gap.title}</h4>
                    <p>{gap.detail}</p>
                  </section>
                );
              })}
            </div>
          </Dialog.Content>
        </Dialog.Portal>
      </Dialog.Root>
      {content}
    </div>
  );
}
