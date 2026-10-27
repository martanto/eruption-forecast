import { CodeBlock } from "react-code-block";
import { Prism, themes, type PrismTheme } from "prism-react-renderer";
import { cn } from "@/lib/utils";
import { ScrollRegion } from "./ScrollRegion";
import "@/styles/code.css";

// Prism's Python grammar only tags `def name(`; add calls so `fit(...)` reads
// as a function. The guard keeps hot reloads from inserting it twice.
const python = Prism.languages.python;
if (python && !("call" in python)) {
  Prism.languages.insertBefore("python", "builtin", {
    call: { pattern: /(?<![.\w])[A-Za-z_]\w*(?=\s*\()/, alias: "function" },
  });
}

// VS Code Dark+ colours; the surface itself is styled in code.css.
const THEME: PrismTheme = {
  plain: { color: themes.vsDark.plain.color },
  styles: [
    ...themes.vsDark.styles,
    { types: ["boolean"], style: { color: "rgb(86, 156, 214)" } },
    { types: ["decorator"], style: { color: "rgb(220, 220, 170)" } },
  ],
};

/** Python snippet with syntax highlighting; `inline` renders a single `<code>` within prose. */
export function CodeSnippet({
  code,
  inline = false,
  lineNumbers = true,
  label = "Python code, scrolls sideways when wide",
  className,
}: {
  code: string;
  inline?: boolean;
  lineNumbers?: boolean;
  label?: string;
  className?: string;
}) {
  const source = code.replace(/\n$/, "");
  if (inline)
    return (
      <CodeBlock code={source} language="python" theme={THEME}>
        <CodeBlock.Code as="code" className={cn("code-inline", className)}>
          <CodeBlock.LineContent as="span">
            <CodeBlock.Token />
          </CodeBlock.LineContent>
        </CodeBlock.Code>
      </CodeBlock>
    );
  return (
    <ScrollRegion label={label} className="code-scroll">
      <CodeBlock code={source} language="python" theme={THEME}>
        <CodeBlock.Code className={cn("code-snippet", className)}>
          <div className="code-line">
            {lineNumbers && <CodeBlock.LineNumber className="code-line-number" />}
            <CodeBlock.LineContent className="code-line-content">
              <CodeBlock.Token />
            </CodeBlock.LineContent>
          </div>
        </CodeBlock.Code>
      </CodeBlock>
    </ScrollRegion>
  );
}
