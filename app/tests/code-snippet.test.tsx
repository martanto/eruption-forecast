import { render } from "@testing-library/react";
import Markdown from "react-markdown";
import { describe, expect, it } from "vitest";
import { CodeSnippet } from "@/components/lab/CodeSnippet";
import { WIKI_MARKDOWN_COMPONENTS } from "@/features/explorer/components/WikiExplanation";

const SETTINGS = `day_to_forecast=3
include_eruption_date=True
window_step=6, window_step_unit="hours"`;

const tokens = (root: HTMLElement, type: string) =>
  [...root.querySelectorAll(`.token.${type}`)].map((el) => el.textContent);

describe("CodeSnippet", () => {
  it("keeps the source text and highlights Python tokens", () => {
    const { container } = render(<CodeSnippet code={SETTINGS} />);
    // The library renders one row per line, so compare line by line.
    const lines = [...container.querySelectorAll(".code-line-content")].map(
      (line) => line.textContent,
    );
    expect(lines).toEqual(SETTINGS.split("\n"));
    expect(
      [...container.querySelectorAll(".code-line-number")].map(
        (number) => number.textContent,
      ),
    ).toEqual(["1", "2", "3"]);
    expect(tokens(container, "boolean")).toEqual(["True"]);
    expect(tokens(container, "number")).toEqual(["3", "6"]);
    expect(tokens(container, "string")).toEqual(['"hours"']);
  });

  it("labels the scrollable region and renders inline snippets in one code element", () => {
    const { container, getByRole } = render(
      <>
        <CodeSnippet code="x = 1" label="Settings example" />
        <CodeSnippet inline code="fit(seeds=25)" />
      </>,
    );
    expect(getByRole("region", { name: "Settings example" })).toBeInTheDocument();
    const inline = container.querySelector("code.code-inline");
    expect(inline?.textContent).toBe("fit(seeds=25)");
    expect(inline?.closest("pre")).toBeNull();
  });
});

describe("wiki markdown overrides", () => {
  it("routes python fences to CodeSnippet and leaves other fences as plain pre", () => {
    const { container } = render(
      <Markdown components={WIKI_MARKDOWN_COMPONENTS}>
        {"```python\ndef fit(seeds=25):\n    pass\n```\n\n```\nplain  diagram\n```"}
      </Markdown>,
    );
    expect(container.querySelectorAll("pre.code-snippet")).toHaveLength(1);
    expect(container.querySelectorAll("pre")).toHaveLength(2);
    expect(tokens(container, "keyword")).toEqual(["def", "pass"]);
    expect(tokens(container, "function")).toEqual(["fit"]);
    expect(container.textContent).toContain("plain  diagram");
  });
});
