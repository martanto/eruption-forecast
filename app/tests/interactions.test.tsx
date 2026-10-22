import { act, render, screen } from "@testing-library/react";
import userEvent from "@testing-library/user-event";
import { createMemoryRouter, RouterProvider } from "react-router-dom";
import { describe, expect, it } from "vitest";
import { useExplorerLocation } from "@/features/explorer";
import { GuidedTour } from "@/features/guided-tour";
import { ScenarioSelector } from "@/features/scenarios";
function Harness() {
  const { location, navigate } = useExplorerLocation();
  return (
    <>
      <output>{JSON.stringify(location)}</output>
      <button onClick={() => navigate({ view: "timeline" })}>Timeline</button>
      <button onClick={() => navigate({ node: "shap" })}>SHAP</button>
      <ScenarioSelector
        selected={location.scenario}
        onSelect={(scenario) => navigate({ scenario })}
      />
      <GuidedTour onSelect={(node) => navigate({ node })} />
    </>
  );
}
describe("URL-owned interactions", () => {
  it("preserves node/scenario across views and restores history without store copies", async () => {
    const router = createMemoryRouter([{ path: "/", element: <Harness /> }], {
      initialEntries: ["/?node=ranking&scenario=scenario-3-2"],
    });
    render(<RouterProvider router={router} />);
    await userEvent.click(screen.getByText("Timeline", { selector: "button" }));
    expect(screen.getByRole("status")).toHaveTextContent(
      '"view":"timeline","scenario":"scenario-3-2","node":"ranking"',
    );
    await userEvent.click(screen.getByText("SHAP", { selector: "button" }));
    await act(() => router.navigate(-1));
    expect(screen.getByRole("status")).toHaveTextContent('"node":"ranking"');
    await act(() => router.navigate(1));
    expect(screen.getByRole("status")).toHaveTextContent('"node":"shap"');
    await userEvent.selectOptions(
      screen.getByLabelText("Example scenario"),
      "scenario-5-2",
    );
    expect(router.state.location.search).toContain("scenario=scenario-5-2");
  });
  it("falls back safely from invalid links and advances/backtracks the guided tour", async () => {
    const router = createMemoryRouter([{ path: "/", element: <Harness /> }], {
      initialEntries: ["/?view=invalid&node=missing&scenario=bad"],
    });
    render(<RouterProvider router={router} />);
    expect(screen.getByRole("status")).toHaveTextContent('"node":"tremor"');
    await userEvent.click(screen.getByText("Take the guided tour"));
    expect(screen.getByLabelText("Previous tour step")).toBeDisabled();
    await userEvent.click(screen.getByText("Next"));
    expect(screen.getByRole("status")).toHaveTextContent('"node":"labels"');
    await userEvent.click(screen.getByLabelText("Previous tour step"));
    expect(screen.getByRole("status")).toHaveTextContent('"node":"tremor"');
    await userEvent.click(screen.getByLabelText("Close tour"));
    expect(screen.getByText("Take the guided tour")).toBeVisible();
  });
});
