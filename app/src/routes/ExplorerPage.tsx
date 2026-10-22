import { Explorer, useExplorerLocation } from "@/features/explorer";
import { ScenarioSelector, ScenarioComparison } from "@/features/scenarios";
import { GuidedTour } from "@/features/guided-tour";
export function ExplorerPage() {
  const { location, navigate } = useExplorerLocation();
  return (
    <Explorer
      location={location}
      navigate={navigate}
      scenarioSelector={
        <ScenarioSelector
          selected={location.scenario}
          onSelect={(scenario) => navigate({ scenario })}
        />
      }
      scenarioComparison={
        <ScenarioComparison
          selected={location.scenario}
          onSelect={(scenario) => navigate({ scenario, view: "timeline" })}
        />
      }
      tour={(onSelect) => <GuidedTour onSelect={onSelect} />}
    />
  );
}
