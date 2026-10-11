import { Navigate, Route, Routes } from "react-router-dom";
import { lazy, Suspense } from "react";
import GeneralSetupPage from "./pages/GeneralSetupPage";
import ScenarioSetupPage from "./pages/ScenarioSetupPage";
import { StagePage } from "./pages/StagePage";
import JobsPage from "./pages/JobsPage";
import SystemPage from "./pages/SystemPage";

const ResultsPage = lazy(() => import("./pages/ResultsPage"));

export function AppRoutes() {
  return (
    <Routes>
      <Route path="/" element={<Navigate to="/setup" replace />} />
      <Route path="/setup" element={<GeneralSetupPage />} />
      <Route path="/scenarios" element={<ScenarioSetupPage />} />
      <Route
        path="/calculate"
        element={
          <StagePage
            stage="calculation"
            title="Calculate"
            definition="Calculation"
          />
        }
      />
      <Route
        path="/training"
        element={
          <StagePage stage="training" title="Training" definition="Training" />
        }
      />
      <Route
        path="/prediction"
        element={
          <StagePage
            stage="prediction"
            title="Prediction"
            definition="Prediction"
          />
        }
      />
      <Route
        path="/evaluation"
        element={
          <StagePage
            stage="evaluation"
            title="Evaluation"
            definition="Evaluation"
          />
        }
      />
      <Route
        path="/explanation"
        element={
          <StagePage
            stage="explanation"
            title="Explanation"
            definition="Explanation"
          />
        }
      />
      <Route
        path="/results"
        element={
          <Suspense fallback={<p role="status">Loading results…</p>}>
            <ResultsPage />
          </Suspense>
        }
      />
      <Route path="/jobs" element={<JobsPage />} />
      <Route path="/system" element={<SystemPage />} />
      <Route path="*" element={<Navigate to="/setup" replace />} />
    </Routes>
  );
}
