import { useState } from "react";
import { Play, FileText, ArrowRight } from "lucide-react";
import { Link, useNavigate } from "react-router-dom";
import { api, type Job, type JobKind } from "../api/client";
import { useWorkspace } from "../hooks/useWorkspace";
import { merge, type JsonObject } from "../lib/configuration";
import { ParameterForm } from "../components/forms/ParameterForm";

const descriptions: Record<string, string> = {
  calculation:
    "Transform raw seismic waveforms into RSAM, DSAR, and Shannon entropy time series.",
  training:
    "Build eruption labels, extract seismic features, and fit a multi-seed classifier ensemble.",
  prediction:
    "Apply a trained ensemble to a forecast window and inspect its probability and uncertainty.",
  evaluation:
    "Measure fitted models against eruption labels. Compare classifiers without retraining.",
  explanation:
    "Explain tree-classifier predictions with SHAP. Inspect aggregate, per-seed, and eruption figures.",
};

export function StagePage({
  stage,
  title,
  definition,
}: {
  stage: JobKind;
  title: string;
  definition: string;
}) {
  const { configuration, setConfiguration, catalog, notice } = useWorkspace();
  const [error, setError] = useState("");
  const [busy, setBusy] = useState(false);
  const navigate = useNavigate();
  const settings = (configuration[stage] as JsonObject) ?? {};
  const shared = merge(catalog.stages[stage] as JsonObject, settings);
  const setup = (configuration.setup ?? {}) as JsonObject;
  const constructor = shared["constructor"] as JsonObject;
  for (const key of Object.keys(constructor))
    if (
      Object.hasOwn(setup, key) &&
      !Object.hasOwn((settings["constructor"] as JsonObject) ?? {}, key)
    )
      constructor[key] = setup[key];
  const run = async () => {
    setBusy(true);
    setError("");
    try {
      const job = await api<Job>("/jobs", { kind: stage, configuration });
      notice(`${title} queued`);
      navigate(`/jobs?selected=${job.id}`);
    } catch (error) {
      setError((error as Error).message);
    } finally {
      setBusy(false);
    }
  };
  return (
    <>
      <div className="page-heading">
        <div>
          <h1>{title}</h1>
          <p>{descriptions[stage]}</p>
        </div>
        <button
          className="button primary"
          disabled={busy}
          onClick={() => void run()}
        >
          <Play size={16} />
          {busy ? "Validating…" : `Run ${title.toLowerCase()}`}
        </button>
      </div>
      <div className="context-line">
        <span>Independent stage</span>
        <Link to="/results">
          Select existing artifacts <ArrowRight size={13} />
        </Link>
        <a
          href={`https://github.com/martanto/eruption-forecast/wiki/${catalog.docs[stage]}`}
          target="_blank"
          rel="noreferrer"
        >
          <FileText size={14} />
          Workflow reference
        </a>
      </div>
      {stage === "explanation" && (
        <p className="info-message">
          SHAP supports RF, lite-RF, GB, and XGB here. Other classifiers are
          skipped by the package with a warning.
        </p>
      )}
      {stage === "evaluation" && (
        <p className="info-message">
          Prediction reuse requires eruption dates and a saved PredictionModel.
          Training reuse uses the training labels.
        </p>
      )}
      {stage === "prediction" && (
        <p className="info-message">
          Choose a saved TrainingModel or ensemble. Files mode requires both the
          feature matrix and matching label CSV.
        </p>
      )}
      {error && (
        <p role="alert" className="error-message">
          {error}
        </p>
      )}
      <div className="form-panel">
        <ParameterForm
          definition={definition}
          value={settings}
          defaults={shared}
          onChange={(value) =>
            setConfiguration({ ...configuration, [stage]: value })
          }
        />
      </div>
      <div className="page-footer">
        <span>
          Settings are saved as a local draft. Submitted jobs keep a separate
          immutable snapshot.
        </span>
        <Link to="/jobs">
          View jobs <ArrowRight size={14} />
        </Link>
      </div>
    </>
  );
}
