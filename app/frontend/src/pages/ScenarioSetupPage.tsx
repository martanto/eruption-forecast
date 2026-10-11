import { useState } from "react";
import { Copy, Plus, Play, Trash2, ArrowUp, ArrowDown } from "lucide-react";
import { useNavigate } from "react-router-dom";
import { api, type Job } from "../api/client";
import { useWorkspace } from "../hooks/useWorkspace";
import { merge, type JsonObject } from "../lib/configuration";
import { ParameterForm } from "../components/forms/ParameterForm";

const stages = ["training", "prediction", "evaluation", "explanation"] as const;

export default function ScenarioSetupPage() {
  const { configuration, setConfiguration, catalog, notice } = useWorkspace();
  const [selected, setSelected] = useState<string>("shared");
  const [stage, setStage] = useState<(typeof stages)[number]>("training");
  const [error, setError] = useState("");
  const [busy, setBusy] = useState(false);
  const [effective, setEffective] = useState<JsonObject | null>(null);
  const navigate = useNavigate();
  const scenarios = (configuration.scenarios as JsonObject[]) ?? [];
  const scenario = scenarios.find((item) => item.id === selected);
  const setScenarios = (values: JsonObject[]) =>
    setConfiguration({ ...configuration, scenarios: values });
  const add = (source?: JsonObject) => {
    const id = crypto.randomUUID();
    setScenarios([
      ...scenarios,
      {
        ...(source ?? {}),
        id,
        name: source
          ? `${source.name} copy`
          : `Scenario ${scenarios.length + 1}`,
      },
    ]);
    setSelected(id);
  };
  const edit = (value: JsonObject) =>
    setScenarios(
      scenarios.map((item) => (item.id === selected ? value : item)),
    );
  const move = (delta: number) => {
    const values = [...scenarios];
    const index = values.findIndex((item) => item.id === selected);
    if (index + delta < 0 || index + delta >= values.length) return;
    [values[index], values[index + delta]] = [
      values[index + delta],
      values[index],
    ];
    setScenarios(values);
  };
  const shared = merge(
    catalog.stages[stage] as JsonObject,
    (configuration[stage] as JsonObject) ?? {},
  );
  const run = async () => {
    setBusy(true);
    setError("");
    try {
      const job = await api<Job>("/jobs", { kind: "scenario", configuration });
      notice("Scenario batch queued");
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
          <h1>Scenario setup</h1>
          <p>
            Compare forecast windows and model settings while sharing one tremor
            calculation.
          </p>
        </div>
        <button
          className="button primary"
          disabled={busy || !scenarios.length}
          onClick={() => void run()}
        >
          <Play size={16} />
          {busy ? "Validatingâ€¦" : "Run scenarios"}
        </button>
      </div>
      {error && (
        <p role="alert" className="error-message">
          {error}
        </p>
      )}
      <div className="batch-options">
        <label>
          <input
            type="checkbox"
            checked={configuration.evaluate_model !== false}
            onChange={(event) =>
              setConfiguration({
                ...configuration,
                evaluate_model: event.target.checked,
              })
            }
          />
          Evaluate models
        </label>
        <label>
          <input
            type="checkbox"
            checked={configuration.explain_model !== false}
            onChange={(event) =>
              setConfiguration({
                ...configuration,
                explain_model: event.target.checked,
              })
            }
          />
          Explain with SHAP
        </label>
        <button
          className="text-button"
          onClick={() =>
            api<JsonObject>("/configurations/resolve", configuration)
              .then(setEffective)
              .catch((error) => setError(error.message))
          }
        >
          Inspect effective configuration
        </button>
      </div>
      <div className="scenario-layout">
        <aside className="scenario-list">
          <button
            aria-pressed={selected === "shared"}
            className={selected === "shared" ? "selected" : ""}
            onClick={() => setSelected("shared")}
          >
            <strong>Shared defaults</strong>
            <span>Used by every scenario</span>
          </button>
          {scenarios.map((item, index) => (
            <button
              key={String(item.id)}
              aria-pressed={selected === item.id}
              className={selected === item.id ? "selected" : ""}
              onClick={() => setSelected(String(item.id))}
            >
              <strong>{String(item.name)}</strong>
              <span>
                Scenario {index + 1} Â·{" "}
                {Object.keys(item).length > 3
                  ? "with overrides"
                  : "shared settings"}
              </span>
            </button>
          ))}
          <button className="add-scenario" onClick={() => add()}>
            <Plus size={16} />
            Add scenario
          </button>
        </aside>
        <section className="scenario-editor">
          {scenario ? (
            <>
              <div className="scenario-toolbar">
                <label>
                  Scenario name
                  <input
                    value={String(scenario.name)}
                    onChange={(event) =>
                      edit({ ...scenario, name: event.target.value })
                    }
                  />
                </label>
                <button
                  className="icon-button"
                  aria-label="Move scenario up"
                  onClick={() => move(-1)}
                >
                  <ArrowUp size={16} />
                </button>
                <button
                  className="icon-button"
                  aria-label="Move scenario down"
                  onClick={() => move(1)}
                >
                  <ArrowDown size={16} />
                </button>
                <button
                  className="icon-button"
                  aria-label="Duplicate scenario"
                  onClick={() => add(scenario)}
                >
                  <Copy size={16} />
                </button>
                <button
                  className="icon-button"
                  aria-label="Delete scenario"
                  onClick={() => {
                    setScenarios(
                      scenarios.filter((item) => item.id !== selected),
                    );
                    setSelected("shared");
                  }}
                >
                  <Trash2 size={16} />
                </button>
              </div>
              <label className="description-field">
                Description
                <input
                  value={String(scenario.description ?? "")}
                  onChange={(event) =>
                    edit({ ...scenario, description: event.target.value })
                  }
                />
              </label>
              <p className="muted">
                Change a value to override it. Reset restores the shared value;
                explicit none remains an override.
              </p>
            </>
          ) : (
            <>
              <h2>Shared stage parameters</h2>
              <p className="muted">
                Start with shared settings, then select a scenario to adjust its
                dates or any stage parameter.
              </p>
            </>
          )}
          <div className="tabs">
            {stages.map((name) => (
              <button
                key={name}
                  aria-pressed={stage === name}
                className={stage === name ? "active" : ""}
                onClick={() => setStage(name)}
              >
                {name[0].toUpperCase() + name.slice(1)}
              </button>
            ))}
          </div>
          <ParameterForm
            key={`${selected}-${stage}`}
            definition={stage[0].toUpperCase() + stage.slice(1)}
            value={((scenario ?? configuration)[stage] as JsonObject) ?? {}}
            defaults={shared}
            overrides={!!scenario}
            onChange={(value) =>
              scenario
                ? edit({ ...scenario, [stage]: value })
                : setConfiguration({ ...configuration, [stage]: value })
            }
          />
        </section>
      </div>
      {effective && (
        <details className="effective-config" open>
          <summary>
            Resolved configuration
            <button className="text-button" onClick={() => setEffective(null)}>
              Close
            </button>
          </summary>
          <pre>{JSON.stringify(effective, null, 2)}</pre>
        </details>
      )}
    </>
  );
}
