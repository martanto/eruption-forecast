import { useCallback, useEffect, useRef, useState } from "react";
import { NavLink } from "react-router-dom";
import {
  Settings2,
  Layers,
  Waves,
  Brain,
  TrendingUp,
  ChartNoAxesCombined,
  ScanSearch,
  FolderOpen,
  ListChecks,
  Activity,
  Menu,
  Download,
  Upload,
  Save,
  FlaskConical,
} from "lucide-react";
import { api, type Artifact } from "./api/client";
import { WorkspaceContext, type Catalog } from "./hooks/useWorkspace";
import { usePolling } from "./hooks/usePolling";
import { type JsonObject } from "./lib/configuration";
import { AppRoutes } from "./router";
import { workerAlive, type SystemData } from "./pages/SystemPage";

const navigation = [
  {
    group: "Workspace",
    path: "/setup",
    label: "General setup",
    icon: Settings2,
  },
  { path: "/scenarios", label: "Scenario setup", icon: Layers },
  { group: "Pipeline", path: "/calculate", label: "Calculate", icon: Waves },
  { path: "/training", label: "Training", icon: Brain },
  { path: "/prediction", label: "Prediction", icon: TrendingUp },
  { path: "/evaluation", label: "Evaluation", icon: ChartNoAxesCombined },
  { path: "/explanation", label: "Explanation", icon: ScanSearch },
  { group: "Research", path: "/results", label: "Results", icon: FolderOpen },
  { path: "/jobs", label: "Jobs", icon: ListChecks },
  { path: "/system", label: "System", icon: Activity },
];

export default function App() {
  const [catalog, setCatalog] = useState<Catalog | null>(null);
  const [configuration, setConfiguration] = useState<JsonObject>(() => {
    try {
      return JSON.parse(localStorage.getItem("eruption-configuration") ?? "{}");
    } catch {
      return {};
    }
  });
  const [artifacts, setArtifacts] = useState<Artifact[]>([]);
  const [error, setError] = useState("");
  const [message, notice] = useState("");
  const [open, setOpen] = useState(false);
  const [profileName, setProfileName] = useState("Research configuration");
  const [profiles, setProfiles] = useState<
    { name: string; configuration: JsonObject }[]
  >([]);
  const upload = useRef<HTMLInputElement>(null);
  const { data: system } = usePolling<SystemData>("/system", 5000);
  const refreshArtifacts = useCallback(async () => {
    setArtifacts(await api<Artifact[]>("/artifacts"));
  }, []);
  useEffect(() => {
    Promise.all([
      api<Catalog>("/parameters"),
      api<JsonObject>("/settings"),
      api<{ name: string; configuration: JsonObject }[]>("/configurations"),
    ])
      .then(([values, setup, stored]) => {
        setCatalog(values);
        setProfiles(stored);
        setConfiguration((current) => ({
          ...current,
          setup: current.setup ?? setup,
        }));
        setError("");
      })
      .catch((error) => setError(error.message));
    void refreshArtifacts().catch(() => {});
  }, [refreshArtifacts]);
  useEffect(() => {
    localStorage.setItem(
      "eruption-configuration",
      JSON.stringify(configuration),
    );
  }, [configuration]);
  useEffect(() => {
    if (message) {
      const timer = setTimeout(() => notice(""), 4000);
      return () => clearTimeout(timer);
    }
  }, [message]);
  const action = async (kind: "save" | "example" | "export", file?: File) => {
    try {
      if (file) {
        setConfiguration(
          await api<JsonObject>("/configurations/import", {
            content: await file.text(),
          }),
        );
        notice("Configuration imported");
        return;
      }
      if (kind === "save") {
        await api("/configurations", { name: profileName, configuration });
        setProfiles(await api("/configurations"));
        notice("Named configuration saved");
      }
      if (kind === "example") {
        setConfiguration(await api("/configurations/example"));
        notice("OJN example loaded; save the directory setup before running");
      }
      if (kind === "export") {
        const response = await fetch(
          "/api/v1/configurations/export?format=yaml",
          {
            method: "POST",
            headers: { "Content-Type": "application/json" },
            body: JSON.stringify(configuration),
          },
        );
        if (!response.ok) throw new Error("Configuration export failed");
        const url = URL.createObjectURL(await response.blob());
        const anchor = document.createElement("a");
        anchor.href = url;
        anchor.download = "forecast-ui.yaml";
        anchor.click();
        URL.revokeObjectURL(url);
      }
      setError("");
    } catch (error) {
      setError((error as Error).message);
    }
  };
  const setup = (configuration.setup as JsonObject) ?? {};
  return (
    <div className="app-shell">
      <a className="skip-link" href="#workspace">
        Skip to workspace
      </a>
      <aside className={`sidebar ${open ? "open" : ""}`}>
        <div className="brand">
          <svg width="32" height="32" viewBox="0 0 32 32" aria-hidden="true">
            <path
              d="M3 23h6l3-7 3 4 4-13 3 16h7"
              fill="none"
              stroke="currentColor"
              strokeWidth="2"
              strokeLinejoin="round"
            />
          </svg>
          <span>
            Eruption
            <br />
            <strong>Forecast</strong>
          </span>
        </div>
        <nav aria-label="Main navigation">
          {navigation.map((item) => (
            <div key={item.path}>
              {item.group && <h2 className="nav-group">{item.group}</h2>}
              <NavLink to={item.path} onClick={() => setOpen(false)}>
                <item.icon size={17} />
                {item.label}
              </NavLink>
            </div>
          ))}
        </nav>
        <div className="sidebar-footer">
          <span
            className={`connection-dot ${workerAlive(system) ? "online" : ""}`}
          />
          <span>
            {workerAlive(system) ? "Worker online" : "Worker offline"}
            <small>Local research workspace</small>
          </span>
        </div>
      </aside>
      <div className="main-shell">
        <header className="topbar">
          <button
            className="icon-button mobile-menu"
            aria-label="Toggle navigation"
            aria-expanded={open}
            onClick={() => setOpen(!open)}
          >
            <Menu size={20} />
          </button>
          <div className="station-context">
            <span>Station</span>
            <strong>
              {[setup.network, setup.station, setup.location, setup.channel]
                .filter((value) => value !== undefined)
                .join(".") || "Not configured"}
            </strong>
          </div>
          <div className="topbar-right">
            <span className="local-tag">Local session</span>
            <a
              href="http://127.0.0.1:8000/docs"
              target="_blank"
              rel="noreferrer"
            >
              API reference
            </a>
          </div>
        </header>
        <div className="configuration-bar">
          <label>
            Configuration
            <input
              aria-label="Configuration name"
              value={profileName}
              onChange={(event) => setProfileName(event.target.value)}
            />
          </label>
          <select
            aria-label="Load saved configuration"
            defaultValue=""
            onChange={(event) => {
              const profile = profiles.find(
                (item) => item.name === event.target.value,
              );
              if (profile) {
                setConfiguration(profile.configuration);
                setProfileName(profile.name);
                notice("Saved configuration loaded");
              }
            }}
          >
            <option value="">Load saved…</option>
            {profiles.map((profile) => (
              <option key={profile.name}>{profile.name}</option>
            ))}
          </select>
          <div className="config-actions">
            <button
              title="Save named configuration"
              aria-label="Save named configuration"
              onClick={() => void action("save")}
            >
              <Save size={15} />
            </button>
            <button
              title="Import JSON or YAML"
              aria-label="Import configuration"
              onClick={() => upload.current?.click()}
            >
              <Upload size={15} />
            </button>
            <button
              title="Export YAML"
              aria-label="Export configuration"
              onClick={() => void action("export")}
            >
              <Download size={15} />
            </button>
            <button
              className="example-button"
              onClick={() => void action("example")}
            >
              <FlaskConical size={15} />
              OJN example
            </button>
            <input
              ref={upload}
              type="file"
              hidden
              accept=".json,.yaml,.yml"
              onChange={(event) => {
                const file = event.target.files?.[0];
                if (file) void action("save", file);
                event.target.value = "";
              }}
            />
          </div>
        </div>
        <main id="workspace" tabIndex={-1}>
          {error && (
            <p role="alert" className="error-message">
              {error}
            </p>
          )}
          {catalog ? (
            <WorkspaceContext.Provider
              value={{
                configuration,
                setConfiguration,
                catalog,
                artifacts,
                refreshArtifacts,
                notice,
              }}
            >
              <AppRoutes />
            </WorkspaceContext.Provider>
          ) : (
            <div className="empty-state">
              <h1>Connecting to your workspace</h1>
              <p>
                Start the backend with{" "}
                <code>uv run --project app/backend eruption-ui dev</code>.
              </p>
              {error && (
                <button
                  className="button secondary"
                  onClick={() => window.location.reload()}
                >
                  Reconnect
                </button>
              )}
            </div>
          )}
        </main>
        <footer className="app-footer">
          <span>Eruption Forecast · Research workspace</span>
          <span>Scientific results require interpretation in context.</span>
        </footer>
      </div>
      {message && (
        <div className="toast" role="status">
          {message}
        </div>
      )}
    </div>
  );
}
