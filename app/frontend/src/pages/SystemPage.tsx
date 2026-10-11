import { Activity, HardDrive, Cpu, MemoryStick } from "lucide-react";
import { usePolling } from "../hooks/usePolling";
import { bytes } from "../lib/configuration";

export interface SystemData {
  cpu_percent: number;
  cpu_count: number;
  package_cpu_budget: number;
  memory: { total: number; available: number; percent: number };
  disks: { path: string; total: number; free: number; percent: number }[];
  queue: Record<string, number>;
  worker: {
    pid?: number;
    heartbeat?: string;
    stopped?: boolean;
    job_id?: string;
  };
}
export function workerAlive(data: SystemData | null): boolean {
  return (
    !!data?.worker.heartbeat &&
    !data.worker.stopped &&
    Date.now() - Date.parse(data.worker.heartbeat) < 15000
  );
}

export default function SystemPage() {
  const { data, error } = usePolling<SystemData>("/system", 5000);
  return (
    <>
      <div className="page-heading">
        <div>
          <h1>System</h1>
          <p>
            Live resources on the machine running your scientific jobs. Updated
            every five seconds.
          </p>
        </div>
      </div>
      {error && (
        <p className="error-message" role="alert">
          {error}
        </p>
      )}
      {data ? (
        <>
          <section className="system-section">
            <div className="section-heading">
              <h2>
                <Activity size={20} />
                Worker & queue
              </h2>
              <span
                className={`status ${workerAlive(data) ? "status-succeeded" : "status-interrupted"}`}
              >
                {workerAlive(data) ? "Worker online" : "Worker offline"}
              </span>
            </div>
            {!workerAlive(data) && (
              <p className="info-message">
                Start the worker with{" "}
                <code>uv run --project app/backend eruption-ui worker</code>.
                Queued jobs remain saved until it starts.
              </p>
            )}
            <dl className="system-list">
              <dt>Queued jobs</dt>
              <dd>{data.queue.queued ?? 0}</dd>
              <dt>Running jobs</dt>
              <dd>{data.queue.running ?? 0}</dd>
              <dt>Failed / interrupted</dt>
              <dd>
                {(data.queue.failed ?? 0) + (data.queue.interrupted ?? 0)}
              </dd>
              <dt>Last heartbeat</dt>
              <dd>
                {data.worker.heartbeat
                  ? new Date(data.worker.heartbeat).toLocaleString()
                  : "Not started"}
              </dd>
            </dl>
          </section>
          <section className="system-section">
            <h2>
              <Cpu size={20} />
              CPU
            </h2>
            <div className="resource-row">
              <span>Current usage</span>
              <progress max={100} value={data.cpu_percent} />
              <strong>{data.cpu_percent.toFixed(1)}%</strong>
            </div>
            <p className="muted">
              {data.cpu_count} logical cores · package budget{" "}
              {data.package_cpu_budget} cores after reserving two for the
              operating system. One heavy job runs at a time.
            </p>
          </section>
          <section className="system-section">
            <h2>
              <MemoryStick size={20} />
              Memory
            </h2>
            <div className="resource-row">
              <span>Current usage</span>
              <progress max={100} value={data.memory.percent} />
              <strong>{data.memory.percent.toFixed(1)}%</strong>
            </div>
            <p className="muted">
              {bytes(data.memory.available)} available of{" "}
              {bytes(data.memory.total)}
            </p>
          </section>
          <section className="system-section">
            <h2>
              <HardDrive size={20} />
              Configured storage
            </h2>
            {data.disks.map((disk) => (
              <div className="disk-row" key={disk.path}>
                <span className="path-text">{disk.path}</span>
                <progress max={100} value={disk.percent} />
                <span>
                  {bytes(disk.free)} free / {bytes(disk.total)}
                </span>
              </div>
            ))}
          </section>
        </>
      ) : (
        <p className="muted">Connecting to the backend…</p>
      )}
    </>
  );
}
