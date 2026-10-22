import { Outlet } from "react-router-dom";
export function ExplorerLayout() {
  return (
    <div className="atlas-shell">
      <a className="skip-link" href="#atlas-content">
        Skip to explorer
      </a>
      <div id="atlas-content">
        <Outlet />
      </div>
    </div>
  );
}
