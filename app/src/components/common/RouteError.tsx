import { Link, useRouteError } from "react-router-dom";
export function RouteError() {
  const error = useRouteError();
  return (
    <main className="route-error">
      <h1>The atlas could not open</h1>
      <p>
        {error instanceof Error ? error.message : "This route is unavailable."}
      </p>
      <Link to="/">Return to the explorer</Link>
    </main>
  );
}
