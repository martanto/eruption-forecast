import { createBrowserRouter } from "react-router-dom";
import { ExplorerLayout } from "@/layouts/ExplorerLayout";
import { ExplorerPage } from "@/routes/ExplorerPage";
import { LabellingPage } from "@/routes/LabellingPage";
import { RouteError } from "@/components/common/RouteError";
import { ENV } from "@/config/env";
export const router = createBrowserRouter(
  [
    {
      path: "/",
      element: <ExplorerLayout />,
      errorElement: <RouteError />,
      children: [{ index: true, element: <ExplorerPage /> }, { path: 'labelling', element: <LabellingPage /> }],
    },
  ],
  { basename: ENV.BASE_URL },
);
