import { createBrowserRouter } from "react-router-dom";
import { ExplorerLayout } from "@/layouts/ExplorerLayout";
import { ExplorerPage } from "@/routes/ExplorerPage";
import { LabellingPage } from "@/routes/LabellingPage";
import { TrainingPage } from "@/routes/TrainingPage";
import { PredictionPage } from "@/routes/PredictionPage";
import { RouteError } from "@/components/common/RouteError";
import { ENV } from "@/config/env";
export const router = createBrowserRouter(
  [
    {
      path: "/",
      element: <ExplorerLayout />,
      errorElement: <RouteError />,
      children: [
        { index: true, element: <ExplorerPage /> },
        { path: "labelling", element: <LabellingPage /> },
        {
          path: "random-under-sampler",
          element: <TrainingPage topic="random-under-sampler" />,
        },
        {
          path: "features-matrix",
          element: <TrainingPage topic="features-matrix" />,
        },
        {
          path: "feature-selection",
          element: <TrainingPage topic="feature-selection" />,
        },
        {
          path: "grid-cv",
          element: <TrainingPage topic="grid-cv" />,
        },
        {
          path: "seed-ensemble",
          element: <TrainingPage topic="seed-ensemble" />,
        },
        {
          path: "classifier-ensemble",
          element: <TrainingPage topic="classifier-ensemble" />,
        },
        { path: "prediction", element: <PredictionPage /> },
      ],
    },
  ],
  { basename: ENV.BASE_URL },
);
