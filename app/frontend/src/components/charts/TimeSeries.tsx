import { useEffect, useRef } from "react";
import * as echarts from "echarts/core";
import { LineChart } from "echarts/charts";
import {
  GridComponent,
  TooltipComponent,
  LegendComponent,
  DataZoomComponent,
  MarkLineComponent,
} from "echarts/components";
import { CanvasRenderer } from "echarts/renderers";
import type { TableData } from "../../api/client";

echarts.use([
  LineChart,
  GridComponent,
  TooltipComponent,
  LegendComponent,
  DataZoomComponent,
  MarkLineComponent,
  CanvasRenderer,
]);

export function TimeSeries({
  data,
  title,
  eruptions = [],
  threshold,
}: {
  data: TableData;
  title: string;
  eruptions?: string[];
  threshold?: number;
}) {
  const element = useRef<HTMLDivElement>(null);
  useEffect(() => {
    if (!element.current) return;
    const chart = echarts.init(element.current);
    const time =
      data.columns.find((column) => /date|time|unnamed/i.test(column)) ??
      data.columns[0];
    const numeric = data.columns.filter(
      (column) =>
        column !== time &&
        data.rows.some((row) => typeof row[column] === "number"),
    );
    chart.setOption({
      color: ["#32634e", "#bc7442", "#556f9e", "#7c6085", "#a68a35", "#457f83"],
      tooltip: { trigger: "axis" },
      legend: { type: "scroll", top: 0, textStyle: { fontFamily: "Inter" } },
      grid: { left: 60, right: 25, top: 60, bottom: 80 },
      xAxis: { type: "time", axisLabel: { color: "#657068" } },
      yAxis: { type: "value", splitLine: { lineStyle: { color: "#e5e9e2" } } },
      dataZoom: [
        { type: "inside" },
        {
          type: "slider",
          bottom: 10,
          height: 24,
          borderColor: "#d4dbd2",
          fillerColor: "#e0e9df",
        },
      ],
      series: numeric.map((column, index) => ({
        name: column,
        type: "line",
        showSymbol: false,
        connectNulls: false,
        data: data.rows.map((row) => [row[time], row[column]]),
        lineStyle: {
          width: /uncertainty/.test(column) ? 1 : 2,
          type: /uncertainty/.test(column) ? "dashed" : "solid",
        },
        markLine:
          index === 0
            ? {
                symbol: "none",
                label: { fontSize: 10 },
                data: [
                  ...eruptions.map((date) => ({
                    xAxis: date,
                    name: "Eruption",
                  })),
                  ...(threshold !== undefined
                    ? [{ yAxis: threshold, name: "Threshold" }]
                    : []),
                ],
              }
            : undefined,
      })),
    });
    const observer = new ResizeObserver(() => chart.resize());
    observer.observe(element.current);
    return () => {
      observer.disconnect();
      chart.dispose();
    };
  }, [data, eruptions, threshold]);
  return (
    <div ref={element} className="time-chart" role="img" aria-label={title} />
  );
}
