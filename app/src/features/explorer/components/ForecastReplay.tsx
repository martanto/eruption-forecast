import { useEffect, useRef, useState } from 'react';
import * as Dialog from '@radix-ui/react-dialog';
import { Maximize2, X } from 'lucide-react';
import { init, use as registerCharts, type EChartsCoreOption } from 'echarts/core';
import { LineChart } from 'echarts/charts';
import { GridComponent, TooltipComponent, MarkLineComponent, DataZoomComponent, AriaComponent } from 'echarts/components';
import { CanvasRenderer } from 'echarts/renderers';
import { Button } from '@/components/ui/button';
import { ERUPTIONS, THRESHOLD, FORECAST_MODELS, FORECAST_ROWS, forecastValues } from '../forecastReplay';

registerCharts([LineChart, GridComponent, TooltipComponent, MarkLineComponent, DataZoomComponent, AriaComponent, CanvasRenderer]);

function ReplayChart() {
  const host = useRef<HTMLDivElement>(null);
  const instance = useRef<ReturnType<typeof init> | null>(null);
  const [model, setModel] = useState('consensus');
  const [count, setCount] = useState(FORECAST_ROWS.length);
  const [playing, setPlaying] = useState(false);
  useEffect(() => {
    if (!host.current) return;
    const chart = init(host.current);
    instance.current = chart;
    const observer = new ResizeObserver(() => chart.resize());
    observer.observe(host.current);
    return () => { observer.disconnect(); chart.dispose(); instance.current = null; };
  }, []);
  useEffect(() => {
    if (!playing) return;
    const timer = window.setInterval(() => setCount((previous) => Math.min(FORECAST_ROWS.length, previous + 72)), 400);
    return () => window.clearInterval(timer);
  }, [playing]);
  useEffect(() => {
    const chart = instance.current;
    if (!chart) return;
    const points = forecastValues(model, count);
    const styles = getComputedStyle(document.documentElement);
    const color = (role: string) => styles.getPropertyValue(`--color-${role}`).trim();
    const option: EChartsCoreOption = {
      animation: false, useUTC: true,
      textStyle: { fontFamily: 'DM Sans, sans-serif', color: color('ink') },
      aria: { enabled: true, description: 'Recorded OJN scenario 5 eruption probabilities. Threshold 0.7. Eruptions on 2 and 18 August 2025. Shading shows mean plus or minus one standard deviation, clipped to zero and one.' },
      grid: { left: 48, right: 24, top: 48, bottom: 88 },
      tooltip: { trigger: 'axis', confine: true, valueFormatter: (value: unknown) => typeof value === 'number' ? value.toFixed(3) : String(value) },
      xAxis: { type: 'time', min: FORECAST_ROWS[0]!.time, max: FORECAST_ROWS.at(-1)!.time, axisLabel: { formatter: '{dd} {MMM}' } },
      yAxis: { type: 'value', min: 0, max: 1, name: 'Probability', splitLine: { lineStyle: { color: color('line') } } },
      dataZoom: [{ type: 'slider', bottom: 20, height: 24, borderColor: color('line'), fillerColor: color('terracotta-soft') }],
      series: [
        { type: 'line', stack: 'band', data: points.map((p) => [p.time, p.lower]), symbol: 'none', lineStyle: { opacity: 0 }, areaStyle: { opacity: 0 }, silent: true, tooltip: { show: false } },
        { type: 'line', stack: 'band', data: points.map((p) => [p.time, p.width]), symbol: 'none', lineStyle: { opacity: 0 }, areaStyle: { color: color('terracotta'), opacity: 0.18 }, silent: true, tooltip: { show: false } },
        { name: 'Mean probability', type: 'line', data: points.map((p) => [p.time, p.mean]), symbol: 'none', lineStyle: { color: color('terracotta'), width: 2 }, itemStyle: { color: color('terracotta') },
          markLine: { symbol: 'none', lineStyle: { color: color('ember'), type: 'dashed' }, label: { color: color('ember'), fontSize: 12 }, data: [
            { yAxis: THRESHOLD, label: { formatter: 'Threshold 0.7', position: 'insideEndTop' } },
            ...ERUPTIONS.map((date) => ({ xAxis: Date.parse(`${date}T00:00:00Z`), label: { formatter: `Eruption · ${date.slice(8)} Aug`, position: 'insideEndTop' } })),
          ] },
        },
      ],
    };
    chart.setOption(option);
  }, [model, count]);
  const ended = count === FORECAST_ROWS.length;
  return <>
    <p>Recorded OJN scenario 5 · 1–22 August 2025 · UTC. Replay of saved predictions, sampled every 10 minutes.</p>
    <div className="forecast-controls">
      <label>Model <select aria-label="Model" value={model} onChange={(event) => setModel(event.target.value)}>{FORECAST_MODELS.map(([value, label]) => <option key={value} value={value}>{label}</option>)}</select></label>
      <Button variant="outline" onClick={() => { if (ended) setCount(1); setPlaying(ended || !playing); }}>{ended ? 'Replay' : playing ? 'Pause' : 'Play'}</Button>
      <Button variant="ghost" onClick={() => { setPlaying(false); setCount(FORECAST_ROWS.length); }}>Show all</Button>
      <a href={new URL('../../../../../assets/forecast-results_2025-08-01_2025-08-22_ws-2.csv', import.meta.url).href} download>Download CSV</a>
    </div>
    <div ref={host} className="forecast-chart" role="img" aria-label="OJN probability chart with 0.7 threshold and August 2 and 18 eruption markers" />
    <label className="forecast-scrubber">Replay position · {new Date(FORECAST_ROWS[count - 1]!.time).toISOString().slice(0, 16).replace('T', ' ')} UTC
      <input type="range" min={1} max={FORECAST_ROWS.length} value={count} onChange={(event) => { setPlaying(false); setCount(Number(event.target.value)); }} />
    </label>
    <p>Terracotta line: mean probability. Shading: ±1 standard deviation, clipped to [0, 1]; this is not a calibrated confidence interval. Dashed burgundy lines: eruption dates and the 0.7 threshold. Drag the chart’s lower handles to zoom. Replay advances 12 hours per step.</p>
    <p>Source: forecast-results_2025-08-01_2025-08-22_ws-2.csv. Eruption dates: scenarios_OJN.py. This example always shows scenario 5, independently of the scenario selected in the atlas.</p>
  </>;
}

export function ForecastReplay() {
  return <section className="forecast-example">
    <h3>OJN · recorded prediction replay</h3>
    <p>Explore real scenario 5 probabilities with the 0.7 eruption threshold and eruption dates on 2 and 18 August.</p>
    <Dialog.Root>
      <Dialog.Trigger asChild><Button variant="outline"><Maximize2 size={16} /> Open prediction replay</Button></Dialog.Trigger>
      <Dialog.Portal>
        <Dialog.Overlay className="sheet-overlay" />
        <Dialog.Content className="explanation-dialog" aria-describedby={undefined}>
          <header className="explanation-dialog-heading"><Dialog.Title>OJN · scenario 5 prediction</Dialog.Title><Dialog.Close asChild><Button variant="ghost" size="icon" aria-label="Close prediction replay"><X size={20} /></Button></Dialog.Close></header>
          <div className="explanation-dialog-body forecast-body"><ReplayChart /></div>
        </Dialog.Content>
      </Dialog.Portal>
    </Dialog.Root>
  </section>;
}

