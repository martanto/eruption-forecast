import type { NodeProps } from "@xyflow/react";
import { ERUPTIONS, SCENARIOS } from "@/features/architecture";
import { dateOffset, shortDate } from "@/lib/time";

const ORIGIN = "2025-01-01";
const X_START = 130;
const SCALE = 3.55;
const x = (date: string) => X_START + dateOffset(date, ORIGIN) * SCALE;
export function TimelinePeriods({ data }: NodeProps) {
  const scenario = SCENARIOS.find((item) => item.id === data.scenario)!;
  return (
    <section className="timeline-periods" aria-label="Scenario periods in UTC">
      <h2>One seismic record, five training end dates</h2>
      <p>
        Calendar time, 2025, UTC. Bars use the dates in forecast-scenario.py,
        including the end day.
      </p>
      <svg
        viewBox="0 0 1000 370"
        role="img"
        aria-label="Five scenario training and prediction periods with eight dated eruption markers"
      >
        {["01", "02", "03", "04", "05", "06", "07", "08"].map((month) => (
          <g key={month}>
            <line
              x1={x(`2025-${month}-01`)}
              x2={x(`2025-${month}-01`)}
              y1={35}
              y2={310}
              className="timeline-grid"
            />
            <text x={x(`2025-${month}-01`)} y={22}>
              {shortDate(`2025-${month}-01`)}
            </text>
          </g>
        ))}
        {SCENARIOS.map((item, index) => (
          <g
            key={item.id}
            className={scenario.id === item.id ? "timeline-selected" : ""}
          >
            <rect
              className="timeline-row"
              x={0}
              y={48 + index * 48}
              width={1000}
              height={42}
              rx={5}
            />
            <text x={14} y={75 + index * 48}>
              {item.name}
            </text>
            <rect
              className="training-bar"
              x={x(item.trainStart)}
              y={58 + index * 48}
              width={(dateOffset(item.trainEnd, item.trainStart) + 1) * SCALE}
              height={23}
              rx={3}
            />
            <rect
              className="prediction-bar"
              x={x(item.predictionStart)}
              y={58 + index * 48}
              width={
                (dateOffset(item.predictionEnd, item.predictionStart) + 1) *
                SCALE
              }
              height={23}
              rx={3}
            />
          </g>
        ))}
        {ERUPTIONS.map((date, index) => (
          <g key={date}>
            <line
              className="eruption-line"
              x1={x(date)}
              x2={x(date)}
              y1={40}
              y2={310 + (index % 2) * 20}
            />
            <circle className="eruption-dot" cx={x(date)} cy={40} r={4} />
            <text
              className="eruption-label"
              x={x(date)}
              y={330 + (index % 2) * 20}
              textAnchor="middle"
            >
              {shortDate(date)}
            </text>
          </g>
        ))}
      </svg>
      <div className="timeline-legend">
        <span>
          <i className="training-bar" />
          Historical training period
        </span>
        <span>
          <i className="prediction-bar" />
          Prediction period
        </span>
        <span>
          <i className="eruption-dot" />
          Eruption date
        </span>
      </div>
      <div className="window-detail">
        <div>
          <h3>Inside one window</h3>
          <p>
            Illustrative window at time t. Features are computed over a
            look-back interval; the target is defined over a look-ahead
            interval. The eruption-day flag affects which windows are labelled
            positive.
          </p>
        </div>
        <div className="window-ruler">
          <div className="lookback">
            Historical feature window
            <br />
            <strong>t − 2 days → t</strong>
          </div>
          <div className="lookahead">
            Label lookahead
            <br />
            <strong>t → t + 2 days</strong>
          </div>
          <span>Window step: training 6 h, prediction 10 min</span>
        </div>
      </div>
      <p className="execution-label">
        Execution order below: calculate once, then train → predict → evaluate
        → explain for each scenario. This order is independent of calendar
        time.
      </p>
    </section>
  );
}
