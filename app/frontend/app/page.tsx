"use client";

import { useMemo, useState } from "react";
import type { MouseEvent } from "react";
import rawData from "../public/dashboard-data.json";

type Metric = {
  label: string;
  value: string;
  sub: string;
};

type TableRow = Record<string, string | number | null>;

type HistogramBin = {
  x0: number;
  x1: number;
  label: string;
  count: number;
};

type Point = {
  x: number;
  y: number;
};

type BoxGroup = {
  label: string;
  min: number;
  q1: number;
  median: number;
  q3: number;
  max: number;
};

type PredictionPoint = {
  actual: number;
  linear: number;
  tree: number;
};

type FeatureImportance = {
  feature: string;
  importance: number;
};

type ModelMetric = {
  r2: number;
  rmse: number;
  mae: number;
  mape: number;
  bestDepth?: number;
};

type DashboardData = {
  overviewMetrics: Metric[];
  dataPreview: TableRow[];
  priceStatistics: TableRow[];
  statisticalSummary: TableRow[];
  charts: {
    priceDistribution: HistogramBin[];
    logPriceDistribution: HistogramBin[];
    incomeScatter: Point[];
    incomeTrend: Point[];
    spaceScatter: Point[];
    spaceTrend: Point[];
    bedBoxes: BoxGroup[];
    bathBoxes: BoxGroup[];
    correlationMatrix: {
      columns: string[];
      values: number[][];
    };
    predictionSample: PredictionPoint[];
    linearResiduals: HistogramBin[];
    treeResiduals: HistogramBin[];
    featureImportances: FeatureImportance[];
  };
  modelMetrics: {
    linear: ModelMetric;
    tree: ModelMetric;
    rmseReduction: number;
  };
  meta: {
    author: string;
    repository: string;
    techStack: string[];
  };
};

const data = rawData as DashboardData;

const sections = [
  "Dataset Overview",
  "Exploratory Analysis",
  "Model Performance",
  "Insights",
] as const;

type Section = (typeof sections)[number];

type TooltipState = {
  x: number;
  y: number;
  title: string;
  rows: string[];
} | null;

const indigo = "#5b6af0";
const teal = "#1d9e75";
const amber = "#ef9f27";

function tooltipFromMouse(
  event: MouseEvent<SVGElement | HTMLDivElement>,
  title: string,
  rows: string[]
): TooltipState {
  return {
    x: event.clientX,
    y: event.clientY,
    title,
    rows,
  };
}

function formatMoney(value: number) {
  return new Intl.NumberFormat("en-US", {
    style: "currency",
    currency: "USD",
    maximumFractionDigits: 0,
  }).format(value);
}

function formatNumber(value: string | number | null) {
  if (value === null) {
    return "";
  }
  if (typeof value === "number") {
    return Number.isInteger(value) ? value.toLocaleString() : value.toFixed(2);
  }
  return value;
}

function Tooltip({ tooltip }: { tooltip: TooltipState }) {
  if (!tooltip) {
    return null;
  }

  return (
    <div
      className="floatingTooltip"
      style={{ left: tooltip.x + 14, top: tooltip.y + 14 }}
    >
      <strong>{tooltip.title}</strong>
      {tooltip.rows.map((row) => (
        <span key={row}>{row}</span>
      ))}
    </div>
  );
}

function domain(values: number[], padding = 0.06): [number, number] {
  const min = Math.min(...values);
  const max = Math.max(...values);
  const span = max - min || 1;
  return [min - span * padding, max + span * padding];
}

function Sidebar({
  active,
  onChange,
}: {
  active: Section;
  onChange: (section: Section) => void;
}) {
  return (
    <aside className="sidebar">
      <div className="brand">
        <div className="brandMark">RE</div>
        <div>
          <div className="brandTitle">Real Estate</div>
          <div className="brandSub">Price Analysis</div>
        </div>
      </div>

      <div className="sideBlock">
        <div className="sideLabel">Navigation</div>
        <nav className="sectionNav" aria-label="Dashboard sections">
          {sections.map((section) => (
            <button
              className={section === active ? "active" : ""}
              key={section}
              onClick={() => onChange(section)}
              type="button"
            >
              {section}
            </button>
          ))}
        </nav>
      </div>

      <div className="sideBlock">
        <div className="sideLabel">Tech Stack</div>
        <div className="tags">
          {data.meta.techStack.map((tag) => (
            <span key={tag}>{tag}</span>
          ))}
        </div>
      </div>

      <div className="sideFooter">
        <div className="authorLabel">Author</div>
        <div className="authorName">{data.meta.author}</div>
        <a href={data.meta.repository} target="_blank" rel="noreferrer">
          GitHub Repository
        </a>
      </div>
    </aside>
  );
}

function SectionHeader({
  badge,
  title,
}: {
  badge: string;
  title: string;
}) {
  return (
    <div className="sectionHeader">
      <div className="sectionBadge">{badge}</div>
      <h1>{title}</h1>
    </div>
  );
}

function MetricGrid({ metrics }: { metrics: Metric[] }) {
  return (
    <div className="metricGrid">
      {metrics.map((metric) => (
        <div className="metricCard" key={metric.label}>
          <div className="metricLabel">{metric.label}</div>
          <div className="metricValue">{metric.value}</div>
          <div className="metricSub">{metric.sub}</div>
        </div>
      ))}
    </div>
  );
}

function DataTable({ rows }: { rows: TableRow[] }) {
  if (!rows.length) {
    return null;
  }

  const columns = Object.keys(rows[0]);

  return (
    <div className="tableWrap">
      <table>
        <thead>
          <tr>
            {columns.map((column) => (
              <th key={column}>{column}</th>
            ))}
          </tr>
        </thead>
        <tbody>
          {rows.map((row, index) => (
            <tr key={index}>
              {columns.map((column) => (
                <td key={column}>{formatNumber(row[column])}</td>
              ))}
            </tr>
          ))}
        </tbody>
      </table>
    </div>
  );
}

function HistogramChart({
  bins,
  color,
  xLabel,
  tooltip,
  setTooltip,
}: {
  bins: HistogramBin[];
  color: string;
  xLabel: string;
  tooltip: TooltipState;
  setTooltip: (tooltip: TooltipState) => void;
}) {
  const width = 640;
  const height = 270;
  const pad = { top: 18, right: 18, bottom: 46, left: 46 };
  const innerW = width - pad.left - pad.right;
  const innerH = height - pad.top - pad.bottom;
  const maxCount = Math.max(...bins.map((bin) => bin.count));
  const barW = innerW / bins.length;

  return (
    <figure className="chart">
      <svg viewBox={`0 0 ${width} ${height}`} role="img" aria-label={xLabel}>
        <line
          className="axis"
          x1={pad.left}
          x2={width - pad.right}
          y1={height - pad.bottom}
          y2={height - pad.bottom}
        />
        <line
          className="axis"
          x1={pad.left}
          x2={pad.left}
          y1={pad.top}
          y2={height - pad.bottom}
        />
        {[0.25, 0.5, 0.75, 1].map((tick) => (
          <line
            className="gridLine"
            key={tick}
            x1={pad.left}
            x2={width - pad.right}
            y1={pad.top + innerH * (1 - tick)}
            y2={pad.top + innerH * (1 - tick)}
          />
        ))}
        {bins.map((bin, index) => {
          const barH = (bin.count / maxCount) * innerH;
          return (
            <rect
              className="interactiveMark"
              fill={color}
              height={barH}
              key={bin.label}
              opacity="0.86"
              rx="2"
              width={Math.max(1, barW - 2)}
              x={pad.left + index * barW}
              y={height - pad.bottom - barH}
              onMouseEnter={(event) =>
                setTooltip(
                  tooltipFromMouse(event, xLabel, [
                    `${bin.count.toLocaleString()} listings`,
                    `From ${formatNumber(bin.x0)} to ${formatNumber(bin.x1)}`,
                  ])
                )
              }
              onMouseMove={(event) =>
                tooltip &&
                setTooltip({ ...tooltip, x: event.clientX, y: event.clientY })
              }
              onMouseLeave={() => setTooltip(null)}
            >
              <title>
                {`${bin.count.toLocaleString()} listings from ${formatNumber(
                  bin.x0
                )} to ${formatNumber(bin.x1)}`}
              </title>
            </rect>
          );
        })}
        <text className="chartLabel" x={width / 2} y={height - 12}>
          {xLabel}
        </text>
      </svg>
    </figure>
  );
}

function ScatterChart({
  points,
  trend,
  color,
  xLabel,
  yLabel,
  moneyAxis = false,
  tooltip,
  setTooltip,
}: {
  points: Point[];
  trend: Point[];
  color: string;
  xLabel: string;
  yLabel: string;
  moneyAxis?: boolean;
  tooltip: TooltipState;
  setTooltip: (tooltip: TooltipState) => void;
}) {
  const width = 640;
  const height = 300;
  const pad = { top: 18, right: 20, bottom: 48, left: 58 };
  const innerW = width - pad.left - pad.right;
  const innerH = height - pad.top - pad.bottom;
  const [xMin, xMax] = domain(points.map((point) => point.x));
  const [yMin, yMax] = domain(points.map((point) => point.y));
  const x = (value: number) => pad.left + ((value - xMin) / (xMax - xMin)) * innerW;
  const y = (value: number) =>
    height - pad.bottom - ((value - yMin) / (yMax - yMin)) * innerH;

  return (
    <figure className="chart">
      <svg viewBox={`0 0 ${width} ${height}`} role="img" aria-label={xLabel}>
        <line
          className="axis"
          x1={pad.left}
          x2={width - pad.right}
          y1={height - pad.bottom}
          y2={height - pad.bottom}
        />
        <line
          className="axis"
          x1={pad.left}
          x2={pad.left}
          y1={pad.top}
          y2={height - pad.bottom}
        />
        {[0.25, 0.5, 0.75, 1].map((tick) => (
          <line
            className="gridLine"
            key={tick}
            x1={pad.left}
            x2={width - pad.right}
            y1={pad.top + innerH * (1 - tick)}
            y2={pad.top + innerH * (1 - tick)}
          />
        ))}
        {points.map((point, index) => (
          <circle
            className="interactiveMark"
            cx={x(point.x)}
            cy={y(point.y)}
            fill={color}
            key={`${point.x}-${point.y}-${index}`}
            opacity="0.18"
            r="3"
            onMouseEnter={(event) =>
              setTooltip(
                tooltipFromMouse(event, "Sample listing", [
                  `${xLabel}: ${
                    moneyAxis ? formatMoney(point.x) : formatNumber(point.x)
                  }`,
                  `${yLabel}: ${formatNumber(point.y)}`,
                ])
              )
            }
            onMouseMove={(event) =>
              tooltip &&
              setTooltip({ ...tooltip, x: event.clientX, y: event.clientY })
            }
            onMouseLeave={() => setTooltip(null)}
          >
            <title>
              {`${xLabel}: ${moneyAxis ? formatMoney(point.x) : formatNumber(point.x)}
${yLabel}: ${formatNumber(point.y)}`}
            </title>
          </circle>
        ))}
        <line
          stroke={amber}
          strokeWidth="3"
          x1={x(trend[0].x)}
          x2={x(trend[1].x)}
          y1={y(trend[0].y)}
          y2={y(trend[1].y)}
        />
        <text className="chartLabel" x={width / 2} y={height - 12}>
          {xLabel}
        </text>
        <text className="chartLabel rotate" x={-height / 2} y="17">
          {yLabel}
        </text>
        <text className="tickLabel" x={pad.left} y={height - 30}>
          {moneyAxis ? `$${Math.round(xMin / 1000)}k` : Math.round(xMin)}
        </text>
        <text className="tickLabel end" x={width - pad.right} y={height - 30}>
          {moneyAxis ? `$${Math.round(xMax / 1000)}k` : Math.round(xMax)}
        </text>
      </svg>
    </figure>
  );
}

function BoxPlotChart({
  groups,
  color,
  xLabel,
  tooltip,
  setTooltip,
}: {
  groups: BoxGroup[];
  color: string;
  xLabel: string;
  tooltip: TooltipState;
  setTooltip: (tooltip: TooltipState) => void;
}) {
  const width = 640;
  const height = 260;
  const pad = { top: 20, right: 24, bottom: 44, left: 48 };
  const innerW = width - pad.left - pad.right;
  const innerH = height - pad.top - pad.bottom;
  const values = groups.flatMap((group) => [
    group.min,
    group.q1,
    group.median,
    group.q3,
    group.max,
  ]);
  const [min, max] = domain(values, 0.02);
  const y = (value: number) =>
    height - pad.bottom - ((value - min) / (max - min)) * innerH;
  const step = innerW / groups.length;

  return (
    <figure className="chart">
      <svg viewBox={`0 0 ${width} ${height}`} role="img" aria-label={xLabel}>
        <line
          className="axis"
          x1={pad.left}
          x2={width - pad.right}
          y1={height - pad.bottom}
          y2={height - pad.bottom}
        />
        <line
          className="axis"
          x1={pad.left}
          x2={pad.left}
          y1={pad.top}
          y2={height - pad.bottom}
        />
        {groups.map((group, index) => {
          const cx = pad.left + step * index + step / 2;
          const boxW = Math.min(46, step * 0.56);
          return (
            <g key={group.label}>
              <line
                className="interactiveMark"
                stroke="#c7c7d4"
                strokeWidth="2"
                x1={cx}
                x2={cx}
                y1={y(group.min)}
                y2={y(group.max)}
                onMouseEnter={(event) =>
                  setTooltip(
                    tooltipFromMouse(event, `${xLabel} ${group.label}`, [
                      `5th pct: ${group.min}`,
                      `Median: ${group.median}`,
                      `95th pct: ${group.max}`,
                    ])
                  )
                }
                onMouseMove={(event) =>
                  tooltip &&
                  setTooltip({ ...tooltip, x: event.clientX, y: event.clientY })
                }
                onMouseLeave={() => setTooltip(null)}
              >
                <title>
                  {`${xLabel} ${group.label}
5th percentile: ${group.min}
Q1: ${group.q1}
Median: ${group.median}
Q3: ${group.q3}
95th percentile: ${group.max}`}
                </title>
              </line>
              <rect
                className="interactiveMark"
                fill={color}
                height={Math.max(2, y(group.q1) - y(group.q3))}
                opacity="0.76"
                rx="3"
                width={boxW}
                x={cx - boxW / 2}
                y={y(group.q3)}
                onMouseEnter={(event) =>
                  setTooltip(
                    tooltipFromMouse(event, `${xLabel} ${group.label}`, [
                      `Q1: ${group.q1}`,
                      `Median: ${group.median}`,
                      `Q3: ${group.q3}`,
                    ])
                  )
                }
                onMouseMove={(event) =>
                  tooltip &&
                  setTooltip({ ...tooltip, x: event.clientX, y: event.clientY })
                }
                onMouseLeave={() => setTooltip(null)}
              >
                <title>
                  {`${xLabel} ${group.label}
Q1: ${group.q1}
Median: ${group.median}
Q3: ${group.q3}`}
                </title>
              </rect>
              <line
                className="interactiveMark"
                stroke={amber}
                strokeWidth="3"
                x1={cx - boxW / 2}
                x2={cx + boxW / 2}
                y1={y(group.median)}
                y2={y(group.median)}
                onMouseEnter={(event) =>
                  setTooltip(
                    tooltipFromMouse(event, `${xLabel} ${group.label}`, [
                      `Median: ${group.median}`,
                    ])
                  )
                }
                onMouseMove={(event) =>
                  tooltip &&
                  setTooltip({ ...tooltip, x: event.clientX, y: event.clientY })
                }
                onMouseLeave={() => setTooltip(null)}
              >
                <title>{`${xLabel} ${group.label} median: ${group.median}`}</title>
              </line>
              <text className="tickLabel middle" x={cx} y={height - 24}>
                {group.label}
              </text>
            </g>
          );
        })}
        <text className="chartLabel" x={width / 2} y={height - 6}>
          {xLabel}
        </text>
      </svg>
    </figure>
  );
}

function PredictionChart({
  points,
  model,
  color,
  title,
  tooltip,
  setTooltip,
}: {
  points: PredictionPoint[];
  model: "linear" | "tree";
  color: string;
  title: string;
  tooltip: TooltipState;
  setTooltip: (tooltip: TooltipState) => void;
}) {
  const chartPoints = points.map((point) => ({
    x: point.actual,
    y: model === "linear" ? point.linear : point.tree,
  }));
  const width = 640;
  const height = 300;
  const pad = { top: 18, right: 20, bottom: 48, left: 56 };
  const innerW = width - pad.left - pad.right;
  const innerH = height - pad.top - pad.bottom;
  const all = chartPoints.flatMap((point) => [point.x, point.y]);
  const [min, max] = domain(all, 0.02);
  const scaleX = (value: number) => pad.left + ((value - min) / (max - min)) * innerW;
  const scaleY = (value: number) =>
    height - pad.bottom - ((value - min) / (max - min)) * innerH;

  return (
    <figure className="chart">
      <figcaption>{title}</figcaption>
      <svg viewBox={`0 0 ${width} ${height}`} role="img" aria-label={title}>
        <line
          className="axis"
          x1={pad.left}
          x2={width - pad.right}
          y1={height - pad.bottom}
          y2={height - pad.bottom}
        />
        <line
          className="axis"
          x1={pad.left}
          x2={pad.left}
          y1={pad.top}
          y2={height - pad.bottom}
        />
        <line
          stroke={amber}
          strokeDasharray="6 6"
          strokeWidth="2.5"
          x1={scaleX(min)}
          x2={scaleX(max)}
          y1={scaleY(min)}
          y2={scaleY(max)}
        />
        {chartPoints.map((point, index) => (
          <circle
            className="interactiveMark"
            cx={scaleX(point.x)}
            cy={scaleY(point.y)}
            fill={color}
            key={index}
            opacity="0.24"
            r="3"
            onMouseEnter={(event) =>
              setTooltip(
                tooltipFromMouse(event, title, [
                  `Actual log price: ${formatNumber(point.x)}`,
                  `Predicted log price: ${formatNumber(point.y)}`,
                ])
              )
            }
            onMouseMove={(event) =>
              tooltip &&
              setTooltip({ ...tooltip, x: event.clientX, y: event.clientY })
            }
            onMouseLeave={() => setTooltip(null)}
          >
            <title>
              {`Actual log price: ${formatNumber(point.x)}
Predicted log price: ${formatNumber(point.y)}`}
            </title>
          </circle>
        ))}
        <text className="chartLabel" x={width / 2} y={height - 12}>
          Actual log price
        </text>
      </svg>
    </figure>
  );
}

function Heatmap({
  matrix,
  setTooltip,
}: {
  matrix: DashboardData["charts"]["correlationMatrix"];
  setTooltip: (tooltip: TooltipState) => void;
}) {
  return (
    <div className="heatmapWrap">
      <div
        className="heatmap"
        style={{
          gridTemplateColumns: `160px repeat(${matrix.columns.length}, minmax(58px, 1fr))`,
        }}
      >
        <div />
        {matrix.columns.map((column) => (
          <div className="heatHead vertical" key={column}>
            {column}
          </div>
        ))}
        {matrix.columns.map((row, rowIndex) => (
          <div className="heatRowGroup" key={row}>
            <div className="heatHead">{row}</div>
            {matrix.values[rowIndex].map((value, colIndex) => {
              const strength = Math.min(1, Math.abs(value));
              const positive = value >= 0;
              const background = positive
                ? `rgba(91, 106, 240, ${0.12 + strength * 0.68})`
                : `rgba(239, 159, 39, ${0.12 + strength * 0.68})`;
              return (
                <div
                  className="heatCell"
                  key={`${row}-${matrix.columns[colIndex]}`}
                  style={{ background }}
                  title={`${row} vs ${matrix.columns[colIndex]}: ${value.toFixed(
                    2
                  )}`}
                  onMouseEnter={(event) =>
                    setTooltip(
                      tooltipFromMouse(event, "Correlation", [
                        `${row} vs ${matrix.columns[colIndex]}`,
                        `Value: ${value.toFixed(2)}`,
                      ])
                    )
                  }
                  onMouseMove={(event) =>
                    setTooltip(({
                      x: event.clientX,
                      y: event.clientY,
                      title: "Correlation",
                      rows: [
                        `${row} vs ${matrix.columns[colIndex]}`,
                        `Value: ${value.toFixed(2)}`,
                      ],
                    }))
                  }
                  onMouseLeave={() => setTooltip(null)}
                >
                  {value.toFixed(2)}
                </div>
              );
            })}
          </div>
        ))}
      </div>
    </div>
  );
}

function ModelCard({
  name,
  badge,
  metric,
  best = false,
}: {
  name: string;
  badge: string;
  metric: ModelMetric;
  best?: boolean;
}) {
  const pct = Math.max(0, Math.min(100, Math.round(metric.r2 * 100)));

  return (
    <div className={`modelCard ${best ? "best" : ""}`}>
      <div className="modelTitle">
        {name}
        <span className={best ? "badgeGreen" : "badgeBlue"}>{badge}</span>
      </div>
      <div className="statRow">
        <span>R2 Score</span>
        <strong>{metric.r2}</strong>
      </div>
      <div className="statRow">
        <span>RMSE</span>
        <strong>{formatMoney(metric.rmse)}</strong>
      </div>
      <div className="statRow">
        <span>MAE</span>
        <strong>{formatMoney(metric.mae)}</strong>
      </div>
      <div className="statRow">
        <span>MAPE</span>
        <strong>{metric.mape}%</strong>
      </div>
      <div className="barWrap">
        <div className="barLabel">R2 score ({pct}%)</div>
        <div className="barTrack">
          <div
            className="barFill"
            style={{ width: `${pct}%`, background: best ? teal : indigo }}
          />
        </div>
      </div>
    </div>
  );
}

function DatasetOverview() {
  return (
    <>
      <SectionHeader badge="01 - Overview" title="American Housing Dataset" />
      <MetricGrid metrics={data.overviewMetrics} />
      <p className="note">
        Note: these values reflect the dataset after preprocessing, outlier
        removal, and feature engineering.
      </p>

      <div className="twoCol wideLeft">
        <section>
          <h2>Data preview</h2>
          <DataTable rows={data.dataPreview} />
        </section>
        <section>
          <h2>Price statistics</h2>
          <DataTable rows={data.priceStatistics} />
        </section>
      </div>

      <section>
        <h2>Full statistical summary</h2>
        <DataTable rows={data.statisticalSummary} />
      </section>
    </>
  );
}

function ExploratoryAnalysis({
  tooltip,
  setTooltip,
}: {
  tooltip: TooltipState;
  setTooltip: (tooltip: TooltipState) => void;
}) {
  return (
    <>
      <SectionHeader badge="02 - EDA" title="Exploratory Data Analysis" />
      <div className="twoCol">
        <section>
          <h2>Price distribution (raw)</h2>
          <HistogramChart
            bins={data.charts.priceDistribution}
            color={indigo}
            xLabel="Price ($)"
            tooltip={tooltip}
            setTooltip={setTooltip}
          />
        </section>
        <section>
          <h2>Log price distribution (transformed)</h2>
          <HistogramChart
            bins={data.charts.logPriceDistribution}
            color={teal}
            xLabel="log(1 + Price)"
            tooltip={tooltip}
            setTooltip={setTooltip}
          />
        </section>
      </div>

      <div className="twoCol">
        <section>
          <h2>Median household income vs log price</h2>
          <ScatterChart
            points={data.charts.incomeScatter}
            trend={data.charts.incomeTrend}
            color={indigo}
            xLabel="Median Household Income ($)"
            yLabel="Log Price"
            moneyAxis
            tooltip={tooltip}
            setTooltip={setTooltip}
          />
        </section>
        <section>
          <h2>Living space vs log price</h2>
          <ScatterChart
            points={data.charts.spaceScatter}
            trend={data.charts.spaceTrend}
            color={teal}
            xLabel="Living Space (sq ft)"
            yLabel="Log Price"
            tooltip={tooltip}
            setTooltip={setTooltip}
          />
        </section>
      </div>

      <div className="twoCol">
        <section>
          <h2>Price by number of bedrooms</h2>
          <BoxPlotChart
            groups={data.charts.bedBoxes}
            color={indigo}
            xLabel="Bedrooms"
            tooltip={tooltip}
            setTooltip={setTooltip}
          />
        </section>
        <section>
          <h2>Price by number of bathrooms</h2>
          <BoxPlotChart
            groups={data.charts.bathBoxes}
            color={teal}
            xLabel="Bathrooms"
            tooltip={tooltip}
            setTooltip={setTooltip}
          />
        </section>
      </div>

      <section>
        <h2>Feature correlation matrix</h2>
        <Heatmap matrix={data.charts.correlationMatrix} setTooltip={setTooltip} />
      </section>
    </>
  );
}

function ModelPerformance({
  tooltip,
  setTooltip,
}: {
  tooltip: TooltipState;
  setTooltip: (tooltip: TooltipState) => void;
}) {
  const comparison = useMemo(
    () => [
      {
        Metric: "R2 Score",
        "Linear Regression": data.modelMetrics.linear.r2,
        "Decision Tree": data.modelMetrics.tree.r2,
      },
      {
        Metric: "RMSE ($)",
        "Linear Regression": formatMoney(data.modelMetrics.linear.rmse),
        "Decision Tree": formatMoney(data.modelMetrics.tree.rmse),
      },
      {
        Metric: "MAE ($)",
        "Linear Regression": formatMoney(data.modelMetrics.linear.mae),
        "Decision Tree": formatMoney(data.modelMetrics.tree.mae),
      },
      {
        Metric: "MAPE (%)",
        "Linear Regression": `${data.modelMetrics.linear.mape}%`,
        "Decision Tree": `${data.modelMetrics.tree.mape}%`,
      },
    ],
    []
  );

  return (
    <>
      <SectionHeader badge="03 - Models" title="Model Evaluation Results" />

      <div className="modelGrid">
        <ModelCard
          name="Linear Regression"
          badge="Baseline"
          metric={data.modelMetrics.linear}
        />
        <ModelCard
          name="Decision Tree Regressor"
          badge={`Best model - depth ${data.modelMetrics.tree.bestDepth}`}
          metric={data.modelMetrics.tree}
          best
        />
      </div>

      <div className="twoCol">
        <section>
          <PredictionChart
            points={data.charts.predictionSample}
            model="linear"
            color={indigo}
            title="Actual vs predicted - Linear Regression"
            tooltip={tooltip}
            setTooltip={setTooltip}
          />
        </section>
        <section>
          <PredictionChart
            points={data.charts.predictionSample}
            model="tree"
            color={teal}
            title="Actual vs predicted - Decision Tree"
            tooltip={tooltip}
            setTooltip={setTooltip}
          />
        </section>
      </div>

      <div className="twoCol">
        <section>
          <h2>Residuals - Linear Regression</h2>
          <HistogramChart
            bins={data.charts.linearResiduals}
            color={indigo}
            xLabel="Residual (log scale)"
            tooltip={tooltip}
            setTooltip={setTooltip}
          />
        </section>
        <section>
          <h2>Residuals - Decision Tree</h2>
          <HistogramChart
            bins={data.charts.treeResiduals}
            color={teal}
            xLabel="Residual (log scale)"
            tooltip={tooltip}
            setTooltip={setTooltip}
          />
        </section>
      </div>

      <section>
        <h2>Feature importances (Decision Tree)</h2>
        <div className="importanceList">
          {data.charts.featureImportances.map((item) => (
            <div className="importanceRow" key={item.feature}>
              <span>{item.feature}</span>
              <div className="importanceTrack">
                <div
                  className="interactiveBar"
                  style={{
                    width: `${Math.max(3, item.importance * 100)}%`,
                    background:
                      item.importance ===
                      Math.max(
                        ...data.charts.featureImportances.map(
                          (feature) => feature.importance
                        )
                      )
                        ? teal
                        : indigo,
                  }}
                  title={`${item.feature}: ${item.importance.toFixed(5)}`}
                  onMouseEnter={(event) =>
                    setTooltip(
                      tooltipFromMouse(event, "Feature importance", [
                        item.feature,
                        `Importance: ${item.importance.toFixed(5)}`,
                      ])
                    )
                  }
                  onMouseMove={(event) =>
                    tooltip &&
                    setTooltip({
                      ...tooltip,
                      x: event.clientX,
                      y: event.clientY,
                    })
                  }
                  onMouseLeave={() => setTooltip(null)}
                />
              </div>
              <strong>{item.importance.toFixed(3)}</strong>
            </div>
          ))}
        </div>
      </section>

      <section>
        <h2>Side-by-side metric comparison</h2>
        <DataTable rows={comparison} />
      </section>
    </>
  );
}

function Insights() {
  return (
    <>
      <SectionHeader badge="04 - Insights" title="Key Findings" />

      <div className="insights">
        <section>
          <h2>Data & Target Variable</h2>
          <p>
            Housing prices in the dataset are heavily right-skewed. Applying a
            log transformation via log1p brought the distribution close to
            normal, which stabilized model training and improved metric
            reliability.
          </p>
        </section>

        <section>
          <h2>Feature Importance</h2>
          <p>
            The engineered Income x Living Space interaction term ranked as the
            single most important Decision Tree feature, ahead of either
            variable on its own. Zip code density contributed meaningfully even
            after income was controlled for, and the distance from the
            expensive-home centroid captured neighborhood prestige effects.
          </p>
        </section>

        <section>
          <h2>Model Comparison</h2>
          <p>
            The Decision Tree Regressor, with optimal depth{" "}
            <strong>{data.modelMetrics.tree.bestDepth}</strong>, outperformed
            Linear Regression across every metric. R2 improved from{" "}
            <strong>{data.modelMetrics.linear.r2}</strong> to{" "}
            <strong>{data.modelMetrics.tree.r2}</strong>, and RMSE dropped from{" "}
            <strong>{formatMoney(data.modelMetrics.linear.rmse)}</strong> to{" "}
            <strong>{formatMoney(data.modelMetrics.tree.rmse)}</strong>, roughly
            a <strong>{data.modelMetrics.rmseReduction}% reduction</strong>.
            This gap exists because housing prices are shaped by non-linear
            interactions between geography, income, and structural variables.
          </p>
        </section>

        <section>
          <h2>Feature Engineering Impact</h2>
          <p>
            Running the same Decision Tree without engineered features produced
            a noticeably lower R2. The performance gains came primarily from
            interaction terms and the geospatial distance feature, not from
            model complexity alone.
          </p>
        </section>

        <section className="conclusion">
          <h2>Conclusion</h2>
          <p>
            US housing prices are determined by a complex interplay of
            socioeconomic, geographic, and structural variables. Tree-based
            models capture these interactions more faithfully than linear
            approaches. Future work could explore gradient-boosted ensembles,
            cross-validation with GridSearchCV, and richer geospatial features.
          </p>
        </section>
      </div>
    </>
  );
}

export default function Home() {
  const [active, setActive] = useState<Section>("Dataset Overview");
  const [tooltip, setTooltip] = useState<TooltipState>(null);

  return (
    <div className="dashboardShell">
      <Sidebar active={active} onChange={setActive} />
      <main className="content">
        {active === "Dataset Overview" && <DatasetOverview />}
        {active === "Exploratory Analysis" && (
          <ExploratoryAnalysis tooltip={tooltip} setTooltip={setTooltip} />
        )}
        {active === "Model Performance" && (
          <ModelPerformance tooltip={tooltip} setTooltip={setTooltip} />
        )}
        {active === "Insights" && <Insights />}
      </main>
      <Tooltip tooltip={tooltip} />
    </div>
  );
}
