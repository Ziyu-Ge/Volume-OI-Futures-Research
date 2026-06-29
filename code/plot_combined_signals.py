import argparse
import json
import os
import re
from pathlib import Path


CODE_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = CODE_DIR.parent
RESULTS_DIR = PROJECT_ROOT / "results"
DEFAULT_RUNS_DIR = RESULTS_DIR
DEFAULT_OUTPUT_DIR = RESULTS_DIR / "combined"
DEFAULT_FACTOR_IDS = ("11", "12", "13", "14")
DEFAULT_CONFIDENCE_WINDOW_DAYS = 10
CONFIDENCE_FULL_SIGNAL_DAYS = 4

os.environ.setdefault("MPLCONFIGDIR", str(PROJECT_ROOT / ".matplotlib"))
os.environ.setdefault("XDG_CACHE_HOME", str(PROJECT_ROOT / ".cache"))
(PROJECT_ROOT / ".matplotlib").mkdir(exist_ok=True)
(PROJECT_ROOT / ".cache").mkdir(exist_ok=True)

import matplotlib
import pandas as pd


matplotlib.use("Agg")
import matplotlib.dates as mdates
import matplotlib.pyplot as plt
from matplotlib.colors import to_hex
import plotly.graph_objects as go
import plotly.io as pio


FACTOR_FILE_PATTERN = re.compile(r"^(.+?)_(\d+)_(.+)\.csv$")
MARKERS = [
    "o",
    "s",
    "^",
    "D",
    "v",
    "P",
    "X",
    "*",
    "<",
    ">",
    "h",
    "8",
    "p",
]
PLOTLY_MARKERS = [
    "circle-open",
    "square-open",
    "triangle-up-open",
    "diamond-open",
    "triangle-down-open",
    "cross-open",
    "x-open",
    "star-open",
    "triangle-left-open",
    "triangle-right-open",
    "hexagram-open",
    "octagon-open",
    "pentagon-open",
]
CLICK_INFO_SCRIPT = """
(function() {
    var graph = document.getElementById('{plot_id}');
    if (!graph) {
        return;
    }

    document.documentElement.style.height = '100%';
    document.documentElement.style.margin = '0';
    document.body.style.height = '100vh';
    document.body.style.margin = '0';
    document.body.style.padding = '10px 12px';
    document.body.style.boxSizing = 'border-box';
    document.body.style.overflow = 'hidden';

    var panel = document.createElement('div');
    panel.style.margin = '0 0 8px 0';
    panel.style.padding = '10px 12px';
    panel.style.border = '1px solid #d1d5db';
    panel.style.borderRadius = '6px';
    panel.style.background = '#f9fafb';
    panel.style.color = '#111827';
    panel.style.font = '14px Arial, sans-serif';
    panel.textContent = 'Click a signal point to show its date and factor.';
    graph.parentNode.insertBefore(panel, graph);

    function resizeGraph() {
        var bodyStyle = window.getComputedStyle(document.body);
        var verticalPadding = (
            parseFloat(bodyStyle.paddingTop) +
            parseFloat(bodyStyle.paddingBottom)
        ) || 0;
        var horizontalPadding = (
            parseFloat(bodyStyle.paddingLeft) +
            parseFloat(bodyStyle.paddingRight)
        ) || 0;
        var panelHeight = panel.getBoundingClientRect().height;
        var availableHeight = window.innerHeight - verticalPadding - panelHeight - 8;
        var availableWidth = document.body.clientWidth - horizontalPadding;

        availableHeight = Math.max(120, Math.floor(availableHeight));
        availableWidth = Math.max(260, Math.floor(availableWidth));
        graph.style.width = availableWidth + 'px';
        graph.style.height = availableHeight + 'px';

        Plotly.relayout(graph, {
            autosize: true,
            width: availableWidth,
            height: availableHeight
        });
    }

    window.addEventListener('resize', resizeGraph);
    window.requestAnimationFrame(resizeGraph);

    graph.on('plotly_click', function(eventData) {
        if (!eventData || !eventData.points || eventData.points.length === 0) {
            return;
        }

        var point = eventData.points[0];
        if (!point.customdata) {
            return;
        }

        var dateText = point.customdata[0];
        var factorLabel = point.customdata[1];
        var factorName = point.customdata[2];
        var factorValue = point.customdata[4] || '';
        var confidenceScore = point.customdata[5] || '';
        var confidenceLevel = point.customdata[6] || '';
        var confidenceSignalDays = point.customdata[7] || '';
        var confidenceRecentDates = point.customdata[8] || '';
        var closeText = Number(point.y).toLocaleString(
            undefined,
            {maximumFractionDigits: 4}
        );

        panel.innerHTML = (
            '<strong>Date:</strong> ' + dateText +
            '&nbsp;&nbsp; <strong>Factor:</strong> ' + factorLabel +
            '&nbsp;&nbsp; <strong>Name:</strong> ' + factorName +
            '&nbsp;&nbsp; <strong>Close:</strong> ' + closeText +
            (factorValue ? '&nbsp;&nbsp; <strong>Factor value:</strong> ' + factorValue : '') +
            (confidenceScore ? '&nbsp;&nbsp; <strong>Confidence:</strong> ' + confidenceScore : '') +
            (confidenceLevel ? '&nbsp;&nbsp; <strong>Level:</strong> ' + confidenceLevel : '') +
            (confidenceSignalDays ? '&nbsp;&nbsp; <strong>Signal days:</strong> ' + confidenceSignalDays : '') +
            (confidenceRecentDates ? '&nbsp;&nbsp; <strong>Recent dates:</strong> ' + confidenceRecentDates : '')
        );
        resizeGraph();

        Plotly.relayout(graph, {
            annotations: [{
                x: point.x,
                y: point.y,
                xref: 'x',
                yref: 'y',
                text: dateText + '<br>' + factorLabel,
                showarrow: true,
                arrowhead: 2,
                ax: 32,
                ay: -46,
                bgcolor: 'rgba(255,255,255,0.94)',
                bordercolor: point.fullData.marker.color || '#111827',
                borderwidth: 1,
                font: {size: 12, color: '#111827'}
            }]
        });
    });
})();
"""
DASHBOARD_HTML_TEMPLATE = """<!doctype html>
<html lang="en">
<head>
    <meta charset="utf-8">
    <meta name="viewport" content="width=device-width, initial-scale=1">
    <title>Combined Factor Signals Dashboard</title>
    <script src="plotly.min.js"></script>
    <style>
        html,
        body {{
            height: 100%;
            width: 100%;
            margin: 0;
            overflow: hidden;
            background: #f7f8fb;
            color: #111827;
            font-family: Arial, sans-serif;
        }}

        body {{
            overflow: hidden;
        }}

        .shell {{
            display: grid;
            grid-template-columns: minmax(220px, 280px) 1fr;
            position: fixed;
            inset: 0;
            width: 100vw;
            height: 100vh;
            height: 100dvh;
            min-height: 0;
            overflow: hidden;
        }}

        .sidebar {{
            display: flex;
            flex-direction: column;
            height: 100%;
            min-width: 0;
            min-height: 0;
            overflow: hidden;
            border-right: 1px solid #d9dee8;
            background: #ffffff;
        }}

        .sidebar-header {{
            flex: 0 0 auto;
            padding: 14px 14px 10px;
            border-bottom: 1px solid #e5e7eb;
        }}

        .sidebar-title {{
            font-size: 16px;
            font-weight: 700;
            line-height: 1.2;
        }}

        .sidebar-meta {{
            margin-top: 4px;
            color: #667085;
            font-size: 12px;
        }}

        .search {{
            width: 100%;
            box-sizing: border-box;
            margin-top: 12px;
            padding: 8px 9px;
            border: 1px solid #cfd5df;
            border-radius: 6px;
            color: #111827;
            background: #ffffff;
            font-size: 13px;
            outline: none;
        }}

        .search:focus {{
            border-color: #2563eb;
            box-shadow: 0 0 0 2px rgba(37, 99, 235, 0.12);
        }}

        .symbol-list {{
            flex: 1 1 auto;
            height: 0;
            min-height: 0;
            overflow-x: hidden;
            overflow-y: auto;
            overscroll-behavior: contain;
            padding: 8px;
        }}

        .symbol-button {{
            display: grid;
            grid-template-columns: 1fr auto;
            gap: 8px;
            align-items: center;
            width: 100%;
            min-height: 36px;
            margin: 0 0 4px;
            padding: 8px 9px;
            border: 1px solid transparent;
            border-radius: 6px;
            background: transparent;
            color: #111827;
            cursor: pointer;
            font: inherit;
            text-align: left;
        }}

        .symbol-button:hover {{
            background: #f2f5f9;
        }}

        .symbol-button.active {{
            border-color: #9bb7f0;
            background: #eaf1ff;
        }}

        .symbol-name {{
            min-width: 0;
            overflow: hidden;
            font-weight: 700;
            text-overflow: ellipsis;
            white-space: nowrap;
        }}

        .signal-count {{
            color: #667085;
            font-size: 12px;
            white-space: nowrap;
        }}

        .content {{
            display: flex;
            flex-direction: column;
            height: 100%;
            min-width: 0;
            min-height: 0;
            overflow: hidden;
        }}

        .topbar {{
            display: grid;
            flex: 0 0 auto;
            grid-template-columns: 1fr auto;
            gap: 16px;
            align-items: center;
            min-height: 44px;
            padding: 8px 14px;
            border-bottom: 1px solid #d9dee8;
            background: #ffffff;
        }}

        .title {{
            min-width: 0;
            overflow: hidden;
            font-size: 17px;
            font-weight: 700;
            text-overflow: ellipsis;
            white-space: nowrap;
        }}

        .summary {{
            color: #667085;
            font-size: 13px;
            white-space: nowrap;
        }}

        .info {{
            flex: 0 0 auto;
            min-height: 18px;
            padding: 6px 14px;
            border-bottom: 1px solid #e5e7eb;
            background: #fbfcfe;
            color: #344054;
            font-size: 13px;
        }}

        .plot-wrap {{
            flex: 1;
            flex-basis: 0;
            min-height: 0;
            max-height: 100%;
            padding: 0;
            overflow: hidden;
            position: relative;
        }}

        #dashboard-plot {{
            position: absolute;
            inset: 0;
            width: 100%;
            height: 100%;
        }}

        @media (max-width: 760px) {{
            body {{
                overflow: hidden;
            }}

            .shell {{
                grid-template-columns: 1fr;
                grid-template-rows: minmax(150px, 32vh) 1fr;
                height: 100vh;
                height: 100dvh;
                min-height: 0;
                overflow: hidden;
            }}

            .sidebar {{
                height: 100%;
                min-height: 0;
                overflow: hidden;
                border-right: 0;
                border-bottom: 1px solid #d9dee8;
            }}

            .topbar {{
                grid-template-columns: 1fr;
                gap: 4px;
            }}

            .summary {{
                white-space: normal;
            }}
        }}
    </style>
</head>
<body>
    <div class="shell">
        <aside class="sidebar">
            <div class="sidebar-header">
                <div class="sidebar-title">Symbols</div>
                <div class="sidebar-meta" id="symbol-total"></div>
                <input class="search" id="symbol-search" placeholder="Filter symbols">
            </div>
            <nav class="symbol-list" id="symbol-list"></nav>
        </aside>
        <main class="content">
            <header class="topbar">
                <div class="title" id="current-title"></div>
                <div class="summary" id="current-summary"></div>
            </header>
            <div class="info" id="point-info">Click a signal point to show its date and factor.</div>
            <section class="plot-wrap">
                <div id="dashboard-plot"></div>
            </section>
        </main>
    </div>
    <script>
        const dashboardData = {dashboard_data};
        const symbolOrder = dashboardData.symbols;
        const symbolData = dashboardData.data;
        let currentSymbol = symbolOrder[0] || null;

        const graph = document.getElementById("dashboard-plot");
        const plotWrap = graph.parentElement;
        const content = document.querySelector(".content");
        const list = document.getElementById("symbol-list");
        const search = document.getElementById("symbol-search");
        const total = document.getElementById("symbol-total");
        const title = document.getElementById("current-title");
        const summary = document.getElementById("current-summary");
        const info = document.getElementById("point-info");
        let clickHandlerBound = false;

        total.textContent = symbolOrder.length + " symbols";

        function formatClose(value) {{
            if (value === null || value === undefined || Number.isNaN(Number(value))) {{
                return "";
            }}
            return Number(value).toLocaleString(undefined, {{
                maximumFractionDigits: 4
            }});
        }}

        function renderSymbolList() {{
            const filter = search.value.trim().toUpperCase();
            list.innerHTML = "";

            symbolOrder.forEach(function(symbol) {{
                if (filter && !symbol.includes(filter)) {{
                    return;
                }}

                const data = symbolData[symbol];
                const button = document.createElement("button");
                button.type = "button";
                button.className = "symbol-button" + (
                    symbol === currentSymbol ? " active" : ""
                );
                button.dataset.symbol = symbol;
                button.innerHTML = (
                    '<span class="symbol-name">' + symbol + '</span>' +
                    '<span class="signal-count">' + data.signal_count + ' signals</span>'
                );
                button.addEventListener("click", function() {{
                    selectSymbol(symbol);
                }});
                list.appendChild(button);
            }});
        }}

        function buildTraces(data) {{
            const traces = [{{
                x: data.price.dates,
                y: data.price.close,
                mode: "lines",
                name: "close",
                line: {{
                    color: "#111827",
                    width: 1.5
                }},
                hovertemplate: (
                    "date=%{{x}}<br>" +
                    "close=%{{y:.4f}}<extra>close</extra>"
                )
            }}];

            data.factors.forEach(function(factor) {{
                if (!factor.dates.length) {{
                    return;
                }}

                traces.push({{
                    x: factor.dates,
                    y: factor.close,
                    mode: "markers",
                    name: factor.label + " (" + factor.dates.length + ")",
                    customdata: factor.customdata,
                    marker: {{
                        symbol: factor.marker,
                        size: 11,
                        color: factor.color,
                        line: {{
                            width: 1.8,
                            color: factor.color
                        }}
                    }},
                    hovertemplate: (
                        "date=%{{customdata[0]}}<br>" +
                        "factor=%{{customdata[1]}}<br>" +
                        "name=%{{customdata[2]}}<br>" +
                        "close=%{{y:.4f}}<br>" +
                        "factor value=%{{customdata[4]}}<br>" +
                        "confidence=%{{customdata[5]}}<br>" +
                        "level=%{{customdata[6]}}<br>" +
                        "signal days=%{{customdata[7]}}<br>" +
                        "recent dates=%{{customdata[8]}}" +
                        "<extra></extra>"
                    )
                }});
            }});

            return traces;
        }}

        function buildLayout(data) {{
            return {{
                title: {{
                    text: ""
                }},
                autosize: true,
                template: "plotly_white",
                clickmode: "event+select",
                hovermode: "closest",
                dragmode: "zoom",
                xaxis: {{
                    title: "Date",
                    showgrid: true,
                    gridcolor: "rgba(17, 24, 39, 0.08)",
                    rangeslider: {{
                        visible: true,
                        thickness: 0.045
                    }}
                }},
                yaxis: {{
                    title: "Close price",
                    showgrid: true,
                    gridcolor: "rgba(17, 24, 39, 0.08)"
                }},
                legend: {{
                    orientation: "v",
                    x: 1.02,
                    xanchor: "left",
                    y: 1,
                    yanchor: "top"
                }},
                margin: {{
                    l: 58,
                    r: 210,
                    t: 8,
                    b: 42
                }}
            }};
        }}

        function dashboardPlotSize() {{
            const viewportHeight = (
                window.visualViewport && window.visualViewport.height
            ) ? window.visualViewport.height : window.innerHeight;
            const viewportWidth = (
                window.visualViewport && window.visualViewport.width
            ) ? window.visualViewport.width : window.innerWidth;
            const contentRect = content.getBoundingClientRect();
            const rect = plotWrap.getBoundingClientRect();
            const rightEdge = Math.min(contentRect.right, viewportWidth);
            const bottomEdge = Math.min(contentRect.bottom, viewportHeight);
            const safeBottom = 12;
            const availableHeight = Math.max(
                160,
                Math.floor(bottomEdge - rect.top - safeBottom)
            );
            const availableWidth = Math.max(
                320,
                Math.floor(rightEdge - rect.left)
            );

            plotWrap.style.height = availableHeight + "px";
            plotWrap.style.maxHeight = availableHeight + "px";

            return {{
                width: availableWidth,
                height: availableHeight
            }};
        }}

        function resizeDashboardPlot() {{
            if (!graph.data) {{
                return;
            }}

            const size = dashboardPlotSize();
            graph.style.width = size.width + "px";
            graph.style.height = size.height + "px";
            Plotly.relayout(graph, {{
                autosize: true,
                width: size.width,
                height: size.height
            }});
        }}

        function updateHeader(data) {{
            title.textContent = data.symbol;
            summary.textContent = (
                data.price.dates.length + " daily points, " +
                data.factor_count + " factors, " +
                data.signal_count + " signals"
            );
            info.textContent = "Click a signal point to show its date and factor.";
        }}

        function selectSymbol(symbol) {{
            currentSymbol = symbol;
            const data = symbolData[symbol];
            renderSymbolList();
            updateHeader(data);

            Plotly.react(
                graph,
                buildTraces(data),
                {{
                    ...buildLayout(data),
                    ...dashboardPlotSize()
                }},
                {{
                    displaylogo: false,
                    responsive: true,
                    scrollZoom: true
                }}
            ).then(function() {{
                bindPlotClick();
                resizeDashboardPlot();
            }});
        }}

        function bindPlotClick() {{
            if (clickHandlerBound || typeof graph.on !== "function") {{
                return;
            }}

            graph.on("plotly_click", handlePlotClick);
            clickHandlerBound = true;
        }}

        function handlePlotClick(eventData) {{
            if (!eventData || !eventData.points || eventData.points.length === 0) {{
                return;
            }}

            const point = eventData.points[0];
            if (!point.customdata) {{
                return;
            }}

            const dateText = point.customdata[0];
            const factorLabel = point.customdata[1];
            const factorName = point.customdata[2];
            const factorValue = point.customdata[4] || "";
            const confidenceScore = point.customdata[5] || "";
            const confidenceLevel = point.customdata[6] || "";
            const confidenceSignalDays = point.customdata[7] || "";
            const confidenceRecentDates = point.customdata[8] || "";
            const closeText = formatClose(point.y);

            info.innerHTML = (
                "<strong>Date:</strong> " + dateText +
                "&nbsp;&nbsp; <strong>Factor:</strong> " + factorLabel +
                "&nbsp;&nbsp; <strong>Name:</strong> " + factorName +
                "&nbsp;&nbsp; <strong>Close:</strong> " + closeText +
                (factorValue ? "&nbsp;&nbsp; <strong>Factor value:</strong> " + factorValue : "") +
                (confidenceScore ? "&nbsp;&nbsp; <strong>Confidence:</strong> " + confidenceScore : "") +
                (confidenceLevel ? "&nbsp;&nbsp; <strong>Level:</strong> " + confidenceLevel : "") +
                (confidenceSignalDays ? "&nbsp;&nbsp; <strong>Signal days:</strong> " + confidenceSignalDays : "") +
                (confidenceRecentDates ? "&nbsp;&nbsp; <strong>Recent dates:</strong> " + confidenceRecentDates : "")
            );

            Plotly.relayout(graph, {{
                annotations: [{{
                    x: point.x,
                    y: point.y,
                    xref: "x",
                    yref: "y",
                    text: dateText + "<br>" + factorLabel,
                    showarrow: true,
                    arrowhead: 2,
                    ax: 32,
                    ay: -46,
                    bgcolor: "rgba(255,255,255,0.94)",
                    bordercolor: point.fullData.marker.color || "#111827",
                    borderwidth: 1,
                    font: {{
                        size: 12,
                        color: "#111827"
                    }}
                }}]
            }});
        }}

        search.addEventListener("input", renderSymbolList);
        window.addEventListener("resize", function() {{
            resizeDashboardPlot();
        }});
        if (window.visualViewport) {{
            window.visualViewport.addEventListener("resize", resizeDashboardPlot);
            window.visualViewport.addEventListener("scroll", resizeDashboardPlot);
        }}

        renderSymbolList();
        if (currentSymbol) {{
            selectSymbol(currentSymbol);
        }}
    </script>
</body>
</html>
"""


def parse_symbols(raw_value):
    if raw_value is None:
        return None

    symbols = [
        item.strip().upper()
        for item in raw_value.split(",")
        if item.strip()
    ]
    return set(symbols) or None


def parse_factor_ids(raw_value):
    if raw_value is None:
        return set(DEFAULT_FACTOR_IDS)

    raw_value = raw_value.strip()
    if raw_value == "" or raw_value.upper() in {"ALL", "*"}:
        return None

    factor_ids = {
        item.strip()
        for item in raw_value.split(",")
        if item.strip()
    }
    return factor_ids or set(DEFAULT_FACTOR_IDS)


def parse_factor_file(path):
    match = FACTOR_FILE_PATTERN.match(path.name)
    if match is None:
        return None

    return {
        "symbol": match.group(1).upper(),
        "factor_id": match.group(2),
        "factor_name": match.group(3),
    }


def discover_factor_files(runs_dir, symbol_filter=None, factor_id_filter=None):
    factor_files = []

    for run_dir in sorted(runs_dir.iterdir()):
        if not run_dir.is_dir():
            continue
        if run_dir.name == "combined":
            continue

        factors_dir = run_dir / "tables" / "factors"
        if not factors_dir.is_dir():
            continue

        for path in sorted(factors_dir.glob("*.csv")):
            metadata = parse_factor_file(path)
            if metadata is None:
                continue
            if symbol_filter is not None and metadata["symbol"] not in symbol_filter:
                continue
            if (
                factor_id_filter is not None and
                metadata["factor_id"] not in factor_id_filter
            ):
                continue

            factor_files.append({
                **metadata,
                "run_dir": run_dir.name,
                "path": path,
            })

    if not factor_files:
        raise FileNotFoundError(
            f"no factor csv found under {runs_dir}"
        )

    return factor_files


def load_factor_daily(file_info):
    columns = [
        "date",
        "close",
        "factor_id",
        "factor_name",
        "factor_value",
        "signal",
    ]
    daily = pd.read_csv(
        file_info["path"],
        usecols=lambda col: col in columns,
    )

    missing_columns = {"date", "close", "signal"} - set(daily.columns)
    if missing_columns:
        raise ValueError(
            f"{file_info['path']} missing columns: "
            f"{','.join(sorted(missing_columns))}"
        )

    daily["date"] = pd.to_datetime(daily["date"])
    daily["close"] = pd.to_numeric(daily["close"], errors="coerce")
    daily["signal"] = (
        pd.to_numeric(daily["signal"], errors="coerce")
        .fillna(0)
        .astype(int)
    )

    if "factor_id" in daily.columns and daily["factor_id"].notna().any():
        factor_id = str(daily["factor_id"].dropna().iloc[0])
    else:
        factor_id = file_info["factor_id"]

    if "factor_name" in daily.columns and daily["factor_name"].notna().any():
        factor_name = str(daily["factor_name"].dropna().iloc[0])
    else:
        factor_name = file_info["factor_name"]

    daily = daily.sort_values("date").reset_index(drop=True)
    daily["symbol"] = file_info["symbol"]
    daily["factor_id"] = factor_id
    daily["factor_name"] = factor_name
    daily["run_dir"] = file_info["run_dir"]
    daily["factor_label"] = f"{factor_id}_{factor_name}"

    return daily


def load_all_factor_data(factor_files):
    frames = []
    errors = []

    for file_info in factor_files:
        try:
            frames.append(load_factor_daily(file_info))
        except Exception as exc:
            errors.append((file_info["path"], exc))

    if not frames:
        error_text = "\n".join(f"- {path}: {exc}" for path, exc in errors)
        raise RuntimeError(f"no factor data could be loaded\n{error_text}")

    return frames, errors


def confidence_level(signal_day_count):
    if signal_day_count >= CONFIDENCE_FULL_SIGNAL_DAYS:
        return "high"
    if signal_day_count >= 2:
        return "medium"
    if signal_day_count >= 1:
        return "low"

    return "none"


def format_date_list(date_values):
    return ",".join(
        pd.Timestamp(value).strftime("%Y-%m-%d")
        for value in date_values
    )


def build_symbol_confidence_by_date(symbol_frames, lookback_days):
    all_dates = sorted({
        value
        for frame in symbol_frames
        for value in frame["date"].dropna()
    })
    if not all_dates:
        return {}

    signal_dates = {
        value
        for frame in symbol_frames
        for value in frame.loc[frame["signal"] == 1, "date"].dropna()
    }
    lookback_days = max(int(lookback_days), 1)
    confidence_by_date = {}

    for index, current_date in enumerate(all_dates):
        window_start = max(0, index - lookback_days + 1)
        window_dates = all_dates[window_start:index + 1]
        recent_signal_dates = [
            value
            for value in window_dates
            if value in signal_dates
        ]
        signal_day_count = len(recent_signal_dates)
        confidence_score = min(
            signal_day_count / CONFIDENCE_FULL_SIGNAL_DAYS,
            1.0,
        )

        confidence_by_date[current_date] = {
            "confidence_score": confidence_score,
            "confidence_signal_days": signal_day_count,
            "confidence_level": confidence_level(signal_day_count),
            "confidence_recent_dates": format_date_list(recent_signal_dates),
        }

    return confidence_by_date


def add_signal_confidence(factor_frames, lookback_days):
    output_frames = []
    symbols = sorted({
        frame["symbol"].iloc[0]
        for frame in factor_frames
    })

    for symbol in symbols:
        symbol_frames = [
            frame
            for frame in factor_frames
            if frame["symbol"].iloc[0] == symbol
        ]
        confidence_by_date = build_symbol_confidence_by_date(
            symbol_frames=symbol_frames,
            lookback_days=lookback_days,
        )

        for frame in symbol_frames:
            frame = frame.copy()
            confidence_values = frame["date"].map(
                lambda value: confidence_by_date.get(value, {})
            )
            frame["confidence_score"] = confidence_values.map(
                lambda item: item.get("confidence_score", 0.0)
            )
            frame["confidence_signal_days"] = confidence_values.map(
                lambda item: item.get("confidence_signal_days", 0)
            )
            frame["confidence_level"] = confidence_values.map(
                lambda item: item.get("confidence_level", "none")
            )
            frame["confidence_recent_dates"] = confidence_values.map(
                lambda item: item.get("confidence_recent_dates", "")
            )
            output_frames.append(frame)

    return output_frames


def build_signal_stats(factor_frames):
    rows = []

    for daily in factor_frames:
        signal_points = daily[daily["signal"] == 1]
        signal_factor_values = (
            pd.to_numeric(signal_points["factor_value"], errors="coerce")
            if "factor_value" in signal_points.columns
            else pd.Series(dtype="float64")
        )

        first_signal = None
        last_signal = None
        if len(signal_points) > 0:
            first_signal = signal_points.iloc[0]
            last_signal = signal_points.iloc[-1]

        rows.append({
            "symbol": daily["symbol"].iloc[0],
            "factor_id": daily["factor_id"].iloc[0],
            "factor_name": daily["factor_name"].iloc[0],
            "factor_label": daily["factor_label"].iloc[0],
            "run_dir": daily["run_dir"].iloc[0],
            "total_days": len(daily),
            "valid_close_days": int(daily["close"].notna().sum()),
            "valid_factor_days": (
                int(pd.to_numeric(daily["factor_value"], errors="coerce").notna().sum())
                if "factor_value" in daily.columns
                else 0
            ),
            "signal_days": int(daily["signal"].sum()),
            "signal_ratio": daily["signal"].mean(),
            "first_signal_date": (
                first_signal["date"].date().isoformat()
                if first_signal is not None
                else ""
            ),
            "last_signal_date": (
                last_signal["date"].date().isoformat()
                if last_signal is not None
                else ""
            ),
            "first_signal_close": (
                first_signal["close"] if first_signal is not None else pd.NA
            ),
            "last_signal_close": (
                last_signal["close"] if last_signal is not None else pd.NA
            ),
            "mean_signal_close": signal_points["close"].mean(),
            "min_signal_close": signal_points["close"].min(),
            "max_signal_close": signal_points["close"].max(),
            "mean_signal_factor_value": signal_factor_values.mean(),
            "min_signal_factor_value": signal_factor_values.min(),
            "max_signal_factor_value": signal_factor_values.max(),
        })

    stats = pd.DataFrame(rows)
    stats = stats.sort_values(["symbol", "factor_id", "factor_name"])

    aggregate = (
        stats
        .groupby(["factor_id", "factor_name", "factor_label"], as_index=False)
        .agg(
            run_dir=("run_dir", "first"),
            total_days=("total_days", "sum"),
            valid_close_days=("valid_close_days", "sum"),
            valid_factor_days=("valid_factor_days", "sum"),
            signal_days=("signal_days", "sum"),
            first_signal_date=("first_signal_date", _min_nonempty),
            last_signal_date=("last_signal_date", _max_nonempty),
            mean_signal_close=("mean_signal_close", "mean"),
            mean_signal_factor_value=("mean_signal_factor_value", "mean"),
        )
    )
    aggregate.insert(0, "symbol", "ALL_SYMBOLS")
    aggregate["signal_ratio"] = (
        aggregate["signal_days"] / aggregate["total_days"]
    )

    aggregate = aggregate[
        [
            "symbol",
            "factor_id",
            "factor_name",
            "factor_label",
            "run_dir",
            "total_days",
            "valid_close_days",
            "valid_factor_days",
            "signal_days",
            "signal_ratio",
            "first_signal_date",
            "last_signal_date",
            "mean_signal_close",
            "mean_signal_factor_value",
        ]
    ]

    stats["row_type"] = "symbol_factor"
    aggregate["row_type"] = "all_symbols_factor"

    ordered_columns = ["row_type"] + [
        col for col in stats.columns if col != "row_type"
    ]
    output = pd.concat([stats, aggregate], ignore_index=True, sort=False)
    return output[ordered_columns]


def _min_nonempty(values):
    values = [value for value in values if isinstance(value, str) and value]
    return min(values) if values else ""


def _max_nonempty(values):
    values = [value for value in values if isinstance(value, str) and value]
    return max(values) if values else ""


def factor_style_map(factor_labels):
    color_cycle = list(plt.get_cmap("tab10").colors)
    color_cycle += list(plt.get_cmap("Set2").colors)
    styles = {}

    for index, label in enumerate(sorted(factor_labels)):
        styles[label] = {
            "marker": MARKERS[index % len(MARKERS)],
            "plotly_marker": PLOTLY_MARKERS[index % len(PLOTLY_MARKERS)],
            "color": to_hex(color_cycle[index % len(color_cycle)]),
        }

    return styles


def plot_symbol_signals(symbol, frames, styles, figures_dir, dpi):
    symbol_frames = [frame for frame in frames if frame["symbol"].iloc[0] == symbol]
    if not symbol_frames:
        return None

    price_frame = max(symbol_frames, key=len)[["date", "close"]].copy()
    price_frame = price_frame.dropna(subset=["date", "close"])
    price_frame = price_frame.sort_values("date")

    fig, ax = plt.subplots(figsize=(22, 10))
    ax.plot(
        price_frame["date"],
        price_frame["close"],
        color="#111827",
        linewidth=1.4,
        label="close",
        zorder=1,
    )

    for daily in sorted(symbol_frames, key=lambda frame: frame["factor_label"].iloc[0]):
        label = daily["factor_label"].iloc[0]
        style = styles[label]
        signals = daily[daily["signal"] == 1].dropna(subset=["date", "close"])
        if signals.empty:
            continue

        ax.scatter(
            signals["date"],
            signals["close"],
            s=84,
            marker=style["marker"],
            facecolors="none",
            edgecolors=style["color"],
            linewidths=1.35,
            alpha=0.95,
            label=f"{label} ({len(signals)})",
            zorder=3,
        )

    ax.set_title(f"{symbol} combined factor signals", fontsize=16)
    ax.set_xlabel("Date")
    ax.set_ylabel("Close price")
    ax.grid(True, alpha=0.22)
    ax.xaxis.set_major_locator(mdates.YearLocator(base=1))
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y"))
    ax.legend(loc="upper left", bbox_to_anchor=(1.01, 1.0), frameon=False)
    fig.tight_layout(rect=[0, 0, 0.84, 1])

    figures_dir.mkdir(parents=True, exist_ok=True)
    figure_path = figures_dir / f"{symbol}_combined_factor_signals.png"
    fig.savefig(figure_path, dpi=dpi)
    plt.close(fig)

    return figure_path


def format_factor_value(value):
    try:
        value = float(value)
    except (TypeError, ValueError):
        return ""

    if pd.isna(value):
        return ""

    return f"{value:.6g}"


def format_confidence_score(value):
    try:
        value = float(value)
    except (TypeError, ValueError):
        return ""

    if pd.isna(value):
        return ""

    return f"{value:.2f}"


def prepare_interactive_signal_points(daily):
    signals = daily[daily["signal"] == 1].dropna(subset=["date", "close"]).copy()
    if signals.empty:
        return signals

    if "factor_value" not in signals.columns:
        signals["factor_value"] = pd.NA
    if "confidence_score" not in signals.columns:
        signals["confidence_score"] = pd.NA
    if "confidence_signal_days" not in signals.columns:
        signals["confidence_signal_days"] = 0
    if "confidence_level" not in signals.columns:
        signals["confidence_level"] = ""
    if "confidence_recent_dates" not in signals.columns:
        signals["confidence_recent_dates"] = ""

    signals["signal_date_text"] = signals["date"].dt.strftime("%Y-%m-%d")
    signals["factor_value_text"] = signals["factor_value"].map(format_factor_value)
    signals["confidence_score_text"] = (
        signals["confidence_score"]
        .map(format_confidence_score)
    )
    signals["confidence_signal_days_text"] = (
        signals["confidence_signal_days"]
        .fillna(0)
        .astype(int)
        .astype(str)
    )
    signals["confidence_level_text"] = (
        signals["confidence_level"]
        .fillna("")
        .astype(str)
    )
    signals["confidence_recent_dates_text"] = (
        signals["confidence_recent_dates"]
        .fillna("")
        .astype(str)
    )

    return signals


def plot_symbol_signals_html(symbol, frames, styles, figures_dir):
    symbol_frames = [frame for frame in frames if frame["symbol"].iloc[0] == symbol]
    if not symbol_frames:
        return None

    price_frame = max(symbol_frames, key=len)[["date", "close"]].copy()
    price_frame = price_frame.dropna(subset=["date", "close"])
    price_frame = price_frame.sort_values("date")

    fig = go.Figure()
    fig.add_trace(
        go.Scatter(
            x=price_frame["date"],
            y=price_frame["close"],
            mode="lines",
            name="close",
            line={"color": "#111827", "width": 1.5},
            hovertemplate=(
                "date=%{x|%Y-%m-%d}<br>"
                "close=%{y:.4f}<extra>close</extra>"
            ),
        )
    )

    for daily in sorted(symbol_frames, key=lambda frame: frame["factor_label"].iloc[0]):
        label = daily["factor_label"].iloc[0]
        style = styles[label]
        signals = prepare_interactive_signal_points(daily)
        if signals.empty:
            continue

        custom_data = signals[
            [
                "signal_date_text",
                "factor_label",
                "factor_name",
                "factor_id",
                "factor_value_text",
                "confidence_score_text",
                "confidence_level_text",
                "confidence_signal_days_text",
                "confidence_recent_dates_text",
            ]
        ].to_numpy()

        fig.add_trace(
            go.Scatter(
                x=signals["date"],
                y=signals["close"],
                mode="markers",
                name=f"{label} ({len(signals)})",
                customdata=custom_data,
                marker={
                    "symbol": style["plotly_marker"],
                    "size": 11,
                    "color": style["color"],
                    "line": {"width": 1.8, "color": style["color"]},
                },
                hovertemplate=(
                    "date=%{customdata[0]}<br>"
                    "factor=%{customdata[1]}<br>"
                    "name=%{customdata[2]}<br>"
                    "close=%{y:.4f}<br>"
                    "factor value=%{customdata[4]}<br>"
                    "confidence=%{customdata[5]}<br>"
                    "level=%{customdata[6]}<br>"
                    "signal days=%{customdata[7]}<br>"
                    "recent dates=%{customdata[8]}"
                    "<extra></extra>"
                ),
            )
        )

    fig.update_layout(
        title=f"{symbol} combined factor signals",
        autosize=True,
        template="plotly_white",
        clickmode="event+select",
        hovermode="closest",
        dragmode="zoom",
        xaxis={
            "title": "Date",
            "showgrid": True,
            "gridcolor": "rgba(17, 24, 39, 0.08)",
            "rangeslider": {"visible": True, "thickness": 0.06},
        },
        yaxis={
            "title": "Close price",
            "showgrid": True,
            "gridcolor": "rgba(17, 24, 39, 0.08)",
        },
        legend={
            "orientation": "v",
            "x": 1.02,
            "xanchor": "left",
            "y": 1,
            "yanchor": "top",
        },
        margin={"l": 70, "r": 260, "t": 70, "b": 90},
    )

    figures_dir.mkdir(parents=True, exist_ok=True)
    html_path = figures_dir / f"{symbol}_combined_factor_signals.html"
    pio.write_html(
        fig,
        file=html_path,
        include_plotlyjs="directory",
        full_html=True,
        config={
            "displaylogo": False,
            "responsive": True,
            "scrollZoom": True,
        },
        post_script=CLICK_INFO_SCRIPT,
        default_width="100%",
        default_height="720px",
    )

    return html_path


def format_date(value):
    if pd.isna(value):
        return ""

    return pd.Timestamp(value).strftime("%Y-%m-%d")


def numeric_values(series):
    values = pd.to_numeric(series, errors="coerce")
    return [
        None if pd.isna(value) else float(value)
        for value in values
    ]


def build_dashboard_data(factor_frames, styles):
    symbols = sorted({
        frame["symbol"].iloc[0]
        for frame in factor_frames
    })
    data = {}

    for symbol in symbols:
        symbol_frames = [
            frame
            for frame in factor_frames
            if frame["symbol"].iloc[0] == symbol
        ]
        price_frame = max(symbol_frames, key=len)[["date", "close"]].copy()
        price_frame = price_frame.dropna(subset=["date", "close"])
        price_frame = price_frame.sort_values("date")

        factors = []
        signal_count = 0
        for daily in sorted(
            symbol_frames,
            key=lambda frame: frame["factor_label"].iloc[0],
        ):
            label = daily["factor_label"].iloc[0]
            factor_id = daily["factor_id"].iloc[0]
            factor_name = daily["factor_name"].iloc[0]
            style = styles[label]
            signals = prepare_interactive_signal_points(daily)
            signal_count += len(signals)

            factors.append({
                "label": label,
                "id": factor_id,
                "name": factor_name,
                "color": style["color"],
                "marker": style["plotly_marker"],
                "dates": [
                    format_date(value)
                    for value in signals["date"]
                ],
                "close": numeric_values(signals["close"]),
                "customdata": [
                    [
                        row["signal_date_text"],
                        label,
                        factor_name,
                        factor_id,
                        row["factor_value_text"],
                        row["confidence_score_text"],
                        row["confidence_level_text"],
                        row["confidence_signal_days_text"],
                        row["confidence_recent_dates_text"],
                    ]
                    for _, row in signals.iterrows()
                ],
            })

        data[symbol] = {
            "symbol": symbol,
            "factor_count": len(symbol_frames),
            "signal_count": signal_count,
            "price": {
                "dates": [
                    format_date(value)
                    for value in price_frame["date"]
                ],
                "close": numeric_values(price_frame["close"]),
            },
            "factors": factors,
        }

    return {
        "symbols": symbols,
        "data": data,
    }


def ensure_plotly_js(figures_dir):
    plotly_js_path = figures_dir / "plotly.min.js"
    if not plotly_js_path.exists():
        plotly_js_path.write_text(pio.get_plotlyjs(), encoding="utf-8")

    return plotly_js_path


def save_dashboard_html(factor_frames, styles, figures_dir):
    figures_dir.mkdir(parents=True, exist_ok=True)
    ensure_plotly_js(figures_dir)

    dashboard_data = json.dumps(
        build_dashboard_data(factor_frames, styles),
        ensure_ascii=False,
    ).replace("</", "<\\/")
    html = DASHBOARD_HTML_TEMPLATE.format(dashboard_data=dashboard_data)

    dashboard_path = figures_dir / "combined_factor_signals_dashboard.html"
    dashboard_path.write_text(html, encoding="utf-8")

    return dashboard_path


def save_outputs(factor_frames, output_dir, dpi, confidence_window_days):
    figures_dir = output_dir / "figures"
    tables_dir = output_dir / "tables"
    figures_dir.mkdir(parents=True, exist_ok=True)
    tables_dir.mkdir(parents=True, exist_ok=True)

    factor_frames = add_signal_confidence(
        factor_frames=factor_frames,
        lookback_days=confidence_window_days,
    )

    stats = build_signal_stats(factor_frames)
    stats_path = tables_dir / "combined_signal_stats.csv"
    stats.to_csv(stats_path, index=False)

    factor_labels = {
        frame["factor_label"].iloc[0]
        for frame in factor_frames
    }
    styles = factor_style_map(factor_labels)

    symbols = sorted({
        frame["symbol"].iloc[0]
        for frame in factor_frames
    })
    figure_paths = []
    html_paths = []
    for symbol in symbols:
        figure_path = plot_symbol_signals(
            symbol=symbol,
            frames=factor_frames,
            styles=styles,
            figures_dir=figures_dir,
            dpi=dpi,
        )
        if figure_path is not None:
            figure_paths.append(figure_path)

        html_path = plot_symbol_signals_html(
            symbol=symbol,
            frames=factor_frames,
            styles=styles,
            figures_dir=figures_dir,
        )
        if html_path is not None:
            html_paths.append(html_path)

    dashboard_path = save_dashboard_html(
        factor_frames=factor_frames,
        styles=styles,
        figures_dir=figures_dir,
    )

    return stats_path, figure_paths, html_paths, dashboard_path


def main():
    parser = argparse.ArgumentParser(
        description="Plot close price with all factor signal points from results."
    )
    parser.add_argument(
        "--runs-dir",
        type=Path,
        default=DEFAULT_RUNS_DIR,
        help=f"directory containing numbered factor result folders, default: {DEFAULT_RUNS_DIR}",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=DEFAULT_OUTPUT_DIR,
        help=f"output directory, default: {DEFAULT_OUTPUT_DIR}",
    )
    parser.add_argument(
        "--symbols",
        "--symbol",
        dest="symbols",
        help="optional comma-separated symbols, for example: PP,CU",
    )
    parser.add_argument(
        "--factor-ids",
        default=",".join(DEFAULT_FACTOR_IDS),
        help=(
            "comma-separated factor ids to plot, default: "
            f"{','.join(DEFAULT_FACTOR_IDS)}; use ALL to plot every factor result"
        ),
    )
    parser.add_argument(
        "--dpi",
        type=int,
        default=300,
        help="figure dpi, default: 300",
    )
    parser.add_argument(
        "--confidence-window-days",
        type=int,
        default=DEFAULT_CONFIDENCE_WINDOW_DAYS,
        help=(
            "trading-day lookback window for display-only confidence, "
            f"default: {DEFAULT_CONFIDENCE_WINDOW_DAYS}"
        ),
    )

    args = parser.parse_args()
    runs_dir = args.runs_dir.resolve()
    output_dir = args.output_dir.resolve()

    if not runs_dir.is_dir():
        raise FileNotFoundError(f"factor results directory not found: {runs_dir}")

    factor_files = discover_factor_files(
        runs_dir=runs_dir,
        symbol_filter=parse_symbols(args.symbols),
        factor_id_filter=parse_factor_ids(args.factor_ids),
    )
    factor_frames, errors = load_all_factor_data(factor_files)
    stats_path, figure_paths, html_paths, dashboard_path = save_outputs(
        factor_frames=factor_frames,
        output_dir=output_dir,
        dpi=args.dpi,
        confidence_window_days=args.confidence_window_days,
    )

    print("combined signal outputs complete.")
    print(f"factor files loaded: {len(factor_frames)}")
    print(f"static png figures saved: {len(figure_paths)}")
    print(f"interactive html figures saved: {len(html_paths)}")
    print(f"dashboard: {dashboard_path}")
    print(f"stats table: {stats_path}")
    print(f"figures dir: {output_dir / 'figures'}")
    print(f"confidence window days: {args.confidence_window_days}")

    if errors:
        print("\nskipped files:")
        for path, exc in errors:
            print(f"- {path}: {exc}")


if __name__ == "__main__":
    main()
