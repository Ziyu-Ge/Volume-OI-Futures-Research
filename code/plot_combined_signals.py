import argparse
import os
import re
from pathlib import Path


CODE_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = CODE_DIR.parent
DEFAULT_RUNS_DIR = PROJECT_ROOT / "results" / "runs"
DEFAULT_OUTPUT_DIR = DEFAULT_RUNS_DIR / "combined"

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

        availableHeight = Math.max(320, Math.floor(availableHeight));
        availableWidth = Math.max(320, Math.floor(availableWidth));
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
        var closeText = Number(point.y).toLocaleString(
            undefined,
            {maximumFractionDigits: 4}
        );

        panel.innerHTML = (
            '<strong>Date:</strong> ' + dateText +
            '&nbsp;&nbsp; <strong>Factor:</strong> ' + factorLabel +
            '&nbsp;&nbsp; <strong>Name:</strong> ' + factorName +
            '&nbsp;&nbsp; <strong>Close:</strong> ' + closeText +
            (factorValue ? '&nbsp;&nbsp; <strong>Factor value:</strong> ' + factorValue : '')
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


def parse_symbols(raw_value):
    if raw_value is None:
        return None

    symbols = [
        item.strip().upper()
        for item in raw_value.split(",")
        if item.strip()
    ]
    return set(symbols) or None


def parse_factor_file(path):
    match = FACTOR_FILE_PATTERN.match(path.name)
    if match is None:
        return None

    return {
        "symbol": match.group(1).upper(),
        "factor_id": match.group(2),
        "factor_name": match.group(3),
    }


def discover_factor_files(runs_dir, symbol_filter=None):
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


def format_datetime_text(value):
    if pd.isna(value):
        return ""

    return pd.Timestamp(value).strftime("%Y-%m-%d %H:%M:%S")


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
                format_datetime_text(first_signal["date"])
                if first_signal is not None
                else ""
            ),
            "last_signal_date": (
                format_datetime_text(last_signal["date"])
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


def configure_datetime_axis(ax, dates):
    dates = pd.Series(dates).dropna()
    if dates.empty:
        return

    span = dates.max() - dates.min()
    total_days = max(span / pd.Timedelta(days=1), 0)

    if total_days > 365 * 2:
        interval = max(1, int(total_days / (365 * 10)) + 1)
        locator = mdates.YearLocator(base=interval)
    elif total_days > 60:
        interval = max(1, int(total_days / (30 * 10)) + 1)
        locator = mdates.MonthLocator(interval=interval)
    elif total_days > 2:
        interval = max(1, int(total_days / 10) + 1)
        locator = mdates.DayLocator(interval=interval)
    elif total_days > 0.25:
        total_hours = total_days * 24
        interval = max(1, int(total_hours / 10) + 1)
        locator = mdates.HourLocator(interval=interval)
    else:
        total_minutes = max(total_days * 24 * 60, 1)
        interval = max(1, int(total_minutes / 10) + 1)
        locator = mdates.MinuteLocator(interval=interval)

    ax.xaxis.set_major_locator(locator)
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y-%m-%d\n%H:%M"))


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
    ax.set_xlabel("Date Time")
    ax.set_ylabel("Close price")
    ax.grid(True, alpha=0.22)
    configure_datetime_axis(ax, price_frame["date"])
    fig.autofmt_xdate(rotation=0, ha="center")
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


def prepare_interactive_signal_points(daily):
    signals = daily[daily["signal"] == 1].dropna(subset=["date", "close"]).copy()
    if signals.empty:
        return signals

    if "factor_value" not in signals.columns:
        signals["factor_value"] = pd.NA

    signals["signal_date_text"] = signals["date"].dt.strftime(
        "%Y-%m-%d %H:%M:%S"
    )
    signals["factor_value_text"] = signals["factor_value"].map(format_factor_value)

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
                "date=%{x|%Y-%m-%d %H:%M:%S}<br>"
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
                    "factor value=%{customdata[4]}"
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
            "title": "Date Time",
            "tickformat": "%Y-%m-%d<br>%H:%M",
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


def save_outputs(factor_frames, output_dir, dpi):
    figures_dir = output_dir / "figures"
    tables_dir = output_dir / "tables"
    figures_dir.mkdir(parents=True, exist_ok=True)
    tables_dir.mkdir(parents=True, exist_ok=True)

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

    return stats_path, figure_paths, html_paths


def main():
    parser = argparse.ArgumentParser(
        description="Plot close price with all factor signal points from results/runs."
    )
    parser.add_argument(
        "--runs-dir",
        type=Path,
        default=DEFAULT_RUNS_DIR,
        help=f"runs directory, default: {DEFAULT_RUNS_DIR}",
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
        "--dpi",
        type=int,
        default=300,
        help="figure dpi, default: 300",
    )

    args = parser.parse_args()
    runs_dir = args.runs_dir.resolve()
    output_dir = args.output_dir.resolve()

    if not runs_dir.is_dir():
        raise FileNotFoundError(f"runs directory not found: {runs_dir}")

    factor_files = discover_factor_files(
        runs_dir=runs_dir,
        symbol_filter=parse_symbols(args.symbols),
    )
    factor_frames, errors = load_all_factor_data(factor_files)
    stats_path, figure_paths, html_paths = save_outputs(
        factor_frames=factor_frames,
        output_dir=output_dir,
        dpi=args.dpi,
    )

    print("combined signal outputs complete.")
    print(f"factor files loaded: {len(factor_frames)}")
    print(f"static png figures saved: {len(figure_paths)}")
    print(f"interactive html figures saved: {len(html_paths)}")
    print(f"stats table: {stats_path}")
    print(f"figures dir: {output_dir / 'figures'}")

    if errors:
        print("\nskipped files:")
        for path, exc in errors:
            print(f"- {path}: {exc}")


if __name__ == "__main__":
    main()
