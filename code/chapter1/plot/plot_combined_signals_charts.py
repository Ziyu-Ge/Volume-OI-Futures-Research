import json

import pandas as pd

# 保存静态 PNG、单品种交互 HTML 和总 dashboard。
from plot_combined_signals_config import setup_cache_dirs
from plot_combined_signals_data import build_signal_stats
from plot_combined_signals_templates import CLICK_INFO_SCRIPT, DASHBOARD_HTML_TEMPLATE


setup_cache_dirs()
import matplotlib

matplotlib.use("Agg")
import matplotlib.dates as mdates
import matplotlib.pyplot as plt
from matplotlib.colors import to_hex
import plotly.graph_objects as go
import plotly.io as pio


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


def prepare_interactive_signal_points(daily):
    signals = daily[daily["signal"] == 1].dropna(subset=["date", "close"]).copy()
    if signals.empty:
        return signals

    if "factor_value" not in signals.columns:
        signals["factor_value"] = pd.NA

    signals["signal_date_text"] = signals["date"].dt.strftime("%Y-%m-%d")
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

    dashboard_path = save_dashboard_html(
        factor_frames=factor_frames,
        styles=styles,
        figures_dir=figures_dir,
    )

    return stats_path, figure_paths, html_paths, dashboard_path
