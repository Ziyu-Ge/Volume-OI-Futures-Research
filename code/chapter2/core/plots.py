from pathlib import Path

import pandas as pd

from core.paths import setup_runtime_dirs


setup_runtime_dirs()

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.ticker import PercentFormatter


def _prepare_path(path):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    return path


def plot_signal_price(frame, trades, path, title):
    """价格走势上标注开仓和平仓。"""
    path = _prepare_path(path)
    fig, ax = plt.subplots(figsize=(12, 5))
    ax.plot(frame["date"], frame["close"], label="close", color="#1f77b4")

    if "trailing_stop_price" in frame.columns:
        trailing = frame["trailing_stop_price"].where(frame["position"] == -1)
        if trailing.notna().any():
            ax.plot(frame["date"], trailing, label="trailing stop", color="#9467bd")

    if not trades.empty:
        _plot_trade_lines(ax, frame, trades)
        entries = trades[trades["entry_price"].notna()]
        ax.scatter(
            entries["entry_time"],
            entries["entry_price"],
            marker="v",
            color="#d62728",
            label="open short",
            zorder=5,
        )
        exits = trades[trades["exit_price"].notna()]
        if not exits.empty:
            ax.scatter(
                exits["exit_time"],
                exits["exit_price"],
                marker="^",
                color="#2ca02c",
                label="cover short",
                zorder=5,
            )

    ax.set_title(title)
    ax.set_xlabel("Date")
    ax.set_ylabel("Price")
    ax.legend()
    fig.tight_layout()
    fig.savefig(path, dpi=220)
    plt.close(fig)


def _plot_trade_lines(ax, frame, trades):
    """开仓到平仓画线，表示一笔空头持仓区间。"""
    line_labeled = False
    open_labeled = False
    last_time = frame["date"].iloc[-1]
    last_close = frame["close"].iloc[-1]

    for _, trade in trades.iterrows():
        if pd.isna(trade["entry_time"]) or pd.isna(trade["entry_price"]):
            continue
        is_closed = pd.notna(trade["exit_time"]) and pd.notna(trade["exit_price"])
        end_time = trade["exit_time"] if is_closed else last_time
        end_price = trade["exit_price"] if is_closed else last_close
        label = None
        if is_closed and not line_labeled:
            label = "entry to cover"
            line_labeled = True
        if not is_closed and not open_labeled:
            label = "open trade to last close"
            open_labeled = True
        ax.plot(
            [trade["entry_time"], end_time],
            [trade["entry_price"], end_price],
            color="#f97316",
            linewidth=1.6,
            linestyle="-" if is_closed else "--",
            alpha=0.85,
            label=label,
            zorder=4,
        )


def plot_strategy_curve(curve, path, title):
    """单品种策略和基准累计收益对比。"""
    path = _prepare_path(path)
    fig, ax = plt.subplots(figsize=(12, 5))
    ax.plot(curve["date"], curve["strategy_cumulative_return"], label="strategy")
    ax.plot(curve["date"], curve["benchmark_cumulative_return"], label="benchmark")
    ax.plot(curve["date"], curve["excess_cumulative_return"], label="excess")
    ax.axhline(0, color="black", linewidth=0.8, alpha=0.5)
    ax.set_title(title)
    ax.set_xlabel("Date")
    ax.set_ylabel("Cumulative return")
    ax.yaxis.set_major_formatter(PercentFormatter(1.0))
    ax.legend()
    fig.tight_layout()
    fig.savefig(path, dpi=220)
    plt.close(fig)


def plot_all_symbols(portfolio, path, title):
    """全品种等权策略走势汇总图。"""
    plot_strategy_curve(portfolio, path, title)


def plot_return_summary(curves, metrics, path, title):
    """只画有交易品种的策略、基准和超额收益对比。"""
    path = _prepare_path(path)
    traded_symbols = set(
        metrics.loc[
            (metrics["trade_count"] > 0)
            & (metrics["symbol"] != "ALL_SYMBOLS_EQUAL_WEIGHT"),
            "symbol",
        ]
    )
    plot_data = _last_curve_rows(curves)
    plot_data = plot_data[plot_data["symbol"].isin(traded_symbols)].copy()

    if plot_data.empty:
        _plot_empty(path, title, "No symbols with trades")
        return

    plot_data = plot_data.sort_values("excess_cumulative_return", ascending=True)
    y_positions = range(len(plot_data))
    figure_height = max(6, min(24, len(plot_data) * 0.35 + 2))

    fig, ax = plt.subplots(figsize=(12, figure_height))
    ax.barh(
        [pos - 0.25 for pos in y_positions],
        plot_data["benchmark_cumulative_return"],
        height=0.25,
        color="#ff7f0e",
        alpha=0.80,
        label="buy-and-hold benchmark",
    )
    ax.barh(
        y_positions,
        plot_data["strategy_cumulative_return"],
        height=0.25,
        color="#1f77b4",
        alpha=0.88,
        label="strategy: buy-and-hold, reverse short on signal",
    )
    ax.barh(
        [pos + 0.25 for pos in y_positions],
        plot_data["excess_cumulative_return"],
        height=0.25,
        color="#2ca02c",
        alpha=0.80,
        label="excess / reverse-short overlay",
    )
    ax.axvline(0, color="black", linewidth=0.8, alpha=0.55)
    ax.set_yticks(list(y_positions))
    ax.set_yticklabels(plot_data["symbol"])
    ax.set_xlabel("Total return")
    ax.set_title(title)
    ax.xaxis.set_major_formatter(PercentFormatter(1.0))
    ax.legend(loc="lower right")
    fig.tight_layout()
    fig.savefig(path, dpi=220)
    plt.close(fig)


def _last_curve_rows(curves):
    data = curves.sort_values(["symbol", "date"]).copy()
    return data.groupby("symbol", sort=False).tail(1)


def _plot_empty(path, title, message):
    fig, ax = plt.subplots(figsize=(12, 5))
    ax.set_title(title)
    ax.text(0.5, 0.5, message, ha="center", va="center", transform=ax.transAxes)
    ax.axis("off")
    fig.tight_layout()
    fig.savefig(path, dpi=220)
    plt.close(fig)
