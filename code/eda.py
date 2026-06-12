import argparse
import os
import sys


PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
FACTORS_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "factors")

if FACTORS_DIR not in sys.path:
    sys.path.insert(0, FACTORS_DIR)

os.environ.setdefault("MPLCONFIGDIR", os.path.join(PROJECT_ROOT, ".matplotlib"))
os.environ.setdefault("XDG_CACHE_HOME", os.path.join(PROJECT_ROOT, ".cache"))

import matplotlib

matplotlib.use("Agg")

import matplotlib.dates as mdates
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from config import SYMBOL
from volume_price_factor_utils import load_daily, past_mad_score


DEFAULT_OUTPUT_DIR = os.path.join(PROJECT_ROOT, "results", "figures", "eda")


def parse_args():
    parser = argparse.ArgumentParser(
        description="Plot EDA charts for price, speculation, OI, and volume.",
    )
    parser.add_argument("--symbol", default=SYMBOL)
    parser.add_argument("--price-column", default="close")
    parser.add_argument("--output-dir", default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--change-days", type=int, default=1)
    parser.add_argument("--mad-window", type=int, default=10)
    parser.add_argument("--min-mad-days", type=int, default=10)
    parser.add_argument("--start-date", default=None)
    parser.add_argument("--end-date", default=None)
    return parser.parse_args()


def add_eda_features(
    daily,
    change_days=1,
    mad_window=10,
    min_mad_days=10,
):
    daily = daily.copy()

    daily["speculation_change"] = (
        daily["speculation"] - daily["speculation"].shift(change_days)
    )
    daily["speculation_change_rate"] = (
        np.exp(daily["speculation_change"]) - 1
    )
    daily["speculation_change_rate_pct"] = (
        daily["speculation_change_rate"] * 100
    )

    (
        daily["speculation_median_history"],
        daily["speculation_mad_history"],
        daily["speculation_mad_score"],
    ) = past_mad_score(
        daily["speculation"],
        window=mad_window,
        min_history_days=min_mad_days,
    )

    previous_open_interest = daily["open_interest"].shift(1)
    daily["oi_change_rate"] = (
        (daily["open_interest"] - previous_open_interest) /
        previous_open_interest *
        100
    )
    daily.loc[previous_open_interest <= 0, "oi_change_rate"] = np.nan

    return daily


def filter_dates(daily, start_date=None, end_date=None):
    daily = daily.copy()

    if start_date is not None:
        daily = daily[daily["date"] >= pd.to_datetime(start_date)]

    if end_date is not None:
        daily = daily[daily["date"] <= pd.to_datetime(end_date)]

    return daily.reset_index(drop=True)


def validate_columns(daily, columns):
    missing_columns = [col for col in columns if col not in daily.columns]
    if missing_columns:
        raise ValueError(f"Missing columns: {missing_columns}")


def format_date_axis(ax):
    locator = mdates.AutoDateLocator(minticks=4, maxticks=9)
    ax.xaxis.set_major_locator(locator)
    ax.xaxis.set_major_formatter(mdates.ConciseDateFormatter(locator))


def plot_indicator_with_price(
    ax,
    daily,
    indicator_column,
    indicator_label,
    price_column,
    title,
    as_bar=False,
    zero_line=False,
):
    dates = daily["date"]

    if as_bar:
        ax.bar(
            dates,
            daily[indicator_column],
            width=1.0,
            color="#60a5fa",
            alpha=0.45,
            label=indicator_label,
        )
    else:
        ax.plot(
            dates,
            daily[indicator_column],
            color="#2563eb",
            linewidth=1.35,
            label=indicator_label,
        )

    if zero_line:
        ax.axhline(0, color="#6b7280", linewidth=0.8, linestyle="--")

    ax.set_title(title)
    ax.set_xlabel("Date")
    ax.set_ylabel(indicator_label)
    ax.grid(True, axis="y", alpha=0.25)

    price_ax = ax.twinx()
    price_ax.plot(
        dates,
        daily[price_column],
        color="#111827",
        linewidth=1.15,
        alpha=0.8,
        label=price_column,
    )
    price_ax.set_ylabel(price_column)

    handles, labels = ax.get_legend_handles_labels()
    price_handles, price_labels = price_ax.get_legend_handles_labels()
    ax.legend(
        handles + price_handles,
        labels + price_labels,
        loc="upper left",
        fontsize=8,
    )

    format_date_axis(ax)
    return price_ax


def get_chart_specs():
    return [
        {
            "column": "speculation",
            "label": "Speculation",
            "title": "Speculation and Close Price",
            "filename": "speculation_price",
            "as_bar": False,
            "zero_line": False,
        },
        {
            "column": "speculation_change_rate_pct",
            "label": "Speculation Change Rate (%)",
            "title": "Speculation Change Rate and Close Price",
            "filename": "speculation_change_rate_price",
            "as_bar": False,
            "zero_line": True,
        },
        {
            "column": "speculation_mad_score",
            "label": "Speculation MAD Score",
            "title": "Speculation MAD Score and Close Price",
            "filename": "speculation_mad_score_price",
            "as_bar": False,
            "zero_line": True,
        },
        {
            "column": "open_interest",
            "label": "Open Interest",
            "title": "Open Interest and Close Price",
            "filename": "open_interest_price",
            "as_bar": False,
            "zero_line": False,
        },
        {
            "column": "oi_change_rate",
            "label": "OI Change Rate (%)",
            "title": "OI Change Rate and Close Price",
            "filename": "oi_change_rate_price",
            "as_bar": False,
            "zero_line": True,
        },
        {
            "column": "volume",
            "label": "Volume",
            "title": "Volume and Close Price",
            "filename": "volume_price",
            "as_bar": True,
            "zero_line": False,
        },
    ]


def save_single_charts(daily, symbol, price_column, output_dir, chart_specs):
    output_paths = []

    for spec in chart_specs:
        fig, ax = plt.subplots(figsize=(14, 5))
        plot_indicator_with_price(
            ax=ax,
            daily=daily,
            indicator_column=spec["column"],
            indicator_label=spec["label"],
            price_column=price_column,
            title=f"{symbol}: {spec['title']}",
            as_bar=spec["as_bar"],
            zero_line=spec["zero_line"],
        )
        fig.tight_layout()

        output_path = os.path.join(
            output_dir,
            f"{symbol}_{spec['filename']}.png",
        )
        fig.savefig(output_path, dpi=300)
        plt.close(fig)
        output_paths.append(output_path)

    return output_paths


def save_overview_chart(daily, symbol, price_column, output_dir, chart_specs):
    fig, axes = plt.subplots(3, 2, figsize=(18, 12), sharex=True)

    for ax, spec in zip(axes.ravel(), chart_specs):
        plot_indicator_with_price(
            ax=ax,
            daily=daily,
            indicator_column=spec["column"],
            indicator_label=spec["label"],
            price_column=price_column,
            title=spec["title"],
            as_bar=spec["as_bar"],
            zero_line=spec["zero_line"],
        )

    fig.suptitle(f"{symbol}: EDA Overview", fontsize=16)
    fig.tight_layout(rect=[0, 0, 1, 0.97])

    output_path = os.path.join(output_dir, f"{symbol}_eda_overview.png")
    fig.savefig(output_path, dpi=300)
    plt.close(fig)

    return output_path


def main():
    args = parse_args()

    daily = load_daily(args.symbol)
    daily = add_eda_features(
        daily,
        change_days=args.change_days,
        mad_window=args.mad_window,
        min_mad_days=args.min_mad_days,
    )
    daily = filter_dates(
        daily,
        start_date=args.start_date,
        end_date=args.end_date,
    )

    chart_specs = get_chart_specs()
    required_columns = ["date", args.price_column]
    required_columns.extend(spec["column"] for spec in chart_specs)
    validate_columns(daily, required_columns)

    os.makedirs(args.output_dir, exist_ok=True)

    output_paths = save_single_charts(
        daily=daily,
        symbol=args.symbol,
        price_column=args.price_column,
        output_dir=args.output_dir,
        chart_specs=chart_specs,
    )
    overview_path = save_overview_chart(
        daily=daily,
        symbol=args.symbol,
        price_column=args.price_column,
        output_dir=args.output_dir,
        chart_specs=chart_specs,
    )
    output_paths.append(overview_path)

    print("EDA figures saved:")
    for output_path in output_paths:
        print(output_path)


if __name__ == "__main__":
    main()
