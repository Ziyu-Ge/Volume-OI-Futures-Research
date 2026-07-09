import argparse
import os
import re
from pathlib import Path

import numpy as np
import pandas as pd


FACTOR_DIR = Path(__file__).resolve().parent
CHAPTER_DIR = FACTOR_DIR.parent
PROJECT_ROOT = CHAPTER_DIR.parents[1]
CHAPTER_RESULTS_DIR = PROJECT_ROOT / "results" / "chapter2"
DEFAULT_HOURLY_DIR = CHAPTER_RESULTS_DIR / "tables" / "hourly"
DEFAULT_OUTPUT_DIR = CHAPTER_RESULTS_DIR / "22_high_bias_oi_drop_hourly_all_symbols"

os.environ.setdefault("MPLCONFIGDIR", str(PROJECT_ROOT / ".matplotlib"))
os.environ.setdefault("XDG_CACHE_HOME", str(PROJECT_ROOT / ".cache"))
(PROJECT_ROOT / ".matplotlib").mkdir(exist_ok=True)
(PROJECT_ROOT / ".cache").mkdir(exist_ok=True)

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt


# =========================
# 参数设置
# =========================

MA_SHORT_WINDOW = 40
MA_LONG_WINDOW = 240
MA_TREND_WINDOW = 600
MA_BIAS_SPREAD_THRESHOLD = 0.01
MA_LONG_BIAS_SPREAD_THRESHOLD = 0.16
REGRESSION_SLOPE_WINDOW = 70
VOLATILITY_WINDOW = 90
TRAILING_VOLATILITY_MULTIPLIER = 5
HOURLY_BARS_PER_YEAR = 252 * 6
BAR_FREQUENCY = "hourly"
WINDOW_UNIT = "hourly_bar"

# 本脚本读取 *_hourly.csv；所有滚动窗口单位都是“小时 bar 数”。
# 例如 ma40 表示最近 40 根小时 bar 的收盘均线，不是 40 个交易日均线。
# 遇到夜盘、午休或节假日等无交易时段时，不额外补空 bar。

MA_SHORT_COLUMN = f"ma{MA_SHORT_WINDOW}"
MA_LONG_COLUMN = f"ma{MA_LONG_WINDOW}"
MA_TREND_COLUMN = f"ma{MA_TREND_WINDOW}"
MA_SHORT_LONG_RATIO_COLUMN = f"{MA_SHORT_COLUMN}_{MA_LONG_COLUMN}_ratio"
BIAS_SHORT_LONG_COLUMN = f"bias_{MA_SHORT_WINDOW}_{MA_LONG_WINDOW}"
BIAS_SHORT_GT_LONG_COLUMN = f"bias_{MA_SHORT_COLUMN}_gt_{MA_LONG_COLUMN}"
MA_LONG_TREND_RATIO_COLUMN = f"{MA_LONG_COLUMN}_{MA_TREND_COLUMN}_ratio"
BIAS_LONG_TREND_COLUMN = f"bias_{MA_LONG_WINDOW}_{MA_TREND_WINDOW}"
BIAS_LONG_GT_TREND_COLUMN = f"bias_{MA_LONG_COLUMN}_gt_{MA_TREND_COLUMN}"
CLOSE_MA_SHORT_BIAS_COLUMN = f"close_{MA_SHORT_COLUMN}_bias"
CLOSE_MA_LONG_BIAS_COLUMN = f"close_{MA_LONG_COLUMN}_bias"
CLOSE_MA_TREND_BIAS_COLUMN = f"close_{MA_TREND_COLUMN}_bias"
AVG_PRICE_RANGE_COLUMN = f"avg_price_range_{VOLATILITY_WINDOW}"
AVG_VOLATILITY_RATE_COLUMN = f"avg_volatility_rate_{VOLATILITY_WINDOW}"

OI_REGRESSION_SLOPE_COLUMN = f"oi_regression_slope_{REGRESSION_SLOPE_WINDOW}"
OI_REGRESSION_MEAN_COLUMN = f"oi_regression_mean_{REGRESSION_SLOPE_WINDOW}"
OI_REGRESSION_SLOPE_RATE_COLUMN = (
    f"oi_regression_slope_rate_{REGRESSION_SLOPE_WINDOW}"
)
OI_REGRESSION_SLOPE_DOWN_COLUMN = (
    f"oi_regression_slope_down_{REGRESSION_SLOPE_WINDOW}"
)
CLOSE_REGRESSION_SLOPE_COLUMN = (
    f"close_regression_slope_{REGRESSION_SLOPE_WINDOW}"
)
CLOSE_REGRESSION_MEAN_COLUMN = (
    f"close_regression_mean_{REGRESSION_SLOPE_WINDOW}"
)
CLOSE_REGRESSION_SLOPE_RATE_COLUMN = (
    f"close_regression_slope_rate_{REGRESSION_SLOPE_WINDOW}"
)
CLOSE_REGRESSION_SLOPE_DOWN_COLUMN = (
    f"close_regression_slope_down_{REGRESSION_SLOPE_WINDOW}"
)


# =========================
# 策略逻辑说明
# =========================
#
# 初始状态为空仓。所有开仓信号在当前小时 bar 收盘后确认，下一小时 bar 开盘执行开空。
#
# 开空：
# 1. ma40/ma240/ma600 分别是最近 40/240/600 根小时 bar 的收盘均线；
# 2. ma240 乖离率 - ma40 乖离率 >= 1%，且 ma600 乖离率 - ma240 乖离率 >= 16%；
# 3. 最近 REGRESSION_SLOPE_WINDOW 根小时 bar 持仓量做回归线，斜率小于 0；
# 4. 最近 REGRESSION_SLOPE_WINDOW 根小时 bar 收盘价做回归线，斜率小于 0。
#
# 平空：
# 1. 收盘价高于开仓价；
# 2. 收盘价高于“开仓以来最低价 + 5 倍历史 90 个 bar 平均波动”。
#
# 两个平空条件任一触发，都会在下一小时 bar 开盘执行平空。历史 90 个 bar 平均
# 波动使用前 90 个小时 bar high-low 相对 close 的比例均值，转换成开仓以来最低价
# 上的价格距离。


def parse_factor_script_metadata(file_path):
    stem = Path(file_path).stem
    match = re.match(r"^(\d+)_?(.+)$", stem)
    if match is None:
        raise ValueError(f"factor script filename must start with an id: {stem}")

    return match.group(1), match.group(2)


factor_id, factor_name = parse_factor_script_metadata(__file__)


def parse_args():
    parser = argparse.ArgumentParser(
        description="运行全部品种的 22 号高乖离持仓回落小时因子。"
    )
    parser.add_argument(
        "--hourly-dir",
        type=Path,
        default=Path(os.environ.get("CHAPTER2_HOURLY_DIR", DEFAULT_HOURLY_DIR)),
        help=f"chapter2 小时频率缓存目录，默认：{DEFAULT_HOURLY_DIR}",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path(os.environ.get("RESULTS_OUTPUT_DIR", DEFAULT_OUTPUT_DIR)),
        help=f"因子结果目录，默认：{DEFAULT_OUTPUT_DIR}",
    )
    parser.add_argument(
        "--collect-only",
        action="store_true",
        help="不重新计算因子，只基于 output-dir 中已有 summary 生成全品种汇总。",
    )
    parser.add_argument(
        "--keep-going",
        action="store_true",
        help="单个品种失败后继续运行后续品种，并在最后汇总失败列表。",
    )
    return parser.parse_args()


def discover_symbols(hourly_dir):
    if not hourly_dir.is_dir():
        raise FileNotFoundError(
            f"小时频率数据目录不存在：{hourly_dir}。"
            "请确认 --hourly-dir 指向已生成的小时频率数据目录。"
        )

    symbols = sorted(
        path.name[: -len("_hourly.csv")].upper()
        for path in hourly_dir.glob("*_hourly.csv")
        if path.is_file()
    )
    if not symbols:
        raise FileNotFoundError(
            f"小时频率数据目录中没有 *_hourly.csv：{hourly_dir}。"
            "请确认该目录中已有可读取的小时频率缓存文件。"
        )

    return symbols


def discover_symbols_from_summaries(output_dir):
    summary_dir = output_dir / "tables" / "summary"
    if not summary_dir.is_dir():
        raise FileNotFoundError(f"summary 目录不存在：{summary_dir}")

    suffix = f"_{factor_id}_{factor_name}_summary.csv"
    symbols = sorted(
        path.name[: -len(suffix)].upper()
        for path in summary_dir.glob(f"*{suffix}")
        if path.is_file() and not path.name.startswith("all_symbols_")
    )
    if not symbols:
        raise FileNotFoundError(
            f"summary 目录中没有 {factor_id} 号因子的单品种汇总：{summary_dir}"
        )

    return symbols


def load_hourly(symbol, hourly_dir):
    hourly_path = hourly_dir / f"{symbol}_hourly.csv"
    if not hourly_path.exists():
        raise FileNotFoundError(f"未找到 {symbol} 小时频率数据：{hourly_path}")

    hourly = pd.read_csv(hourly_path)
    required_columns = {
        "date",
        "open",
        "close",
        "high",
        "low",
        "volume",
        "total_turnover",
        "open_interest",
    }
    missing_columns = required_columns - set(hourly.columns)
    if missing_columns:
        missing_text = ", ".join(sorted(missing_columns))
        raise ValueError(f"{hourly_path} 缺少必要列：{missing_text}")

    hourly["date"] = pd.to_datetime(hourly["date"])
    if "trading_date" in hourly.columns:
        hourly["trading_date"] = pd.to_datetime(hourly["trading_date"])

    numeric_columns = [
        "open",
        "close",
        "high",
        "low",
        "volume",
        "total_turnover",
        "open_interest",
    ]
    for column in numeric_columns:
        hourly[column] = pd.to_numeric(hourly[column], errors="coerce")

    return hourly.sort_values("date").reset_index(drop=True)


def positive_part(series):
    return series.clip(lower=0).fillna(0)


def linear_regression_slope(values):
    values = np.asarray(values, dtype=float)
    if np.isnan(values).any():
        return np.nan

    x = np.arange(len(values), dtype=float)
    x = x - x.mean()
    denominator = np.square(x).sum()
    if denominator == 0:
        return np.nan

    return np.dot(x, values - values.mean()) / denominator


def exit_reason_from_signals(price_above_entry_signal, trailing_rebound_signal):
    if price_above_entry_signal and trailing_rebound_signal:
        return "price_above_entry_and_trailing_rebound"
    if price_above_entry_signal:
        return "price_above_entry"
    if trailing_rebound_signal:
        return "trailing_rebound"
    return ""


def build_short_state_machine(frame):
    """根据前一根 bar 信号在当前 bar 开盘执行交易，并输出逐 bar 持仓状态。"""
    bars = frame.copy()
    bars["actual_open_short_signal"] = (
        bars["open_short_signal"].shift(1).fillna(0).astype(int)
    )

    position = 0
    entry_price = np.nan
    low_since_entry = np.nan
    pending_cover = False
    pending_price_above_entry = False
    pending_trailing_rebound = False
    pending_exit_reason = ""

    positions = []
    trade_signals = []
    trade_actions = []
    entry_prices = []
    exit_prices = []
    exit_reasons = []
    low_since_entry_values = []
    trailing_stop_distances = []
    trailing_stop_prices = []
    price_above_entry_signals = []
    trailing_rebound_signals = []
    cover_short_signals = []
    cover_signal_reasons = []
    actual_cover_short_signals = []
    actual_price_above_entry_signals = []
    actual_trailing_rebound_signals = []

    for _, row in bars.iterrows():
        open_price = row["open"]
        close = row["close"]
        low = row["low"]
        avg_volatility_rate = row[AVG_VOLATILITY_RATE_COLUMN]

        actual_open_signal = int(row["actual_open_short_signal"]) == 1
        actual_cover_signal = bool(pending_cover)
        actual_price_above_entry_signal = int(pending_price_above_entry)
        actual_trailing_rebound_signal = int(pending_trailing_rebound)
        actual_exit_reason = pending_exit_reason

        pending_cover = False
        pending_price_above_entry = False
        pending_trailing_rebound = False
        pending_exit_reason = ""

        trade_signal = 0
        trade_action = ""
        exit_reason = ""
        row_exit_price = np.nan
        opened_current_bar = False
        row_entry_price = entry_price if position == -1 else np.nan
        row_low_since_entry = low_since_entry if position == -1 else np.nan

        if position == -1 and actual_cover_signal and pd.notna(open_price):
            trade_signal = 1
            trade_action = "cover_short"
            exit_reason = actual_exit_reason or "cover_short"
            row_entry_price = entry_price
            row_low_since_entry = low_since_entry
            row_exit_price = open_price
            position = 0
            entry_price = np.nan
            low_since_entry = np.nan
        elif position == 0 and actual_open_signal and pd.notna(open_price):
            trade_signal = -1
            trade_action = "open_short"
            entry_price = open_price
            low_since_entry = open_price
            row_entry_price = entry_price
            row_low_since_entry = low_since_entry
            position = -1
            opened_current_bar = True

        price_above_entry_signal = 0
        trailing_rebound_signal = 0
        cover_short_signal = 0
        cover_signal_reason = ""
        trailing_stop_distance = np.nan
        trailing_stop_price = np.nan

        if position == -1:
            low_candidate = low
            if pd.isna(low_candidate):
                low_candidate = open_price if opened_current_bar else close
            if pd.notna(low_candidate):
                low_since_entry = min(low_since_entry, low_candidate)

            row_low_since_entry = low_since_entry
            row_entry_price = entry_price
            price_above_entry_signal = int(
                pd.notna(close)
                and pd.notna(entry_price)
                and close > entry_price
            )

            if (
                pd.notna(close)
                and pd.notna(low_since_entry)
                and pd.notna(avg_volatility_rate)
            ):
                trailing_stop_distance = (
                    low_since_entry
                    * avg_volatility_rate
                    * TRAILING_VOLATILITY_MULTIPLIER
                )
                trailing_stop_price = low_since_entry + trailing_stop_distance
                trailing_rebound_signal = int(close > trailing_stop_price)

            cover_short_signal = int(
                price_above_entry_signal or trailing_rebound_signal
            )
            cover_signal_reason = exit_reason_from_signals(
                price_above_entry_signal,
                trailing_rebound_signal,
            )
            pending_cover = bool(cover_short_signal)
            pending_price_above_entry = bool(price_above_entry_signal)
            pending_trailing_rebound = bool(trailing_rebound_signal)
            pending_exit_reason = cover_signal_reason

        positions.append(position)
        trade_signals.append(trade_signal)
        trade_actions.append(trade_action)
        entry_prices.append(row_entry_price)
        exit_prices.append(row_exit_price)
        exit_reasons.append(exit_reason)
        low_since_entry_values.append(row_low_since_entry)
        trailing_stop_distances.append(trailing_stop_distance)
        trailing_stop_prices.append(trailing_stop_price)
        price_above_entry_signals.append(price_above_entry_signal)
        trailing_rebound_signals.append(trailing_rebound_signal)
        cover_short_signals.append(cover_short_signal)
        cover_signal_reasons.append(cover_signal_reason)
        actual_cover_short_signals.append(int(actual_cover_signal))
        actual_price_above_entry_signals.append(actual_price_above_entry_signal)
        actual_trailing_rebound_signals.append(actual_trailing_rebound_signal)

    bars["actual_cover_short_signal"] = actual_cover_short_signals
    bars["position"] = positions
    bars["trade_signal"] = trade_signals
    bars["trade_action"] = trade_actions
    bars["entry_price"] = entry_prices
    bars["exit_price"] = exit_prices
    bars["exit_reason"] = exit_reasons
    bars["low_since_entry"] = low_since_entry_values
    bars["trailing_stop_distance"] = trailing_stop_distances
    bars["trailing_stop_price"] = trailing_stop_prices
    bars["price_above_entry_signal"] = price_above_entry_signals
    bars["trailing_rebound_signal"] = trailing_rebound_signals
    bars["cover_short_signal"] = cover_short_signals
    bars["cover_signal_reason"] = cover_signal_reasons
    bars["actual_price_above_entry_signal"] = actual_price_above_entry_signals
    bars["actual_trailing_rebound_signal"] = actual_trailing_rebound_signals
    bars["short_entry_signal"] = (bars["trade_signal"] == -1).astype(int)
    bars["short_exit_signal"] = (bars["trade_signal"] == 1).astype(int)

    return bars


def calculate_max_drawdown(return_series):
    equity_curve = (1 + return_series.fillna(0)).cumprod()
    if equity_curve.empty:
        return np.nan

    running_high = equity_curve.cummax()
    drawdown = equity_curve / running_high - 1
    return drawdown.min()


def annualized_sharpe(return_series):
    returns = return_series.fillna(0)
    std = returns.std(ddof=1)
    if pd.isna(std) or std == 0:
        return np.nan

    return returns.mean() / std * np.sqrt(HOURLY_BARS_PER_YEAR)


def build_signal_table(bars, feature_columns):
    base_columns = [
        "factor_id",
        "factor_name",
        "signal_date",
    ]
    if "trading_date" in bars.columns:
        base_columns.append("trading_date")
    base_columns += [
        "signal_close",
        "factor_value",
    ]
    output_columns = base_columns.copy()

    for col in feature_columns:
        if col in bars.columns and col not in output_columns:
            output_columns.append(col)

    signal_points = bars[bars["signal"] == 1].copy()
    if len(signal_points) == 0:
        return pd.DataFrame(columns=output_columns)

    signal_table = signal_points.rename(
        columns={
            "date": "signal_date",
            "close": "signal_close",
        }
    )

    return signal_table[output_columns].copy()


def build_summary_table(bars, feature_columns):
    signal_points = bars[bars["signal"] == 1].copy()

    if len(signal_points) > 0:
        first_signal_date = signal_points["date"].min()
        first_signal_close = signal_points.loc[
            signal_points["date"].idxmin(),
            "close",
        ]
    else:
        first_signal_date = pd.NaT
        first_signal_close = np.nan

    row = {
        "factor_id": factor_id,
        "factor_name": factor_name,
        "total_bars": len(bars),
        "valid_factor_bars": int(bars["factor_value"].notna().sum()),
        "signal_bars": int(bars["signal"].sum()),
        "signal_ratio": bars["signal"].mean(),
        "first_signal_date": first_signal_date,
        "first_signal_close": first_signal_close,
        "mean_factor_value": bars["factor_value"].mean(),
        "max_factor_value": bars["factor_value"].max(),
        "min_factor_value": bars["factor_value"].min(),
    }

    for col in feature_columns:
        if col not in bars.columns:
            continue
        if not pd.api.types.is_numeric_dtype(bars[col]):
            continue

        row[f"mean_{col}"] = bars[col].mean()

    return pd.DataFrame([row])


def build_plot_trade_table(bars):
    columns = [
        "status",
        "entry_date",
        "exit_date",
        "entry_price",
        "exit_price",
    ]
    if "trade_signal" not in bars.columns:
        return pd.DataFrame(columns=columns)

    plot_bars = bars.copy()
    plot_bars["trade_signal"] = (
        pd.to_numeric(plot_bars["trade_signal"], errors="coerce")
        .fillna(0)
        .astype(int)
    )

    trades = []
    open_trade = None

    for _, row in plot_bars.iterrows():
        if row["trade_signal"] == -1 and open_trade is None:
            entry_price = row.get("entry_price", np.nan)
            if pd.isna(entry_price):
                entry_price = row["open"]
            open_trade = {
                "entry_date": row["date"],
                "entry_price": entry_price,
            }
            continue

        if row["trade_signal"] == 1 and open_trade is not None:
            exit_price = row.get("exit_price", np.nan)
            if pd.isna(exit_price):
                exit_price = row["open"]
            trades.append(
                {
                    "status": "closed",
                    "entry_date": open_trade["entry_date"],
                    "exit_date": row["date"],
                    "entry_price": open_trade["entry_price"],
                    "exit_price": exit_price,
                }
            )
            open_trade = None

    if open_trade is not None and not plot_bars.empty:
        last_row = plot_bars.iloc[-1]
        trades.append(
            {
                "status": "open",
                "entry_date": open_trade["entry_date"],
                "exit_date": last_row["date"],
                "entry_price": open_trade["entry_price"],
                "exit_price": last_row["close"],
            }
        )

    return pd.DataFrame(trades, columns=columns)


def plot_signal_trades(bars, figure_path, title):
    fig, ax = plt.subplots(figsize=(12, 6))
    ax.plot(bars["date"], bars["close"], color="#1f77b4", label="close")

    if "trailing_stop_price" in bars.columns:
        trailing_line = bars["trailing_stop_price"].where(bars["position"] == -1)
        if trailing_line.notna().any():
            ax.plot(
                bars["date"],
                trailing_line,
                color="#9467bd",
                linewidth=1.0,
                alpha=0.65,
                label="trailing stop",
            )

    trades = build_plot_trade_table(bars)
    if not trades.empty:
        interval_labeled = False
        segment_labeled = False

        for _, trade in trades.iterrows():
            is_closed = trade["status"] == "closed"
            ax.axvspan(
                trade["entry_date"],
                trade["exit_date"],
                color="#f59e0b",
                alpha=0.12,
                label="short interval" if not interval_labeled else None,
            )
            interval_labeled = True

            ax.plot(
                [trade["entry_date"], trade["exit_date"]],
                [trade["entry_price"], trade["exit_price"]],
                color="#f97316",
                linewidth=1.8,
                linestyle="-" if is_closed else "--",
                alpha=0.9,
                label="entry to cover" if is_closed and not segment_labeled else None,
            )
            if is_closed:
                segment_labeled = True

        ax.scatter(
            trades["entry_date"],
            trades["entry_price"],
            marker="v",
            s=46,
            color="#d62728",
            edgecolor="white",
            linewidth=0.6,
            zorder=5,
            label="open short",
        )

        closed_trades = trades[trades["status"] == "closed"]
        if not closed_trades.empty:
            ax.scatter(
                closed_trades["exit_date"],
                closed_trades["exit_price"],
                marker="^",
                s=46,
                color="#2ca02c",
                edgecolor="white",
                linewidth=0.6,
                zorder=5,
                label="cover short",
            )

    ax.set_title(title)
    ax.set_xlabel("Date")
    ax.set_ylabel("Close Price")
    ax.legend()
    fig.tight_layout()
    fig.savefig(figure_path, dpi=300)
    plt.close(fig)


def save_factor_outputs(
    bars,
    symbol,
    factor_value_column,
    signal_column,
    feature_columns,
    output_dir,
):
    bars = bars.copy()

    bars["factor_id"] = factor_id
    bars["factor_name"] = factor_name
    bars["factor_value"] = bars[factor_value_column]
    bars["signal"] = bars[signal_column].fillna(0).astype(int)

    tables_dir = output_dir / "tables"
    factor_figures_dir = output_dir / "figures" / "factors"

    factor_output_path = (
        tables_dir / "factors" / f"{symbol}_{factor_id}_{factor_name}.csv"
    )
    signal_output_path = (
        tables_dir
        / "signals"
        / f"{symbol}_{factor_id}_{factor_name}_signals.csv"
    )
    summary_output_path = (
        tables_dir
        / "summary"
        / f"{symbol}_{factor_id}_{factor_name}_summary.csv"
    )
    price_figure_path = (
        factor_figures_dir
        / f"{symbol}_{factor_id}_{factor_name}_signal_on_price.png"
    )

    base_columns = [
        "date",
        "trading_date",
        "open",
        "close",
        "high",
        "low",
        "volume",
        "total_turnover",
        "open_interest",
        "speculation",
        "threshold",
        "factor_id",
        "factor_name",
        "factor_value",
    ]

    output_columns = []
    for col in base_columns + feature_columns + ["signal"]:
        if col in bars.columns and col not in output_columns:
            output_columns.append(col)

    factor_bar = bars[output_columns].copy()
    signal_table = build_signal_table(bars, feature_columns)
    summary_table = build_summary_table(bars, feature_columns)
    summary_table.insert(0, "symbol", symbol)

    factor_output_path.parent.mkdir(parents=True, exist_ok=True)
    signal_output_path.parent.mkdir(parents=True, exist_ok=True)
    summary_output_path.parent.mkdir(parents=True, exist_ok=True)
    price_figure_path.parent.mkdir(parents=True, exist_ok=True)

    factor_bar.to_csv(factor_output_path, index=False)
    signal_table.to_csv(signal_output_path, index=False)
    summary_table.to_csv(summary_output_path, index=False)

    plot_signal_trades(
        bars,
        price_figure_path,
        f"{symbol} Factor {factor_id}: {factor_name}",
    )

    print(f"[{symbol}] factor {factor_id}: {factor_name} complete.", flush=True)
    print(f"[{symbol}] factor bar table: {factor_output_path}", flush=True)
    print(f"[{symbol}] signal table: {signal_output_path}", flush=True)
    print(f"[{symbol}] summary table: {summary_output_path}", flush=True)
    print(f"[{symbol}] signal bars: {int(bars['signal'].sum())}", flush=True)

    return {
        "factor_bar": factor_bar,
        "signal_table": signal_table,
        "summary_table": summary_table,
        "factor_output_path": factor_output_path,
        "signal_output_path": signal_output_path,
        "summary_output_path": summary_output_path,
        "price_figure_path": price_figure_path,
    }


def calculate_symbol(symbol, hourly_dir, output_dir):
    bars = load_hourly(symbol, hourly_dir)

    bars[MA_SHORT_COLUMN] = (
        bars["close"]
        .rolling(window=MA_SHORT_WINDOW, min_periods=MA_SHORT_WINDOW)
        .mean()
    )
    bars[MA_LONG_COLUMN] = (
        bars["close"]
        .rolling(window=MA_LONG_WINDOW, min_periods=MA_LONG_WINDOW)
        .mean()
    )
    bars[MA_TREND_COLUMN] = (
        bars["close"]
        .rolling(window=MA_TREND_WINDOW, min_periods=MA_TREND_WINDOW)
        .mean()
    )
    bars[MA_SHORT_LONG_RATIO_COLUMN] = (
        bars[MA_SHORT_COLUMN] / bars[MA_LONG_COLUMN]
    )
    bars[BIAS_SHORT_LONG_COLUMN] = bars[MA_SHORT_LONG_RATIO_COLUMN] - 1
    bars[BIAS_SHORT_GT_LONG_COLUMN] = (
        bars[MA_SHORT_COLUMN] > bars[MA_LONG_COLUMN]
    ).astype(int)
    bars[MA_LONG_TREND_RATIO_COLUMN] = (
        bars[MA_LONG_COLUMN] / bars[MA_TREND_COLUMN]
    )
    bars[BIAS_LONG_TREND_COLUMN] = bars[MA_LONG_TREND_RATIO_COLUMN] - 1
    bars[BIAS_LONG_GT_TREND_COLUMN] = (
        bars[MA_LONG_COLUMN] > bars[MA_TREND_COLUMN]
    ).astype(int)
    bars[CLOSE_MA_SHORT_BIAS_COLUMN] = (
        bars["close"] / bars[MA_SHORT_COLUMN] - 1
    )
    bars[CLOSE_MA_LONG_BIAS_COLUMN] = (
        bars["close"] / bars[MA_LONG_COLUMN] - 1
    )
    bars[CLOSE_MA_TREND_BIAS_COLUMN] = (
        bars["close"] / bars[MA_TREND_COLUMN] - 1
    )
    bars["ma_bias_spread"] = (
        bars[CLOSE_MA_LONG_BIAS_COLUMN] - bars[CLOSE_MA_SHORT_BIAS_COLUMN]
    )
    bars["ma_bias_spread_signal"] = (
        bars["ma_bias_spread"] >= MA_BIAS_SPREAD_THRESHOLD
    ).astype(int)
    bars["ma_long_bias_spread"] = (
        bars[CLOSE_MA_TREND_BIAS_COLUMN] - bars[CLOSE_MA_LONG_BIAS_COLUMN]
    )
    bars["ma_long_bias_spread_signal"] = (
        bars["ma_long_bias_spread"] >= MA_LONG_BIAS_SPREAD_THRESHOLD
    ).astype(int)

    bars[OI_REGRESSION_SLOPE_COLUMN] = (
        bars["open_interest"]
        .rolling(
            window=REGRESSION_SLOPE_WINDOW,
            min_periods=REGRESSION_SLOPE_WINDOW,
        )
        .apply(linear_regression_slope, raw=True)
    )
    bars[OI_REGRESSION_MEAN_COLUMN] = (
        bars["open_interest"]
        .rolling(
            window=REGRESSION_SLOPE_WINDOW,
            min_periods=REGRESSION_SLOPE_WINDOW,
        )
        .mean()
    )
    bars[OI_REGRESSION_SLOPE_RATE_COLUMN] = (
        bars[OI_REGRESSION_SLOPE_COLUMN]
        / bars[OI_REGRESSION_MEAN_COLUMN].replace(0, np.nan)
    )
    bars[OI_REGRESSION_SLOPE_DOWN_COLUMN] = (
        bars[OI_REGRESSION_SLOPE_COLUMN] < 0
    ).astype(int)

    bars[CLOSE_REGRESSION_SLOPE_COLUMN] = (
        bars["close"]
        .rolling(
            window=REGRESSION_SLOPE_WINDOW,
            min_periods=REGRESSION_SLOPE_WINDOW,
        )
        .apply(linear_regression_slope, raw=True)
    )
    bars[CLOSE_REGRESSION_MEAN_COLUMN] = (
        bars["close"]
        .rolling(
            window=REGRESSION_SLOPE_WINDOW,
            min_periods=REGRESSION_SLOPE_WINDOW,
        )
        .mean()
    )
    bars[CLOSE_REGRESSION_SLOPE_RATE_COLUMN] = (
        bars[CLOSE_REGRESSION_SLOPE_COLUMN]
        / bars[CLOSE_REGRESSION_MEAN_COLUMN].replace(0, np.nan)
    )
    bars[CLOSE_REGRESSION_SLOPE_DOWN_COLUMN] = (
        bars[CLOSE_REGRESSION_SLOPE_COLUMN] < 0
    ).astype(int)

    bars["bar_return"] = bars["close"].pct_change()
    bars["gap_return"] = (
        bars["open"] / bars["close"].shift(1).replace(0, np.nan) - 1
    )
    bars["intrabar_return"] = (
        bars["close"] / bars["open"].replace(0, np.nan) - 1
    )
    bars["price_range"] = bars["high"] - bars["low"]
    bars["price_range_rate"] = bars["price_range"] / bars["close"].replace(
        0,
        np.nan,
    )
    bars[AVG_PRICE_RANGE_COLUMN] = (
        bars["price_range"]
        .shift(1)
        .rolling(window=VOLATILITY_WINDOW, min_periods=VOLATILITY_WINDOW)
        .mean()
    )
    bars[AVG_VOLATILITY_RATE_COLUMN] = (
        bars["price_range_rate"]
        .shift(1)
        .rolling(window=VOLATILITY_WINDOW, min_periods=VOLATILITY_WINDOW)
        .mean()
    )

    bars["open_short_signal"] = (
        (bars["ma_bias_spread_signal"] == 1)
        & (bars["ma_long_bias_spread_signal"] == 1)
        & (bars[OI_REGRESSION_SLOPE_DOWN_COLUMN] == 1)
        & (bars[CLOSE_REGRESSION_SLOPE_DOWN_COLUMN] == 1)
    ).astype(int)

    bars["high_bias_oi_drop_score"] = (
        positive_part(bars[BIAS_SHORT_LONG_COLUMN])
        + positive_part(bars["ma_bias_spread"])
        + positive_part(bars["ma_long_bias_spread"])
        + positive_part(-bars[OI_REGRESSION_SLOPE_RATE_COLUMN])
        + positive_part(-bars[CLOSE_REGRESSION_SLOPE_RATE_COLUMN])
    )

    bars = build_short_state_machine(bars)
    bars["strategy_bar_return"] = (
        bars["position"].shift(1).fillna(0)
        * bars["gap_return"].fillna(0)
        + bars["position"].fillna(0) * bars["intrabar_return"].fillna(0)
    )
    bars["strategy_cumulative_return"] = (
        (1 + bars["strategy_bar_return"]).cumprod() - 1
    )

    feature_columns = [
        "bar_return",
        "gap_return",
        "intrabar_return",
        MA_SHORT_COLUMN,
        MA_LONG_COLUMN,
        MA_TREND_COLUMN,
        MA_SHORT_LONG_RATIO_COLUMN,
        BIAS_SHORT_LONG_COLUMN,
        BIAS_SHORT_GT_LONG_COLUMN,
        MA_LONG_TREND_RATIO_COLUMN,
        BIAS_LONG_TREND_COLUMN,
        BIAS_LONG_GT_TREND_COLUMN,
        CLOSE_MA_SHORT_BIAS_COLUMN,
        CLOSE_MA_LONG_BIAS_COLUMN,
        CLOSE_MA_TREND_BIAS_COLUMN,
        "ma_bias_spread",
        "ma_bias_spread_signal",
        "ma_long_bias_spread",
        "ma_long_bias_spread_signal",
        OI_REGRESSION_SLOPE_COLUMN,
        OI_REGRESSION_MEAN_COLUMN,
        OI_REGRESSION_SLOPE_RATE_COLUMN,
        OI_REGRESSION_SLOPE_DOWN_COLUMN,
        CLOSE_REGRESSION_SLOPE_COLUMN,
        CLOSE_REGRESSION_MEAN_COLUMN,
        CLOSE_REGRESSION_SLOPE_RATE_COLUMN,
        CLOSE_REGRESSION_SLOPE_DOWN_COLUMN,
        "price_range",
        "price_range_rate",
        AVG_PRICE_RANGE_COLUMN,
        AVG_VOLATILITY_RATE_COLUMN,
        "open_short_signal",
        "actual_open_short_signal",
        "cover_short_signal",
        "cover_signal_reason",
        "actual_cover_short_signal",
        "price_above_entry_signal",
        "trailing_rebound_signal",
        "actual_price_above_entry_signal",
        "actual_trailing_rebound_signal",
        "position",
        "trade_signal",
        "trade_action",
        "entry_price",
        "exit_price",
        "low_since_entry",
        "trailing_stop_distance",
        "trailing_stop_price",
        "exit_reason",
        "short_entry_signal",
        "short_exit_signal",
        "strategy_bar_return",
        "strategy_cumulative_return",
        "high_bias_oi_drop_score",
    ]

    result = save_factor_outputs(
        bars=bars,
        symbol=symbol,
        factor_value_column="high_bias_oi_drop_score",
        signal_column="short_entry_signal",
        feature_columns=feature_columns,
        output_dir=output_dir,
    )

    summary_table = result["summary_table"].copy()
    summary_table["bar_frequency"] = BAR_FREQUENCY
    summary_table["window_unit"] = WINDOW_UNIT
    summary_table["ma_short_window"] = MA_SHORT_WINDOW
    summary_table["ma_long_window"] = MA_LONG_WINDOW
    summary_table["ma_trend_window"] = MA_TREND_WINDOW
    summary_table["ma_bias_spread_threshold"] = MA_BIAS_SPREAD_THRESHOLD
    summary_table["ma_long_bias_spread_threshold"] = (
        MA_LONG_BIAS_SPREAD_THRESHOLD
    )
    summary_table["regression_slope_window"] = REGRESSION_SLOPE_WINDOW
    summary_table["volatility_window"] = VOLATILITY_WINDOW
    summary_table["trailing_volatility_multiplier"] = (
        TRAILING_VOLATILITY_MULTIPLIER
    )
    summary_table["raw_open_short_signal_bars"] = int(
        bars["open_short_signal"].sum()
    )
    summary_table["raw_cover_short_signal_bars"] = int(
        bars["cover_short_signal"].sum()
    )
    summary_table["short_entry_count"] = int(bars["short_entry_signal"].sum())
    summary_table["short_exit_count"] = int(bars["short_exit_signal"].sum())
    summary_table["price_above_entry_exit_count"] = int(
        bars["exit_reason"].str.contains("price_above_entry", na=False).sum()
    )
    summary_table["trailing_rebound_exit_count"] = int(
        bars["exit_reason"].str.contains("trailing_rebound", na=False).sum()
    )
    summary_table["combined_exit_count"] = int(
        (
            bars["exit_reason"]
            == "price_above_entry_and_trailing_rebound"
        ).sum()
    )
    summary_table["final_position"] = int(bars["position"].iloc[-1])
    summary_table["mean_strategy_bar_return"] = bars[
        "strategy_bar_return"
    ].mean()
    summary_table["strategy_cumulative_return"] = bars[
        "strategy_cumulative_return"
    ].iloc[-1]
    summary_table["strategy_max_drawdown"] = calculate_max_drawdown(
        bars["strategy_bar_return"]
    )
    summary_table["strategy_sharpe"] = annualized_sharpe(
        bars["strategy_bar_return"]
    )
    summary_table.to_csv(result["summary_output_path"], index=False)

    return summary_table


def read_factor_summary(symbol, output_dir):
    summary_path = (
        output_dir
        / "tables"
        / "summary"
        / f"{symbol}_{factor_id}_{factor_name}_summary.csv"
    )
    if not summary_path.exists():
        return None

    summary = pd.read_csv(summary_path)
    if "symbol" not in summary.columns:
        summary.insert(0, "symbol", symbol)
    return summary


def save_combined_summaries(symbols, output_dir):
    summaries = []
    for symbol in symbols:
        summary = read_factor_summary(symbol, output_dir)
        if summary is not None:
            summaries.append(summary)

    if not summaries:
        raise FileNotFoundError(f"没有找到可汇总的 {factor_id} 号因子 summary。")

    combined_summary = pd.concat(summaries, ignore_index=True)
    combined_summary = combined_summary.sort_values("symbol")

    summary_dir = output_dir / "tables" / "summary"
    summary_dir.mkdir(parents=True, exist_ok=True)

    summary_path = summary_dir / f"all_symbols_{factor_id}_{factor_name}_summary.csv"
    combined_summary.to_csv(summary_path, index=False)

    return {
        "summary_path": summary_path,
        "summary_count": len(summaries),
    }


def main():
    args = parse_args()
    hourly_dir = args.hourly_dir.resolve()
    output_dir = args.output_dir.resolve()
    failures = []

    if args.collect_only:
        symbols = discover_symbols_from_summaries(output_dir)
    else:
        symbols = discover_symbols(hourly_dir)

    output_dir.mkdir(parents=True, exist_ok=True)

    print(f"chapter2 小时频率目录：{hourly_dir}", flush=True)
    print(f"本次运行品种数量：{len(symbols)}", flush=True)
    print(f"品种列表：{','.join(symbols)}", flush=True)
    print(f"因子结果目录：{output_dir}", flush=True)

    if not args.collect_only:
        for symbol in symbols:
            print(f"\n###### 开始处理品种：{symbol} ######", flush=True)
            try:
                calculate_symbol(symbol, hourly_dir, output_dir)
            except Exception as exc:
                if not args.keep_going:
                    raise

                failures.append((symbol, exc))
                print(f"\n!!!!!! 品种 {symbol} 处理失败：{exc} !!!!!!", flush=True)

    summary_info = save_combined_summaries(symbols, output_dir)

    print("\n全部因子任务完成。", flush=True)
    print(f"成功处理品种数量：{len(symbols) - len(failures)}", flush=True)
    print(f"汇总品种数量：{summary_info['summary_count']}", flush=True)
    print(f"汇总表：{summary_info['summary_path']}", flush=True)

    if failures:
        print(f"\n失败品种数量：{len(failures)}", flush=True)
        for symbol, exc in failures:
            print(f"- {symbol}: {exc}", flush=True)
        raise SystemExit(1)


if __name__ == "__main__":
    main()
