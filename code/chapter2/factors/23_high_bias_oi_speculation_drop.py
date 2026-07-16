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
DEFAULT_DAILY_DIR = CHAPTER_RESULTS_DIR / "tables" / "daily"
DEFAULT_OUTPUT_DIR = (
    CHAPTER_RESULTS_DIR / "23_high_bias_oi_speculation_drop_all_symbols"
)

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

MA_SHORT_WINDOW = 5
MA_LONG_WINDOW = 20
MA_TREND_WINDOW = 60
MA_BIAS_SPREAD_THRESHOLD = 0.04
MA_LONG_BIAS_SPREAD_THRESHOLD = 0.10
MA_LONG_BIAS_SPREAD_CAP_THRESHOLD = 0.18
OI_REGRESSION_SLOPE_WINDOW = 5
CLOSE_REGRESSION_SLOPE_WINDOW = 7
SPECULATION_REGRESSION_SLOPE_WINDOW = 5
SPECULATION_REGRESSION_SLOPE_THRESHOLD = -0.01
VOLATILITY_WINDOW = 10
TRAILING_VOLATILITY_MULTIPLIER = 4
TRADING_DAYS_PER_YEAR = 252

OI_REGRESSION_SLOPE_COLUMN = f"oi_regression_slope_{OI_REGRESSION_SLOPE_WINDOW}"
OI_REGRESSION_MEAN_COLUMN = f"oi_regression_mean_{OI_REGRESSION_SLOPE_WINDOW}"
OI_REGRESSION_SLOPE_RATE_COLUMN = (
    f"oi_regression_slope_rate_{OI_REGRESSION_SLOPE_WINDOW}"
)
OI_REGRESSION_SLOPE_DOWN_COLUMN = (
    f"oi_regression_slope_down_{OI_REGRESSION_SLOPE_WINDOW}"
)
CLOSE_REGRESSION_SLOPE_COLUMN = (
    f"close_regression_slope_{CLOSE_REGRESSION_SLOPE_WINDOW}"
)
CLOSE_REGRESSION_MEAN_COLUMN = (
    f"close_regression_mean_{CLOSE_REGRESSION_SLOPE_WINDOW}"
)
CLOSE_REGRESSION_SLOPE_RATE_COLUMN = (
    f"close_regression_slope_rate_{CLOSE_REGRESSION_SLOPE_WINDOW}"
)
CLOSE_REGRESSION_SLOPE_DOWN_COLUMN = (
    f"close_regression_slope_down_{CLOSE_REGRESSION_SLOPE_WINDOW}"
)
SPECULATION_REGRESSION_SLOPE_COLUMN = (
    f"speculation_regression_slope_{SPECULATION_REGRESSION_SLOPE_WINDOW}"
)
SPECULATION_REGRESSION_SLOPE_DOWN_COLUMN = (
    f"speculation_regression_slope_down_{SPECULATION_REGRESSION_SLOPE_WINDOW}"
)
SPECULATION_REGRESSION_SLOPE_SIGNAL_COLUMN = (
    f"speculation_regression_slope_signal_{SPECULATION_REGRESSION_SLOPE_WINDOW}"
)


# =========================
# 策略逻辑说明
# =========================
#
# 初始状态为空仓。所有开仓信号在当天收盘后确认，下一交易日开盘执行开空。
#
# 开空：
# 1. ma20 乖离率 - ma5 乖离率 >= 4%；
# 2. ma60 乖离率 - ma20 乖离率在 [10%, 18%] 内，避免追空过度延伸的强趋势；
# 3. 最近 OI_REGRESSION_SLOPE_WINDOW 个交易日持仓量做回归线，斜率小于 0；
# 4. 最近 CLOSE_REGRESSION_SLOPE_WINDOW 个交易日收盘价做回归线，斜率小于 0；
# 5. 最近 SPECULATION_REGRESSION_SLOPE_WINDOW 个交易日投机度做回归线，斜率 <= -0.01。
#
# 平空：
# 1. 收盘价高于开仓价；
# 2. 收盘价高于“开仓以来最低价 + 4 倍历史 10 日平均波动”。
#
# 两个平空条件任一触发，都会在下一交易日开盘执行平空。历史 10 日平均
# 波动使用前 10 日 high-low 相对 close 的比例均值，转换成开仓以来最低价
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
        description="运行全部品种的 23 号高乖离持仓投机度回落因子。"
    )
    parser.add_argument(
        "--daily-dir",
        type=Path,
        default=Path(os.environ.get("CHAPTER2_DAILY_DIR", DEFAULT_DAILY_DIR)),
        help=f"chapter2 日频缓存目录，默认：{DEFAULT_DAILY_DIR}",
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


def discover_symbols(daily_dir):
    if not daily_dir.is_dir():
        raise FileNotFoundError(
            f"日频数据目录不存在：{daily_dir}。"
            "请确认 --daily-dir 指向已生成的日频数据目录。"
        )

    symbols = sorted(
        path.name[: -len("_daily.csv")].upper()
        for path in daily_dir.glob("*_daily.csv")
        if path.is_file()
    )
    if not symbols:
        raise FileNotFoundError(
            f"日频数据目录中没有 *_daily.csv：{daily_dir}。"
            "请确认该目录中已有可读取的日频缓存文件。"
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


def load_daily(symbol, daily_dir):
    daily_path = daily_dir / f"{symbol}_daily.csv"
    if not daily_path.exists():
        raise FileNotFoundError(f"未找到 {symbol} 日频数据：{daily_path}")

    daily = pd.read_csv(daily_path)
    required_columns = {
        "date",
        "open",
        "close",
        "high",
        "low",
        "volume",
        "total_turnover",
        "open_interest",
        "speculation",
    }
    missing_columns = required_columns - set(daily.columns)
    if missing_columns:
        missing_text = ", ".join(sorted(missing_columns))
        raise ValueError(f"{daily_path} 缺少必要列：{missing_text}")

    daily["date"] = pd.to_datetime(daily["date"])
    numeric_columns = [
        "open",
        "close",
        "high",
        "low",
        "volume",
        "total_turnover",
        "open_interest",
        "speculation",
    ]
    for column in numeric_columns:
        daily[column] = pd.to_numeric(daily[column], errors="coerce")

    return daily.sort_values("date").reset_index(drop=True)


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
    """根据前一日信号在当日开盘执行交易，并输出每日持仓状态。"""
    daily = frame.copy()
    daily["actual_open_short_signal"] = (
        daily["open_short_signal"].shift(1).fillna(0).astype(int)
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

    for _, row in daily.iterrows():
        open_price = row["open"]
        close = row["close"]
        low = row["low"]
        avg_volatility_rate = row["avg_volatility_rate_10"]

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
        opened_today = False
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
            opened_today = True

        price_above_entry_signal = 0
        trailing_rebound_signal = 0
        cover_short_signal = 0
        cover_signal_reason = ""
        trailing_stop_distance = np.nan
        trailing_stop_price = np.nan

        if position == -1:
            low_candidate = low
            if pd.isna(low_candidate):
                low_candidate = open_price if opened_today else close
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

    daily["actual_cover_short_signal"] = actual_cover_short_signals
    daily["position"] = positions
    daily["trade_signal"] = trade_signals
    daily["trade_action"] = trade_actions
    daily["entry_price"] = entry_prices
    daily["exit_price"] = exit_prices
    daily["exit_reason"] = exit_reasons
    daily["low_since_entry"] = low_since_entry_values
    daily["trailing_stop_distance"] = trailing_stop_distances
    daily["trailing_stop_price"] = trailing_stop_prices
    daily["price_above_entry_signal"] = price_above_entry_signals
    daily["trailing_rebound_signal"] = trailing_rebound_signals
    daily["cover_short_signal"] = cover_short_signals
    daily["cover_signal_reason"] = cover_signal_reasons
    daily["actual_price_above_entry_signal"] = actual_price_above_entry_signals
    daily["actual_trailing_rebound_signal"] = actual_trailing_rebound_signals
    daily["short_entry_signal"] = (daily["trade_signal"] == -1).astype(int)
    daily["short_exit_signal"] = (daily["trade_signal"] == 1).astype(int)

    return daily


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

    return returns.mean() / std * np.sqrt(TRADING_DAYS_PER_YEAR)


def build_signal_table(daily, feature_columns):
    base_columns = [
        "factor_id",
        "factor_name",
        "signal_date",
        "signal_close",
        "factor_value",
    ]
    output_columns = base_columns.copy()

    for col in feature_columns:
        if col in daily.columns and col not in output_columns:
            output_columns.append(col)

    signal_points = daily[daily["signal"] == 1].copy()
    if len(signal_points) == 0:
        return pd.DataFrame(columns=output_columns)

    signal_table = signal_points.rename(
        columns={
            "date": "signal_date",
            "close": "signal_close",
        }
    )

    return signal_table[output_columns].copy()


def build_summary_table(daily, feature_columns):
    signal_points = daily[daily["signal"] == 1].copy()

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
        "total_days": len(daily),
        "valid_factor_days": int(daily["factor_value"].notna().sum()),
        "signal_days": int(daily["signal"].sum()),
        "signal_ratio": daily["signal"].mean(),
        "first_signal_date": first_signal_date,
        "first_signal_close": first_signal_close,
        "mean_factor_value": daily["factor_value"].mean(),
        "max_factor_value": daily["factor_value"].max(),
        "min_factor_value": daily["factor_value"].min(),
    }

    for col in feature_columns:
        if col not in daily.columns:
            continue
        if not pd.api.types.is_numeric_dtype(daily[col]):
            continue

        row[f"mean_{col}"] = daily[col].mean()

    return pd.DataFrame([row])


def build_plot_trade_table(daily):
    columns = [
        "status",
        "entry_date",
        "exit_date",
        "entry_price",
        "exit_price",
    ]
    if "trade_signal" not in daily.columns:
        return pd.DataFrame(columns=columns)

    plot_daily = daily.copy()
    plot_daily["trade_signal"] = (
        pd.to_numeric(plot_daily["trade_signal"], errors="coerce")
        .fillna(0)
        .astype(int)
    )

    trades = []
    open_trade = None

    for _, row in plot_daily.iterrows():
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

    if open_trade is not None and not plot_daily.empty:
        last_row = plot_daily.iloc[-1]
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


def plot_signal_trades(daily, figure_path, title):
    fig, ax = plt.subplots(figsize=(12, 6))
    ax.plot(daily["date"], daily["close"], color="#1f77b4", label="close")

    if "trailing_stop_price" in daily.columns:
        trailing_line = daily["trailing_stop_price"].where(daily["position"] == -1)
        if trailing_line.notna().any():
            ax.plot(
                daily["date"],
                trailing_line,
                color="#9467bd",
                linewidth=1.0,
                alpha=0.65,
                label="trailing stop",
            )

    trades = build_plot_trade_table(daily)
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
    daily,
    symbol,
    factor_value_column,
    signal_column,
    feature_columns,
    output_dir,
):
    daily = daily.copy()

    daily["factor_id"] = factor_id
    daily["factor_name"] = factor_name
    daily["factor_value"] = daily[factor_value_column]
    daily["signal"] = daily[signal_column].fillna(0).astype(int)

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
        if col in daily.columns and col not in output_columns:
            output_columns.append(col)

    factor_daily = daily[output_columns].copy()
    signal_table = build_signal_table(daily, feature_columns)
    summary_table = build_summary_table(daily, feature_columns)
    summary_table.insert(0, "symbol", symbol)

    factor_output_path.parent.mkdir(parents=True, exist_ok=True)
    signal_output_path.parent.mkdir(parents=True, exist_ok=True)
    summary_output_path.parent.mkdir(parents=True, exist_ok=True)
    price_figure_path.parent.mkdir(parents=True, exist_ok=True)

    factor_daily.to_csv(factor_output_path, index=False)
    signal_table.to_csv(signal_output_path, index=False)
    summary_table.to_csv(summary_output_path, index=False)

    plot_signal_trades(
        daily,
        price_figure_path,
        f"{symbol} Factor {factor_id}: {factor_name}",
    )

    print(f"[{symbol}] factor {factor_id}: {factor_name} complete.", flush=True)
    print(f"[{symbol}] factor daily table: {factor_output_path}", flush=True)
    print(f"[{symbol}] signal table: {signal_output_path}", flush=True)
    print(f"[{symbol}] summary table: {summary_output_path}", flush=True)
    print(f"[{symbol}] signal days: {int(daily['signal'].sum())}", flush=True)

    return {
        "factor_daily": factor_daily,
        "signal_table": signal_table,
        "summary_table": summary_table,
        "factor_output_path": factor_output_path,
        "signal_output_path": signal_output_path,
        "summary_output_path": summary_output_path,
        "price_figure_path": price_figure_path,
    }


def calculate_symbol(symbol, daily_dir, output_dir):
    daily = load_daily(symbol, daily_dir)

    daily["ma5"] = (
        daily["close"]
        .rolling(window=MA_SHORT_WINDOW, min_periods=MA_SHORT_WINDOW)
        .mean()
    )
    daily["ma20"] = (
        daily["close"]
        .rolling(window=MA_LONG_WINDOW, min_periods=MA_LONG_WINDOW)
        .mean()
    )
    daily["ma60"] = (
        daily["close"]
        .rolling(window=MA_TREND_WINDOW, min_periods=MA_TREND_WINDOW)
        .mean()
    )
    daily["ma5_ma20_ratio"] = daily["ma5"] / daily["ma20"]
    daily["bias_5_20"] = daily["ma5_ma20_ratio"] - 1
    daily["bias_ma5_gt_ma20"] = (daily["ma5"] > daily["ma20"]).astype(int)
    daily["ma20_ma60_ratio"] = daily["ma20"] / daily["ma60"]
    daily["bias_20_60"] = daily["ma20_ma60_ratio"] - 1
    daily["bias_ma20_gt_ma60"] = (daily["ma20"] > daily["ma60"]).astype(int)
    daily["close_ma5_bias"] = daily["close"] / daily["ma5"] - 1
    daily["close_ma20_bias"] = daily["close"] / daily["ma20"] - 1
    daily["close_ma60_bias"] = daily["close"] / daily["ma60"] - 1
    daily["ma_bias_spread"] = (
        daily["close_ma20_bias"] - daily["close_ma5_bias"]
    )
    daily["ma_bias_spread_signal"] = (
        daily["ma_bias_spread"] >= MA_BIAS_SPREAD_THRESHOLD
    ).astype(int)
    daily["ma_long_bias_spread"] = (
        daily["close_ma60_bias"] - daily["close_ma20_bias"]
    )
    daily["ma_long_bias_spread_signal"] = (
        daily["ma_long_bias_spread"] >= MA_LONG_BIAS_SPREAD_THRESHOLD
    ).astype(int)
    daily["ma_long_bias_spread_cap_signal"] = (
        daily["ma_long_bias_spread"] <= MA_LONG_BIAS_SPREAD_CAP_THRESHOLD
    ).astype(int)

    daily[OI_REGRESSION_SLOPE_COLUMN] = (
        daily["open_interest"]
        .rolling(
            window=OI_REGRESSION_SLOPE_WINDOW,
            min_periods=OI_REGRESSION_SLOPE_WINDOW,
        )
        .apply(linear_regression_slope, raw=True)
    )
    daily[OI_REGRESSION_MEAN_COLUMN] = (
        daily["open_interest"]
        .rolling(
            window=OI_REGRESSION_SLOPE_WINDOW,
            min_periods=OI_REGRESSION_SLOPE_WINDOW,
        )
        .mean()
    )
    daily[OI_REGRESSION_SLOPE_RATE_COLUMN] = (
        daily[OI_REGRESSION_SLOPE_COLUMN]
        / daily[OI_REGRESSION_MEAN_COLUMN].replace(0, np.nan)
    )
    daily[OI_REGRESSION_SLOPE_DOWN_COLUMN] = (
        daily[OI_REGRESSION_SLOPE_COLUMN] < 0
    ).astype(int)

    daily[CLOSE_REGRESSION_SLOPE_COLUMN] = (
        daily["close"]
        .rolling(
            window=CLOSE_REGRESSION_SLOPE_WINDOW,
            min_periods=CLOSE_REGRESSION_SLOPE_WINDOW,
        )
        .apply(linear_regression_slope, raw=True)
    )
    daily[CLOSE_REGRESSION_MEAN_COLUMN] = (
        daily["close"]
        .rolling(
            window=CLOSE_REGRESSION_SLOPE_WINDOW,
            min_periods=CLOSE_REGRESSION_SLOPE_WINDOW,
        )
        .mean()
    )
    daily[CLOSE_REGRESSION_SLOPE_RATE_COLUMN] = (
        daily[CLOSE_REGRESSION_SLOPE_COLUMN]
        / daily[CLOSE_REGRESSION_MEAN_COLUMN].replace(0, np.nan)
    )
    daily[CLOSE_REGRESSION_SLOPE_DOWN_COLUMN] = (
        daily[CLOSE_REGRESSION_SLOPE_COLUMN] < 0
    ).astype(int)

    daily[SPECULATION_REGRESSION_SLOPE_COLUMN] = (
        daily["speculation"]
        .rolling(
            window=SPECULATION_REGRESSION_SLOPE_WINDOW,
            min_periods=SPECULATION_REGRESSION_SLOPE_WINDOW,
        )
        .apply(linear_regression_slope, raw=True)
    )
    daily[SPECULATION_REGRESSION_SLOPE_DOWN_COLUMN] = (
        daily[SPECULATION_REGRESSION_SLOPE_COLUMN] < 0
    ).astype(int)
    daily[SPECULATION_REGRESSION_SLOPE_SIGNAL_COLUMN] = (
        daily[SPECULATION_REGRESSION_SLOPE_COLUMN]
        <= SPECULATION_REGRESSION_SLOPE_THRESHOLD
    ).astype(int)

    daily["daily_return"] = daily["close"].pct_change()
    daily["overnight_return"] = (
        daily["open"] / daily["close"].shift(1).replace(0, np.nan) - 1
    )
    daily["intraday_return"] = (
        daily["close"] / daily["open"].replace(0, np.nan) - 1
    )
    daily["price_range"] = daily["high"] - daily["low"]
    daily["price_range_rate"] = daily["price_range"] / daily["close"].replace(
        0,
        np.nan,
    )
    daily["avg_price_range_10"] = (
        daily["price_range"]
        .shift(1)
        .rolling(window=VOLATILITY_WINDOW, min_periods=VOLATILITY_WINDOW)
        .mean()
    )
    daily["avg_volatility_rate_10"] = (
        daily["price_range_rate"]
        .shift(1)
        .rolling(window=VOLATILITY_WINDOW, min_periods=VOLATILITY_WINDOW)
        .mean()
    )

    daily["open_short_signal"] = (
        (daily["ma_bias_spread_signal"] == 1)
        & (daily["ma_long_bias_spread_signal"] == 1)
        & (daily["ma_long_bias_spread_cap_signal"] == 1)
        & (daily[OI_REGRESSION_SLOPE_DOWN_COLUMN] == 1)
        & (daily[CLOSE_REGRESSION_SLOPE_DOWN_COLUMN] == 1)
        & (daily[SPECULATION_REGRESSION_SLOPE_SIGNAL_COLUMN] == 1)
    ).astype(int)

    daily["high_bias_oi_speculation_drop_score"] = (
        positive_part(daily["bias_5_20"])
        + positive_part(daily["ma_bias_spread"])
        + positive_part(daily["ma_long_bias_spread"])
        + positive_part(-daily[OI_REGRESSION_SLOPE_RATE_COLUMN])
        + positive_part(-daily[CLOSE_REGRESSION_SLOPE_RATE_COLUMN])
        + positive_part(-daily[SPECULATION_REGRESSION_SLOPE_COLUMN])
    )

    daily = build_short_state_machine(daily)
    daily["strategy_daily_return"] = (
        daily["position"].shift(1).fillna(0)
        * daily["overnight_return"].fillna(0)
        + daily["position"].fillna(0) * daily["intraday_return"].fillna(0)
    )
    daily["strategy_cumulative_return"] = (
        (1 + daily["strategy_daily_return"]).cumprod() - 1
    )

    feature_columns = [
        "daily_return",
        "overnight_return",
        "intraday_return",
        "ma5",
        "ma20",
        "ma60",
        "ma5_ma20_ratio",
        "bias_5_20",
        "bias_ma5_gt_ma20",
        "ma20_ma60_ratio",
        "bias_20_60",
        "bias_ma20_gt_ma60",
        "close_ma5_bias",
        "close_ma20_bias",
        "close_ma60_bias",
        "ma_bias_spread",
        "ma_bias_spread_signal",
        "ma_long_bias_spread",
        "ma_long_bias_spread_signal",
        "ma_long_bias_spread_cap_signal",
        OI_REGRESSION_SLOPE_COLUMN,
        OI_REGRESSION_MEAN_COLUMN,
        OI_REGRESSION_SLOPE_RATE_COLUMN,
        OI_REGRESSION_SLOPE_DOWN_COLUMN,
        CLOSE_REGRESSION_SLOPE_COLUMN,
        CLOSE_REGRESSION_MEAN_COLUMN,
        CLOSE_REGRESSION_SLOPE_RATE_COLUMN,
        CLOSE_REGRESSION_SLOPE_DOWN_COLUMN,
        SPECULATION_REGRESSION_SLOPE_COLUMN,
        SPECULATION_REGRESSION_SLOPE_DOWN_COLUMN,
        SPECULATION_REGRESSION_SLOPE_SIGNAL_COLUMN,
        "price_range",
        "price_range_rate",
        "avg_price_range_10",
        "avg_volatility_rate_10",
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
        "strategy_daily_return",
        "strategy_cumulative_return",
        "high_bias_oi_speculation_drop_score",
    ]

    result = save_factor_outputs(
        daily=daily,
        symbol=symbol,
        factor_value_column="high_bias_oi_speculation_drop_score",
        signal_column="short_entry_signal",
        feature_columns=feature_columns,
        output_dir=output_dir,
    )

    summary_table = result["summary_table"].copy()
    summary_table["ma_short_window"] = MA_SHORT_WINDOW
    summary_table["ma_long_window"] = MA_LONG_WINDOW
    summary_table["ma_trend_window"] = MA_TREND_WINDOW
    summary_table["ma_bias_spread_threshold"] = MA_BIAS_SPREAD_THRESHOLD
    summary_table["ma_long_bias_spread_threshold"] = (
        MA_LONG_BIAS_SPREAD_THRESHOLD
    )
    summary_table["ma_long_bias_spread_cap_threshold"] = (
        MA_LONG_BIAS_SPREAD_CAP_THRESHOLD
    )
    summary_table["oi_regression_slope_window"] = OI_REGRESSION_SLOPE_WINDOW
    summary_table["close_regression_slope_window"] = (
        CLOSE_REGRESSION_SLOPE_WINDOW
    )
    summary_table["speculation_regression_slope_window"] = (
        SPECULATION_REGRESSION_SLOPE_WINDOW
    )
    summary_table["speculation_regression_slope_threshold"] = (
        SPECULATION_REGRESSION_SLOPE_THRESHOLD
    )
    summary_table["volatility_window"] = VOLATILITY_WINDOW
    summary_table["trailing_volatility_multiplier"] = (
        TRAILING_VOLATILITY_MULTIPLIER
    )
    summary_table["raw_open_short_signal_days"] = int(
        daily["open_short_signal"].sum()
    )
    summary_table["raw_cover_short_signal_days"] = int(
        daily["cover_short_signal"].sum()
    )
    summary_table["short_entry_count"] = int(daily["short_entry_signal"].sum())
    summary_table["short_exit_count"] = int(daily["short_exit_signal"].sum())
    summary_table["price_above_entry_exit_count"] = int(
        daily["exit_reason"].str.contains("price_above_entry", na=False).sum()
    )
    summary_table["trailing_rebound_exit_count"] = int(
        daily["exit_reason"].str.contains("trailing_rebound", na=False).sum()
    )
    summary_table["combined_exit_count"] = int(
        (
            daily["exit_reason"]
            == "price_above_entry_and_trailing_rebound"
        ).sum()
    )
    summary_table["final_position"] = int(daily["position"].iloc[-1])
    summary_table["mean_strategy_daily_return"] = daily[
        "strategy_daily_return"
    ].mean()
    summary_table["strategy_cumulative_return"] = daily[
        "strategy_cumulative_return"
    ].iloc[-1]
    summary_table["strategy_max_drawdown"] = calculate_max_drawdown(
        daily["strategy_daily_return"]
    )
    summary_table["strategy_sharpe"] = annualized_sharpe(
        daily["strategy_daily_return"]
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
    daily_dir = args.daily_dir.resolve()
    output_dir = args.output_dir.resolve()
    failures = []

    if args.collect_only:
        symbols = discover_symbols_from_summaries(output_dir)
    else:
        symbols = discover_symbols(daily_dir)

    output_dir.mkdir(parents=True, exist_ok=True)

    print(f"chapter2 日频目录：{daily_dir}", flush=True)
    print(f"本次运行品种数量：{len(symbols)}", flush=True)
    print(f"品种列表：{','.join(symbols)}", flush=True)
    print(f"因子结果目录：{output_dir}", flush=True)

    if not args.collect_only:
        for symbol in symbols:
            print(f"\n###### 开始处理品种：{symbol} ######", flush=True)
            try:
                calculate_symbol(symbol, daily_dir, output_dir)
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
