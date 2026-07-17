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
DEFAULT_HOURLY_DIR = CHAPTER_RESULTS_DIR / "tables" / "hourly"
DEFAULT_OUTPUT_DIR = (
    CHAPTER_RESULTS_DIR / "24_high_bias_oi_speculation_drop_mixed_all_symbols"
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
# 24 号因子的开仓条件与 23 号完全一致，但信号确认与平仓执行使用混合频率：
#
# 开空：
# 1. 只用完整日频数据计算 23 号开仓条件；
# 2. 完整日频开仓信号在该交易日最后一根小时 bar 收盘后确认；
# 3. 下一小时 bar 开盘执行开空。
#
# 平空：
# 1. 小时数据逐小时合成“截至当前小时”的日 K；
# 2. 当前小时收盘价高于开仓价，或当前小时收盘价高于开仓以来最低价
#    + 4 倍历史 10 日平均波动；
# 3. 任一条件触发后，下一小时 bar 开盘执行平空。


def parse_factor_script_metadata(file_path):
    stem = Path(file_path).stem
    match = re.match(r"^(\d+)_?(.+)$", stem)
    if match is None:
        raise ValueError(f"factor script filename must start with an id: {stem}")

    return match.group(1), match.group(2)


factor_id, factor_name = parse_factor_script_metadata(__file__)


def parse_args():
    parser = argparse.ArgumentParser(
        description=(
            "运行全部品种的 24 号高乖离持仓投机度回落混合频率因子。"
        )
    )
    parser.add_argument(
        "--daily-dir",
        type=Path,
        default=Path(os.environ.get("CHAPTER2_DAILY_DIR", DEFAULT_DAILY_DIR)),
        help=f"chapter2 日频缓存目录，默认：{DEFAULT_DAILY_DIR}",
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
        "--symbol",
        action="append",
        help="指定要运行的品种，可重复传入；默认运行日频和小时频率都有数据的全部品种。",
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


def discover_symbols(daily_dir, hourly_dir, selected_symbols=None):
    if not daily_dir.is_dir():
        raise FileNotFoundError(f"日频数据目录不存在：{daily_dir}")
    if not hourly_dir.is_dir():
        raise FileNotFoundError(f"小时频率数据目录不存在：{hourly_dir}")

    daily_symbols = {
        path.name[: -len("_daily.csv")].upper()
        for path in daily_dir.glob("*_daily.csv")
        if path.is_file()
    }
    hourly_symbols = {
        path.name[: -len("_hourly.csv")].upper()
        for path in hourly_dir.glob("*_hourly.csv")
        if path.is_file()
    }
    symbols = sorted(daily_symbols & hourly_symbols)
    if selected_symbols is not None:
        selected = {symbol.strip().upper() for symbol in selected_symbols}
        symbols = [symbol for symbol in symbols if symbol in selected]
        if not symbols:
            raise FileNotFoundError(
                f"未找到指定品种的数据：{','.join(sorted(selected))}"
            )
    if not symbols:
        raise FileNotFoundError(
            f"没有同时存在日频和小时频率缓存的品种：{daily_dir}, {hourly_dir}"
        )

    return symbols


def discover_symbols_from_summaries(output_dir, selected_symbols=None):
    summary_dir = output_dir / "tables" / "summary"
    if not summary_dir.is_dir():
        raise FileNotFoundError(f"summary 目录不存在：{summary_dir}")

    suffix = f"_{factor_id}_{factor_name}_summary.csv"
    symbols = sorted(
        path.name[: -len(suffix)].upper()
        for path in summary_dir.glob(f"*{suffix}")
        if path.is_file() and not path.name.startswith("all_symbols_")
    )
    if selected_symbols is not None:
        selected = {symbol.strip().upper() for symbol in selected_symbols}
        symbols = [symbol for symbol in symbols if symbol in selected]
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


def load_hourly(symbol, hourly_dir):
    hourly_path = hourly_dir / f"{symbol}_hourly.csv"
    if not hourly_path.exists():
        raise FileNotFoundError(f"未找到 {symbol} 小时频率数据：{hourly_path}")

    hourly = pd.read_csv(hourly_path)
    required_columns = {
        "date",
        "trading_date",
        "open",
        "close",
        "high",
        "low",
    }
    missing_columns = required_columns - set(hourly.columns)
    if missing_columns:
        missing_text = ", ".join(sorted(missing_columns))
        raise ValueError(f"{hourly_path} 缺少必要列：{missing_text}")

    hourly["date"] = pd.to_datetime(hourly["date"])
    hourly["trading_date"] = pd.to_datetime(hourly["trading_date"])
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
        if column in hourly.columns:
            hourly[column] = pd.to_numeric(hourly[column], errors="coerce")

    return hourly.sort_values(["trading_date", "date"]).reset_index(drop=True)


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


def add_complete_daily_entry_features(daily):
    daily = daily.copy()

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

    daily["complete_daily_open_short_signal"] = (
        (daily["ma_bias_spread_signal"] == 1)
        & (daily["ma_long_bias_spread_signal"] == 1)
        & (daily["ma_long_bias_spread_cap_signal"] == 1)
        & (daily[OI_REGRESSION_SLOPE_DOWN_COLUMN] == 1)
        & (daily[CLOSE_REGRESSION_SLOPE_DOWN_COLUMN] == 1)
        & (daily[SPECULATION_REGRESSION_SLOPE_SIGNAL_COLUMN] == 1)
    ).astype(int)

    daily["high_bias_oi_speculation_drop_mixed_score"] = (
        positive_part(daily["bias_5_20"])
        + positive_part(daily["ma_bias_spread"])
        + positive_part(daily["ma_long_bias_spread"])
        + positive_part(-daily[OI_REGRESSION_SLOPE_RATE_COLUMN])
        + positive_part(-daily[CLOSE_REGRESSION_SLOPE_RATE_COLUMN])
        + positive_part(-daily[SPECULATION_REGRESSION_SLOPE_COLUMN])
    )

    return daily


def build_intraday_daily_bars(hourly):
    bars = hourly.sort_values(["trading_date", "date"]).reset_index(drop=True)
    grouped = bars.groupby("trading_date", sort=False)

    columns = {
        "date": bars["date"],
        "trading_date": bars["trading_date"],
        "hourly_open": bars["open"],
        "hourly_low": bars["low"],
        "open": grouped["open"].transform("first"),
        "close": bars["close"],
        "high": grouped["high"].cummax(),
        "low": grouped["low"].cummin(),
    }
    for column in ["volume", "total_turnover"]:
        if column in bars.columns:
            columns[column] = grouped[column].cumsum()
    for column in ["open_interest", "speculation"]:
        if column in bars.columns:
            columns[column] = bars[column]

    return pd.DataFrame(columns)


def attach_complete_daily_entry_signal(frame, entry_daily):
    daily_features = entry_daily.set_index("date")
    signal_by_date = daily_features["complete_daily_open_short_signal"]
    score_by_date = daily_features["high_bias_oi_speculation_drop_mixed_score"]

    frame = frame.copy()
    frame["complete_daily_open_short_signal"] = (
        frame["trading_date"].map(signal_by_date).fillna(0).astype(int)
    )
    frame["factor_value"] = frame["trading_date"].map(score_by_date)
    frame["is_complete_daily_bar"] = (
        frame["trading_date"] != frame["trading_date"].shift(-1)
    ).astype(int)
    frame["open_short_signal"] = (
        (frame["is_complete_daily_bar"] == 1)
        & (frame["complete_daily_open_short_signal"] == 1)
    ).astype(int)

    feature_columns = [
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
    ]
    for column in feature_columns:
        frame[f"entry_{column}"] = frame["trading_date"].map(
            daily_features[column]
        )

    return frame


def add_hourly_exit_features(frame):
    frame = frame.copy()
    full_daily = (
        frame.groupby("trading_date", sort=False)
        .tail(1)
        .copy()
        .set_index("trading_date", drop=False)
    )

    full_daily["exit_price_range"] = full_daily["high"] - full_daily["low"]
    full_daily["exit_price_range_rate"] = (
        full_daily["exit_price_range"] / full_daily["close"].replace(0, np.nan)
    )
    avg_price_range = (
        full_daily["exit_price_range"]
        .shift(1)
        .rolling(window=VOLATILITY_WINDOW, min_periods=VOLATILITY_WINDOW)
        .mean()
    )
    avg_volatility_rate = (
        full_daily["exit_price_range_rate"]
        .shift(1)
        .rolling(window=VOLATILITY_WINDOW, min_periods=VOLATILITY_WINDOW)
        .mean()
    )
    frame["exit_price_range"] = frame["high"] - frame["low"]
    frame["exit_price_range_rate"] = (
        frame["exit_price_range"] / frame["close"].replace(0, np.nan)
    )
    frame["avg_price_range_10"] = frame["trading_date"].map(avg_price_range)
    frame["avg_volatility_rate_10"] = frame["trading_date"].map(
        avg_volatility_rate
    )
    return frame


def exit_reason_from_signals(price_above_entry_signal, trailing_rebound_signal):
    if price_above_entry_signal and trailing_rebound_signal:
        return "price_above_entry_and_trailing_rebound"
    if price_above_entry_signal:
        return "price_above_entry"
    if trailing_rebound_signal:
        return "trailing_rebound"
    return ""


def build_short_state_machine(frame):
    hourly = frame.copy()

    position = 0
    entry_price = np.nan
    low_since_entry = np.nan
    pending_open = False
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
    actual_open_short_signals = []
    actual_cover_short_signals = []
    actual_price_above_entry_signals = []
    actual_trailing_rebound_signals = []

    for _, row in hourly.iterrows():
        open_price = row["hourly_open"]
        close = row["close"]
        low = row["hourly_low"]
        avg_volatility_rate = row["avg_volatility_rate_10"]

        actual_open_signal = bool(pending_open)
        actual_cover_signal = bool(pending_cover)
        actual_price_above_entry_signal = int(pending_price_above_entry)
        actual_trailing_rebound_signal = int(pending_trailing_rebound)
        actual_exit_reason = pending_exit_reason

        pending_open = False
        pending_cover = False
        pending_price_above_entry = False
        pending_trailing_rebound = False
        pending_exit_reason = ""

        trade_signal = 0
        trade_action = ""
        exit_reason = ""
        row_exit_price = np.nan
        opened_this_bar = False
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
            opened_this_bar = True

        price_above_entry_signal = 0
        trailing_rebound_signal = 0
        cover_short_signal = 0
        cover_signal_reason = ""
        trailing_stop_distance = np.nan
        trailing_stop_price = np.nan

        if position == -1:
            low_candidate = low
            if pd.isna(low_candidate):
                low_candidate = open_price if opened_this_bar else close
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

        pending_open = bool(row["open_short_signal"])
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
        actual_open_short_signals.append(int(actual_open_signal))
        actual_cover_short_signals.append(int(actual_cover_signal))
        actual_price_above_entry_signals.append(actual_price_above_entry_signal)
        actual_trailing_rebound_signals.append(actual_trailing_rebound_signal)

    hourly["actual_open_short_signal"] = actual_open_short_signals
    hourly["actual_cover_short_signal"] = actual_cover_short_signals
    hourly["position"] = positions
    hourly["trade_signal"] = trade_signals
    hourly["trade_action"] = trade_actions
    hourly["entry_price"] = entry_prices
    hourly["exit_price"] = exit_prices
    hourly["exit_reason"] = exit_reasons
    hourly["low_since_entry"] = low_since_entry_values
    hourly["trailing_stop_distance"] = trailing_stop_distances
    hourly["trailing_stop_price"] = trailing_stop_prices
    hourly["price_above_entry_signal"] = price_above_entry_signals
    hourly["trailing_rebound_signal"] = trailing_rebound_signals
    hourly["cover_short_signal"] = cover_short_signals
    hourly["cover_signal_reason"] = cover_signal_reasons
    hourly["actual_price_above_entry_signal"] = actual_price_above_entry_signals
    hourly["actual_trailing_rebound_signal"] = actual_trailing_rebound_signals
    hourly["short_entry_signal"] = (hourly["trade_signal"] == -1).astype(int)
    hourly["short_exit_signal"] = (hourly["trade_signal"] == 1).astype(int)
    hourly["signal"] = hourly["short_entry_signal"]

    return hourly


def build_trade_table(symbol, hourly):
    columns = [
        "symbol",
        "status",
        "entry_date",
        "entry_trading_date",
        "entry_price",
        "exit_date",
        "exit_trading_date",
        "exit_price",
        "exit_reason",
    ]
    if "trade_signal" not in hourly.columns:
        return pd.DataFrame(columns=columns)

    trades = []
    open_trade = None
    trade_frame = hourly.copy()
    trade_frame["trade_signal"] = (
        pd.to_numeric(trade_frame["trade_signal"], errors="coerce")
        .fillna(0)
        .astype(int)
    )

    for _, row in trade_frame.iterrows():
        if row["trade_signal"] == -1 and open_trade is None:
            open_trade = {
                "entry_date": row["date"],
                "entry_trading_date": row["trading_date"],
                "entry_price": row["entry_price"],
            }
            continue

        if row["trade_signal"] == 1 and open_trade is not None:
            trades.append(
                {
                    "symbol": symbol,
                    "status": "closed",
                    "entry_date": open_trade["entry_date"],
                    "entry_trading_date": open_trade["entry_trading_date"],
                    "entry_price": open_trade["entry_price"],
                    "exit_date": row["date"],
                    "exit_trading_date": row["trading_date"],
                    "exit_price": row["exit_price"],
                    "exit_reason": row["exit_reason"],
                }
            )
            open_trade = None

    if open_trade is not None:
        trades.append(
            {
                "symbol": symbol,
                "status": "open",
                "entry_date": open_trade["entry_date"],
                "entry_trading_date": open_trade["entry_trading_date"],
                "entry_price": open_trade["entry_price"],
                "exit_date": pd.NaT,
                "exit_trading_date": pd.NaT,
                "exit_price": np.nan,
                "exit_reason": "",
            }
        )

    return pd.DataFrame(trades, columns=columns)


def plot_signal_trades(hourly, trades, figure_path, title):
    fig, ax = plt.subplots(figsize=(12, 6))
    ax.plot(hourly["date"], hourly["close"], color="#1f77b4", label="close")

    trailing_line = hourly["trailing_stop_price"].where(hourly["position"] == -1)
    if trailing_line.notna().any():
        ax.plot(
            hourly["date"],
            trailing_line,
            color="#9467bd",
            linewidth=1.0,
            alpha=0.65,
            label="trailing stop",
        )

    if not trades.empty:
        interval_labeled = False
        segment_labeled = False
        last_date = hourly["date"].iloc[-1]
        last_close = hourly["close"].iloc[-1]

        for _, trade in trades.iterrows():
            is_closed = trade["status"] == "closed"
            exit_date = trade["exit_date"] if is_closed else last_date
            exit_price = trade["exit_price"] if is_closed else last_close

            ax.axvspan(
                trade["entry_date"],
                exit_date,
                color="#f59e0b",
                alpha=0.12,
                label="short interval" if not interval_labeled else None,
            )
            interval_labeled = True

            ax.plot(
                [trade["entry_date"], exit_date],
                [trade["entry_price"], exit_price],
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


def calculate_max_drawdown(return_series):
    equity_curve = (1 + return_series.fillna(0)).cumprod()
    if equity_curve.empty:
        return np.nan

    running_high = equity_curve.cummax()
    drawdown = equity_curve / running_high - 1
    return drawdown.min()


def annualized_sharpe(return_series, periods_per_year):
    returns = return_series.fillna(0)
    std = returns.std(ddof=1)
    if pd.isna(std) or std == 0:
        return np.nan

    return returns.mean() / std * np.sqrt(periods_per_year)


def infer_trading_bars_per_year(hourly):
    trading_days = hourly["trading_date"].nunique()
    if trading_days <= 0:
        return TRADING_DAYS_PER_YEAR
    return TRADING_DAYS_PER_YEAR * len(hourly) / trading_days


def add_return_columns(hourly):
    hourly = hourly.copy()
    bar_open = hourly["hourly_open"].replace(0, np.nan)
    previous_close = hourly["close"].shift(1).replace(0, np.nan)
    previous_position = hourly["position"].shift(1).fillna(0)

    hourly["gap_return"] = (bar_open / previous_close - 1).fillna(0)
    hourly["intraday_return"] = (hourly["close"] / bar_open - 1).fillna(0)
    if not hourly.empty:
        hourly.loc[hourly.index[0], "gap_return"] = 0
        hourly.loc[hourly.index[0], "intraday_return"] = 0

    hourly["strategy_return"] = (
        previous_position * hourly["gap_return"]
        + hourly["position"] * hourly["intraday_return"]
    )
    hourly["strategy_cumulative_return"] = (
        (1 + hourly["strategy_return"]).cumprod() - 1
    )
    return hourly


def build_summary_table(symbol, hourly, trades):
    closed = trades[trades["status"] == "closed"]
    periods_per_year = infer_trading_bars_per_year(hourly)
    complete_daily_signal_days = hourly.loc[
        hourly["complete_daily_open_short_signal"] == 1,
        "trading_date",
    ].nunique()
    signal_points = hourly[hourly["signal"] == 1]
    if signal_points.empty:
        first_signal_date = pd.NaT
        first_signal_close = np.nan
    else:
        first_signal_date = signal_points["date"].min()
        first_signal_close = signal_points.loc[
            signal_points["date"].idxmin(),
            "close",
        ]

    return pd.DataFrame(
        [
            {
                "symbol": symbol,
                "factor_id": factor_id,
                "factor_name": factor_name,
                "total_bars": len(hourly),
                "trading_days": hourly["trading_date"].nunique(),
                "bars_per_year": periods_per_year,
                "valid_factor_bars": int(hourly["factor_value"].notna().sum()),
                "complete_daily_open_signal_days": int(
                    complete_daily_signal_days
                ),
                "raw_open_short_signal_bars": int(hourly["open_short_signal"].sum()),
                "raw_cover_short_signal_bars": int(hourly["cover_short_signal"].sum()),
                "signal_days": int(hourly["signal"].sum()),
                "short_entry_count": int(hourly["short_entry_signal"].sum()),
                "short_exit_count": int(hourly["short_exit_signal"].sum()),
                "trade_count": len(trades),
                "closed_trade_count": len(closed),
                "first_signal_date": first_signal_date,
                "first_signal_close": first_signal_close,
                "mean_factor_value": hourly["factor_value"].mean(),
                "max_factor_value": hourly["factor_value"].max(),
                "min_factor_value": hourly["factor_value"].min(),
                "price_above_entry_exit_count": int(
                    hourly["exit_reason"]
                    .str.contains("price_above_entry", na=False)
                    .sum()
                ),
                "trailing_rebound_exit_count": int(
                    hourly["exit_reason"]
                    .str.contains("trailing_rebound", na=False)
                    .sum()
                ),
                "combined_exit_count": int(
                    (
                        hourly["exit_reason"]
                        == "price_above_entry_and_trailing_rebound"
                    ).sum()
                ),
                "final_position": int(hourly["position"].iloc[-1]),
                "mean_strategy_return": hourly["strategy_return"].mean(),
                "strategy_cumulative_return": hourly[
                    "strategy_cumulative_return"
                ].iloc[-1],
                "strategy_max_drawdown": calculate_max_drawdown(
                    hourly["strategy_return"]
                ),
                "strategy_sharpe": annualized_sharpe(
                    hourly["strategy_return"],
                    periods_per_year,
                ),
                "mean_trade_return": (
                    closed["entry_price"] / closed["exit_price"] - 1
                ).mean()
                if not closed.empty
                else np.nan,
                "win_rate": (
                    (closed["entry_price"] / closed["exit_price"] - 1) > 0
                ).mean()
                if not closed.empty
                else np.nan,
                "ma_short_window": MA_SHORT_WINDOW,
                "ma_long_window": MA_LONG_WINDOW,
                "ma_trend_window": MA_TREND_WINDOW,
                "ma_bias_spread_threshold": MA_BIAS_SPREAD_THRESHOLD,
                "ma_long_bias_spread_threshold": MA_LONG_BIAS_SPREAD_THRESHOLD,
                "ma_long_bias_spread_cap_threshold": (
                    MA_LONG_BIAS_SPREAD_CAP_THRESHOLD
                ),
                "oi_regression_slope_window": OI_REGRESSION_SLOPE_WINDOW,
                "close_regression_slope_window": CLOSE_REGRESSION_SLOPE_WINDOW,
                "speculation_regression_slope_window": (
                    SPECULATION_REGRESSION_SLOPE_WINDOW
                ),
                "speculation_regression_slope_threshold": (
                    SPECULATION_REGRESSION_SLOPE_THRESHOLD
                ),
                "volatility_window": VOLATILITY_WINDOW,
                "trailing_volatility_multiplier": (
                    TRAILING_VOLATILITY_MULTIPLIER
                ),
            }
        ]
    )


def save_symbol_outputs(symbol, hourly, trades, summary, output_dir):
    hourly = hourly.copy()
    hourly["factor_id"] = factor_id
    hourly["factor_name"] = factor_name

    tables_dir = output_dir / "tables"
    factor_path = tables_dir / "factors" / f"{symbol}_{factor_id}_{factor_name}.csv"
    signal_path = (
        tables_dir
        / "signals"
        / f"{symbol}_{factor_id}_{factor_name}_signals.csv"
    )
    trade_path = (
        tables_dir / "trades" / f"{symbol}_{factor_id}_{factor_name}_trades.csv"
    )
    summary_path = (
        tables_dir
        / "summary"
        / f"{symbol}_{factor_id}_{factor_name}_summary.csv"
    )
    figure_path = (
        output_dir
        / "figures"
        / "factors"
        / f"{symbol}_{factor_id}_{factor_name}_signal_on_price.png"
    )

    factor_path.parent.mkdir(parents=True, exist_ok=True)
    signal_path.parent.mkdir(parents=True, exist_ok=True)
    trade_path.parent.mkdir(parents=True, exist_ok=True)
    summary_path.parent.mkdir(parents=True, exist_ok=True)
    figure_path.parent.mkdir(parents=True, exist_ok=True)

    hourly.to_csv(factor_path, index=False)
    hourly.loc[hourly["signal"] == 1].to_csv(signal_path, index=False)
    trades.to_csv(trade_path, index=False)
    summary.to_csv(summary_path, index=False)
    plot_signal_trades(
        hourly,
        trades,
        figure_path,
        f"{symbol} Factor {factor_id}: {factor_name}",
    )

    print(f"[{symbol}] factor {factor_id}: {factor_name} complete.", flush=True)
    print(f"[{symbol}] factor hourly table: {factor_path}", flush=True)
    print(f"[{symbol}] trade table: {trade_path}", flush=True)
    print(f"[{symbol}] summary table: {summary_path}", flush=True)
    print(f"[{symbol}] entries: {int(hourly['short_entry_signal'].sum())}", flush=True)


def calculate_symbol(symbol, daily_dir, hourly_dir, output_dir):
    daily = load_daily(symbol, daily_dir)
    entry_daily = add_complete_daily_entry_features(daily)

    hourly = load_hourly(symbol, hourly_dir)
    if hourly.empty:
        raise ValueError(f"{symbol} 小时频率数据为空。")

    frame = build_intraday_daily_bars(hourly)
    frame = attach_complete_daily_entry_signal(frame, entry_daily)
    frame = add_hourly_exit_features(frame)
    frame = build_short_state_machine(frame)
    frame = add_return_columns(frame)

    trades = build_trade_table(symbol, frame)
    summary = build_summary_table(symbol, frame, trades)
    save_symbol_outputs(symbol, frame, trades, summary, output_dir)
    return summary


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

    combined_summary = pd.concat(summaries, ignore_index=True, sort=False)
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
    hourly_dir = args.hourly_dir.resolve()
    output_dir = args.output_dir.resolve()
    failures = []

    if args.collect_only:
        symbols = discover_symbols_from_summaries(output_dir, args.symbol)
    else:
        symbols = discover_symbols(daily_dir, hourly_dir, args.symbol)

    output_dir.mkdir(parents=True, exist_ok=True)

    print(f"chapter2 日频目录：{daily_dir}", flush=True)
    print(f"chapter2 小时频率目录：{hourly_dir}", flush=True)
    print(f"本次运行品种数量：{len(symbols)}", flush=True)
    print(f"品种列表：{','.join(symbols)}", flush=True)
    print(f"因子结果目录：{output_dir}", flush=True)

    if not args.collect_only:
        for symbol in symbols:
            print(f"\n###### 开始处理品种：{symbol} ######", flush=True)
            try:
                calculate_symbol(symbol, daily_dir, hourly_dir, output_dir)
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
