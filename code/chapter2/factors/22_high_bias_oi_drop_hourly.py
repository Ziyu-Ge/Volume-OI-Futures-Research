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

MA_SHORT_WINDOW = 5
MA_LONG_WINDOW = 20
MA_TREND_WINDOW = 60
MA_BIAS_SPREAD_THRESHOLD = 0.04
MA_LONG_BIAS_SPREAD_THRESHOLD = 0.10
MA_LONG_BIAS_SPREAD_CAP_THRESHOLD = 0.18
OI_REGRESSION_SLOPE_WINDOW = 15
CLOSE_REGRESSION_SLOPE_WINDOW = 7
VOLATILITY_WINDOW = 10
TRAILING_VOLATILITY_MULTIPLIER = 3.5

OI_REGRESSION_SLOPE_DOWN_COLUMN = (
    f"oi_regression_slope_down_{OI_REGRESSION_SLOPE_WINDOW}"
)
CLOSE_REGRESSION_SLOPE_DOWN_COLUMN = (
    f"close_regression_slope_down_{CLOSE_REGRESSION_SLOPE_WINDOW}"
)


# =========================
# 策略逻辑说明
# =========================
#
# 读取小时数据，但滚动窗口仍按交易日计算。每个小时把当日已发生的小时
# bar 累计成一根“截至当前小时”的日 K；当日以前只使用完整交易日。
#
# 初始状态为空仓。每个小时收盘后确认一次信号，下一小时 bar 开盘执行。
#
# 开空：
# 1. ma20 乖离率 - ma5 乖离率 >= 4%；
# 2. ma60 乖离率 - ma20 乖离率在 [10%, 18%] 内，避免追空过度延伸；
# 3. 最近 OI_REGRESSION_SLOPE_WINDOW 个交易日持仓量做回归线，斜率小于 0；
# 4. 最近 CLOSE_REGRESSION_SLOPE_WINDOW 个交易日收盘价做回归线，斜率小于 0。
#
# 平空：
# 1. 收盘价高于开仓价；
# 2. 收盘价高于“开仓以来最低价 + 3.5 倍历史 10 日平均波动”。
#
# 两个平空条件任一触发，都会在下一小时 bar 开盘执行平空。历史 10 日平均
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
        description="基于小时数据按日频口径滚动计算 22 号高乖离持仓回落因子。"
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
        "open_interest",
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
        "open_interest",
    ]
    for column in numeric_columns:
        hourly[column] = pd.to_numeric(hourly[column], errors="coerce")

    return hourly.sort_values(["trading_date", "date"]).reset_index(drop=True)


def build_intraday_daily_bars(hourly):
    bars = hourly.sort_values(["trading_date", "date"]).reset_index(drop=True)
    grouped = bars.groupby("trading_date", sort=False)

    # 把小时 bar 转成“截至当前小时”的日 K：
    # open 取当日第一根小时 bar 开盘价，close 取当前小时收盘价。
    # high/low 分别取当日截至当前小时的最高价/最低价。
    # open_interest 使用当前小时持仓量，用来判断截至当前小时的持仓趋势。
    daily = pd.DataFrame(
        {
            "date": bars["date"],
            "trading_date": bars["trading_date"],
            "hourly_open": bars["open"],
            "hourly_low": bars["low"],
            "open": grouped["open"].transform("first"),
            "close": bars["close"],
            "high": grouped["high"].cummax(),
            "low": grouped["low"].cummin(),
            "open_interest": bars["open_interest"],
        }
    )

    return daily


def build_full_daily_bars(daily):
    # 每个交易日最后一根“截至当前小时”的日 K 等同于完整日 K，
    # 后续滚动窗口只用这些完整日 K 做历史部分，避免引用未来小时。
    return (
        daily.groupby("trading_date", sort=False)
        .tail(1)
        .copy()
        .set_index("trading_date", drop=False)
    )


def map_daily_series(daily, series):
    return daily["trading_date"].map(series)


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


def add_daily_rolling_mean(daily, full_daily, source_column, output_column, window):
    if window <= 1:
        daily[output_column] = daily[source_column]
        return

    # 截至当前小时的日频均线：
    # MA_n = (前 n-1 个完整交易日 source 之和 + 当前截至小时 source) / n。
    # 这样当前交易日只使用已发生的小时信息，不会偷看当天后续 bar。
    prior_sum = (
        full_daily[source_column]
        .shift(1)
        .rolling(window=window - 1, min_periods=window - 1)
        .sum()
    )
    daily[output_column] = (
        map_daily_series(daily, prior_sum) + daily[source_column]
    ) / window


def add_daily_regression_down_signal(
    daily,
    full_daily,
    source_column,
    output_column,
    window,
):
    if window <= 1:
        daily[output_column] = 0
        return

    # 用普通最小二乘的一元线性回归斜率判断趋势方向。
    # x 取 0,1,...,window-1 并中心化，斜率公式为：
    # slope = sum((x - mean(x)) * y) / sum((x - mean(x))^2)。
    # y 由前 window-1 个完整交易日值 + 当前截至小时值组成。
    weights = np.arange(window, dtype=float)
    weights = weights - weights.mean()
    denominator = np.square(weights).sum()

    # 历史部分先在完整日 K 上滚动计算加权和，当前小时再补上最后一个权重。
    prior_values = full_daily[source_column].shift(1)
    prior_window = window - 1
    prior_weighted_sum = prior_values.rolling(
        window=prior_window,
        min_periods=prior_window,
    ).apply(lambda values: np.dot(weights[:-1], values), raw=True)

    slope = (
        map_daily_series(daily, prior_weighted_sum)
        + weights[-1] * daily[source_column]
    ) / denominator
    # 斜率小于 0 表示最近 window 个交易日口径下该变量在回落。
    daily[output_column] = (slope < 0).astype(int)


def build_complete_daily_entry_signal(full_daily):
    entry_daily = full_daily.copy()

    entry_daily["entry_ma5"] = (
        entry_daily["close"]
        .rolling(window=MA_SHORT_WINDOW, min_periods=MA_SHORT_WINDOW)
        .mean()
    )
    entry_daily["entry_ma20"] = (
        entry_daily["close"]
        .rolling(window=MA_LONG_WINDOW, min_periods=MA_LONG_WINDOW)
        .mean()
    )
    entry_daily["entry_ma60"] = (
        entry_daily["close"]
        .rolling(window=MA_TREND_WINDOW, min_periods=MA_TREND_WINDOW)
        .mean()
    )

    entry_close_ma5_bias = entry_daily["close"] / entry_daily["entry_ma5"] - 1
    entry_close_ma20_bias = entry_daily["close"] / entry_daily["entry_ma20"] - 1
    entry_close_ma60_bias = entry_daily["close"] / entry_daily["entry_ma60"] - 1
    entry_ma_bias_spread = entry_close_ma20_bias - entry_close_ma5_bias
    entry_ma_long_bias_spread = entry_close_ma60_bias - entry_close_ma20_bias
    entry_ma_bias_spread_signal = entry_ma_bias_spread >= MA_BIAS_SPREAD_THRESHOLD
    entry_ma_long_bias_spread_signal = (
        entry_ma_long_bias_spread >= MA_LONG_BIAS_SPREAD_THRESHOLD
    )
    entry_ma_long_bias_spread_cap_signal = (
        entry_ma_long_bias_spread <= MA_LONG_BIAS_SPREAD_CAP_THRESHOLD
    )

    entry_oi_slope_down = (
        entry_daily["open_interest"]
        .rolling(
            window=OI_REGRESSION_SLOPE_WINDOW,
            min_periods=OI_REGRESSION_SLOPE_WINDOW,
        )
        .apply(linear_regression_slope, raw=True)
        < 0
    )
    entry_close_slope_down = (
        entry_daily["close"]
        .rolling(
            window=CLOSE_REGRESSION_SLOPE_WINDOW,
            min_periods=CLOSE_REGRESSION_SLOPE_WINDOW,
        )
        .apply(linear_regression_slope, raw=True)
        < 0
    )

    entry_daily["complete_daily_open_short_signal"] = (
        entry_ma_bias_spread_signal
        & entry_ma_long_bias_spread_signal
        & entry_ma_long_bias_spread_cap_signal
        & entry_oi_slope_down
        & entry_close_slope_down
    ).astype(int)

    return entry_daily["complete_daily_open_short_signal"]


def add_daily_frequency_features(daily):
    daily = daily.copy()
    full_daily = build_full_daily_bars(daily)

    # 三条日频均线都按“前完整交易日 + 当前截至小时日 K”的口径计算。
    # ma5 代表短期成本，ma20 代表中期成本，ma60 代表更长期趋势成本。
    add_daily_rolling_mean(daily, full_daily, "close", "ma5", MA_SHORT_WINDOW)
    add_daily_rolling_mean(daily, full_daily, "close", "ma20", MA_LONG_WINDOW)
    add_daily_rolling_mean(daily, full_daily, "close", "ma60", MA_TREND_WINDOW)

    # 价格相对均线的乖离率：
    # close_maN_bias = close / maN - 1。
    # 数值越负，表示当前价格越低于对应均线。
    close_ma5_bias = daily["close"] / daily["ma5"] - 1
    close_ma20_bias = daily["close"] / daily["ma20"] - 1
    close_ma60_bias = daily["close"] / daily["ma60"] - 1

    # 中短期乖离差：
    # close_ma20_bias - close_ma5_bias >= 4%。
    # 当价格相对 ma5 更弱、相对 ma20 没那么弱时，说明短期下跌更急。
    daily["ma_bias_spread"] = close_ma20_bias - close_ma5_bias
    daily["ma_bias_spread_signal"] = (
        daily["ma_bias_spread"] >= MA_BIAS_SPREAD_THRESHOLD
    ).astype(int)

    # 长中期乖离差：
    # close_ma60_bias - close_ma20_bias 在 [10%, 18%]。
    # 下限确认中期下跌足够明显，上限过滤过度延伸后的追空。
    daily["ma_long_bias_spread"] = close_ma60_bias - close_ma20_bias
    daily["ma_long_bias_spread_signal"] = (
        daily["ma_long_bias_spread"] >= MA_LONG_BIAS_SPREAD_THRESHOLD
    ).astype(int)
    daily["ma_long_bias_spread_cap_signal"] = (
        daily["ma_long_bias_spread"] <= MA_LONG_BIAS_SPREAD_CAP_THRESHOLD
    ).astype(int)

    # 持仓量回归斜率小于 0：持仓量趋势在下降。
    add_daily_regression_down_signal(
        daily,
        full_daily,
        "open_interest",
        OI_REGRESSION_SLOPE_DOWN_COLUMN,
        OI_REGRESSION_SLOPE_WINDOW,
    )
    # 收盘价回归斜率小于 0：价格趋势也在下降。
    add_daily_regression_down_signal(
        daily,
        full_daily,
        "close",
        CLOSE_REGRESSION_SLOPE_DOWN_COLUMN,
        CLOSE_REGRESSION_SLOPE_WINDOW,
    )

    # 单日波动率近似为 (high - low) / close，
    # 衡量当天价格振幅占收盘价的比例。
    full_daily["price_range_rate"] = (
        (full_daily["high"] - full_daily["low"])
        / full_daily["close"].replace(0, np.nan)
    )
    # 历史平均波动只用前 VOLATILITY_WINDOW 个完整交易日，
    # 后面平仓止损会把这个比例换算成价格距离。
    avg_volatility_rate = (
        full_daily["price_range_rate"]
        .shift(1)
        .rolling(window=VOLATILITY_WINDOW, min_periods=VOLATILITY_WINDOW)
        .mean()
    )
    daily["avg_volatility_rate_10"] = map_daily_series(
        daily,
        avg_volatility_rate,
    )

    # 小时合成日频开仓信号保留为诊断列；真实开仓只使用完整日频信号。
    # 开空信号四个条件同时满足：
    # 1. 中短期乖离差达标；2. 长中期乖离差达标；
    # 3. 持仓量回归斜率向下；4. 价格回归斜率向下。
    daily["intraday_open_short_signal"] = (
        (daily["ma_bias_spread_signal"] == 1)
        & (daily["ma_long_bias_spread_signal"] == 1)
        & (daily["ma_long_bias_spread_cap_signal"] == 1)
        & (daily[OI_REGRESSION_SLOPE_DOWN_COLUMN] == 1)
        & (daily[CLOSE_REGRESSION_SLOPE_DOWN_COLUMN] == 1)
    ).astype(int)

    complete_entry_signal = build_complete_daily_entry_signal(full_daily)
    daily["complete_daily_open_short_signal"] = (
        map_daily_series(daily, complete_entry_signal).fillna(0).astype(int)
    )
    daily["is_complete_daily_bar"] = (
        daily["trading_date"] != daily["trading_date"].shift(-1)
    ).astype(int)
    # 完整日频开仓信号只能在当日最后一根小时 bar 收盘后确认，
    # 状态机会在下一小时 bar 开盘执行。
    daily["open_short_signal"] = (
        (daily["is_complete_daily_bar"] == 1)
        & (daily["complete_daily_open_short_signal"] == 1)
    ).astype(int)

    return daily


def exit_reason_from_signals(price_above_entry_signal, trailing_rebound_signal):
    if price_above_entry_signal and trailing_rebound_signal:
        return "price_above_entry_and_trailing_rebound"
    if price_above_entry_signal:
        return "price_above_entry"
    if trailing_rebound_signal:
        return "trailing_rebound"
    return ""


def build_short_state_machine(frame):
    """根据上一小时信号在当前小时开盘执行交易。"""
    daily = frame.copy()

    position = 0
    entry_price = np.nan
    low_since_entry = np.nan
    pending_open = False
    pending_cover = False
    pending_exit_reason = ""

    positions = []
    trade_signals = []
    entry_prices = []
    exit_prices = []
    exit_reasons = []
    trailing_stop_prices = []
    cover_short_signals = []

    for _, row in daily.iterrows():
        open_price = row.get("hourly_open", row["open"])
        close = row["close"]
        low = row.get("hourly_low", row["low"])
        avg_volatility_rate = row["avg_volatility_rate_10"]

        # 上一小时收盘确认的信号，在当前小时开盘执行。
        # actual_open_signal/actual_cover_signal 只负责“本小时是否执行交易”。
        actual_open_signal = pending_open
        actual_cover_signal = pending_cover
        actual_exit_reason = pending_exit_reason
        pending_open = False
        pending_cover = False
        pending_exit_reason = ""

        trade_signal = 0
        exit_reason = ""
        row_entry_price = entry_price if position == -1 else np.nan
        row_exit_price = np.nan
        opened_this_bar = False

        if position == -1 and actual_cover_signal and pd.notna(open_price):
            # 平空执行价 = 当前小时开盘价。
            trade_signal = 1
            exit_reason = actual_exit_reason or "cover_short"
            row_entry_price = entry_price
            row_exit_price = open_price
            position = 0
            entry_price = np.nan
            low_since_entry = np.nan
        elif position == 0 and actual_open_signal and pd.notna(open_price):
            # 开空执行价 = 当前小时开盘价。
            trade_signal = -1
            entry_price = open_price
            low_since_entry = open_price
            row_entry_price = entry_price
            position = -1
            opened_this_bar = True

        cover_short_signal = 0
        cover_signal_reason = ""
        trailing_stop_price = np.nan

        if position == -1:
            low_candidate = low
            if pd.isna(low_candidate):
                low_candidate = open_price if opened_this_bar else close
            if pd.notna(low_candidate):
                # 开仓以来最低价：
                # low_since_entry = min(原最低价, 当前小时最低价)。
                # 空头移动止损线会跟随这个最低价向下移动。
                low_since_entry = min(low_since_entry, low_candidate)

            row_entry_price = entry_price
            # 平仓条件 1：当前收盘价高于开仓价，说明空头已经亏损或回到不利区间。
            price_above_entry_signal = (
                pd.notna(close)
                and pd.notna(entry_price)
                and close > entry_price
            )
            trailing_rebound_signal = False

            if (
                pd.notna(close)
                and pd.notna(low_since_entry)
                and pd.notna(avg_volatility_rate)
            ):
                # 平仓条件 2 的移动止损价：
                # trailing_stop_price =
                #     开仓以来最低价 * (1 + 历史平均波动率 * 倍数)。
                # 当价格从低点反弹超过这个距离时，认为下跌动能减弱，准备平空。
                trailing_stop_price = low_since_entry * (
                    1 + avg_volatility_rate * TRAILING_VOLATILITY_MULTIPLIER
                )
                trailing_rebound_signal = close > trailing_stop_price

            # 两个平仓条件任意一个触发，就在下一小时开盘平空。
            cover_short_signal = int(
                price_above_entry_signal or trailing_rebound_signal
            )
            cover_signal_reason = exit_reason_from_signals(
                price_above_entry_signal,
                trailing_rebound_signal,
            )

        # 当前小时收盘后确认下一小时要执行的开仓/平仓信号。
        pending_open = bool(row["open_short_signal"])
        pending_cover = bool(cover_short_signal)
        pending_exit_reason = cover_signal_reason

        positions.append(position)
        trade_signals.append(trade_signal)
        entry_prices.append(row_entry_price)
        exit_prices.append(row_exit_price)
        exit_reasons.append(exit_reason)
        trailing_stop_prices.append(trailing_stop_price)
        cover_short_signals.append(cover_short_signal)

    daily["position"] = positions
    daily["trade_signal"] = trade_signals
    daily["entry_price"] = entry_prices
    daily["exit_price"] = exit_prices
    daily["exit_reason"] = exit_reasons
    daily["trailing_stop_price"] = trailing_stop_prices
    daily["cover_short_signal"] = cover_short_signals
    # trade_signal: -1 表示本小时开空，1 表示本小时平空，0 表示无交易。
    daily["short_entry_signal"] = (daily["trade_signal"] == -1).astype(int)
    daily["short_exit_signal"] = (daily["trade_signal"] == 1).astype(int)

    return daily


def build_trade_table(symbol, daily):
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
    if "trade_signal" not in daily.columns:
        return pd.DataFrame(columns=columns)

    trades = []
    open_trade = None
    trade_frame = daily.copy()
    trade_frame["trade_signal"] = (
        pd.to_numeric(trade_frame["trade_signal"], errors="coerce")
        .fillna(0)
        .astype(int)
    )

    for _, row in trade_frame.iterrows():
        if row["trade_signal"] == -1 and open_trade is None:
            # 记录一笔空头交易的开仓时间和开仓价。
            open_trade = {
                "entry_date": row["date"],
                "entry_trading_date": row["trading_date"],
                "entry_price": row["entry_price"],
            }
            continue

        if row["trade_signal"] == 1 and open_trade is not None:
            # 遇到平仓信号后，把开仓信息和平仓信息合并成一行交易记录。
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
        # 样本结束时仍未平仓的交易保留为 open，不填平仓价。
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


def plot_signal_trades(daily, trades, figure_path, title):
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

    if not trades.empty:
        interval_labeled = False
        segment_labeled = False
        last_date = daily["date"].iloc[-1]
        last_close = daily["close"].iloc[-1]

        for _, trade in trades.iterrows():
            is_closed = trade["status"] == "closed"
            # 未平仓交易在图上延伸到样本最后一根 bar，
            # 只用于展示持仓区间，不写入交易表的平仓字段。
            exit_date = trade["exit_date"] if is_closed else last_date
            exit_price = trade["exit_price"] if is_closed else last_close

            # 橙色阴影表示空头持仓区间。
            ax.axvspan(
                trade["entry_date"],
                exit_date,
                color="#f59e0b",
                alpha=0.12,
                label="short interval" if not interval_labeled else None,
            )
            interval_labeled = True

            # 橙色线段连接开仓价和平仓价；未平仓交易用虚线连接到最后价格。
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


def save_symbol_figure(daily, trades, symbol, output_dir):
    figure_path = (
        output_dir
        / "figures"
        / "factors"
        / f"{symbol}_{factor_id}_{factor_name}_signal_on_price.png"
    )
    figure_path.parent.mkdir(parents=True, exist_ok=True)
    plot_signal_trades(
        daily,
        trades,
        figure_path,
        f"{symbol} Factor {factor_id}: {factor_name}",
    )
    return figure_path


def calculate_symbol(symbol, hourly_dir, output_dir):
    hourly = load_hourly(symbol, hourly_dir)
    if hourly.empty:
        raise ValueError(f"{symbol} 小时频率数据为空。")

    daily = build_intraday_daily_bars(hourly)
    daily = add_daily_frequency_features(daily)
    daily = build_short_state_machine(daily)

    trades = build_trade_table(symbol, daily)
    save_symbol_figure(daily, trades, symbol, output_dir)
    return trades


def save_trade_table(trade_tables, output_dir):
    if trade_tables:
        trades = pd.concat(trade_tables, ignore_index=True)
        if not trades.empty:
            trades = trades.sort_values(["symbol", "entry_date"])
    else:
        trades = pd.DataFrame(
            columns=[
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
        )

    table_dir = output_dir / "tables"
    table_dir.mkdir(parents=True, exist_ok=True)

    table_path = table_dir / f"all_symbols_{factor_id}_{factor_name}_trades.csv"
    trades.to_csv(table_path, index=False)

    return {
        "table_path": table_path,
        "trade_count": len(trades),
    }


def main():
    args = parse_args()
    hourly_dir = args.hourly_dir.resolve()
    output_dir = args.output_dir.resolve()
    symbols = discover_symbols(hourly_dir)
    trade_tables = []
    failures = []

    output_dir.mkdir(parents=True, exist_ok=True)

    for symbol in symbols:
        try:
            trade_tables.append(calculate_symbol(symbol, hourly_dir, output_dir))
        except Exception as exc:
            if not args.keep_going:
                raise

            failures.append((symbol, exc))
            print(f"[{symbol}] 失败：{exc}", flush=True)

    table_info = save_trade_table(trade_tables, output_dir)

    print(f"成功处理品种数量：{len(symbols) - len(failures)}", flush=True)
    print(f"交易数量：{table_info['trade_count']}", flush=True)
    print(f"交易表：{table_info['table_path']}", flush=True)
    print(f"图片目录：{output_dir / 'figures' / 'factors'}", flush=True)

    if failures:
        print(f"失败品种数量：{len(failures)}", flush=True)
        for symbol, exc in failures:
            print(f"- {symbol}: {exc}", flush=True)
        raise SystemExit(1)


if __name__ == "__main__":
    main()
