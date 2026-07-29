import os
import re
import sys

import numpy as np
import pandas as pd


PROJECT_ROOT = os.path.dirname(
    os.path.dirname(
        os.path.dirname(
            os.path.dirname(os.path.abspath(__file__))
        )
    )
)
CODE_DIR = os.path.join(PROJECT_ROOT, "code", "chapter1")

if CODE_DIR not in sys.path:
    sys.path.insert(0, CODE_DIR)

SYMBOL_ENV_VAR = "SYMBOL"
SYMBOL = os.environ.get(SYMBOL_ENV_VAR, "").strip().upper()

RESULTS_OUTPUT_ENV_VAR = "RESULTS_OUTPUT_DIR"
DAILY_DATA_ENV_VAR = "CHAPTER1_DAILY_DIR"
DEFAULT_RESULTS_DIR = os.path.join(PROJECT_ROOT, "results", "chapter1")
DEFAULT_DAILY_DIR = os.path.join(DEFAULT_RESULTS_DIR, "tables")

MAD_SCALE = 1.4826
MAD_EPSILON = 1e-12


def require_symbol(symbol=None):
    """读取并校验品种代码。"""
    if symbol is None:
        symbol = SYMBOL

    symbol = str(symbol).strip().upper()
    if not symbol:
        raise RuntimeError(
            f"缺少品种代码。请通过环境变量 {SYMBOL_ENV_VAR} 运行因子脚本，"
            "或使用 code/chapter1/run/ 下的批量入口。"
        )

    return symbol


def get_results_dir(results_dir=None):
    """确定本次因子结果输出目录。"""
    if results_dir is None:
        results_dir = os.environ.get(RESULTS_OUTPUT_ENV_VAR)

    if results_dir is None:
        results_dir = DEFAULT_RESULTS_DIR

    return os.path.abspath(os.path.expanduser(str(results_dir)))


def get_daily_dir():
    """确定日频数据输入目录。"""
    daily_dir = os.environ.get(DAILY_DATA_ENV_VAR)
    if daily_dir:
        return os.path.abspath(os.path.expanduser(daily_dir))

    return os.path.abspath(DEFAULT_DAILY_DIR)


def parse_factor_script_metadata(file_path):
    """从脚本文件名解析因子编号和因子名。"""
    stem = os.path.splitext(os.path.basename(file_path))[0]
    match = re.match(r"^(\d+)_?(.+)$", stem)
    if match is None:
        raise ValueError(f"factor script filename must start with an id: {stem}")

    return match.group(1), match.group(2)


def load_daily(symbol, results_dir=None):
    """读取单个品种的日频表。"""
    symbol = require_symbol(symbol)
    daily_filename = f"{symbol}_daily.csv"
    daily_input_candidates = [os.path.join(get_daily_dir(), daily_filename)]

    legacy_tables_dir = os.path.join(PROJECT_ROOT, "results", "tables")
    daily_input_candidates.extend([
        os.path.join(legacy_tables_dir, "daily", daily_filename),
        os.path.join(legacy_tables_dir, daily_filename),
    ])

    daily_input_path = next(
        (
            candidate_path
            for candidate_path in daily_input_candidates
            if os.path.exists(candidate_path)
        ),
        None,
    )
    if daily_input_path is None:
        raise FileNotFoundError(
            "未找到日频数据，请先运行 code/chapter1/00_prepare_data.py。"
            f" 尝试路径：{daily_input_candidates}"
        )

    daily = pd.read_csv(daily_input_path)
    daily["date"] = pd.to_datetime(daily["date"])
    return daily.sort_values("date").reset_index(drop=True)


def past_rank(series, window, min_history_days):
    """今天的值在过去窗口中的历史分位，不包含今天。"""
    ranks = []

    for i, current_value in enumerate(series):
        history = series.iloc[max(0, i - window):i].dropna()

        if len(history) < min_history_days or pd.isna(current_value):
            ranks.append(np.nan)
            continue

        ranks.append((history <= current_value).mean())

    return pd.Series(ranks, index=series.index)


def mad_score(
    series,
    window,
    min_history_days,
    mad_scale=MAD_SCALE,
    mad_epsilon=MAD_EPSILON,
):
    """用过去窗口的中位数和 MAD 计算稳健异常分数。"""
    median_past = (
        series
        .rolling(window=window, min_periods=min_history_days)
        .median()
        .shift(1)
    )

    mad_values = []
    for i in range(len(series)):
        history = series.iloc[max(0, i - window):i].dropna()

        if len(history) < min_history_days:
            mad_values.append(np.nan)
            continue

        history_median = history.median()
        mad_values.append((history - history_median).abs().median())

    mad_past = pd.Series(mad_values, index=series.index)
    score = (series - median_past) / (mad_scale * mad_past + mad_epsilon)
    score[mad_past <= 0] = np.nan

    return median_past, mad_past, score


def positive_part(series):
    """只保留正向贡献，缺失值按 0 处理。"""
    return series.clip(lower=0).fillna(0)


def add_price_ma_features(
    daily,
    ma_gap_threshold,
    short_window=5,
    mid_window=10,
    long_window=20,
    trend_window=120,
):
    """计算 11-14 号因子共用的价格均线特征。"""
    daily = daily.copy()

    for window in [short_window, mid_window, long_window, trend_window]:
        daily[f"ma{window}"] = (
            daily["close"]
            .rolling(window=window, min_periods=window)
            .mean()
        )

    daily["is_ma_bullish"] = (
        (daily[f"ma{short_window}"] > daily[f"ma{mid_window}"]) &
        (daily[f"ma{mid_window}"] > daily[f"ma{long_window}"])
    ).astype(int)

    daily["ma5_ma10_bias"] = (
        daily[f"ma{short_window}"] / daily[f"ma{mid_window}"] - 1
    )
    daily["ma10_ma20_bias"] = (
        daily[f"ma{mid_window}"] / daily[f"ma{long_window}"] - 1
    )
    daily["close_ma20_bias"] = (
        daily["close"] / daily[f"ma{long_window}"] - 1
    )
    daily["ma_bull_stack_filter"] = (
        (daily[f"ma{short_window}"] > daily[f"ma{mid_window}"]) &
        (daily[f"ma{mid_window}"] > daily[f"ma{long_window}"]) &
        (daily["ma5_ma10_bias"] >= ma_gap_threshold) &
        (daily["ma10_ma20_bias"] >= ma_gap_threshold)
    )

    return daily


def add_volume_price_features(
    daily,
    price_rank_window=20,
    mad_window=10,
    min_rank_days=8,
    min_mad_days=5,
):
    """计算 14 号因子需要的量价和持仓特征。"""
    daily = daily.copy()
    daily["daily_return"] = daily["close"].pct_change()
    daily["ret_5"] = daily["close"].pct_change(5)

    safe_open_interest = daily["open_interest"].where(
        daily["open_interest"] > 0,
        np.nan,
    )
    safe_volume = daily["volume"].where(daily["volume"] > 0, np.nan)

    daily["log_open_interest"] = np.log(safe_open_interest)
    daily["log_volume"] = np.log(safe_volume)
    daily["oi_ret_5"] = (
        daily["log_open_interest"] - daily["log_open_interest"].shift(5)
    )

    daily["range_pct"] = (
        (daily["high"] - daily["low"]) /
        daily["close"].replace(0, np.nan)
    )
    daily["close_location"] = (
        (daily["close"] - daily["low"]) /
        (daily["high"] - daily["low"]).replace(0, np.nan)
    )

    daily["close_rank_20"] = past_rank(
        daily["close"],
        window=price_rank_window,
        min_history_days=min_rank_days,
    )
    daily["close_rank_60"] = past_rank(
        daily["close"],
        window=60,
        min_history_days=20,
    )

    (
        daily["log_volume_median_past"],
        daily["log_volume_mad_past"],
        daily["volume_mad_score"],
    ) = mad_score(
        daily["log_volume"],
        window=mad_window,
        min_history_days=min_mad_days,
    )
    (
        daily["range_pct_median_past"],
        daily["range_pct_mad_past"],
        daily["range_mad_score"],
    ) = mad_score(
        daily["range_pct"],
        window=mad_window,
        min_history_days=min_mad_days,
    )
    (
        daily["log_open_interest_median_past"],
        daily["log_open_interest_mad_past"],
        daily["open_interest_mad_score"],
    ) = mad_score(
        daily["log_open_interest"],
        window=mad_window,
        min_history_days=min_mad_days,
    )

    return daily
