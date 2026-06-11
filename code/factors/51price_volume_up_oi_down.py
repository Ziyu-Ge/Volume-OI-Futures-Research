import numpy as np

from volume_price_factor_utils import (
    load_daily_and_segments,
    mad_score,
    past_rank,
    positive_part,
    save_factor_outputs,
)


symbol = "LC"
factor_id = "51"
factor_name = "price_volume_up_oi_down"

price_rank_threshold = 0.70
oi_change_mad_threshold = 1.0
volume_mad_threshold = 1.0

price_rank_window = 20
price_rank_long_window = 60
min_rank_days = 8
min_rank_long_days = 20

price_short_window = 5
price_mid_window = 10
price_long_window = 20

open_interest_short_window = 5
open_interest_mid_window = 10

oi_change_window = 5
oi_change_mad_window = 10
volume_mad_window = 10
min_mad_days = 5

signal_position_scale = 0
signal_holding_days = 5


daily, segments = load_daily_and_segments(symbol)

daily["daily_return"] = daily["close"].pct_change()
daily["ret_5"] = daily["close"].pct_change(5)

daily["close_rank_20"] = past_rank(
    daily["close"],
    window=price_rank_window,
    min_history_days=min_rank_days,
)
daily["close_rank_60"] = past_rank(
    daily["close"],
    window=price_rank_long_window,
    min_history_days=min_rank_long_days,
)

safe_volume = daily["volume"].where(daily["volume"] > 0, np.nan)
daily["log_volume"] = np.log(safe_volume)

(
    daily["log_volume_median_past"],
    daily["log_volume_mad_past"],
    daily["volume_mad_score"],
) = mad_score(
    daily["log_volume"],
    window=volume_mad_window,
    min_history_days=min_mad_days,
)

# 价格条件：
# 1. 价格处于 20 日历史高位
# 2. 价格均线多头排列：MA5 > MA10 > MA20
# 3. 价格 5 日收益为正
daily["price_ma_5"] = (
    daily["close"]
    .rolling(window=price_short_window, min_periods=price_short_window)
    .mean()
)
daily["price_ma_10"] = (
    daily["close"]
    .rolling(window=price_mid_window, min_periods=price_mid_window)
    .mean()
)
daily["price_ma_20"] = (
    daily["close"]
    .rolling(window=price_long_window, min_periods=price_long_window)
    .mean()
)
daily["is_price_ma_bullish"] = (
    (daily["price_ma_5"] > daily["price_ma_10"]) &
    (daily["price_ma_10"] > daily["price_ma_20"])
).astype(int)

# 持仓量偏高条件：持仓量 MA5 > MA10
daily["open_interest_ma_5"] = (
    daily["open_interest"]
    .rolling(
        window=open_interest_short_window,
        min_periods=open_interest_short_window
    )
    .mean()
)
daily["open_interest_ma_10"] = (
    daily["open_interest"]
    .rolling(
        window=open_interest_mid_window,
        min_periods=open_interest_mid_window
    )
    .mean()
)
daily["is_open_interest_ma_bullish"] = (
    daily["open_interest_ma_5"] > daily["open_interest_ma_10"]
).astype(int)

# OI 变化率的 MAD score：
# 使用 5 日持仓量变化率，并和过去 oi_change_mad_window 天比较。
daily["oi_change_5"] = (
    daily["open_interest"] -
    daily["open_interest"].shift(oi_change_window)
)
daily["oi_change_5_rate"] = (
    daily["oi_change_5"] /
    daily["open_interest"].shift(oi_change_window).replace(0, np.nan)
)

(
    daily["oi_change_5_rate_median_past"],
    daily["oi_change_5_rate_mad_past"],
    daily["oi_change_5_rate_mad_score"],
) = mad_score(
    daily["oi_change_5_rate"],
    window=oi_change_mad_window,
    min_history_days=min_mad_days,
)

daily["oi_change_5_rate_mad_abs_score"] = (
    daily["oi_change_5_rate_mad_score"].abs()
)

daily["price_volume_oi_score"] = (
    daily["close_rank_20"].fillna(0)
    + daily["is_price_ma_bullish"].fillna(0)
    + positive_part(daily["ret_5"])
    + daily["is_open_interest_ma_bullish"].fillna(0)
    + positive_part(daily["oi_change_5_rate_mad_abs_score"])
    + positive_part(daily["volume_mad_score"])
)

daily["price_volume_up_oi_down_signal"] = 0
daily.loc[
    (
        daily["close_rank_20"] >= price_rank_threshold
    ) & (
        daily["is_price_ma_bullish"] == 1
    ) & (
        daily["ret_5"] > 0
    ) & (
        daily["is_open_interest_ma_bullish"] == 1
    ) & (
        daily["oi_change_5_rate_mad_abs_score"] >= oi_change_mad_threshold
    ) & (
        daily["volume_mad_score"] >= volume_mad_threshold
    ),
    "price_volume_up_oi_down_signal",
] = 1

feature_columns = [
    "daily_return",
    "ret_5",
    "close_rank_20",
    "close_rank_60",
    "price_ma_5",
    "price_ma_10",
    "price_ma_20",
    "is_price_ma_bullish",
    "open_interest_ma_5",
    "open_interest_ma_10",
    "is_open_interest_ma_bullish",
    "oi_change_5",
    "oi_change_5_rate",
    "oi_change_5_rate_median_past",
    "oi_change_5_rate_mad_past",
    "oi_change_5_rate_mad_score",
    "oi_change_5_rate_mad_abs_score",
    "log_volume",
    "log_volume_median_past",
    "log_volume_mad_past",
    "volume_mad_score",
    "price_volume_oi_score",
]

save_factor_outputs(
    daily=daily,
    segments=segments,
    symbol=symbol,
    factor_id=factor_id,
    factor_name=factor_name,
    factor_value_column="price_volume_oi_score",
    signal_column="price_volume_up_oi_down_signal",
    position_scale_on_signal=signal_position_scale,
    feature_columns=feature_columns,
    figure_feature_columns=[
        "close_rank_20",
        "ret_5",
        "oi_change_5_rate_mad_abs_score",
        "volume_mad_score",
    ],
    signal_holding_days=signal_holding_days,
)
