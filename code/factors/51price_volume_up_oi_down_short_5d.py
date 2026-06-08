import numpy as np

from volume_price_factor_utils import (
    load_daily_and_segments,
    positive_part,
    save_factor_outputs,
)


symbol = "LC"
factor_id = "51"
factor_name = "price_volume_up_oi_down_short_5d"

compare_window = 5
signal_position_scale = -1
signal_holding_days = 5


daily, segments = load_daily_and_segments(symbol)

daily["daily_return"] = daily["close"].pct_change()

daily["price_ma_5"] = (
    daily["close"]
    .rolling(window=compare_window, min_periods=compare_window)
    .mean()
)
daily["price_ma_5_prev"] = daily["price_ma_5"].shift(compare_window)
daily["price_ma_5_change"] = (
    daily["price_ma_5"] /
    daily["price_ma_5_prev"].replace(0, np.nan) -
    1
)

daily["volume_ma_5"] = (
    daily["volume"]
    .rolling(window=compare_window, min_periods=compare_window)
    .mean()
)
daily["volume_ma_5_prev"] = daily["volume_ma_5"].shift(compare_window)
daily["volume_ma_5_change"] = (
    daily["volume_ma_5"] /
    daily["volume_ma_5_prev"].replace(0, np.nan) -
    1
)

daily["oi_change_5"] = (
    daily["open_interest"] -
    daily["open_interest"].shift(compare_window)
)
daily["oi_change_5_rate"] = (
    daily["oi_change_5"] /
    daily["open_interest"].shift(compare_window).replace(0, np.nan)
)

daily["is_price_ma_5_up"] = (
    daily["price_ma_5"] > daily["price_ma_5_prev"]
).astype(int)
daily["is_volume_ma_5_up"] = (
    daily["volume_ma_5"] > daily["volume_ma_5_prev"]
).astype(int)
daily["is_oi_down_5"] = (daily["oi_change_5"] < 0).astype(int)

daily["price_volume_oi_short_score"] = (
    positive_part(daily["price_ma_5_change"])
    + positive_part(daily["volume_ma_5_change"])
    + positive_part(-daily["oi_change_5_rate"])
)

daily["price_volume_up_oi_down_short_5d_signal"] = 0
daily.loc[
    (
        daily["is_price_ma_5_up"] == 1
    ) & (
        daily["is_oi_down_5"] == 1
    ) & (
        daily["is_volume_ma_5_up"] == 1
    ),
    "price_volume_up_oi_down_short_5d_signal",
] = 1

feature_columns = [
    "daily_return",
    "price_ma_5",
    "price_ma_5_prev",
    "price_ma_5_change",
    "volume_ma_5",
    "volume_ma_5_prev",
    "volume_ma_5_change",
    "oi_change_5",
    "oi_change_5_rate",
    "is_price_ma_5_up",
    "is_volume_ma_5_up",
    "is_oi_down_5",
    "price_volume_oi_short_score",
]

save_factor_outputs(
    daily=daily,
    segments=segments,
    symbol=symbol,
    factor_id=factor_id,
    factor_name=factor_name,
    factor_value_column="price_volume_oi_short_score",
    signal_column="price_volume_up_oi_down_short_5d_signal",
    position_scale_on_signal=signal_position_scale,
    feature_columns=feature_columns,
    figure_feature_columns=[
        "price_ma_5_change",
        "volume_ma_5_change",
        "oi_change_5_rate",
    ],
    target_trend="up_trend",
    signal_holding_days=signal_holding_days,
)
