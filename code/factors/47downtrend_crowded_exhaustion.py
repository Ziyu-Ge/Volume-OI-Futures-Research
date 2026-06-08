from volume_price_factor_utils import (
    add_volume_price_features,
    load_daily_and_segments,
    positive_part,
    save_factor_outputs,
)


symbol = "LC"
factor_id = "47"
factor_name = "downtrend_crowded_exhaustion"

close_rank_threshold = 0.25
ret_5_threshold = -0.02
oi_ret_5_min_threshold = 0.02
oi_ret_5_max_threshold = 0.09
open_interest_mad_max_threshold = 2.0
range_mad_threshold = 0.8
close_location_threshold = 0.45
signal_position_scale = -1


daily, segments = load_daily_and_segments(symbol)
daily = add_volume_price_features(daily)

low_price_rank_score = 1 - daily["close_rank_20"]

daily["downtrend_exhaustion_score"] = (
    low_price_rank_score.fillna(0)
    + positive_part(daily["range_mad_score"])
    + positive_part(daily["oi_ret_5"] * 10)
    + daily["close_location"].fillna(0)
    - positive_part(
        daily["open_interest_mad_score"] -
        open_interest_mad_max_threshold
    )
)

daily["downtrend_crowded_exhaustion_signal"] = 0
daily.loc[
    (
        daily["realtime_trend"] == "down_trend"
    ) & (
        daily["close_rank_20"] <= close_rank_threshold
    ) & (
        daily["ret_5"] < ret_5_threshold
    ) & (
        daily["oi_ret_5"] > oi_ret_5_min_threshold
    ) & (
        daily["oi_ret_5"] <= oi_ret_5_max_threshold
    ) & (
        daily["open_interest_mad_score"] <= (
            open_interest_mad_max_threshold
        )
    ) & (
        daily["range_mad_score"] >= range_mad_threshold
    ) & (
        daily["close_location"] >= close_location_threshold
    ),
    "downtrend_crowded_exhaustion_signal",
] = 1

feature_columns = [
    "ret_5",
    "realtime_trend_age",
    "close_rank_20",
    "close_rank_60",
    "oi_ret_5",
    "open_interest_mad_score",
    "range_pct",
    "range_mad_score",
    "close_location",
    "downtrend_exhaustion_score",
]

save_factor_outputs(
    daily=daily,
    segments=segments,
    symbol=symbol,
    factor_id=factor_id,
    factor_name=factor_name,
    factor_value_column="downtrend_exhaustion_score",
    signal_column="downtrend_crowded_exhaustion_signal",
    position_scale_on_signal=signal_position_scale,
    feature_columns=feature_columns,
    figure_feature_columns=[
        "close_rank_20",
        "open_interest_mad_score",
        "range_mad_score",
        "close_location",
    ],
    target_trend="down_trend",
)
