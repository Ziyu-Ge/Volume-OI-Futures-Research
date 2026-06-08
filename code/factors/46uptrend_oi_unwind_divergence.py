from volume_price_factor_utils import (
    add_volume_price_features,
    load_daily_and_segments,
    positive_part,
    save_factor_outputs,
)


symbol = "LC"
factor_id = "46"
factor_name = "uptrend_oi_unwind_divergence"

ret_10_threshold = 0.08
price_rank_threshold = 0.60
volume_mad_threshold = 0.5
speculation_mad_threshold = 0.5
signal_position_scale = -1


daily, segments = load_daily_and_segments(symbol)
daily = add_volume_price_features(daily)

daily["oi_unwind_divergence_score"] = (
    daily["close_rank_20"].fillna(0)
    + positive_part(-daily["oi_ret_5"]) * 10
    + positive_part(daily["volume_mad_score"])
    + positive_part(daily["speculation_mad_score"])
)

daily["uptrend_oi_unwind_divergence_signal"] = 0
daily.loc[
    (
        daily["realtime_trend"] == "up_trend"
    ) & (
        daily["close_rank_20"] >= price_rank_threshold
    ) & (
        daily["ret_10"] > ret_10_threshold
    ) & (
        daily["oi_ret_5"] < 0
    ) & (
        daily["volume_mad_score"] >= volume_mad_threshold
    ) & (
        daily["speculation_mad_score"] >= speculation_mad_threshold
    ),
    "uptrend_oi_unwind_divergence_signal",
] = 1

feature_columns = [
    "ret_10",
    "realtime_trend_age",
    "close_rank_20",
    "close_rank_60",
    "oi_ret_5",
    "volume_mad_score",
    "speculation_mad_score",
    "volume_to_open_interest",
    "oi_unwind_divergence_score",
]

save_factor_outputs(
    daily=daily,
    segments=segments,
    symbol=symbol,
    factor_id=factor_id,
    factor_name=factor_name,
    factor_value_column="oi_unwind_divergence_score",
    signal_column="uptrend_oi_unwind_divergence_signal",
    position_scale_on_signal=signal_position_scale,
    feature_columns=feature_columns,
    figure_feature_columns=[
        "ret_10",
        "oi_ret_5",
        "volume_mad_score",
        "speculation_mad_score",
    ],
    target_trend="up_trend",
)
