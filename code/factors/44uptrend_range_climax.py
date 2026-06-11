from volume_price_factor_utils import (
    add_volume_price_features,
    load_daily,
    positive_part,
    save_factor_outputs,
)


symbol = "LC"
factor_id = "44"
factor_name = "uptrend_range_climax"

price_rank_threshold = 0.70
range_mad_threshold = 1.0
volume_mad_threshold = 0.8
close_location_threshold = 0.55
signal_position_scale = 0


daily = load_daily(symbol)
daily = add_volume_price_features(daily)

weak_close_score = 1 - daily["close_location"].clip(lower=0, upper=1)

daily["range_climax_score"] = (
    daily["close_rank_20"].fillna(0)
    + positive_part(daily["range_mad_score"]) * weak_close_score.fillna(0)
    + 0.5 * positive_part(daily["open_interest_mad_score"])
    + 0.5 * positive_part(daily["volume_mad_score"])
)

daily["uptrend_range_climax_signal"] = 0
daily.loc[
    (
        daily["close_rank_20"] >= price_rank_threshold
    ) & (
        daily["range_mad_score"] >= range_mad_threshold
    ) & (
        (daily["oi_ret_3"] > 0) |
        (daily["volume_mad_score"] >= volume_mad_threshold)
    ) & (
        daily["close_location"] <= close_location_threshold
    ),
    "uptrend_range_climax_signal",
] = 1

feature_columns = [
    "ret_3",
    "close_rank_20",
    "close_rank_60",
    "oi_ret_3",
    "volume_mad_score",
    "range_pct",
    "range_mad_score",
    "close_location",
    "range_climax_score",
]

save_factor_outputs(
    daily=daily,
    symbol=symbol,
    factor_id=factor_id,
    factor_name=factor_name,
    factor_value_column="range_climax_score",
    signal_column="uptrend_range_climax_signal",
    position_scale_on_signal=signal_position_scale,
    feature_columns=feature_columns,
    figure_feature_columns=[
        "range_mad_score",
        "close_location",
        "volume_mad_score",
    ],
)
