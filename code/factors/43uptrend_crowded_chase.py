from volume_price_factor_utils import (
    add_volume_price_features,
    load_daily,
    positive_part,
    save_factor_outputs,
)


symbol = "LC"
factor_id = "43"
factor_name = "uptrend_crowded_chase"

price_rank_threshold = 0.75 
oi_ret_5_threshold = 0.05
volume_mad_threshold = 1.2
range_mad_threshold = 1.2
close_location_threshold = 0.95
signal_position_scale = 0


daily = load_daily(symbol)
daily = add_volume_price_features(daily)

daily["crowded_chase_score"] = (
    daily["close_rank_20"].fillna(0)
    + positive_part(daily["open_interest_mad_score"])
    + 0.5 * positive_part(daily["volume_mad_score"])
    + 0.5 * positive_part(daily["range_mad_score"])
)

daily["uptrend_crowded_chase_signal"] = 0
daily.loc[
    (
        daily["close_rank_20"] >= price_rank_threshold
    ) & (
        daily["ret_5"] > 0
    ) & (
        daily["oi_ret_5"] > oi_ret_5_threshold
    ) & (
        (daily["volume_mad_score"] >= volume_mad_threshold) |
        (daily["range_mad_score"] >= range_mad_threshold)
    ) & (
        daily["close_location"] <= close_location_threshold
    ),
    "uptrend_crowded_chase_signal",
] = 1

feature_columns = [
    "daily_return",
    "ret_5",
    "close_rank_20",
    "close_rank_60",
    "oi_ret_5",
    "open_interest_mad_score",
    "volume_mad_score",
    "range_pct",
    "range_mad_score",
    "close_location",
    "crowded_chase_score",
]

save_factor_outputs(
    daily=daily,
    symbol=symbol,
    factor_id=factor_id,
    factor_name=factor_name,
    factor_value_column="crowded_chase_score",
    signal_column="uptrend_crowded_chase_signal",
    position_scale_on_signal=signal_position_scale,
    feature_columns=feature_columns,
    figure_feature_columns=[
        "close_rank_20",
        "close_rank_60",
        "open_interest_mad_score",
        "volume_mad_score",
        "range_mad_score",
    ],
)
