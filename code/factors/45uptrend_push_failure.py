from volume_price_factor_utils import (
    add_volume_price_features,
    load_daily,
    positive_part,
    save_factor_outputs,
)


symbol = "LC"
factor_id = "45"
factor_name = "uptrend_push_failure"

price_rank_threshold = 0.75
ret_3_threshold = 0.015
close_location_threshold = 0.55
volume_mad_threshold = 0.3
price_efficiency_low_rank_threshold = 0.4
signal_position_scale = 0


daily = load_daily(symbol)
daily = add_volume_price_features(daily)

weak_close_score = 1 - daily["close_location"].clip(lower=0, upper=1)

daily["push_failure_score"] = (
    daily["close_rank_20"].fillna(0)
    + daily["price_efficiency_low_rank_20"].fillna(0)
    + 0.5 * positive_part(daily["open_interest_mad_score"])
    + 0.25 * positive_part(daily["volume_mad_score"])
    + weak_close_score.fillna(0)
)

daily["uptrend_push_failure_signal"] = 0
daily.loc[
    (
        daily["close_rank_20"] >= price_rank_threshold
    ) & (
        daily["oi_ret_3"] > 0
    ) & (
        daily["volume_mad_score"] >= volume_mad_threshold
    ) & (
        daily["price_efficiency_low_rank_20"] >= (
            price_efficiency_low_rank_threshold
        )
    ) & (
        (daily["ret_3"] <= ret_3_threshold) |
        (daily["close_location"] <= close_location_threshold)
    ),
    "uptrend_push_failure_signal",
] = 1

feature_columns = [
    "ret_3",
    "ret_5",
    "close_rank_20",
    "close_rank_60",
    "oi_ret_3",
    "volume_mad_score",
    "close_location",
    "volume_to_open_interest",
    "price_efficiency_5",
    "price_efficiency_low_rank_20",
    "push_failure_score",
]

save_factor_outputs(
    daily=daily,
    symbol=symbol,
    factor_id=factor_id,
    factor_name=factor_name,
    factor_value_column="push_failure_score",
    signal_column="uptrend_push_failure_signal",
    position_scale_on_signal=signal_position_scale,
    feature_columns=feature_columns,
    figure_feature_columns=[
        "price_efficiency_low_rank_20",
        "close_location",
        "open_interest_mad_score",
    ],
)
