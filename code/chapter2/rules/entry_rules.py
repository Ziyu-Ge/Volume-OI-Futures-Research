from dataclasses import dataclass

import numpy as np

from core.indicators import (
    add_ma_features,
    add_slope_features,
    add_volatility_features,
    positive_part,
)


@dataclass(frozen=True)
class EntryConfig:
    ma_short: int = 5
    ma_long: int = 20
    ma_trend: int = 60
    ma_bias_threshold: float = 0.04
    ma_long_bias_threshold: float = 0.10
    ma_long_bias_cap: float | None = None
    oi_slope_window: int = 7
    close_slope_window: int = 7
    speculation_slope_window: int = 5
    speculation_slope_threshold: float = -0.01
    volatility_window: int = 10


def add_entry_signals(daily, config, use_speculation=False):
    """计算完整日频开仓信号；信号在收盘确认，下一根 bar 执行。"""
    data = daily.copy()
    data = add_ma_features(data, config.ma_short, config.ma_long, config.ma_trend)
    data = add_slope_features(
        data, "open_interest", "oi", config.oi_slope_window
    )
    data = add_slope_features(data, "close", "close", config.close_slope_window)
    data = add_volatility_features(data, config.volatility_window)

    data["ma_bias_spread_signal"] = (
        data["ma_bias_spread"] >= config.ma_bias_threshold
    ).astype(int)
    data["ma_long_bias_spread_signal"] = (
        data["ma_long_bias_spread"] >= config.ma_long_bias_threshold
    ).astype(int)
    if config.ma_long_bias_cap is None:
        data["ma_long_bias_spread_cap_signal"] = 1
    else:
        data["ma_long_bias_spread_cap_signal"] = (
            data["ma_long_bias_spread"] <= config.ma_long_bias_cap
        ).astype(int)

    speculation_score = 0
    if use_speculation:
        if "speculation" not in data.columns:
            raise ValueError("启用投机度条件时，日频数据必须包含 speculation 字段")
        data = add_slope_features(
            data, "speculation", "speculation", config.speculation_slope_window
        )
        data["speculation_slope_signal"] = (
            data["speculation_slope"] <= config.speculation_slope_threshold
        ).astype(int)
        speculation_score = positive_part(-data["speculation_slope"])
    else:
        data["speculation_slope_signal"] = 1

    data["open_short_signal"] = (
        (data["ma_bias_spread_signal"] == 1)
        & (data["ma_long_bias_spread_signal"] == 1)
        & (data["ma_long_bias_spread_cap_signal"] == 1)
        & (data["oi_slope_down"] == 1)
        & (data["close_slope_down"] == 1)
        & (data["speculation_slope_signal"] == 1)
    ).astype(int)

    # 分数只用于排序和排查，不参与交易执行。
    data["factor_value"] = (
        positive_part(data["ma_bias_spread"])
        + positive_part(data["ma_long_bias_spread"])
        + positive_part(-data["oi_slope_rate"])
        + positive_part(-data["close_slope_rate"])
        + speculation_score
    )
    return data

