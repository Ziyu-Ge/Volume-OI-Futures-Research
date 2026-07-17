from dataclasses import dataclass

import numpy as np
import pandas as pd


@dataclass(frozen=True)
class ExitConfig:
    trailing_multiplier: float = 4.0
    entry_loss_volatility_multiplier: float = 0.0


def exit_reason(price_above_entry, trailing_rebound):
    if price_above_entry and trailing_rebound:
        return "price_above_entry_and_trailing_rebound"
    if price_above_entry:
        return "price_above_entry"
    if trailing_rebound:
        return "trailing_rebound"
    return ""


def check_short_exit(close, entry_price, low_since_entry, avg_volatility_rate, config):
    """空头平仓规则：突破波动调整后的开仓线，或从低点反弹超过阈值。"""
    entry_loss_stop_price = np.nan
    if pd.notna(entry_price):
        if config.entry_loss_volatility_multiplier == 0:
            entry_loss_stop_price = entry_price
        elif pd.notna(avg_volatility_rate):
            entry_loss_stop_price = entry_price * (
                1
                + avg_volatility_rate
                * config.entry_loss_volatility_multiplier
            )
    price_above_entry = bool(
        pd.notna(close)
        and pd.notna(entry_loss_stop_price)
        and close > entry_loss_stop_price
    )
    trailing_stop_price = np.nan
    trailing_rebound = False

    if pd.notna(close) and pd.notna(low_since_entry) and pd.notna(avg_volatility_rate):
        trailing_stop_price = low_since_entry * (
            1 + avg_volatility_rate * config.trailing_multiplier
        )
        trailing_rebound = bool(close > trailing_stop_price)

    return {
        "cover_signal": int(price_above_entry or trailing_rebound),
        "price_above_entry_signal": int(price_above_entry),
        "trailing_rebound_signal": int(trailing_rebound),
        "entry_loss_stop_price": entry_loss_stop_price,
        "trailing_stop_price": trailing_stop_price,
        "exit_reason": exit_reason(price_above_entry, trailing_rebound),
    }
