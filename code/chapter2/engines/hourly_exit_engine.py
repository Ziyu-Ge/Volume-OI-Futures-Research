import numpy as np
import pandas as pd

from rules.entry_rules import add_entry_signals
from rules.exit_rules import check_short_exit


def run_hourly_exit_engine(daily, hourly, entry_config, exit_config, use_speculation=False):
    """日频确认开仓，小时频执行和平仓。"""
    entry_daily = add_entry_signals(daily, entry_config, use_speculation)
    frame = build_intraday_frame(hourly)
    frame = attach_daily_entry(frame, entry_daily)
    return run_hourly_state_machine(frame, exit_config)


def build_intraday_frame(hourly):
    bars = hourly.sort_values(["trading_date", "date"]).reset_index(drop=True)
    grouped = bars.groupby("trading_date", sort=False)
    frame = pd.DataFrame(
        {
            "date": bars["date"],
            "trading_date": bars["trading_date"],
            "hourly_open": bars["open"],
            "hourly_low": bars["low"],
            "open": grouped["open"].transform("first"),
            "close": bars["close"],
            "high": grouped["high"].cummax(),
            "low": grouped["low"].cummin(),
        }
    )
    return frame


def attach_daily_entry(frame, entry_daily):
    daily_index = entry_daily.set_index("date")
    data = frame.copy()
    data["complete_daily_open_short_signal"] = (
        data["trading_date"]
        .map(daily_index["open_short_signal"])
        .fillna(0)
        .astype(int)
    )
    data["factor_value"] = data["trading_date"].map(daily_index["factor_value"])
    data["avg_volatility_rate"] = data["trading_date"].map(
        daily_index["avg_volatility_rate"]
    )
    data["is_complete_daily_bar"] = (
        data["trading_date"] != data["trading_date"].shift(-1)
    ).astype(int)
    data["open_short_signal"] = (
        (data["is_complete_daily_bar"] == 1)
        & (data["complete_daily_open_short_signal"] == 1)
    ).astype(int)
    return data


def run_hourly_state_machine(frame, exit_config):
    position = 0
    entry_price = np.nan
    low_since_entry = np.nan
    pending_open = False
    pending_cover = False
    pending_exit_reason = ""

    rows = []
    for _, row in frame.iterrows():
        result = _empty_result(position, entry_price, low_since_entry)
        open_price = row["hourly_open"]
        close = row["close"]
        low = row["hourly_low"]

        if position == -1 and pending_cover and pd.notna(open_price):
            result.update(
                _close_short(open_price, entry_price, low_since_entry, pending_exit_reason)
            )
            position = 0
            entry_price = np.nan
            low_since_entry = np.nan
        elif position == 0 and pending_open and pd.notna(open_price):
            position = -1
            entry_price = open_price
            low_since_entry = open_price
            result.update(_open_short(entry_price, low_since_entry))

        pending_open = False
        pending_cover = False
        pending_exit_reason = ""

        if position == -1:
            low_candidate = low if pd.notna(low) else close
            if pd.notna(low_candidate):
                low_since_entry = min(low_since_entry, low_candidate)
            exit_info = check_short_exit(
                close,
                entry_price,
                low_since_entry,
                row["avg_volatility_rate"],
                exit_config,
            )
            result.update(exit_info)
            result["position"] = -1
            result["entry_price"] = entry_price
            result["low_since_entry"] = low_since_entry
            pending_cover = bool(exit_info["cover_signal"])
            pending_exit_reason = exit_info["exit_reason"]

        pending_open = bool(row["open_short_signal"])
        rows.append(result)

    result = pd.concat([frame.copy(), pd.DataFrame(rows, index=frame.index)], axis=1)
    result["short_entry_signal"] = (result["trade_signal"] == -1).astype(int)
    result["short_exit_signal"] = (result["trade_signal"] == 1).astype(int)
    result["signal"] = result["short_entry_signal"]
    return result


def _empty_result(position, entry_price, low_since_entry):
    return {
        "position": position,
        "trade_signal": 0,
        "trade_action": "",
        "entry_price": entry_price if position == -1 else np.nan,
        "exit_price": np.nan,
        "exit_reason": "",
        "low_since_entry": low_since_entry if position == -1 else np.nan,
        "cover_signal": 0,
        "price_above_entry_signal": 0,
        "trailing_rebound_signal": 0,
        "entry_loss_stop_price": np.nan,
        "trailing_stop_price": np.nan,
    }


def _open_short(entry_price, low_since_entry):
    return {
        "position": -1,
        "trade_signal": -1,
        "trade_action": "open_short",
        "entry_price": entry_price,
        "low_since_entry": low_since_entry,
    }


def _close_short(exit_price, entry_price, low_since_entry, reason):
    return {
        "position": 0,
        "trade_signal": 1,
        "trade_action": "cover_short",
        "entry_price": entry_price,
        "exit_price": exit_price,
        "exit_reason": reason or "cover_short",
        "low_since_entry": low_since_entry,
    }
