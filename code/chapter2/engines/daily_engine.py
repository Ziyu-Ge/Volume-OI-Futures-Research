import numpy as np
import pandas as pd

from rules.entry_rules import add_entry_signals
from rules.exit_rules import check_short_exit


def run_daily_engine(daily, entry_config, exit_config, use_speculation=False):
    """21/23 共用：日频收盘确认，下一交易日开盘执行。"""
    frame = add_entry_signals(daily, entry_config, use_speculation)

    position = 0
    entry_price = np.nan
    low_since_entry = np.nan
    pending_open = False
    pending_cover = False
    pending_exit_reason = ""

    rows = []
    for _, row in frame.iterrows():
        result = _empty_result(position, entry_price, low_since_entry)
        open_price = row["open"]
        close = row["close"]
        low = row["low"]

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

    return _attach_results(frame, rows)


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


def _attach_results(frame, rows):
    result = frame.copy()
    result = pd.concat([result, pd.DataFrame(rows, index=result.index)], axis=1)
    result["short_entry_signal"] = (result["trade_signal"] == -1).astype(int)
    result["short_exit_signal"] = (result["trade_signal"] == 1).astype(int)
    result["signal"] = result["short_entry_signal"]
    return result
