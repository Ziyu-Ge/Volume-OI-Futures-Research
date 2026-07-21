from pathlib import Path
import sys

import pandas as pd
import pytest


CHAPTER2_DIR = Path(__file__).resolve().parents[1] / "code" / "chapter2"
sys.path.insert(0, str(CHAPTER2_DIR))

from engines import backtest
from run_factor import combine_outputs


class Factor:
    FACTOR_ID = "test"
    FACTOR_NAME = "test"


def _frame(symbol, trade_signals, positions, closes):
    dates = pd.to_datetime(
        ["2026-01-01 09:00:00", "2026-01-02 09:00:00", "2026-01-02 10:00:00"]
    )
    data = pd.DataFrame(
        {
            "date": dates,
            "trading_date": dates.date,
            "open": [100.0, 100.0, 100.0],
            "close": closes,
            "position": positions,
            "trade_signal": trade_signals,
            "entry_price": [pd.NA, 100.0, pd.NA],
            "exit_price": [pd.NA, pd.NA, 100.0],
            "exit_reason": ["", "", "test"],
        }
    )
    trades = backtest.build_trade_table(
        data, symbol, Factor.FACTOR_ID, Factor.FACTOR_NAME
    )
    return {"symbol": symbol, "frame": data, "trades": trades, "periods": 252}


def test_combine_outputs_starts_at_global_first_actual_entry():
    first = _frame("A", [0, -1, 1], [0, -1, 0], [100.0, 110.0, 100.0])
    no_trade = _frame("B", [0, 0, 0], [0, 0, 0], [100.0, 101.0, 102.0])

    metrics, trades, curves, portfolio, processed, start_time = combine_outputs(
        [first, no_trade], Factor
    )

    assert start_time == pd.Timestamp("2026-01-02 09:00:00")
    assert curves["date"].min() == start_time
    assert all(item["frame"]["date"].min() == start_time for item in processed)

    first_rows = curves.sort_values(["symbol", "date"]).groupby("symbol").head(1)
    strategy_by_symbol = first_rows.set_index("symbol")["strategy_return"]
    cumulative_by_symbol = first_rows.set_index("symbol")[
        "strategy_cumulative_return"
    ]
    assert strategy_by_symbol["A"] == pytest.approx(-0.10)
    assert strategy_by_symbol["B"] == pytest.approx(0.01)
    assert cumulative_by_symbol["A"] == pytest.approx(strategy_by_symbol["A"])
    assert portfolio["date"].iloc[0] == start_time
    assert portfolio["symbol_count"].iloc[0] == 2
    assert set(metrics["symbol"]) == {"A", "B", "ALL_SYMBOLS_EQUAL_WEIGHT"}
    assert trades["entry_time"].min() == start_time


def test_combine_outputs_errors_when_no_symbol_has_actual_entry():
    no_trade = _frame("B", [0, 0, 0], [0, 0, 0], [100.0, 101.0, 102.0])

    with pytest.raises(RuntimeError, match="没有任何品种产生实际开仓"):
        combine_outputs([no_trade], Factor)
