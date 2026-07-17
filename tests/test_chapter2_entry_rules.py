from pathlib import Path
import sys

import pandas as pd


CHAPTER2_DIR = Path(__file__).resolve().parents[1] / "code" / "chapter2"
sys.path.insert(0, str(CHAPTER2_DIR))

from rules.entry_rules import EntryConfig, add_entry_signals


def _daily_frame():
    close = [100.0, 105.0, 110.0, 108.0, 106.0, 104.0]
    return pd.DataFrame(
        {
            "date": pd.date_range("2024-01-01", periods=len(close)),
            "open": close,
            "close": close,
            "high": [value + 1 for value in close],
            "low": [value - 1 for value in close],
            "volume": [1000.0] * len(close),
            "open_interest": [100.0, 110.0, 120.0, 119.0, 118.0, 117.0],
        }
    )


def _config(**overrides):
    values = {
        "ma_short": 2,
        "ma_long": 3,
        "ma_trend": 4,
        "ma_bias_threshold": -1.0,
        "ma_long_bias_threshold": -1.0,
        "oi_slope_window": 3,
        "close_slope_window": 3,
        "volatility_window": 2,
    }
    values.update(overrides)
    return EntryConfig(**values)


def test_zero_slope_thresholds_preserve_downward_slope_filter():
    result = add_entry_signals(_daily_frame(), _config())

    assert result.loc[5, "oi_slope_rate"] < 0
    assert result.loc[5, "close_slope_rate"] < 0
    assert result.loc[5, "open_short_signal"] == 1


def test_slope_thresholds_can_require_a_stronger_decline():
    baseline = add_entry_signals(_daily_frame(), _config())
    strict = add_entry_signals(
        _daily_frame(),
        _config(
            oi_slope_threshold=baseline.loc[5, "oi_slope_rate"] - 0.001,
            close_slope_threshold=baseline.loc[5, "close_slope_rate"] - 0.001,
        ),
    )

    assert strict.loc[5, "oi_slope_signal"] == 0
    assert strict.loc[5, "close_slope_signal"] == 0
    assert strict.loc[5, "open_short_signal"] == 0
