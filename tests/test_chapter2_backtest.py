from pathlib import Path
import sys

import pandas as pd
import pytest


CHAPTER2_DIR = Path(__file__).resolve().parents[1] / "code" / "chapter2"
sys.path.insert(0, str(CHAPTER2_DIR))

from engines.backtest import add_return_columns


def test_strategy_uses_full_long_or_full_short_positions():
    frame = pd.DataFrame(
        {
            "open": [100.0, 101.0, 99.0],
            "close": [101.0, 99.0, 100.0],
            "position": [0, -1, 0],
        }
    )

    result = add_return_columns(frame)

    assert result["strategy_net_position"].tolist() == [1, -1, 1]
    assert result["previous_strategy_net_position"].tolist() == [1, 1, -1]


def test_first_period_keeps_open_to_close_return_after_trim():
    frame = pd.DataFrame(
        {
            "open": [100.0],
            "close": [101.0],
            "position": [-1],
        }
    )

    result = add_return_columns(frame)

    assert result["gap_return"].iloc[0] == 0
    assert result["intraday_return"].iloc[0] == pytest.approx(0.01)
    assert result["benchmark_return"].iloc[0] == pytest.approx(0.01)
    assert result["strategy_return"].iloc[0] == pytest.approx(-0.01)
