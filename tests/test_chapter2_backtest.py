from pathlib import Path
import sys

import pandas as pd


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
