from pathlib import Path
import sys

import numpy as np
import pytest


CHAPTER2_DIR = Path(__file__).resolve().parents[1] / "code" / "chapter2"
sys.path.insert(0, str(CHAPTER2_DIR))

from rules.exit_rules import ExitConfig, check_short_exit


def test_volatility_buffer_avoids_exit_just_above_entry():
    config = ExitConfig(
        trailing_multiplier=10.0,
        entry_loss_volatility_multiplier=1.5,
    )

    result = check_short_exit(
        close=101.0,
        entry_price=100.0,
        low_since_entry=100.0,
        avg_volatility_rate=0.02,
        config=config,
    )

    assert result["entry_loss_stop_price"] == pytest.approx(103.0)
    assert result["price_above_entry_signal"] == 0
    assert result["cover_signal"] == 0


def test_volatility_buffer_exits_after_adjusted_entry_line_breaks():
    config = ExitConfig(
        trailing_multiplier=10.0,
        entry_loss_volatility_multiplier=1.5,
    )

    result = check_short_exit(
        close=103.01,
        entry_price=100.0,
        low_since_entry=100.0,
        avg_volatility_rate=0.02,
        config=config,
    )

    assert result["price_above_entry_signal"] == 1
    assert result["cover_signal"] == 1
    assert result["exit_reason"] == "price_above_entry"


def test_zero_buffer_preserves_original_entry_price_exit():
    result = check_short_exit(
        close=101.0,
        entry_price=100.0,
        low_since_entry=100.0,
        avg_volatility_rate=np.nan,
        config=ExitConfig(),
    )

    assert result["entry_loss_stop_price"] == 100.0
    assert result["price_above_entry_signal"] == 1
    assert result["cover_signal"] == 1


def test_trailing_rebound_still_exits_inside_entry_loss_buffer():
    config = ExitConfig(
        trailing_multiplier=1.0,
        entry_loss_volatility_multiplier=1.5,
    )

    result = check_short_exit(
        close=93.0,
        entry_price=100.0,
        low_since_entry=90.0,
        avg_volatility_rate=0.02,
        config=config,
    )

    assert result["price_above_entry_signal"] == 0
    assert result["trailing_rebound_signal"] == 1
    assert result["cover_signal"] == 1
    assert result["exit_reason"] == "trailing_rebound"
