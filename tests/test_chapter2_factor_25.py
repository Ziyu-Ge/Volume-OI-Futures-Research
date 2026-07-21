from pathlib import Path
import sys


CHAPTER2_DIR = Path(__file__).resolve().parents[1] / "code" / "chapter2"
sys.path.insert(0, str(CHAPTER2_DIR))

from factors import factor_24, factor_25
from run_factor import load_factor, parse_args
from rules.entry_rules import EntryConfig


def test_factor_25_uses_restored_entry_conditions():
    assert factor_25.FACTOR_ID == "25"
    assert factor_25.FACTOR_NAME == "high_bias_oi_speculation_drop_hourly"
    assert factor_25.ENGINE == "hourly_exit"
    assert factor_25.USE_SPECULATION is True
    assert factor_25.ENTRY_CONFIG == EntryConfig(
        ma_short=5,
        ma_long=20,
        ma_trend=60,
        ma_bias_threshold=0.04,
        ma_long_bias_threshold=0.10,
        ma_long_bias_cap=0.18,
        oi_slope_window=5,
        oi_slope_threshold=0.0,
        close_slope_window=7,
        close_slope_threshold=0.0,
        speculation_slope_window=5,
        speculation_slope_threshold=-0.01,
        volatility_window=10,
    )


def test_factor_25_keeps_factor_24_exit_conditions():
    assert factor_25.EXIT_CONFIG == factor_24.EXIT_CONFIG
    assert factor_25.EXIT_CONFIG.trailing_multiplier == 1.225
    assert factor_25.EXIT_CONFIG.entry_loss_volatility_multiplier == 1.5


def test_factor_25_is_available_from_cli(monkeypatch):
    monkeypatch.setattr(sys, "argv", ["run_factor.py", "--factor", "25"])

    assert parse_args().factor == "25"
    assert load_factor("25").FACTOR_ID == "25"
