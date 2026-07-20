from pathlib import Path
import sys

import numpy as np
import pandas as pd
import pytest


CHAPTER2_DIR = Path(__file__).resolve().parents[1] / "code" / "chapter2"
sys.path.insert(0, str(CHAPTER2_DIR))

from core.metrics import add_curve_columns, annual_return, max_drawdown
from engines.backtest import build_portfolio


def test_curve_uses_simple_cumulative_returns():
    frame = pd.DataFrame({"return": [0.10, 0.10, -0.05, np.nan]})

    result = add_curve_columns(frame, "return", "strategy")

    assert result["strategy_cumulative_return"].tolist() == pytest.approx(
        [0.10, 0.20, 0.15, 0.15]
    )
    assert result["strategy_equity"].tolist() == pytest.approx(
        [1.10, 1.20, 1.15, 1.15]
    )


def test_annual_return_uses_linear_annualization():
    assert annual_return([0.10, 0.10], periods_per_year=4) == pytest.approx(0.40)


def test_max_drawdown_uses_simple_equity_curve():
    returns = [0.0, 0.10, -0.20, 0.05]

    assert max_drawdown(returns) == pytest.approx(0.90 / 1.10 - 1)


def test_drawdown_includes_initial_equity_as_high_water_mark():
    frame = pd.DataFrame({"return": [-0.10, 0.05]})

    result = add_curve_columns(frame, "return", "strategy")

    assert result["strategy_drawdown"].tolist() == pytest.approx([-0.10, -0.05])


def test_portfolio_excess_is_strategy_minus_benchmark():
    curves = pd.DataFrame(
        {
            "date": ["2026-01-01", "2026-01-01", "2026-01-02", "2026-01-02"],
            "symbol": ["A", "B", "A", "B"],
            "strategy_return": [0.10, 0.30, -0.20, 0.00],
            "benchmark_return": [0.05, 0.10, 0.10, 0.00],
        }
    )

    portfolio, metrics = build_portfolio(
        curves, factor_id="test", factor_name="test", periods_per_year=2
    )

    assert portfolio["excess_cumulative_return"].tolist() == pytest.approx(
        portfolio["strategy_cumulative_return"]
        - portfolio["benchmark_cumulative_return"]
    )
    assert metrics["excess_annual_return"] == pytest.approx(
        metrics["annual_return"] - metrics["benchmark_annual_return"]
    )
