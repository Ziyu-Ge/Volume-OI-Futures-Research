from pathlib import Path
import sys

import pandas as pd
import pytest


CHAPTER2_DIR = Path(__file__).resolve().parents[1] / "code" / "chapter2"
sys.path.insert(0, str(CHAPTER2_DIR))

from rolling.factor_24_param_space import (
    ENTRY_PARAM_OPTIONS,
    EXIT_PARAM_OPTIONS,
    build_param_candidates,
)
from rolling.walk_forward import make_windows, score_metric
from rules.entry_rules import EntryConfig
from rules.exit_rules import ExitConfig


def test_param_candidates_allow_every_entry_and_exit_field_to_change():
    entry_fields = set(EntryConfig.__dataclass_fields__)
    exit_fields = set(ExitConfig.__dataclass_fields__)

    changed_fields = {item["changed_field"] for item in build_param_candidates()}

    assert set(ENTRY_PARAM_OPTIONS) == entry_fields
    assert set(EXIT_PARAM_OPTIONS) == exit_fields
    assert {"entry." + field for field in entry_fields}.issubset(changed_fields)
    assert {"exit." + field for field in exit_fields}.issubset(changed_fields)


def test_make_windows_uses_three_year_train_and_six_month_test_by_default():
    windows = make_windows(
        pd.Timestamp("2016-01-01"),
        pd.Timestamp("2020-01-10"),
        train_years=3,
        test_months=6,
    )

    assert windows[0]["train_start"] == pd.Timestamp("2016-01-01")
    assert windows[0]["train_end"] == pd.Timestamp("2019-01-01")
    assert windows[0]["test_start"] == pd.Timestamp("2019-01-01")
    assert windows[0]["test_end"] == pd.Timestamp("2019-07-01")
    assert windows[1]["train_start"] == pd.Timestamp("2016-07-01")
    assert windows[1]["test_start"] == pd.Timestamp("2019-07-01")


def test_score_metric_prefers_same_sharpe_with_smaller_drawdown():
    shallow = {"sharpe": 1.5, "max_drawdown": -0.05}
    deep = {"sharpe": 1.5, "max_drawdown": -0.20}

    assert score_metric(shallow, drawdown_penalty=2.0) > score_metric(
        deep, drawdown_penalty=2.0
    )
