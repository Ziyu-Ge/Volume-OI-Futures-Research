from dataclasses import asdict

from factors.factor_24 import ENTRY_CONFIG, EXIT_CONFIG
from rules.entry_rules import EntryConfig
from rules.exit_rules import ExitConfig


# 基准参数就是当前 factor_24.py 的参数。
BASE_ENTRY_PARAMS = asdict(ENTRY_CONFIG)
BASE_EXIT_PARAMS = asdict(EXIT_CONFIG)


# 每个 EntryConfig 字段都放在这里，所以每个开仓参数都可以参与滚动调参。
ENTRY_PARAM_OPTIONS = {
    "ma_short": [7, 9, 10],
    "ma_long": [10, 12, 15],
    "ma_trend": [60, 75, 90, 120],

    "ma_bias_threshold": [
        -0.020,
        -0.017,
        -0.014,
        -0.011,
        -0.008,
    ],
    "ma_long_bias_threshold": [
        -0.14,
        -0.12,
        -0.10,
        -0.08,
        -0.06,
    ],

    "oi_slope_window": [2, 3, 5],
    "oi_slope_threshold": [
        -0.018,
        -0.015,
        -0.01225,
        -0.010,
        -0.0075,
    ],

    "close_slope_window": [2, 3, 5],
    "close_slope_threshold": [
        -0.007,
        -0.006,
        -0.00475,
        -0.0035,
        -0.0025,
    ],

    "speculation_slope_window": [24, 30, 36],
    "speculation_slope_threshold": [
        -0.009,
        -0.007,
        -0.0055,
        -0.004,
        -0.002,
    ],

    "volatility_window": [5, 7, 10, 14],
}



# 每个 ExitConfig 字段也放在这里，所以每个平仓参数都可以参与滚动调参。
EXIT_PARAM_OPTIONS = {
    "trailing_multiplier": [2.30, 2.55, 2.80, 3.20],
    "entry_loss_volatility_multiplier": [1.50, 1.80, 2.00, 2.20, 2.40],
}


def build_param_candidates():
    """生成候选参数。

    为了让代码简单且可运行，这里使用“基准参数 + 单字段变化”的搜索方式。
    如果直接对 15 个字段做笛卡尔积，组合数量会指数级增长，跑全品种会非常慢。
    """
    candidates = []
    seen = set()

    def add_candidate(entry_params, exit_params, changed_field):
        key = (
            tuple(sorted(entry_params.items())),
            tuple(sorted(exit_params.items())),
        )
        if key in seen:
            return
        seen.add(key)
        candidates.append(
            {
                "changed_field": changed_field,
                "entry_config": EntryConfig(**entry_params),
                "exit_config": ExitConfig(**exit_params),
                **{f"entry_{key}": value for key, value in entry_params.items()},
                **{f"exit_{key}": value for key, value in exit_params.items()},
            }
        )

    add_candidate(BASE_ENTRY_PARAMS, BASE_EXIT_PARAMS, "base")

    for field, values in ENTRY_PARAM_OPTIONS.items():
        for value in values:
            if value == BASE_ENTRY_PARAMS[field]:
                continue
            entry_params = {**BASE_ENTRY_PARAMS, field: value}
            add_candidate(entry_params, BASE_EXIT_PARAMS, f"entry.{field}")

    for field, values in EXIT_PARAM_OPTIONS.items():
        for value in values:
            if value == BASE_EXIT_PARAMS[field]:
                continue
            exit_params = {**BASE_EXIT_PARAMS, field: value}
            add_candidate(BASE_ENTRY_PARAMS, exit_params, f"exit.{field}")

    return candidates
