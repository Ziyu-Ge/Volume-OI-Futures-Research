from dataclasses import asdict

from factors.factor_24 import ENTRY_CONFIG, EXIT_CONFIG
from rules.entry_rules import EntryConfig
from rules.exit_rules import ExitConfig


# 基准参数就是当前 factor_24.py 的参数。
BASE_ENTRY_PARAMS = asdict(ENTRY_CONFIG)
BASE_EXIT_PARAMS = asdict(EXIT_CONFIG)


# 每个 EntryConfig 字段都放在这里，所以每个开仓参数都可以参与滚动调参。
# 默认只给少量候选值，是为了先让 walk-forward 能在全品种上跑得动。
ENTRY_PARAM_OPTIONS = {
    "ma_short": [2, 3],
    "ma_long": [5, 7],
    "ma_trend": [7, 10],
    "ma_bias_threshold": [-0.006, -0.003],
    "ma_long_bias_threshold": [-0.05, -0.03],
    "oi_slope_window": [7, 10],
    "oi_slope_threshold": [0.003, 0.001],
    "close_slope_window": [7, 10],
    "close_slope_threshold": [0.0065, 0.004],
    "speculation_slope_window": [3, 5],
    "speculation_slope_threshold": [-0.013, -0.010],
    "volatility_window": [10, 15],
}


# 每个 ExitConfig 字段也放在这里，所以每个平仓参数都可以参与滚动调参。
EXIT_PARAM_OPTIONS = {
    "trailing_multiplier": [1.215, 1.40],
    "entry_loss_volatility_multiplier": [0.83, 1.00],
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
