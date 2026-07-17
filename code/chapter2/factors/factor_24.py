from rules.entry_rules import EntryConfig
from rules.exit_rules import ExitConfig


FACTOR_ID = "24"
FACTOR_NAME = "high_bias_oi_speculation_drop_mixed"
ENGINE = "hourly_exit"
USE_SPECULATION = True

# 24 号：23 号开仓条件 + 小时频执行和平仓。
ENTRY_CONFIG = EntryConfig(
    ma_short=2,
    ma_long=5,
    ma_trend=7,
    ma_bias_threshold=-0.006,
    ma_long_bias_threshold=-0.25,
    ma_long_bias_cap=0.12,
    oi_slope_window=7,
    oi_slope_threshold=0.003,
    close_slope_window=7,
    close_slope_threshold=0.0065,
    speculation_slope_window=3,
    speculation_slope_threshold=-0.02,
    volatility_window=10,
)
EXIT_CONFIG = ExitConfig(
    trailing_multiplier=1.225,
    entry_loss_volatility_multiplier=1.5,
)
