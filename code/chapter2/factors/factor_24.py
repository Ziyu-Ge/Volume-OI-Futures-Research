from rules.entry_rules import EntryConfig
from rules.exit_rules import ExitConfig


FACTOR_ID = "24"
FACTOR_NAME = "high_bias_oi_speculation_drop_mixed"
ENGINE = "hourly_exit"
USE_SPECULATION = True

# 24 号：23 号开仓条件 + 小时频执行和平仓。
ENTRY_CONFIG = EntryConfig(
    ma_short=4,
    ma_long=13,
    ma_trend=17,
    ma_bias_threshold=0.011,
    ma_long_bias_threshold=0,
    oi_slope_window=11,
    oi_slope_threshold=0.0065,
    close_slope_window=6,
    close_slope_threshold=0.007,
    speculation_slope_window=3,
    speculation_slope_threshold=-0.004,
    volatility_window=16,
)
EXIT_CONFIG = ExitConfig(
    trailing_multiplier=1.7,
    entry_loss_volatility_multiplier=1.8,
)
