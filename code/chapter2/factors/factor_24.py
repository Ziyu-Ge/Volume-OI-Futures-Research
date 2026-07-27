from rules.entry_rules import EntryConfig
from rules.exit_rules import ExitConfig


FACTOR_ID = "24"
FACTOR_NAME = "high_bias_oi_speculation_drop_mixed"
ENGINE = "hourly_exit"
USE_SPECULATION = True

# 24 号：23 号开仓条件 + 小时频执行和平仓。
ENTRY_CONFIG = EntryConfig(
    ma_short=9,
    ma_long=12,
    ma_trend=90,
    ma_bias_threshold=-0.014,
    ma_long_bias_threshold=-0.1,
    oi_slope_window=2,
    oi_slope_threshold=-0.01225,
    close_slope_window=2,
    close_slope_threshold=-0.00475,
    speculation_slope_window=30,
    speculation_slope_threshold=-0.0055,
    volatility_window=7,
)
EXIT_CONFIG = ExitConfig(
    trailing_multiplier=2.55,
    entry_loss_volatility_multiplier=2.2,
)
