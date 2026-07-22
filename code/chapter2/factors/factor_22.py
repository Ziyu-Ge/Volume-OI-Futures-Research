from rules.entry_rules import EntryConfig
from rules.exit_rules import ExitConfig


FACTOR_ID = "22"
FACTOR_NAME = "high_bias_oi_drop_mixed"
ENGINE = "hourly_exit"
USE_SPECULATION = False

# 22 号：日频确认高乖离持仓回落，小时频执行和平仓。
ENTRY_CONFIG = EntryConfig(
    ma_short=5,
    ma_long=20,
    ma_trend=60,
    ma_bias_threshold=0.04,
    ma_long_bias_threshold=0.10,
    oi_slope_window=7,
    close_slope_window=7,
    volatility_window=10,
)
EXIT_CONFIG = ExitConfig(trailing_multiplier=3.5)
