from rules.entry_rules import EntryConfig
from rules.exit_rules import ExitConfig


FACTOR_ID = "21"
FACTOR_NAME = "high_bias_oi_drop"
ENGINE = "daily"
USE_SPECULATION = False

# 21 号：高乖离 + 持仓回落 + 价格回落，日频执行。
ENTRY_CONFIG = EntryConfig(
    ma_short=5,
    ma_long=20,
    ma_trend=60,
    ma_bias_threshold=0.04,
    ma_long_bias_threshold=0.10,
    ma_long_bias_cap=None,
    oi_slope_window=7,
    close_slope_window=7,
    volatility_window=10,
)
EXIT_CONFIG = ExitConfig(trailing_multiplier=4.0)

