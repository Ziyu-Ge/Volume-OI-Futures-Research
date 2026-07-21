from rules.entry_rules import EntryConfig
from rules.exit_rules import ExitConfig


FACTOR_ID = "25"
FACTOR_NAME = "high_bias_oi_speculation_drop_hourly"
ENGINE = "hourly_exit"
USE_SPECULATION = True

# 25 号：恢复高乖离、持仓和价格回落条件，小时频执行和平仓。
ENTRY_CONFIG = EntryConfig(
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

EXIT_CONFIG = ExitConfig(
    trailing_multiplier=0,
    entry_loss_volatility_multiplier=0,
)
