import numpy as np
import pandas as pd

from volume_price_factor_utils import (
    SYMBOL,
    load_daily,
    parse_factor_script_metadata,
    past_rank,
    positive_part,
    save_factor_outputs,
)


# =========================
# 参数设置
# =========================

symbol = SYMBOL

factor_id, factor_name = parse_factor_script_metadata(__file__)

# 价格乖离率窗口。
bias_short_window = 5
bias_long_window = 20
bias_rank_window = 120

# 持仓量指标窗口。
oi_rank_window = 60
oi_recent_high_window = 5
oi_drawdown_window = 20

# 价格变化和平仓观察窗口。
return_window = 3
oi_change_window = 3

# 开空、平空和止损阈值。
bias_rank_threshold = 0.8
oi_recent_high_threshold = 0.8
oi_drawdown_threshold = -0.02
oi_flat_change_threshold = 0.005
cover_return_threshold = -0.003
stop_loss_threshold = 0.03


def build_short_state_machine(frame):
    """根据前一日信号执行交易，并输出每日持仓状态。"""
    daily = frame.copy()

    daily["actual_open_short_signal"] = (
        daily["open_short_signal"].shift(1).fillna(0).astype(int)
    )
    daily["actual_cover_short_signal"] = (
        daily["cover_short_signal"].shift(1).fillna(0).astype(int)
    )

    position = 0
    entry_price = np.nan
    pending_stop_loss = False

    positions = []
    trade_signals = []
    trade_actions = []
    entry_prices = []
    exit_reasons = []
    stop_loss_signals = []
    actual_stop_loss_signals = []

    for _, row in daily.iterrows():
        close = row["close"]
        actual_open_signal = int(row["actual_open_short_signal"]) == 1
        actual_cover_signal = int(row["actual_cover_short_signal"]) == 1

        trade_signal = 0
        trade_action = ""
        exit_reason = ""
        row_entry_price = entry_price
        actual_stop_loss_signal = int(pending_stop_loss)

        if position == -1 and (actual_stop_loss_signal or actual_cover_signal):
            trade_signal = 1
            trade_action = "cover_short"
            exit_reason = (
                "stop_loss"
                if actual_stop_loss_signal
                else "cover_signal"
            )
            row_entry_price = entry_price
            position = 0
            entry_price = np.nan
            pending_stop_loss = False

        elif position == 0 and actual_open_signal:
            trade_signal = -1
            trade_action = "open_short"
            entry_price = close
            row_entry_price = entry_price
            position = -1
            pending_stop_loss = False

        if position == -1 and pd.notna(entry_price) and pd.notna(close):
            stop_loss_signal = int(
                close / entry_price - 1 >= stop_loss_threshold
            )
        else:
            stop_loss_signal = 0

        pending_stop_loss = bool(stop_loss_signal)

        if position == 0 and trade_action != "cover_short":
            row_entry_price = np.nan

        positions.append(position)
        trade_signals.append(trade_signal)
        trade_actions.append(trade_action)
        entry_prices.append(row_entry_price)
        exit_reasons.append(exit_reason)
        stop_loss_signals.append(stop_loss_signal)
        actual_stop_loss_signals.append(actual_stop_loss_signal)

    daily["position"] = positions
    daily["trade_signal"] = trade_signals
    daily["trade_action"] = trade_actions
    daily["entry_price"] = entry_prices
    daily["exit_reason"] = exit_reasons
    daily["stop_loss_signal"] = stop_loss_signals
    daily["actual_stop_loss_signal"] = actual_stop_loss_signals
    daily["short_entry_signal"] = (daily["trade_signal"] == -1).astype(int)
    daily["short_exit_signal"] = (daily["trade_signal"] == 1).astype(int)

    return daily


def calculate_max_drawdown(return_series):
    equity_curve = (1 + return_series.fillna(0)).cumprod()
    running_high = equity_curve.cummax()
    drawdown = equity_curve / running_high - 1
    return drawdown.min()


# =========================
# 1. 读取日频数据
# =========================
# 日频数据由 code/01_prepare_data.py 生成，公共读取函数会优先读取
# RESULTS_OUTPUT_DIR/tables/daily/{symbol}_daily.csv。

daily = load_daily(symbol)


# =========================
# 2. 计算价格乖离率
# =========================
# past_rank() 用今天的 bias_5_20 与此前窗口内的历史值比较，
# 不包含今天自身，避免把当前观测放进历史基准。

daily["ma5"] = (
    daily["close"]
    .rolling(window=bias_short_window, min_periods=bias_short_window)
    .mean()
)
daily["ma20"] = (
    daily["close"]
    .rolling(window=bias_long_window, min_periods=bias_long_window)
    .mean()
)
daily["bias_5_20"] = daily["ma5"] / daily["ma20"] - 1
daily["bias_rank_120"] = past_rank(
    daily["bias_5_20"],
    window=bias_rank_window,
    min_history_days=bias_rank_window,
)


# =========================
# 3. 计算持仓量指标
# =========================
# open_interest 非正值无法取 log，统一设为 NaN 后由滚动窗口自然过滤。

safe_open_interest = daily["open_interest"].where(
    daily["open_interest"] > 0,
    np.nan,
)
daily["log_oi"] = np.log(safe_open_interest)
daily["oi_rank_60"] = past_rank(
    daily["log_oi"],
    window=oi_rank_window,
    min_history_days=oi_rank_window,
)
daily["oi_recent_high"] = (
    daily["oi_rank_60"]
    .rolling(window=oi_recent_high_window, min_periods=oi_recent_high_window)
    .max()
)
daily["oi_max20"] = (
    daily["log_oi"]
    .rolling(window=oi_drawdown_window, min_periods=oi_drawdown_window)
    .max()
)
daily["oi_drawdown20"] = daily["log_oi"] - daily["oi_max20"]


# =========================
# 4. 计算价格变化和平仓指标
# =========================

daily["daily_return"] = daily["close"].pct_change()
daily["ret_3"] = daily["close"] / daily["close"].shift(return_window) - 1
daily["oi_chg3"] = daily["log_oi"] - daily["log_oi"].shift(oi_change_window)


# =========================
# 5. 生成原始开空和平空信号
# =========================
# 原始信号使用当天收盘后可确认的信息；实际交易在状态机里整体 shift(1)。

daily["open_short_signal"] = (
    (daily["bias_rank_120"] >= bias_rank_threshold) &
    (daily["oi_recent_high"] >= oi_recent_high_threshold) &
    (daily["oi_drawdown20"] <= oi_drawdown_threshold) &
    (daily["ret_3"] < 0)
).astype(int)

daily["cover_short_signal"] = (
    (daily["oi_chg3"].abs() <= oi_flat_change_threshold) &
    (daily["ret_3"] >= cover_return_threshold)
).astype(int)

daily["high_bias_oi_drop_score"] = (
    daily["bias_rank_120"].fillna(0)
    + daily["oi_recent_high"].fillna(0)
    + positive_part(-daily["oi_drawdown20"])
    + positive_part(-daily["ret_3"])
)


# =========================
# 6. 状态机执行交易
# =========================
# position = -1 表示持有空单，0 表示无持仓；
# trade_signal = -1 表示开空，1 表示平空，0 表示无交易。

daily = build_short_state_machine(daily)
daily["strategy_daily_return"] = (
    daily["position"].shift(1).fillna(0) *
    daily["daily_return"].fillna(0)
)
daily["strategy_cumulative_return"] = (
    (1 + daily["strategy_daily_return"]).cumprod() - 1
)


# =========================
# 7. 保存标准化结果
# =========================

feature_columns = [
    "daily_return",
    "ma5",
    "ma20",
    "bias_5_20",
    "bias_rank_120",
    "log_oi",
    "oi_rank_60",
    "oi_recent_high",
    "oi_max20",
    "oi_drawdown20",
    "ret_3",
    "oi_chg3",
    "open_short_signal",
    "cover_short_signal",
    "actual_open_short_signal",
    "actual_cover_short_signal",
    "stop_loss_signal",
    "actual_stop_loss_signal",
    "position",
    "trade_signal",
    "trade_action",
    "entry_price",
    "exit_reason",
    "short_entry_signal",
    "short_exit_signal",
    "strategy_daily_return",
    "strategy_cumulative_return",
    "high_bias_oi_drop_score",
]

result = save_factor_outputs(
    daily=daily,
    symbol=symbol,
    factor_id=factor_id,
    factor_name=factor_name,
    factor_value_column="high_bias_oi_drop_score",
    signal_column="short_entry_signal",
    feature_columns=feature_columns,
    figure_feature_columns=[
        "bias_rank_120",
        "oi_rank_60",
        "oi_recent_high",
        "oi_drawdown20",
        "ret_3",
        "strategy_cumulative_return",
    ],
)


# =========================
# 8. 补充策略参数和绩效摘要
# =========================

summary_table = result["summary_table"].copy()
summary_table["bias_short_window"] = bias_short_window
summary_table["bias_long_window"] = bias_long_window
summary_table["bias_rank_window"] = bias_rank_window
summary_table["oi_rank_window"] = oi_rank_window
summary_table["oi_recent_high_window"] = oi_recent_high_window
summary_table["oi_drawdown_window"] = oi_drawdown_window
summary_table["return_window"] = return_window
summary_table["oi_change_window"] = oi_change_window
summary_table["bias_rank_threshold"] = bias_rank_threshold
summary_table["oi_recent_high_threshold"] = oi_recent_high_threshold
summary_table["oi_drawdown_threshold"] = oi_drawdown_threshold
summary_table["oi_flat_change_threshold"] = oi_flat_change_threshold
summary_table["cover_return_threshold"] = cover_return_threshold
summary_table["stop_loss_threshold"] = stop_loss_threshold
summary_table["raw_open_short_signal_days"] = int(
    daily["open_short_signal"].sum()
)
summary_table["raw_cover_short_signal_days"] = int(
    daily["cover_short_signal"].sum()
)
summary_table["short_entry_count"] = int(daily["short_entry_signal"].sum())
summary_table["short_exit_count"] = int(daily["short_exit_signal"].sum())
summary_table["stop_loss_exit_count"] = int(
    (daily["exit_reason"] == "stop_loss").sum()
)
summary_table["cover_signal_exit_count"] = int(
    (daily["exit_reason"] == "cover_signal").sum()
)
summary_table["final_position"] = int(daily["position"].iloc[-1])
summary_table["mean_strategy_daily_return"] = daily[
    "strategy_daily_return"
].mean()
summary_table["strategy_cumulative_return"] = daily[
    "strategy_cumulative_return"
].iloc[-1]
summary_table["strategy_max_drawdown"] = calculate_max_drawdown(
    daily["strategy_daily_return"]
)
summary_table.to_csv(result["summary_output_path"], index=False)
