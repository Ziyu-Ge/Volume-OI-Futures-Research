import os
import pandas as pd
import numpy as np

# =========================
# 参数设置
# =========================

symbol = "LC"

factor_id = "22"
factor_name = "speculation_first_difference"
# 11 high_speculation
# 12 speculation_mad
# 21 speculation_change_rate
# 22 speculation_first_difference
# 23  speculation_continuous_drop

annual_days = 252

# 当前文件在 code/backtest/ 下面
project_root = os.path.dirname(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
)

factor_input_path = os.path.join(
    project_root,
    "results",
    "tables",
    "factors",
    f"{symbol}_{factor_id}_{factor_name}.csv"
)

segments_input_path = os.path.join(
    project_root,
    "results",
    "tables",
    "daily",
    f"{symbol}_trend_segments.csv"
)

# 如果趋势段文件还在旧位置，则自动读旧路径
if not os.path.exists(segments_input_path):
    segments_input_path = os.path.join(
        project_root,
        "results",
        "tables",
        f"{symbol}_trend_segments.csv"
    )

backtest_daily_output_path = os.path.join(
    project_root,
    "results",
    "tables",
    "backtest",
    f"{symbol}_{factor_id}_{factor_name}_backtest_daily.csv"
)

backtest_trades_output_path = os.path.join(
    project_root,
    "results",
    "tables",
    "backtest",
    f"{symbol}_{factor_id}_{factor_name}_backtest_trades.csv"
)

summary_output_path = os.path.join(
    project_root,
    "results",
    "tables",
    "backtest",
    f"{symbol}_{factor_id}_{factor_name}_backtest_summary.csv"
)


# =========================
# 1. 读取数据
# =========================

daily = pd.read_csv(factor_input_path)
segments = pd.read_csv(segments_input_path)

daily["date"] = pd.to_datetime(daily["date"])
segments["start_date"] = pd.to_datetime(segments["start_date"])
segments["end_date"] = pd.to_datetime(segments["end_date"])

if "end_signal_date" in segments.columns:
    segments["end_signal_date"] = pd.to_datetime(segments["end_signal_date"])

daily = daily.sort_values("date").reset_index(drop=True)


# =========================
# 2. 生成原始趋势仓位
# =========================

daily["base_position"] = 0.0

daily.loc[daily["trend"] == "up_trend", "base_position"] = 1.0
daily.loc[daily["trend"] == "down_trend", "base_position"] = -1.0


# =========================
# 3. 生成单因子减仓仓位
# =========================

# 如果没有 position_scale，则默认不减仓
if "position_scale" not in daily.columns:
    daily["position_scale"] = 1.0

daily["factor_position"] = (
    daily["base_position"] * daily["position_scale"]
)


# =========================
# 4. 计算每日收益和净值
# =========================

daily["daily_return"] = daily["close"].pct_change()

# 用昨天的仓位赚今天的收益，避免未来函数
daily["strategy_return"] = (
    daily["factor_position"].shift(1) * daily["daily_return"]
)

daily["base_strategy_return"] = (
    daily["base_position"].shift(1) * daily["daily_return"]
)

daily["strategy_return"] = daily["strategy_return"].fillna(0)
daily["base_strategy_return"] = daily["base_strategy_return"].fillna(0)

daily["nav"] = (1 + daily["strategy_return"]).cumprod()
daily["base_nav"] = (1 + daily["base_strategy_return"]).cumprod()

daily["running_max"] = daily["nav"].cummax()
daily["drawdown"] = daily["nav"] / daily["running_max"] - 1


# =========================
# 5. 计算整体绩效指标
# =========================

num_days = len(daily)

annual_return = daily["nav"].iloc[-1] ** (annual_days / num_days) - 1

max_drawdown = daily["drawdown"].min()

return_std = daily["strategy_return"].std()

if return_std == 0 or pd.isna(return_std):
    sharpe_ratio = np.nan
else:
    sharpe_ratio = (
        daily["strategy_return"].mean()
        / return_std
        * np.sqrt(annual_days)
    )


# =========================
# 6. 按趋势段计算交易表现
# =========================

trade_rows = []

for _, seg in segments.iterrows():

    segment_id = seg["segment_id"]

    part = daily[daily["segment_id"] == segment_id].copy()

    if len(part) == 0:
        continue

    trend = seg["trend"]

    start_close = seg["start_close"]
    end_close = seg["end_close"]

    end_signal_close = seg.get("end_signal_close", np.nan)

    # 如果没有 end_signal_close，就用趋势结束价格
    if pd.isna(end_signal_close):
        end_signal_close = end_close

    # 单因子策略在这一段的收益
    trade_return = (1 + part["strategy_return"]).prod() - 1

    # 原始趋势策略在这一段的收益
    base_trade_return = (1 + part["base_strategy_return"]).prod() - 1

    # 是否触发过信号
    has_signal = int(part["signal"].sum() > 0)

    if "is_effective_signal" in part.columns:
        has_effective_signal = int(part["is_effective_signal"].sum() > 0)
    else:
        has_effective_signal = np.nan

    # =========================
    # 趋势结束后的利润回吐比例
    # =========================

    if trend == "up_trend":

        max_profit_before_end = end_close / start_close - 1
        profit_at_exit = end_signal_close / start_close - 1

    elif trend == "down_trend":

        max_profit_before_end = start_close / end_close - 1
        profit_at_exit = start_close / end_signal_close - 1

    else:
        max_profit_before_end = np.nan
        profit_at_exit = np.nan

    if (
        pd.isna(max_profit_before_end)
        or max_profit_before_end <= 0
    ):
        profit_giveback = np.nan
    else:
        profit_giveback = (
            max_profit_before_end - profit_at_exit
        ) / max_profit_before_end

    win = int(trade_return > 0)

    trade_rows.append({
        "symbol": symbol,
        "factor_id": factor_id,
        "factor_name": factor_name,

        "segment_id": segment_id,
        "trend": trend,

        "start_date": seg["start_date"],
        "end_date": seg["end_date"],
        "end_signal_date": seg.get("end_signal_date", pd.NaT),

        "start_close": start_close,
        "end_close": end_close,
        "end_signal_close": end_signal_close,

        "has_signal": has_signal,
        "has_effective_signal": has_effective_signal,

        "trade_return": trade_return,
        "base_trade_return": base_trade_return,

        "max_profit_before_end": max_profit_before_end,
        "profit_at_exit": profit_at_exit,
        "profit_giveback": profit_giveback,

        "win": win,
    })

trade_table = pd.DataFrame(trade_rows)


# =========================
# 7. 计算交易层面的指标
# =========================

if len(trade_table) > 0:
    win_rate = trade_table["win"].mean()
    avg_profit_giveback = trade_table["profit_giveback"].mean()
    num_trades = len(trade_table)
    num_signal_trades = trade_table["has_signal"].sum()

    if "has_effective_signal" in trade_table.columns:
        num_effective_signal_trades = (
            trade_table["has_effective_signal"].sum()
        )
    else:
        num_effective_signal_trades = np.nan
else:
    win_rate = np.nan
    avg_profit_giveback = np.nan
    num_trades = 0
    num_signal_trades = 0
    num_effective_signal_trades = np.nan


summary = pd.DataFrame([{
    "symbol": symbol,
    "factor_id": factor_id,
    "factor_name": factor_name,

    "annual_return": annual_return,
    "max_drawdown": max_drawdown,
    "sharpe_ratio": sharpe_ratio,
    "win_rate": win_rate,
    "avg_profit_giveback": avg_profit_giveback,

    "num_trades": num_trades,
    "num_signal_trades": num_signal_trades,
    "num_effective_signal_trades": num_effective_signal_trades,
}])


# =========================
# 8. 保存结果
# =========================

os.makedirs(os.path.dirname(backtest_daily_output_path), exist_ok=True)
os.makedirs(os.path.dirname(backtest_trades_output_path), exist_ok=True)
os.makedirs(os.path.dirname(summary_output_path), exist_ok=True)

daily.to_csv(backtest_daily_output_path, index=False)
trade_table.to_csv(backtest_trades_output_path, index=False)
summary.to_csv(summary_output_path, index=False)

print("单因子回测完成。")
print(f"每日回测表保存为：{backtest_daily_output_path}")
print(f"趋势交易表保存为：{backtest_trades_output_path}")
print(f"绩效汇总表保存为：{summary_output_path}")

print("\n绩效汇总：")
print(summary)
