import os

project_root = os.path.dirname(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
)

os.environ.setdefault(
    "MPLCONFIGDIR",
    os.path.join(project_root, ".matplotlib")
)
os.environ.setdefault(
    "XDG_CACHE_HOME",
    os.path.join(project_root, ".cache")
)

os.makedirs(os.environ["MPLCONFIGDIR"], exist_ok=True)
os.makedirs(os.environ["XDG_CACHE_HOME"], exist_ok=True)

import pandas as pd
import numpy as np
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt


# =========================
# 参数设置
# =========================

symbol = os.environ.get("SYMBOL", "LC")

factor_name_by_id = {
    "42": "price_up_oi_down",
    "43": "uptrend_crowded_chase",
    "44": "uptrend_range_climax",
    "45": "uptrend_push_failure",
    "46": "uptrend_oi_unwind_divergence",
    "47": "downtrend_crowded_exhaustion",
}

factor_id = os.environ.get("FACTOR_ID", "43")
factor_name = os.environ.get(
    "FACTOR_NAME",
    factor_name_by_id.get(factor_id, "uptrend_crowded_chase")
)

reduce_days = 3  # 一有信号，就从下一天开始连续减仓 3 天
annual_days = 252
plot_dpi = 300
figsize = (12, 6)

factor_input_path = (
    f"../../results/tables/factors/"
    f"{symbol}_{factor_id}_{factor_name}.csv"
)

prepared_daily_input_path = (
    f"../../results/tables/daily/"
    f"{symbol}_daily_with_trend.csv"
)

backtest_output_path = (
    f"../../results/tables/backtest/"
    f"{symbol}_{factor_id}_{factor_name}_long_only_simple_backtest.csv"
)

summary_output_path = (
    f"../../results/tables/backtest/"
    f"{symbol}_{factor_id}_{factor_name}_long_only_simple_summary.csv"
)

nav_figure_path = (
    f"../../results/figures/backtest/"
    f"{symbol}_{factor_id}_simple_nav.png"
)

drawdown_figure_path = (
    f"../../results/figures/backtest/"
    f"{symbol}_{factor_id}_simple_drawdown.png"
)


# =========================
# 1. 读取因子每日表
# =========================

daily = pd.read_csv(factor_input_path)

daily["date"] = pd.to_datetime(daily["date"])
daily = daily.sort_values("date").reset_index(drop=True)

if "open" not in daily.columns:
    prepared_daily = pd.read_csv(prepared_daily_input_path)
    prepared_daily["date"] = pd.to_datetime(prepared_daily["date"])

    daily = daily.merge(
        prepared_daily[["date", "open"]],
        on="date",
        how="left"
    )


# =========================
# 2. 计算每日单利收益
# =========================
# 这里不用百分比收益率，而是用价格差：
# 每天收益 = 当天平仓价 - 当天开仓价
#
# 做多情况下：
# close > open，当天赚钱
# close < open，当天亏钱

daily["daily_return"] = (
    (daily["close"] - daily["open"]) / daily["open"]
)

# 用第一天开盘价作为标准化基准
# 这样可以把“价格收益”转换成类似净值的比例
initial_price = daily["open"].iloc[0]


# =========================
# 3. 设置两个策略的仓位
# =========================

# 原始策略：一直满仓做多
daily["base_position"] = 1.0

# 信号减仓策略：
# 平时仓位 = 1
# 一旦检测到信号，从下一天开始连续 reduce_days 天减仓
#
# 重点：
# 信号是今天收盘后才知道的，
# 所以要 shift(1)，不能用今天的信号交易今天。

daily["reduce_signal"] = (
    daily["signal"]
    .shift(1)
    .rolling(window=reduce_days, min_periods=1)
    .max()
    .fillna(0)
)

daily["strategy_position"] = 1.0

daily.loc[
    daily["reduce_signal"] == 1,
    "strategy_position"
] = daily["position_scale"].min()


# =========================
# 4. 计算策略每日单利收益
# =========================
# 单利收益不做复利滚动
# 每天只计算：仓位 * 当天价格差

daily["base_return"] = (
    daily["base_position"] * daily["daily_return"]
)

daily["strategy_return"] = (
    daily["strategy_position"] * daily["daily_return"]
)

daily["base_return"] = daily["base_return"].fillna(0)
daily["strategy_return"] = daily["strategy_return"].fillna(0)


# =========================
# 5. 计算单利净值曲线
# =========================
# 初始净值 = 1
# 单利净值 = 1 + 累计价格收益 / 初始价格
#
# 注意：
# 这里是相加，不是 cumprod

daily["base_nav"] = 1 + daily["base_return"].cumsum()

daily["strategy_nav"] = (
    1 + daily["strategy_return"].cumsum()
)


# =========================
# 6. 计算回撤
# =========================

daily["base_running_max"] = daily["base_nav"].cummax()
daily["strategy_running_max"] = daily["strategy_nav"].cummax()

daily["base_drawdown"] = (
    daily["base_nav"] / daily["base_running_max"] - 1
)

daily["strategy_drawdown"] = (
    daily["strategy_nav"] / daily["strategy_running_max"] - 1
)


# =========================
# 7. 计算回测指标
# =========================

num_days = len(daily)

base_total_return = daily["base_nav"].iloc[-1] - 1
strategy_total_return = daily["strategy_nav"].iloc[-1] - 1

# 单利年化收益：
# 先算平均每天价格收益，再乘 annual_days
# 再除以初始价格，变成比例

base_annual_return = (
    daily["base_return"].mean() * annual_days
)

strategy_annual_return = (
    daily["strategy_return"].mean() * annual_days
)

base_max_drawdown = daily["base_drawdown"].min()
strategy_max_drawdown = daily["strategy_drawdown"].min()

# 单利波动率：
# 每日价格收益 / 初始价格
# 再年化

base_volatility = daily["base_return"].std() * np.sqrt(annual_days)
strategy_volatility = daily["strategy_return"].std() * np.sqrt(annual_days)

base_sharpe = (
    base_annual_return / base_volatility
    if base_volatility != 0 else np.nan
)

strategy_sharpe = (
    strategy_annual_return / strategy_volatility
    if strategy_volatility != 0 else np.nan
)

signal_days = daily["signal"].sum()
signal_ratio = daily["signal"].mean()


summary = pd.DataFrame([
    {
        "strategy": "base_long_only_simple",
        "total_return": base_total_return,
        "annual_return": base_annual_return,
        "max_drawdown": base_max_drawdown,
        "annual_volatility": base_volatility,
        "sharpe": base_sharpe,
        "signal_days": signal_days,
        "signal_ratio": signal_ratio,
    },
    {
        "strategy": "long_only_reduce_position_simple",
        "total_return": strategy_total_return,
        "annual_return": strategy_annual_return,
        "max_drawdown": strategy_max_drawdown,
        "annual_volatility": strategy_volatility,
        "sharpe": strategy_sharpe,
        "signal_days": signal_days,
        "signal_ratio": signal_ratio,
    }
])


# =========================
# 8. 保存表格
# =========================

os.makedirs(os.path.dirname(backtest_output_path), exist_ok=True)
os.makedirs(os.path.dirname(summary_output_path), exist_ok=True)

daily.to_csv(backtest_output_path, index=False)
summary.to_csv(summary_output_path, index=False)


# =========================
# 9. 画单利净值曲线
# =========================

plt.figure(figsize=figsize)

plt.plot(
    daily["date"],
    daily["base_nav"],
    label="Base: Always Long Simple NAV"
)

plt.plot(
    daily["date"],
    daily["strategy_nav"],
    label=f"Reduce Position for {reduce_days} Days After Signal Simple NAV"
)

plt.title(f"{symbol} Long Only Backtest Simple NAV")
plt.xlabel("Date")
plt.ylabel("Simple NAV")
plt.legend()
plt.grid(alpha=0.3)
plt.tight_layout()

os.makedirs(os.path.dirname(nav_figure_path), exist_ok=True)
plt.savefig(nav_figure_path, dpi=plot_dpi)
plt.close()


# # =========================
# # 10. 画回撤曲线
# # =========================

# plt.figure(figsize=figsize)

# plt.plot(
#     daily["date"],
#     daily["base_drawdown"],
#     label="Base Drawdown"
# )

# plt.plot(
#     daily["date"],
#     daily["strategy_drawdown"],
#     label="Strategy Drawdown"
# )

# plt.title(f"{symbol} Long Only Backtest Simple Drawdown")
# plt.xlabel("Date")
# plt.ylabel("Drawdown")
# plt.legend()
# plt.grid(alpha=0.3)
# plt.tight_layout()

# os.makedirs(os.path.dirname(drawdown_figure_path), exist_ok=True)
# plt.savefig(drawdown_figure_path, dpi=plot_dpi)
# plt.close()


# =========================
# 11. 打印结果
# =========================

print("长期做多 + 信号减仓单利回测完成。")
print(f"每日回测结果保存为：{backtest_output_path}")
print(f"回测汇总结果保存为：{summary_output_path}")
print(f"净值图保存为：{nav_figure_path}")
print(f"回撤图保存为：{drawdown_figure_path}")

print("\n回测结果：")
print(summary)
