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
    "51": "price_volume_up_oi_down",
}

factor_id = os.environ.get("FACTOR_ID", "42")
factor_name = os.environ.get(
    "FACTOR_NAME",
    factor_name_by_id.get(factor_id, "price_up_oi_down")
)

annual_days = 252
plot_dpi = 300
figsize = (12, 6)

factor_input_path = (
    f"../../results/tables/factors/"
    f"{symbol}_{factor_id}_{factor_name}.csv"
)

backtest_output_path = (
    f"../../results/tables/backtest/"
    f"{symbol}_{factor_id}_{factor_name}_long_only_backtest.csv"
)

summary_output_path = (
    f"../../results/tables/backtest/"
    f"{symbol}_{factor_id}_{factor_name}_long_only_summary.csv"
)

nav_figure_path = (
    f"../../results/figures/backtest/"
    f"{symbol}_{factor_id}_multi_nav.png"
)

drawdown_figure_path = (
    f"../../results/figures/backtest/"
    f"{symbol}_{factor_id}_multi_drawdown.png"
)


# =========================
# 1. 读取因子每日表
# =========================

daily = pd.read_csv(factor_input_path)

daily["date"] = pd.to_datetime(daily["date"])
daily = daily.sort_values("date").reset_index(drop=True)


# =========================
# 2. 计算每日收益率
# =========================
# 这里用最简单的收盘价收益率：
# 今天收益 = 今天收盘价 / 昨天收盘价 - 1

daily["daily_return"] = daily["close"].pct_change()


# =========================
# 3. 设置两个策略的仓位
# =========================

# 原始策略：一直满仓做多
daily["base_position"] = 1.0

# 因子仓位策略：
# 仓位完全由因子每日表中的 position/position_scale 决定。
#
# 重点：
# 因子仓位是今天收盘后才知道的，
# 所以要 shift(1)，不能用今天收盘后的仓位交易今天。

if "position" in daily.columns:
    factor_position_column = "position"
elif "position_scale" in daily.columns:
    factor_position_column = "position_scale"
else:
    raise ValueError(
        "factor daily table must contain 'position' or 'position_scale'"
    )

daily["factor_position_source"] = factor_position_column
daily["strategy_position"] = (
    daily[factor_position_column]
    .shift(1)
    .fillna(1.0)
)


# =========================
# 4. 计算策略每日收益
# =========================

daily["base_return"] = (
    daily["base_position"] * daily["daily_return"]
)

daily["strategy_return"] = (
    daily["strategy_position"] * daily["daily_return"]
)

daily["base_return"] = daily["base_return"].fillna(0)
daily["strategy_return"] = daily["strategy_return"].fillna(0)


# =========================
# 5. 计算净值曲线
# =========================
# 初始净值 = 1
# 每天按收益率复利增长

daily["base_nav"] = (1 + daily["base_return"]).cumprod()
daily["strategy_nav"] = (1 + daily["strategy_return"]).cumprod()


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

base_annual_return = (
    daily["base_nav"].iloc[-1] ** (annual_days / num_days) - 1
)

strategy_annual_return = (
    daily["strategy_nav"].iloc[-1] ** (annual_days / num_days) - 1
)

base_max_drawdown = daily["base_drawdown"].min()
strategy_max_drawdown = daily["strategy_drawdown"].min()

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
        "strategy": "base_long_only",
        "total_return": base_total_return,
        "annual_return": base_annual_return,
        "max_drawdown": base_max_drawdown,
        "annual_volatility": base_volatility,
        "sharpe": base_sharpe,
        "signal_days": signal_days,
        "signal_ratio": signal_ratio,
    },
    {
        "strategy": "factor_position",
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
# 9. 画净值曲线
# =========================

plt.figure(figsize=figsize)

plt.plot(
    daily["date"],
    daily["base_nav"],
    label="Base: Always Long"
)

plt.plot(
    daily["date"],
    daily["strategy_nav"],
    label="Strategy: Factor Position"
)

plt.title(f"{symbol} Factor Position Backtest NAV")
plt.xlabel("Date")
plt.ylabel("NAV")
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

# plt.title(f"{symbol} Long Only Backtest Drawdown")
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

print("长期做多基准 + 因子仓位回测完成。")
print(f"每日回测结果保存为：{backtest_output_path}")
print(f"回测汇总结果保存为：{summary_output_path}")
print(f"净值图保存为：{nav_figure_path}")
print(f"回撤图保存为：{drawdown_figure_path}")
print(f"实际仓位来源：{factor_position_column}")

print("\n回测结果：")
print(summary)
