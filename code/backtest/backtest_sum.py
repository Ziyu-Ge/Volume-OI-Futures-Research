import os
import sys

project_root = os.path.dirname(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
)
code_dir = os.path.join(project_root, "code")

if code_dir not in sys.path:
    sys.path.insert(0, code_dir)

from config import SYMBOL

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

symbol = os.environ.get("SYMBOL", SYMBOL)
factor_id_filter = os.environ.get("FACTOR_ID", "ALL")
factor_name_filter = os.environ.get("FACTOR_NAME")

annual_days = 252
plot_dpi = 300
figsize = (12, 6)

factor_dir = os.path.join(project_root, "results", "tables", "factors")
daily_dir = os.path.join(project_root, "results", "tables", "daily")
backtest_dir = os.path.join(project_root, "results", "tables", "backtest")
figure_dir = os.path.join(project_root, "results", "figures", "backtest")
daily_input_path = os.path.join(daily_dir, f"{symbol}_daily.csv")


def parse_filter(raw_value):
    if raw_value is None:
        return None

    raw_value = raw_value.strip()
    if raw_value == "" or raw_value.upper() in {"ALL", "*"}:
        return None

    return {
        item.strip()
        for item in raw_value.split(",")
        if item.strip()
    }


def discover_factor_files():
    factor_ids = parse_filter(factor_id_filter)
    factor_names = parse_filter(factor_name_filter)
    prefix = f"{symbol}_"
    suffix = ".csv"

    factors = []
    if not os.path.isdir(factor_dir):
        raise FileNotFoundError(f"factor directory not found: {factor_dir}")

    for filename in sorted(os.listdir(factor_dir)):
        if not filename.startswith(prefix) or not filename.endswith(suffix):
            continue

        stem = filename[:-len(suffix)]
        factor_part = stem[len(prefix):]
        if "_" not in factor_part:
            continue

        factor_id, factor_name = factor_part.split("_", 1)
        if factor_ids is not None and factor_id not in factor_ids:
            continue
        if factor_names is not None and factor_name not in factor_names:
            continue

        factors.append({
            "factor_id": factor_id,
            "factor_name": factor_name,
            "path": os.path.join(factor_dir, filename),
        })

    if not factors:
        raise FileNotFoundError(
            f"no factor csv found for SYMBOL={symbol}, "
            f"FACTOR_ID={factor_id_filter}, FACTOR_NAME={factor_name_filter}"
        )

    return factors


def get_factor_position_column(daily, factor_path):
    if "position" in daily.columns:
        return "position"
    if "position_scale" in daily.columns:
        return "position_scale"

    raise ValueError(
        f"{factor_path} must contain 'position' or 'position_scale'"
    )


def ensure_open_price(daily):
    if "open" in daily.columns and daily["open"].notna().all():
        return daily

    if not os.path.exists(daily_input_path):
        raise FileNotFoundError(
            f"daily table not found for filling open price: {daily_input_path}"
        )

    prepared_daily = pd.read_csv(daily_input_path)
    prepared_daily["date"] = pd.to_datetime(prepared_daily["date"])
    prepared_daily = prepared_daily[["date", "open"]].rename(
        columns={"open": "prepared_open"}
    )

    daily = daily.merge(prepared_daily, on="date", how="left")
    if "open" not in daily.columns:
        daily["open"] = daily["prepared_open"]
    else:
        daily["open"] = daily["open"].fillna(daily["prepared_open"])

    daily = daily.drop(columns=["prepared_open"])
    if daily["open"].isna().any():
        raise ValueError("open price still has missing values after merge")

    return daily


def run_backtest(factor_info):
    factor_id = factor_info["factor_id"]
    factor_name = factor_info["factor_name"]
    factor_path = factor_info["path"]

    backtest_output_path = os.path.join(
        backtest_dir,
        f"{symbol}_{factor_id}_{factor_name}_long_only_simple_backtest.csv"
    )
    summary_output_path = os.path.join(
        backtest_dir,
        f"{symbol}_{factor_id}_{factor_name}_long_only_simple_summary.csv"
    )
    nav_figure_path = os.path.join(
        figure_dir,
        f"{symbol}_{factor_id}_{factor_name}_simple_nav.png"
    )

    # =========================
    # 1. 读取因子每日表
    # =========================

    daily = pd.read_csv(factor_path)

    daily["date"] = pd.to_datetime(daily["date"])
    daily = daily.sort_values("date").reset_index(drop=True)
    daily = ensure_open_price(daily)

    # =========================
    # 2. 计算每日单利收益
    # =========================
    # 这里不用收盘价之间的复利收益率，而是用日内收益：
    # 每天收益 = (当天平仓价 - 当天开仓价) / 当天开仓价

    daily["daily_return"] = (
        (daily["close"] - daily["open"]) / daily["open"]
    )

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

    factor_position_column = get_factor_position_column(daily, factor_path)

    daily["factor_position_source"] = factor_position_column
    daily["strategy_position"] = (
        daily[factor_position_column]
        .shift(1)
        .fillna(1.0)
    )

    # =========================
    # 4. 计算策略每日单利收益
    # =========================
    # 单利收益不做复利滚动
    # 每天只计算：仓位 * 当天收益

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
    # 单利净值 = 1 + 每日收益累加

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

    base_total_return = daily["base_nav"].iloc[-1] - 1
    strategy_total_return = daily["strategy_nav"].iloc[-1] - 1

    # 单利年化收益：
    # 先算平均每天收益，再乘 annual_days

    base_annual_return = (
        daily["base_return"].mean() * annual_days
    )

    strategy_annual_return = (
        daily["strategy_return"].mean() * annual_days
    )

    base_max_drawdown = daily["base_drawdown"].min()
    strategy_max_drawdown = daily["strategy_drawdown"].min()

    # 单利波动率：
    # 每日收益标准差再年化

    base_volatility = daily["base_return"].std() * np.sqrt(annual_days)
    strategy_volatility = (
        daily["strategy_return"].std() * np.sqrt(annual_days)
    )

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
            "factor_id": factor_id,
            "factor_name": factor_name,
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
            "factor_id": factor_id,
            "factor_name": factor_name,
            "strategy": "factor_position_simple",
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
        label="Strategy: Factor Position Simple NAV"
    )

    plt.title(f"{symbol} {factor_id} {factor_name} Simple NAV")
    plt.xlabel("Date")
    plt.ylabel("Simple NAV")
    plt.legend()
    plt.grid(alpha=0.3)
    plt.tight_layout()

    os.makedirs(os.path.dirname(nav_figure_path), exist_ok=True)
    plt.savefig(nav_figure_path, dpi=plot_dpi)
    plt.close()

    print(f"factor {factor_id}: {factor_name} simple backtest complete.")
    print(f"每日回测结果保存为：{backtest_output_path}")
    print(f"回测汇总结果保存为：{summary_output_path}")
    print(f"净值图保存为：{nav_figure_path}")
    print(f"实际仓位来源：{factor_position_column}")

    return summary


factor_files = discover_factor_files()
all_summaries = []

for factor_info in factor_files:
    all_summaries.append(run_backtest(factor_info))

combined_summary = pd.concat(all_summaries, ignore_index=True)
combined_summary_path = os.path.join(
    backtest_dir,
    f"{symbol}_all_factors_long_only_simple_summary.csv"
)
combined_summary.to_csv(combined_summary_path, index=False)

print("\n长期做多基准 + 因子仓位单利回测完成。")
print(f"本次回测因子数量：{len(factor_files)}")
print(f"总汇总结果保存为：{combined_summary_path}")
print("\n回测结果：")
print(combined_summary)
