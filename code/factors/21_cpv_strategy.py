import argparse
import os
import tempfile
from pathlib import Path

import numpy as np
import pandas as pd

from config import SYMBOL as CONFIG_SYMBOL


PROJECT_ROOT = Path(__file__).resolve().parents[1]
# 年化指标统一按 252 个交易日估算。
ANNUAL_DAYS = 252

# 输出字段顺序集中定义，便于各个结果 CSV 保持稳定列序。
FACTOR_COLUMNS = ["date", "pv", "signal", "valid_minutes"]
BACKTEST_COLUMNS = [
    "date",
    "open_price",
    "next_open_price",
    "pv",
    "signal",
    "position",
    "ret_open_to_open",
    "strategy_return",
    "position_change",
    "cost",
    "strategy_return_net",
    "strategy_nav",
    "benchmark_nav",
]
SUMMARY_COLUMNS = [
    "symbol",
    "start_date",
    "end_date",
    "annual_return",
    "annual_volatility",
    "sharpe_ratio",
    "max_drawdown",
    "win_rate",
    "num_trading_days",
    "num_long_days",
    "num_short_days",
    "num_flat_days",
    "num_turnovers",
]


def parse_args():
    # 解析命令行参数；默认品种优先取环境变量 SYMBOL，其次取 config.py。
    parser = argparse.ArgumentParser(
        description="Run the minute-level commodity futures CPV strategy."
    )
    parser.add_argument(
        "--symbol",
        default=os.environ.get("SYMBOL", CONFIG_SYMBOL),
        help="Symbol to run. Defaults to SYMBOL in code/config.py.",
    )
    parser.add_argument(
        "--data-path",
        default=None,
        help="Minute CSV path. Defaults to data/{symbol}.csv.",
    )
    parser.add_argument(
        "--output-root",
        default=str(PROJECT_ROOT / "results" / "tables"),
        help="Root directory for output CSV files.",
    )
    parser.add_argument(
        "--figure-root",
        default=str(PROJECT_ROOT / "results" / "figures" / "backtest"),
        help="Root directory for NAV figure files.",
    )
    parser.add_argument(
        "--min-minutes",
        type=int,
        default=30,
        help="Minimum valid minute deltas required for one daily PV value.",
    )
    parser.add_argument(
        "--cost-rate",
        type=float,
        default=0.0,
        help="Trading cost charged for one unit of position change.",
    )
    return parser.parse_args()


def normalize_dates_for_output(df):
    # 输出到 CSV 前统一把日期格式化为 YYYY-MM-DD，避免带上时间部分。
    df = df.copy()
    if "date" in df.columns:
        df["date"] = pd.to_datetime(df["date"]).dt.strftime("%Y-%m-%d")
    return df


def load_minute_data(data_path):
    # 读取分钟数据，并检查策略计算必需字段是否存在。
    required_columns = {"datetime", "open", "close", "open_interest"}
    df = pd.read_csv(data_path)
    df.columns = df.columns.str.strip()

    missing_columns = required_columns - set(df.columns)
    if missing_columns:
        missing_text = ", ".join(sorted(missing_columns))
        raise ValueError(f"{data_path} missing required columns: {missing_text}")

    df["datetime"] = pd.to_datetime(df["datetime"], errors="coerce")

    # 若原始数据自带 date，则优先使用；解析失败时回退到 datetime 的自然日。
    if "date" in df.columns:
        parsed_date = pd.to_datetime(
            df["date"].astype(str).str.strip(),
            errors="coerce",
        )
        fallback_date = df["datetime"].dt.normalize()
        df["date"] = parsed_date.fillna(fallback_date)
    else:
        df["date"] = df["datetime"].dt.normalize()

    # 价格和持仓量转成数值，无法转换的数据后续会被过滤掉。
    for column in ["open", "close", "open_interest"]:
        df[column] = pd.to_numeric(df[column], errors="coerce")

    df = df.dropna(subset=["datetime", "date", "close", "open_interest"])
    df["date"] = pd.to_datetime(df["date"]).dt.normalize()
    df = df.sort_values(["date", "datetime"]).reset_index(drop=True)

    return df


def calculate_daily_pv(minute_df, min_minutes=30):
    # CPV 因子：每日分钟收盘价变化与持仓量变化的相关系数。
    records = []

    for trade_date, day_df in minute_df.groupby("date", sort=True):
        day_df = day_df.sort_values("datetime")
        # 使用分钟差分而不是水平值，衡量价格变化和持仓变化是否同向。
        delta_price = day_df["close"].diff().iloc[1:]
        delta_oi = day_df["open_interest"].diff().iloc[1:]
        valid_minutes = len(delta_price)
        pv = np.nan

        # 有效分钟数不足，或任一序列无波动时，不计算相关系数。
        if valid_minutes >= min_minutes:
            price_std = delta_price.std()
            oi_std = delta_oi.std()

            if (
                pd.notna(price_std)
                and pd.notna(oi_std)
                and price_std != 0
                and oi_std != 0
            ):
                pv = delta_price.corr(delta_oi)

        records.append({
            "date": trade_date,
            "pv": pv,
            "valid_minutes": valid_minutes,
        })

    factor = pd.DataFrame(records)
    # pv 为正做多、为负做空；无效或为 0 的 pv 保持空仓信号。
    factor["signal"] = np.select(
        [factor["pv"] > 0, factor["pv"] < 0],
        [1, -1],
        default=0,
    ).astype(int)

    return factor[FACTOR_COLUMNS]


def build_daily_open(minute_df):
    # 每个交易日取第一条分钟记录的开盘价，作为开盘到开盘收益的基准价格。
    ordered = minute_df.sort_values(["date", "datetime"])
    daily_open = (
        ordered
        .drop_duplicates(subset=["date"], keep="first")
        .loc[:, ["date", "open"]]
        .rename(columns={"open": "open_price"})
        .reset_index(drop=True)
    )

    return daily_open


def run_backtest(factor, daily_open, cost_rate=0.0):
    # 将每日因子信号和每日开盘价对齐，形成回测主表。
    backtest = (
        daily_open
        .merge(factor, on="date", how="left")
        .sort_values("date")
        .reset_index(drop=True)
    )

    backtest["signal"] = backtest["signal"].fillna(0).astype(int)
    # 信号滞后一日执行，避免使用当天分钟数据后又在当天开盘成交。
    backtest["position"] = backtest["signal"].shift(1).fillna(0).astype(int)
    backtest["next_open_price"] = backtest["open_price"].shift(-1)
    backtest["ret_open_to_open"] = (
        backtest["next_open_price"] / backtest["open_price"] - 1
    )

    backtest = backtest.dropna(
        subset=["open_price", "next_open_price", "ret_open_to_open"]
    ).copy()

    backtest["strategy_return"] = (
        backtest["position"] * backtest["ret_open_to_open"]
    )
    # 持仓变化绝对值用于估算换手和交易成本。
    previous_position = backtest["position"].shift(1).fillna(0)
    backtest["position_change"] = (
        backtest["position"] - previous_position
    ).abs()
    backtest["cost"] = backtest["position_change"] * cost_rate
    backtest["strategy_return_net"] = (
        backtest["strategy_return"] - backtest["cost"]
    )
    backtest["strategy_nav"] = (1 + backtest["strategy_return_net"]).cumprod()
    backtest["benchmark_nav"] = (1 + backtest["ret_open_to_open"]).cumprod()

    return backtest[BACKTEST_COLUMNS]


def calculate_summary(symbol, backtest):
    # 若无可回测样本，仍返回固定列结构，方便后续批量汇总。
    if backtest.empty:
        return pd.DataFrame([{
            "symbol": symbol,
            "start_date": np.nan,
            "end_date": np.nan,
            "annual_return": np.nan,
            "annual_volatility": np.nan,
            "sharpe_ratio": np.nan,
            "max_drawdown": np.nan,
            "win_rate": np.nan,
            "num_trading_days": 0,
            "num_long_days": 0,
            "num_short_days": 0,
            "num_flat_days": 0,
            "num_turnovers": 0,
        }])[SUMMARY_COLUMNS]

    num_days = len(backtest)
    nav_final = backtest["strategy_nav"].iloc[-1]
    # 年化收益按最终净值折算；净值非正时不计算。
    annual_return = (
        nav_final ** (ANNUAL_DAYS / num_days) - 1
        if nav_final > 0 and num_days > 0
        else np.nan
    )
    annual_volatility = backtest["strategy_return_net"].std() * np.sqrt(
        ANNUAL_DAYS
    )
    sharpe_ratio = (
        annual_return / annual_volatility
        if pd.notna(annual_volatility) and annual_volatility != 0
        else np.nan
    )
    running_max = backtest["strategy_nav"].cummax()
    max_drawdown = (backtest["strategy_nav"] / running_max - 1).min()

    # 胜率只在实际有持仓的交易日上统计，空仓日不参与分母。
    has_position = backtest["position"] != 0
    num_trading_days = int(has_position.sum())
    win_rate = (
        (backtest.loc[has_position, "strategy_return_net"] > 0).sum()
        / num_trading_days
        if num_trading_days > 0
        else np.nan
    )

    summary = pd.DataFrame([{
        "symbol": symbol,
        "start_date": backtest["date"].min(),
        "end_date": backtest["date"].max(),
        "annual_return": annual_return,
        "annual_volatility": annual_volatility,
        "sharpe_ratio": sharpe_ratio,
        "max_drawdown": max_drawdown,
        "win_rate": win_rate,
        "num_trading_days": num_trading_days,
        "num_long_days": int((backtest["position"] == 1).sum()),
        "num_short_days": int((backtest["position"] == -1).sum()),
        "num_flat_days": int((backtest["position"] == 0).sum()),
        "num_turnovers": int((backtest["position_change"] > 0).sum()),
    }])

    return summary[SUMMARY_COLUMNS]


def build_output_paths(output_root, figure_root, symbol):
    # 按项目约定生成因子、回测、汇总和净值图的输出路径。
    output_root = Path(output_root)
    figure_root = Path(figure_root)
    return {
        "factor": output_root / "factors" / f"{symbol}_cpv_factor.csv",
        "backtest": output_root / "backtest" / f"{symbol}_cpv_backtest_daily.csv",
        "summary": output_root / "summary" / f"{symbol}_cpv_summary.csv",
        "simple_nav": figure_root / f"{symbol}_cpv_simple_nav.png",
        "compound_nav": figure_root / f"{symbol}_cpv_compound_nav.png",
    }


def save_outputs(factor, backtest, summary, output_paths):
    # 保存前确保所有目标目录存在。
    for output_path in output_paths.values():
        output_path.parent.mkdir(parents=True, exist_ok=True)

    normalize_dates_for_output(factor).to_csv(
        output_paths["factor"],
        index=False,
    )
    normalize_dates_for_output(backtest).to_csv(
        output_paths["backtest"],
        index=False,
    )

    summary_output = summary.copy()
    # 汇总表的起止日期同样格式化，便于直接阅读和跨品种拼接。
    for column in ["start_date", "end_date"]:
        summary_output[column] = pd.to_datetime(
            summary_output[column],
            errors="coerce",
        ).dt.strftime("%Y-%m-%d")
    summary_output.to_csv(output_paths["summary"], index=False)


def plot_nav_curves(backtest, symbol, output_paths):
    # matplotlib 在无界面环境下使用临时配置目录和 Agg 后端，便于批量运行。
    mpl_config_dir = Path(tempfile.gettempdir()) / "lc_research_matplotlib"
    mpl_config_dir.mkdir(parents=True, exist_ok=True)
    os.environ.setdefault("MPLCONFIGDIR", str(mpl_config_dir))

    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    plot_df = backtest.copy()
    plot_df["date"] = pd.to_datetime(plot_df["date"])
    # 简单净值用于观察收益累加，复利净值对应实际资金滚动。
    plot_df["strategy_simple_nav"] = 1 + plot_df["strategy_return_net"].cumsum()
    plot_df["benchmark_simple_nav"] = 1 + plot_df["ret_open_to_open"].cumsum()

    figure_specs = [
        (
            "simple_nav",
            "strategy_simple_nav",
            "benchmark_simple_nav",
            f"{symbol} CPV Simple NAV",
        ),
        (
            "compound_nav",
            "strategy_nav",
            "benchmark_nav",
            f"{symbol} CPV Compound NAV",
        ),
    ]

    for path_key, strategy_column, benchmark_column, title in figure_specs:
        output_paths[path_key].parent.mkdir(parents=True, exist_ok=True)

        fig, ax = plt.subplots(figsize=(12, 6))
        ax.plot(
            plot_df["date"],
            plot_df[strategy_column],
            label="Strategy",
            linewidth=1.6,
        )
        ax.plot(
            plot_df["date"],
            plot_df[benchmark_column],
            label="Benchmark",
            linewidth=1.2,
            alpha=0.8,
        )
        ax.set_title(title)
        ax.set_xlabel("Date")
        ax.set_ylabel("NAV")
        ax.grid(True, linestyle="--", alpha=0.3)
        ax.legend()
        fig.autofmt_xdate()
        fig.tight_layout()
        fig.savefig(output_paths[path_key], dpi=150)
        plt.close(fig)


def run_cpv_strategy(symbol, data_path, output_root, figure_root, min_minutes, cost_rate):
    # 串联完整流程：读数据、算因子、回测、汇总、落盘和画图。
    minute_df = load_minute_data(data_path)
    factor = calculate_daily_pv(minute_df, min_minutes=min_minutes)
    daily_open = build_daily_open(minute_df)
    backtest = run_backtest(factor, daily_open, cost_rate=cost_rate)
    summary = calculate_summary(symbol, backtest)
    output_paths = build_output_paths(output_root, figure_root, symbol)
    save_outputs(factor, backtest, summary, output_paths)
    plot_nav_curves(backtest, symbol, output_paths)

    return factor, backtest, summary, output_paths


def main():
    # 命令行入口：补全默认数据路径后运行策略并打印关键输出位置。
    args = parse_args()
    symbol = args.symbol.upper()
    data_path = (
        Path(args.data_path)
        if args.data_path is not None
        else PROJECT_ROOT / "data" / f"{symbol}.csv"
    )

    factor, backtest, summary, output_paths = run_cpv_strategy(
        symbol=symbol,
        data_path=data_path,
        output_root=args.output_root,
        figure_root=args.figure_root,
        min_minutes=args.min_minutes,
        cost_rate=args.cost_rate,
    )

    print("CPV strategy backtest complete.")
    print(f"symbol: {symbol}")
    print(f"factor rows: {len(factor)}")
    print(f"backtest rows: {len(backtest)}")
    print(f"factor file: {output_paths['factor']}")
    print(f"backtest_daily file: {output_paths['backtest']}")
    print(f"summary file: {output_paths['summary']}")
    print(f"simple NAV figure: {output_paths['simple_nav']}")
    print(f"compound NAV figure: {output_paths['compound_nav']}")
    print(summary.to_string(index=False))


if __name__ == "__main__":
    main()
