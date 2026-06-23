import argparse
import os
import tempfile
from pathlib import Path

import numpy as np
import pandas as pd

from config import SYMBOL as CONFIG_SYMBOL


PROJECT_ROOT = Path(__file__).resolve().parents[1]
ANNUAL_DAYS = 252

FACTOR_COLUMNS = [
    "date",
    "pv",
    "dov_mean",
    "dov_std",
    "dov_mean_threshold",
    "dov_std_threshold",
    "raw_signal",
    "final_signal",
    "is_reversed",
    "valid_minutes",
    "valid_dov_minutes",
]

BACKTEST_COLUMNS = [
    "date",
    "signal_date",
    "open_price",
    "close_price",
    "pv",
    "dov_mean",
    "dov_std",
    "raw_signal",
    "final_signal",
    "executed_signal",
    "position",
    "ret_open_to_close",
    "strategy_return",
    "round_turnover",
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
    "num_backtest_days",
    "num_trading_days",
    "num_long_days",
    "num_short_days",
    "num_flat_days",
    "num_round_trips",
    "final_nav",
    "benchmark_final_nav",
]


def parse_args():
    parser = argparse.ArgumentParser(
        description="Run the CPV + DOV commodity futures intraday strategy."
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
        "--dov-mean-window",
        type=int,
        default=30,
        help="Lookback days for the DOV_mean reversal threshold.",
    )
    parser.add_argument(
        "--dov-mean-quantile",
        type=float,
        default=0.85,
        help="Rolling quantile for the DOV_mean reversal threshold.",
    )
    parser.add_argument(
        "--dov-std-window",
        type=int,
        default=50,
        help="Lookback days for the DOV_std reversal threshold.",
    )
    parser.add_argument(
        "--dov-std-quantile",
        type=float,
        default=0.80,
        help="Rolling quantile for the DOV_std reversal threshold.",
    )
    parser.add_argument(
        "--cost-rate",
        type=float,
        default=0.0,
        help="Trading cost charged for one unit of one-way turnover.",
    )
    return parser.parse_args()


def normalize_dates_for_output(df):
    out = df.copy()
    for column in ["date", "signal_date", "start_date", "end_date"]:
        if column in out.columns:
            out[column] = pd.to_datetime(
                out[column],
                errors="coerce",
            ).dt.strftime("%Y-%m-%d")
    return out


def load_minute_data(data_path):
    required_columns = {"datetime", "open", "close", "volume", "open_interest"}
    df = pd.read_csv(data_path)
    df.columns = df.columns.str.strip()

    missing_columns = required_columns - set(df.columns)
    if missing_columns:
        missing_text = ", ".join(sorted(missing_columns))
        raise ValueError(f"{data_path} missing required columns: {missing_text}")

    df["datetime"] = pd.to_datetime(df["datetime"], errors="coerce")

    if "date" in df.columns:
        parsed_date = pd.to_datetime(
            df["date"].astype(str).str.strip(),
            errors="coerce",
        )
        fallback_date = df["datetime"].dt.normalize()
        df["date"] = parsed_date.fillna(fallback_date)
    else:
        df["date"] = df["datetime"].dt.normalize()

    for column in ["open", "close", "volume", "open_interest"]:
        df[column] = pd.to_numeric(df[column], errors="coerce")

    df = df.dropna(
        subset=["datetime", "date", "open", "close", "volume", "open_interest"]
    ).copy()
    df["date"] = pd.to_datetime(df["date"]).dt.normalize()
    df = df.sort_values(["date", "datetime"]).reset_index(drop=True)

    if df.empty:
        raise ValueError(f"{data_path} has no valid minute rows")

    return df


def calculate_daily_factors(
    minute_df,
    min_minutes=30,
    dov_mean_window=30,
    dov_mean_quantile=0.85,
    dov_std_window=50,
    dov_std_quantile=0.80,
):
    records = []

    for trade_date, day_df in minute_df.groupby("date", sort=True):
        day_df = day_df.sort_values("datetime")
        delta_close = day_df["close"].diff()
        delta_oi = day_df["open_interest"].diff()

        valid_pv = pd.DataFrame({
            "delta_close": delta_close,
            "delta_oi": delta_oi,
        }).dropna()
        valid_minutes = len(valid_pv)
        pv = np.nan

        if valid_minutes >= min_minutes:
            close_std = valid_pv["delta_close"].std()
            oi_std = valid_pv["delta_oi"].std()
            if (
                pd.notna(close_std)
                and pd.notna(oi_std)
                and close_std != 0
                and oi_std != 0
            ):
                pv = valid_pv["delta_close"].corr(valid_pv["delta_oi"])

        abs_delta_oi = delta_oi.abs().replace(0, np.nan)
        dov = day_df["volume"] / abs_delta_oi
        dov = dov.replace([np.inf, -np.inf], np.nan).dropna()
        valid_dov_minutes = len(dov)
        dov_mean = dov.mean() if valid_dov_minutes > 0 else np.nan
        dov_std = dov.std() if valid_dov_minutes > 1 else np.nan

        records.append({
            "date": trade_date,
            "pv": pv,
            "dov_mean": dov_mean,
            "dov_std": dov_std,
            "valid_minutes": valid_minutes,
            "valid_dov_minutes": valid_dov_minutes,
        })

    factor = pd.DataFrame(records).sort_values("date").reset_index(drop=True)
    factor["raw_signal"] = np.select(
        [factor["pv"] > 0, factor["pv"] < 0],
        [1, -1],
        default=0,
    ).astype(int)

    factor["dov_mean_threshold"] = (
        factor["dov_mean"]
        .shift(1)
        .rolling(dov_mean_window, min_periods=dov_mean_window)
        .quantile(dov_mean_quantile)
    )
    factor["dov_std_threshold"] = (
        factor["dov_std"]
        .shift(1)
        .rolling(dov_std_window, min_periods=dov_std_window)
        .quantile(dov_std_quantile)
    )

    mean_reversed = factor["dov_mean"] > factor["dov_mean_threshold"]
    std_reversed = factor["dov_std"] > factor["dov_std_threshold"]
    factor["is_reversed"] = (mean_reversed | std_reversed).fillna(False)
    factor["final_signal"] = np.where(
        factor["is_reversed"],
        -factor["raw_signal"],
        factor["raw_signal"],
    ).astype(int)

    return factor[FACTOR_COLUMNS]


def build_daily_prices(minute_df):
    ordered = minute_df.sort_values(["date", "datetime"])
    daily_open = (
        ordered
        .drop_duplicates(subset=["date"], keep="first")
        .loc[:, ["date", "open"]]
        .rename(columns={"open": "open_price"})
    )
    daily_close = (
        ordered
        .drop_duplicates(subset=["date"], keep="last")
        .loc[:, ["date", "close"]]
        .rename(columns={"close": "close_price"})
    )

    daily_prices = (
        daily_open
        .merge(daily_close, on="date", how="inner")
        .sort_values("date")
        .reset_index(drop=True)
    )
    return daily_prices


def run_backtest(factor, daily_prices, cost_rate=0.0):
    backtest = (
        daily_prices
        .merge(factor, on="date", how="left")
        .sort_values("date")
        .reset_index(drop=True)
    )

    backtest["raw_signal"] = backtest["raw_signal"].fillna(0).astype(int)
    backtest["final_signal"] = backtest["final_signal"].fillna(0).astype(int)

    signal_date = backtest["date"].shift(1)
    executed_signal = backtest["final_signal"].shift(1).fillna(0).astype(int)
    backtest["signal_date"] = signal_date
    backtest["executed_signal"] = executed_signal
    backtest["position"] = executed_signal
    backtest["ret_open_to_close"] = (
        backtest["close_price"] / backtest["open_price"] - 1
    )

    backtest = backtest.dropna(
        subset=["open_price", "close_price", "ret_open_to_close"]
    ).copy()

    backtest["strategy_return"] = (
        backtest["position"] * backtest["ret_open_to_close"]
    )
    backtest["round_turnover"] = backtest["position"].abs() * 2
    backtest["cost"] = backtest["round_turnover"] * cost_rate
    backtest["strategy_return_net"] = (
        backtest["strategy_return"] - backtest["cost"]
    )
    backtest["strategy_nav"] = (1 + backtest["strategy_return_net"]).cumprod()
    backtest["benchmark_nav"] = (1 + backtest["ret_open_to_close"]).cumprod()

    return backtest[BACKTEST_COLUMNS]


def calculate_summary(symbol, backtest):
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
            "num_backtest_days": 0,
            "num_trading_days": 0,
            "num_long_days": 0,
            "num_short_days": 0,
            "num_flat_days": 0,
            "num_round_trips": 0,
            "final_nav": np.nan,
            "benchmark_final_nav": np.nan,
        }])[SUMMARY_COLUMNS]

    num_days = len(backtest)
    nav_final = backtest["strategy_nav"].iloc[-1]
    benchmark_final_nav = backtest["benchmark_nav"].iloc[-1]
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
        "num_backtest_days": num_days,
        "num_trading_days": num_trading_days,
        "num_long_days": int((backtest["position"] == 1).sum()),
        "num_short_days": int((backtest["position"] == -1).sum()),
        "num_flat_days": int((backtest["position"] == 0).sum()),
        "num_round_trips": int(has_position.sum()),
        "final_nav": nav_final,
        "benchmark_final_nav": benchmark_final_nav,
    }])

    return summary[SUMMARY_COLUMNS]


def build_output_paths(output_root, figure_root, symbol):
    output_root = Path(output_root)
    figure_root = Path(figure_root)
    return {
        "factor": output_root / "factors" / f"{symbol}_cpv_dov_factor.csv",
        "backtest": output_root / "backtest" / f"{symbol}_cpv_dov_backtest_daily.csv",
        "summary": output_root / "summary" / f"{symbol}_cpv_dov_summary.csv",
        "simple_nav": figure_root / f"{symbol}_cpv_dov_simple_nav.png",
        "compound_nav": figure_root / f"{symbol}_cpv_dov_compound_nav.png",
    }


def save_outputs(factor, backtest, summary, output_paths):
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
    normalize_dates_for_output(summary).to_csv(
        output_paths["summary"],
        index=False,
    )


def plot_nav_curves(backtest, symbol, output_paths):
    mpl_config_dir = Path(tempfile.gettempdir()) / "lc_research_matplotlib"
    mpl_config_dir.mkdir(parents=True, exist_ok=True)
    os.environ.setdefault("MPLCONFIGDIR", str(mpl_config_dir))

    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    plot_df = backtest.copy()
    plot_df["date"] = pd.to_datetime(plot_df["date"])
    plot_df["strategy_simple_nav"] = 1 + plot_df["strategy_return_net"].cumsum()
    plot_df["benchmark_simple_nav"] = 1 + plot_df["ret_open_to_close"].cumsum()

    figure_specs = [
        (
            "simple_nav",
            "strategy_simple_nav",
            "benchmark_simple_nav",
            f"{symbol} CPV DOV Intraday Simple NAV",
        ),
        (
            "compound_nav",
            "strategy_nav",
            "benchmark_nav",
            f"{symbol} CPV DOV Intraday Compound NAV",
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


def run_cpv_dov_strategy(
    symbol,
    data_path,
    output_root,
    figure_root,
    min_minutes=30,
    dov_mean_window=30,
    dov_mean_quantile=0.85,
    dov_std_window=50,
    dov_std_quantile=0.80,
    cost_rate=0.0,
):
    minute_df = load_minute_data(data_path)
    factor = calculate_daily_factors(
        minute_df,
        min_minutes=min_minutes,
        dov_mean_window=dov_mean_window,
        dov_mean_quantile=dov_mean_quantile,
        dov_std_window=dov_std_window,
        dov_std_quantile=dov_std_quantile,
    )
    daily_prices = build_daily_prices(minute_df)
    backtest = run_backtest(factor, daily_prices, cost_rate=cost_rate)
    summary = calculate_summary(symbol, backtest)
    output_paths = build_output_paths(output_root, figure_root, symbol)
    save_outputs(factor, backtest, summary, output_paths)
    plot_nav_curves(backtest, symbol, output_paths)

    return factor, backtest, summary, output_paths


def main():
    args = parse_args()
    symbol = args.symbol.upper()
    data_path = (
        Path(args.data_path)
        if args.data_path is not None
        else PROJECT_ROOT / "data" / f"{symbol}.csv"
    )

    factor, backtest, summary, output_paths = run_cpv_dov_strategy(
        symbol=symbol,
        data_path=data_path,
        output_root=args.output_root,
        figure_root=args.figure_root,
        min_minutes=args.min_minutes,
        dov_mean_window=args.dov_mean_window,
        dov_mean_quantile=args.dov_mean_quantile,
        dov_std_window=args.dov_std_window,
        dov_std_quantile=args.dov_std_quantile,
        cost_rate=args.cost_rate,
    )

    print("CPV DOV intraday strategy backtest complete.")
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
