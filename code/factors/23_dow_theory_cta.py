import argparse
import os
import tempfile
from pathlib import Path

import numpy as np
import pandas as pd

from config import SYMBOL as CONFIG_SYMBOL


PROJECT_ROOT = Path(__file__).resolve().parents[1]
ANNUAL_DAYS = 252

# 单组回测默认参数；参数扫描使用下面的 M_GRID 和 X_GRID。
M = 180
x = 10
initial_position = 0
M_GRID = [10, 20, 30, 60, 90, 120, 180, 270, 360]
X_GRID = [1, 2, 3, 5, 10, 20]

DAILY_COLUMNS = [
    "date",
    "close",
    "is_peak",
    "is_trough",
    "confirmed_peak",
    "confirmed_trough",
    "upper_channel",
    "lower_channel",
    "signal",
    "position",
    "daily_return",
    "strategy_return",
    "strategy_nav",
    "benchmark_nav",
]

SUMMARY_COLUMNS = [
    "start_date",
    "end_date",
    "M",
    "x",
    "annual_return",
    "annual_volatility",
    "sharpe_ratio",
    "max_drawdown",
    "win_rate",
    "psychological_win_rate",
    "daily_win_rate",
    "traditional_win_rate",
    "num_trading_days",
    "num_long_days",
    "num_short_days",
    "num_flat_days",
    "num_trades",
    "num_completed_trades",
    "final_nav",
    "benchmark_final_nav",
]

SWEEP_COLUMNS = [
    "rank",
    "symbol",
    "start_date",
    "end_date",
    "M",
    "x",
    "annual_return",
    "annual_volatility",
    "sharpe_ratio",
    "max_drawdown",
    "psychological_win_rate",
    "daily_win_rate",
    "traditional_win_rate",
    "num_trading_days",
    "num_trades",
    "num_completed_trades",
    "final_nav",
    "benchmark_final_nav",
]


def parse_int_list(value):
    if isinstance(value, (list, tuple)):
        return [int(item) for item in value]

    values = []
    for item in str(value).split(","):
        item = item.strip()
        if item:
            values.append(int(item))

    if not values:
        raise ValueError("parameter list cannot be empty")

    return values


def parse_args():
    parser = argparse.ArgumentParser(
        description="Run a pure price-based Dow Theory CTA strategy."
    )
    parser.add_argument(
        "--symbol",
        default=os.environ.get("SYMBOL", CONFIG_SYMBOL),
        help="Symbol to run. Defaults to SYMBOL in code/config.py.",
    )
    parser.add_argument(
        "--data-path",
        default=None,
        help="Daily or minute CSV path. Defaults to data/{symbol}.csv.",
    )
    parser.add_argument(
        "--output-root",
        default=str(PROJECT_ROOT / "results" / "tables"),
        help="Root directory for output CSV files.",
    )
    parser.add_argument(
        "--figure-root",
        default=str(PROJECT_ROOT / "results" / "figures" / "backtest"),
        help="Root directory for output figure files.",
    )
    parser.add_argument(
        "--M",
        dest="M",
        type=int,
        default=M,
        help="Lookback length for Dow Theory channel regressions.",
    )
    parser.add_argument(
        "--x",
        dest="x",
        type=int,
        default=x,
        help="Half-window length for local peak/trough confirmation.",
    )
    parser.add_argument(
        "--initial-position",
        type=int,
        choices=[-1, 0, 1],
        default=initial_position,
        help="Initial position before the first signal.",
    )
    parser.add_argument(
        "--symbol-prefix",
        action="store_true",
        help="Prefix output filenames with the symbol to avoid overwrites.",
    )
    parser.add_argument(
        "--sweep",
        action="store_true",
        help="Run an M/x parameter sweep. This is the default unless --single is set.",
    )
    parser.add_argument(
        "--single",
        action="store_true",
        help="Run only one M/x pair instead of the default parameter sweep.",
    )
    parser.add_argument(
        "--M-list",
        default=",".join(str(value) for value in M_GRID),
        help="Comma-separated M values used by --sweep.",
    )
    parser.add_argument(
        "--x-list",
        default=",".join(str(value) for value in X_GRID),
        help="Comma-separated x values used by --sweep.",
    )
    return parser.parse_args()


def load_data(path):
    df = pd.read_csv(path)
    df.columns = df.columns.str.strip()

    if "close" not in df.columns:
        raise ValueError(f"{path} missing required column: close")

    if "date" not in df.columns and "datetime" not in df.columns:
        raise ValueError(f"{path} must contain either date or datetime")

    datetime_values = None
    if "datetime" in df.columns:
        datetime_values = pd.to_datetime(df["datetime"], errors="coerce")

    if "date" in df.columns:
        date_values = pd.to_datetime(
            df["date"].astype(str).str.strip(),
            errors="coerce",
        )
        if datetime_values is not None:
            date_values = date_values.fillna(datetime_values.dt.normalize())
    else:
        date_values = datetime_values.dt.normalize()

    work = pd.DataFrame({
        "date": date_values,
        "close": pd.to_numeric(df["close"], errors="coerce"),
    })
    if datetime_values is not None:
        work["datetime"] = datetime_values

    work = work.dropna(subset=["date", "close"]).copy()
    work["date"] = pd.to_datetime(work["date"]).dt.normalize()

    if "datetime" in work.columns:
        work = work.dropna(subset=["datetime"])
        work = work.sort_values(["date", "datetime"])
    else:
        work = work.sort_values("date")

    daily = (
        work
        .drop_duplicates(subset=["date"], keep="last")
        .loc[:, ["date", "close"]]
        .reset_index(drop=True)
    )

    if daily.empty:
        raise ValueError(f"{path} has no valid date/close rows")

    return daily


def detect_local_extrema(df, x):
    if x < 1:
        raise ValueError("x must be at least 1")

    out = df.copy()
    close = out["close"].to_numpy(dtype=float)
    n = len(out)

    is_peak = np.zeros(n, dtype=bool)
    is_trough = np.zeros(n, dtype=bool)

    for i in range(x, n - x):
        window = close[i - x:i + x + 1]
        is_peak[i] = close[i] == np.max(window)
        is_trough[i] = close[i] == np.min(window)

    confirmed_peak = np.zeros(n, dtype=bool)
    confirmed_trough = np.zeros(n, dtype=bool)
    if n > x:
        confirmed_peak[x:] = is_peak[:-x]
        confirmed_trough[x:] = is_trough[:-x]

    out["is_peak"] = is_peak
    out["is_trough"] = is_trough
    # Confirmation is recorded on the day the information becomes available.
    out["confirmed_peak"] = confirmed_peak
    out["confirmed_trough"] = confirmed_trough

    return out


def calculate_dow_channels(df, M, x):
    if M < 1:
        raise ValueError("M must be at least 1")

    out = df.copy()
    if "is_peak" not in out.columns or "is_trough" not in out.columns:
        out = detect_local_extrema(out, x)

    close = out["close"].to_numpy(dtype=float)
    n = len(out)
    all_indices = np.arange(n)
    peak_indices = all_indices[out["is_peak"].to_numpy(dtype=bool)]
    trough_indices = all_indices[out["is_trough"].to_numpy(dtype=bool)]

    upper_channel = np.full(n, np.nan)
    lower_channel = np.full(n, np.nan)

    for T in range(n):
        if T < M:
            continue

        lookback_start = max(0, T - M)

        eligible_peaks = peak_indices[
            (peak_indices >= lookback_start)
            & (peak_indices <= T)
            & (peak_indices + x <= T)
        ]
        if len(eligible_peaks) >= 2:
            slope, intercept = np.polyfit(
                eligible_peaks.astype(float),
                close[eligible_peaks],
                1,
            )
            upper_channel[T] = intercept + slope * T

        eligible_troughs = trough_indices[
            (trough_indices >= lookback_start)
            & (trough_indices <= T)
            & (trough_indices + x <= T)
        ]
        if len(eligible_troughs) >= 2:
            slope, intercept = np.polyfit(
                eligible_troughs.astype(float),
                close[eligible_troughs],
                1,
            )
            lower_channel[T] = intercept + slope * T

    out["upper_channel"] = upper_channel
    out["lower_channel"] = lower_channel

    return out


def generate_signals(df, initial_position=initial_position):
    out = df.copy()
    signals = np.zeros(len(out), dtype=int)
    previous_signal = initial_position

    for i, (_, row) in enumerate(out.iterrows()):
        close = row["close"]
        upper = row["upper_channel"]
        lower = row["lower_channel"]

        if pd.notna(upper) and pd.notna(lower):
            if close > upper:
                previous_signal = 1
            elif close < lower:
                previous_signal = -1

        signals[i] = previous_signal

    out["signal"] = signals
    out["position"] = out["signal"]

    return out


def backtest(df, initial_position=initial_position):
    out = df.copy()
    if "position" not in out.columns:
        out["position"] = out.get("signal", initial_position)

    out["daily_return"] = out["close"].pct_change().fillna(0.0)
    previous_position = out["position"].shift(1).fillna(initial_position)
    out["strategy_return"] = previous_position * out["daily_return"]
    out["strategy_nav"] = (1 + out["strategy_return"]).cumprod()
    out["benchmark_nav"] = out["close"] / out["close"].iloc[0]

    return out


def calculate_report_win_rates(df, initial_position=initial_position):
    if df.empty:
        return {
            "psychological_win_rate": np.nan,
            "daily_win_rate": np.nan,
            "traditional_win_rate": np.nan,
            "num_completed_trades": 0,
        }

    position = df["position"].astype(int).reset_index(drop=True)
    previous_position = position.shift(1).fillna(initial_position).astype(int)
    close = df["close"].reset_index(drop=True)
    strategy_nav = df["strategy_nav"].reset_index(drop=True)
    strategy_return = df["strategy_return"].reset_index(drop=True)

    # 日频胜率：实际持仓隔夜后，当日收盘净值高于昨日收盘净值。
    held_overnight = previous_position != 0
    daily_win_rate = (
        (strategy_return.loc[held_overnight] > 0).mean()
        if held_overnight.any()
        else np.nan
    )

    # 心理胜率：持仓期间，当日净值高于最近一次调仓时的净值。
    psychological_outcomes = []
    last_trade_nav = np.nan
    for i in range(len(df)):
        if position.iloc[i] != previous_position.iloc[i]:
            last_trade_nav = strategy_nav.iloc[i]

        if position.iloc[i] != 0 and pd.notna(last_trade_nav):
            psychological_outcomes.append(strategy_nav.iloc[i] > last_trade_nav)

    psychological_win_rate = (
        float(np.mean(psychological_outcomes))
        if psychological_outcomes
        else np.nan
    )

    # 传统胜率：一次完整持仓从开仓到平仓/反手，方向收益为正则胜。
    completed_trade_wins = []
    current_position = initial_position
    entry_price = close.iloc[0] if initial_position != 0 else np.nan

    for i in range(len(df)):
        new_position = position.iloc[i]
        if new_position == current_position:
            continue

        if current_position != 0 and pd.notna(entry_price):
            trade_return = current_position * (close.iloc[i] / entry_price - 1)
            completed_trade_wins.append(trade_return > 0)

        entry_price = close.iloc[i] if new_position != 0 else np.nan
        current_position = new_position

    traditional_win_rate = (
        float(np.mean(completed_trade_wins))
        if completed_trade_wins
        else np.nan
    )

    return {
        "psychological_win_rate": psychological_win_rate,
        "daily_win_rate": daily_win_rate,
        "traditional_win_rate": traditional_win_rate,
        "num_completed_trades": len(completed_trade_wins),
    }


def calculate_summary(
    df,
    M_value=M,
    x_value=x,
    annual_days=ANNUAL_DAYS,
    initial_position=initial_position,
):
    if df.empty:
        return pd.DataFrame([{
            "start_date": np.nan,
            "end_date": np.nan,
            "M": M_value,
            "x": x_value,
            "annual_return": np.nan,
            "annual_volatility": np.nan,
            "sharpe_ratio": np.nan,
            "max_drawdown": np.nan,
            "win_rate": np.nan,
            "psychological_win_rate": np.nan,
            "daily_win_rate": np.nan,
            "traditional_win_rate": np.nan,
            "num_trading_days": 0,
            "num_long_days": 0,
            "num_short_days": 0,
            "num_flat_days": 0,
            "num_trades": 0,
            "num_completed_trades": 0,
            "final_nav": np.nan,
            "benchmark_final_nav": np.nan,
        }])[SUMMARY_COLUMNS]

    num_trading_days = len(df)
    final_nav = df["strategy_nav"].iloc[-1]
    annual_return = (
        final_nav ** (annual_days / num_trading_days) - 1
        if final_nav > 0 and num_trading_days > 0
        else np.nan
    )
    annual_volatility = df["strategy_return"].std() * np.sqrt(annual_days)
    sharpe_ratio = (
        annual_return / annual_volatility
        if pd.notna(annual_volatility) and annual_volatility != 0
        else np.nan
    )
    drawdown = df["strategy_nav"] / df["strategy_nav"].cummax() - 1
    max_drawdown = drawdown.min()
    win_rates = calculate_report_win_rates(
        df,
        initial_position=initial_position,
    )

    position = df["position"]
    previous_position = position.shift(1).fillna(initial_position)
    num_trades = int((position != previous_position).sum())

    summary = pd.DataFrame([{
        "start_date": df["date"].min(),
        "end_date": df["date"].max(),
        "M": M_value,
        "x": x_value,
        "annual_return": annual_return,
        "annual_volatility": annual_volatility,
        "sharpe_ratio": sharpe_ratio,
        "max_drawdown": max_drawdown,
        # 兼容旧字段：win_rate 指向报告口径中的日频胜率。
        "win_rate": win_rates["daily_win_rate"],
        "psychological_win_rate": win_rates["psychological_win_rate"],
        "daily_win_rate": win_rates["daily_win_rate"],
        "traditional_win_rate": win_rates["traditional_win_rate"],
        "num_trading_days": num_trading_days,
        "num_long_days": int((position == 1).sum()),
        "num_short_days": int((position == -1).sum()),
        "num_flat_days": int((position == 0).sum()),
        "num_trades": num_trades,
        "num_completed_trades": win_rates["num_completed_trades"],
        "final_nav": final_nav,
        "benchmark_final_nav": df["benchmark_nav"].iloc[-1],
    }])

    return summary[SUMMARY_COLUMNS]


def setup_matplotlib():
    mpl_config_dir = Path(tempfile.gettempdir()) / "lc_research_matplotlib"
    mpl_config_dir.mkdir(parents=True, exist_ok=True)
    os.environ.setdefault("MPLCONFIGDIR", str(mpl_config_dir))

    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    return plt


def plot_nav(df, output_path=None, symbol=None):
    output_path = (
        Path(output_path)
        if output_path is not None
        else PROJECT_ROOT / "results" / "figures" / "backtest" / "dow_theory_cta_nav.png"
    )
    output_path.parent.mkdir(parents=True, exist_ok=True)

    plt = setup_matplotlib()
    plot_df = df.copy()
    plot_df["date"] = pd.to_datetime(plot_df["date"])

    fig, ax = plt.subplots(figsize=(12, 6))
    ax.plot(
        plot_df["date"],
        plot_df["strategy_nav"],
        label="Strategy",
        linewidth=1.6,
    )
    ax.plot(
        plot_df["date"],
        plot_df["benchmark_nav"],
        label="Benchmark",
        linewidth=1.2,
        alpha=0.85,
    )
    title = "Dow Theory CTA NAV"
    if symbol:
        title = f"{symbol.upper()} Dow Theory CTA NAV"
    ax.set_title(title)
    ax.set_xlabel("Date")
    ax.set_ylabel("NAV")
    ax.grid(True, linestyle="--", alpha=0.3)
    ax.legend()
    fig.autofmt_xdate()
    fig.tight_layout()
    fig.savefig(output_path, dpi=150)
    plt.close(fig)

    return output_path


def plot_channels(
    df,
    output_path=None,
    initial_position=initial_position,
    symbol=None,
):
    output_path = (
        Path(output_path)
        if output_path is not None
        else PROJECT_ROOT / "results" / "figures" / "backtest" / "dow_theory_cta_channels.png"
    )
    output_path.parent.mkdir(parents=True, exist_ok=True)

    plt = setup_matplotlib()
    plot_df = df.copy()
    plot_df["date"] = pd.to_datetime(plot_df["date"])

    previous_signal = plot_df["signal"].shift(1).fillna(initial_position)
    long_signal = (plot_df["signal"] == 1) & (previous_signal != 1)
    short_signal = (plot_df["signal"] == -1) & (previous_signal != -1)

    fig, ax = plt.subplots(figsize=(13, 7))
    ax.plot(plot_df["date"], plot_df["close"], label="Close", linewidth=1.2)
    ax.plot(
        plot_df["date"],
        plot_df["upper_channel"],
        label="Upper Channel",
        linewidth=1.0,
        alpha=0.85,
    )
    ax.plot(
        plot_df["date"],
        plot_df["lower_channel"],
        label="Lower Channel",
        linewidth=1.0,
        alpha=0.85,
    )
    ax.scatter(
        plot_df.loc[long_signal, "date"],
        plot_df.loc[long_signal, "close"],
        label="Long Signal",
        marker="^",
        s=36,
        color="#2ca02c",
        zorder=4,
    )
    ax.scatter(
        plot_df.loc[short_signal, "date"],
        plot_df.loc[short_signal, "close"],
        label="Short Signal",
        marker="v",
        s=36,
        color="#d62728",
        zorder=4,
    )
    title = "Dow Theory CTA Channels"
    if symbol:
        title = f"{symbol.upper()} Dow Theory CTA Channels"
    ax.set_title(title)
    ax.set_xlabel("Date")
    ax.set_ylabel("Close")
    ax.grid(True, linestyle="--", alpha=0.3)
    ax.legend()
    fig.autofmt_xdate()
    fig.tight_layout()
    fig.savefig(output_path, dpi=150)
    plt.close(fig)

    return output_path


def normalize_dates_for_output(df):
    out = df.copy()
    if "date" in out.columns:
        out["date"] = pd.to_datetime(out["date"]).dt.strftime("%Y-%m-%d")
    for column in ["start_date", "end_date"]:
        if column in out.columns:
            out[column] = pd.to_datetime(
                out[column],
                errors="coerce",
            ).dt.strftime("%Y-%m-%d")
    return out


def build_output_paths(output_root, figure_root, symbol=None, symbol_prefix=False):
    stem = "dow_theory_cta"
    if symbol_prefix and symbol:
        stem = f"{symbol}_{stem}"

    output_root = Path(output_root)
    figure_root = Path(figure_root)

    return {
        "daily": output_root / "backtest" / f"{stem}_daily.csv",
        "summary": output_root / "summary" / f"{stem}_summary.csv",
        "nav": figure_root / f"{stem}_nav.png",
        "channels": figure_root / f"{stem}_channels.png",
    }


def build_sweep_output_path(output_root, symbol):
    output_root = Path(output_root)
    symbol_text = symbol.upper() if symbol else "UNKNOWN"
    return output_root / "summary" / f"{symbol_text}_dow_theory_cta_param_compare.csv"


def save_outputs(daily, summary, output_paths):
    for key in ["daily", "summary"]:
        output_paths[key].parent.mkdir(parents=True, exist_ok=True)

    normalize_dates_for_output(daily[DAILY_COLUMNS]).to_csv(
        output_paths["daily"],
        index=False,
    )
    normalize_dates_for_output(summary).to_csv(
        output_paths["summary"],
        index=False,
    )


def run_dow_theory_cta(
    data_path,
    output_root,
    figure_root,
    M_value=M,
    x_value=x,
    initial_position_value=initial_position,
    symbol=None,
    symbol_prefix=False,
):
    daily = load_data(data_path)
    daily = detect_local_extrema(daily, x_value)
    daily = calculate_dow_channels(daily, M_value, x_value)
    daily = generate_signals(daily, initial_position=initial_position_value)
    daily = backtest(daily, initial_position=initial_position_value)
    summary = calculate_summary(
        daily,
        M_value=M_value,
        x_value=x_value,
        initial_position=initial_position_value,
    )

    output_paths = build_output_paths(
        output_root,
        figure_root,
        symbol=symbol,
        symbol_prefix=symbol_prefix,
    )
    save_outputs(daily, summary, output_paths)
    plot_nav(daily, output_paths["nav"], symbol=symbol)
    plot_channels(
        daily,
        output_paths["channels"],
        initial_position=initial_position_value,
        symbol=symbol,
    )

    return daily, summary, output_paths


def run_parameter_sweep(
    data_path,
    output_root,
    M_values,
    x_values,
    initial_position_value=initial_position,
    symbol=None,
):
    base_daily = load_data(data_path)
    summaries = []

    for x_value in x_values:
        extrema = detect_local_extrema(base_daily, x_value)

        for M_value in M_values:
            daily = calculate_dow_channels(extrema, M_value, x_value)
            daily = generate_signals(
                daily,
                initial_position=initial_position_value,
            )
            daily = backtest(
                daily,
                initial_position=initial_position_value,
            )
            summary = calculate_summary(
                daily,
                M_value=M_value,
                x_value=x_value,
                initial_position=initial_position_value,
            )
            summary.insert(0, "symbol", symbol.upper() if symbol else "")
            summaries.append(summary)

    comparison = pd.concat(summaries, ignore_index=True)
    comparison = comparison.sort_values(
        ["annual_return", "sharpe_ratio", "final_nav"],
        ascending=[False, False, False],
        na_position="last",
    ).reset_index(drop=True)
    comparison.insert(0, "rank", np.arange(1, len(comparison) + 1))

    output_path = build_sweep_output_path(output_root, symbol)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    normalize_dates_for_output(comparison[SWEEP_COLUMNS]).to_csv(
        output_path,
        index=False,
    )

    return comparison[SWEEP_COLUMNS], output_path


def run_best_from_comparison(
    comparison,
    data_path,
    output_root,
    figure_root,
    initial_position_value=initial_position,
    symbol=None,
):
    if comparison.empty:
        raise ValueError("comparison is empty, cannot run best parameter backtest")

    best = comparison.iloc[0]
    best_M = int(best["M"])
    best_x = int(best["x"])

    daily, summary, output_paths = run_dow_theory_cta(
        data_path=data_path,
        output_root=output_root,
        figure_root=figure_root,
        M_value=best_M,
        x_value=best_x,
        initial_position_value=initial_position_value,
        symbol=symbol,
        symbol_prefix=True,
    )

    return daily, summary, output_paths, best_M, best_x


def main():
    args = parse_args()
    symbol = args.symbol.upper()
    data_path = (
        Path(args.data_path)
        if args.data_path is not None
        else PROJECT_ROOT / "data" / f"{symbol}.csv"
    )

    if args.sweep or not args.single:
        M_values = parse_int_list(args.M_list)
        x_values = parse_int_list(args.x_list)
        comparison, output_path = run_parameter_sweep(
            data_path=data_path,
            output_root=args.output_root,
            M_values=M_values,
            x_values=x_values,
            initial_position_value=args.initial_position,
            symbol=symbol,
        )
        best_daily, best_summary, best_output_paths, best_M, best_x = (
            run_best_from_comparison(
                comparison=comparison,
                data_path=data_path,
                output_root=args.output_root,
                figure_root=args.figure_root,
                initial_position_value=args.initial_position,
                symbol=symbol,
            )
        )

        print("Dow Theory CTA parameter sweep complete.")
        print(f"symbol: {symbol}")
        print(f"M values: {M_values}")
        print(f"x values: {x_values}")
        print(f"comparison file: {output_path}")
        print(f"best annual return params: M={best_M}, x={best_x}")
        print(f"best daily rows: {len(best_daily)}")
        print(f"best daily file: {best_output_paths['daily']}")
        print(f"best summary file: {best_output_paths['summary']}")
        print(f"best NAV figure: {best_output_paths['nav']}")
        print(f"best channels figure: {best_output_paths['channels']}")
        print(
            comparison.loc[
                :,
                [
                    "rank",
                    "M",
                    "x",
                    "annual_return",
                    "sharpe_ratio",
                    "max_drawdown",
                    "num_trades",
                    "final_nav",
                ],
            ].to_string(index=False)
        )
        print(best_summary.to_string(index=False))
        return

    daily, summary, output_paths = run_dow_theory_cta(
        data_path=data_path,
        output_root=args.output_root,
        figure_root=args.figure_root,
        M_value=args.M,
        x_value=args.x,
        initial_position_value=args.initial_position,
        symbol=symbol,
        symbol_prefix=args.symbol_prefix,
    )

    print("Dow Theory CTA backtest complete.")
    print(f"symbol: {symbol}")
    print(f"daily rows: {len(daily)}")
    print(f"daily file: {output_paths['daily']}")
    print(f"summary file: {output_paths['summary']}")
    print(f"NAV figure: {output_paths['nav']}")
    print(f"channels figure: {output_paths['channels']}")
    print(summary.to_string(index=False))


if __name__ == "__main__":
    main()
