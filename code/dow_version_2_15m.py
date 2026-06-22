import argparse
import os
import tempfile
from pathlib import Path

import numpy as np
import pandas as pd

from config import SYMBOL as CONFIG_SYMBOL


PROJECT_ROOT = Path(__file__).resolve().parents[1]
TRADING_DAYS_PER_YEAR = 252
BARS_PER_DAY = 15
ANNUAL_DAYS = TRADING_DAYS_PER_YEAR * BARS_PER_DAY
STRATEGY_NAME = "dow_version_2_15m"
BAR_FREQ = "15min"
OUTPUT_DATETIME_FORMAT = "%Y-%m-%d %H:%M:%S"

# 单组回测默认参数；参数扫描使用下面的 M_GRID 和 X_GRID。
# 15 分钟版本的 M/x 单位是 15 分钟 bar，默认近似沿用日线版本的时间跨度。
M = 180 * BARS_PER_DAY
x = 10 * BARS_PER_DAY
initial_position = 0
STOP_LOSS_PCT = 0.05
M_GRID = [
    value * BARS_PER_DAY
    for value in [10, 20, 30, 60, 90, 120, 180, 270, 360]
]
X_GRID = [
    value * BARS_PER_DAY
    for value in [1, 2, 3, 5, 10, 20]
]

BAR_COLUMNS = [
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
    "bar_return",
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
    "bar_win_rate",
    "traditional_win_rate",
    "num_bars",
    "num_long_bars",
    "num_short_bars",
    "num_flat_bars",
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
    "bar_win_rate",
    "traditional_win_rate",
    "num_bars",
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
        description="Run a 15-minute pure price-based Dow Theory CTA strategy."
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
        default=str(PROJECT_ROOT / "results" / "tables" / STRATEGY_NAME),
        help="Root directory for output CSV files.",
    )
    parser.add_argument(
        "--figure-root",
        default=str(PROJECT_ROOT / "results" / "figures" / "backtest" / STRATEGY_NAME),
        help="Root directory for output figure files.",
    )
    parser.add_argument(
        "--M",
        dest="M",
        type=int,
        default=M,
        help="Lookback length in 15-minute bars for Dow Theory channel regressions.",
    )
    parser.add_argument(
        "--x",
        dest="x",
        type=int,
        default=x,
        help="Half-window length in 15-minute bars for local peak/trough confirmation.",
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
        help="Comma-separated M values in 15-minute bars used by --sweep.",
    )
    parser.add_argument(
        "--x-list",
        default=",".join(str(value) for value in X_GRID),
        help="Comma-separated x values in 15-minute bars used by --sweep.",
    )
    return parser.parse_args()


def load_data(path):
    df = pd.read_csv(path)
    df.columns = df.columns.str.strip()

    if "close" not in df.columns:
        raise ValueError(f"{path} missing required column: close")

    if "datetime" not in df.columns and "date" not in df.columns:
        raise ValueError(f"{path} must contain either datetime or date")

    time_column = "datetime" if "datetime" in df.columns else "date"
    datetime_values = pd.to_datetime(
        df[time_column].astype(str).str.strip(),
        errors="coerce",
    )
    if time_column == "date" and datetime_values.dt.time.nunique(dropna=True) <= 1:
        raise ValueError(f"{path} needs intraday datetime values for 15m resampling")

    work = pd.DataFrame({"datetime": datetime_values})
    price_columns = ["open", "high", "low", "close"]
    flow_columns = ["volume", "total_turnover"]
    carry_columns = ["open_interest"]

    for column in price_columns + flow_columns + carry_columns:
        if column in df.columns:
            work[column] = pd.to_numeric(df[column], errors="coerce")

    work = work.dropna(subset=["datetime", "close"]).copy()
    work = work.sort_values("datetime")
    work = work.drop_duplicates(subset=["datetime"], keep="last")

    agg = {"close": "last"}
    if "open" in work.columns:
        agg["open"] = "first"
    if "high" in work.columns:
        agg["high"] = "max"
    if "low" in work.columns:
        agg["low"] = "min"
    for column in flow_columns:
        if column in work.columns:
            agg[column] = "sum"
    for column in carry_columns:
        if column in work.columns:
            agg[column] = "last"

    bars = (
        work
        .set_index("datetime")
        .resample(BAR_FREQ, label="right", closed="right")
        .agg(agg)
        .dropna(subset=["close"])
        .reset_index()
    )

    daily = bars.rename(columns={"datetime": "date"}).loc[:, ["date", "close"]]

    if daily.empty:
        raise ValueError(f"{path} has no valid 15m date/close rows")

    return daily.reset_index(drop=True)


def detect_local_extrema(df, x):
    if x < 1:
        raise ValueError("x must be at least 1")

    out = df.copy()
    close = out["close"].astype(float)
    n = len(out)

    is_peak = np.zeros(n, dtype=bool)
    is_trough = np.zeros(n, dtype=bool)
    if n >= 2 * x + 1:
        rolling_window = 2 * x + 1
        rolling_max = close.rolling(
            rolling_window,
            center=True,
            min_periods=rolling_window,
        ).max()
        rolling_min = close.rolling(
            rolling_window,
            center=True,
            min_periods=rolling_window,
        ).min()
        close_values = close.to_numpy(dtype=float)
        is_peak = (
            (close_values == rolling_max.to_numpy(dtype=float))
            & rolling_max.notna().to_numpy()
        )
        is_trough = (
            (close_values == rolling_min.to_numpy(dtype=float))
            & rolling_min.notna().to_numpy()
        )

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


def calculate_regression_channel(close, extrema_mask, M, x):
    n = len(close)
    channel = np.full(n, np.nan)
    if n == 0 or M < 1:
        return channel

    extrema_indices = np.flatnonzero(extrema_mask)
    if len(extrema_indices) < 2:
        return channel

    point_count = np.zeros(n, dtype=float)
    point_x = np.zeros(n, dtype=float)
    point_y = np.zeros(n, dtype=float)
    point_xx = np.zeros(n, dtype=float)
    point_xy = np.zeros(n, dtype=float)

    x_values = extrema_indices.astype(float)
    y_values = close[extrema_indices]
    point_count[extrema_indices] = 1.0
    point_x[extrema_indices] = x_values
    point_y[extrema_indices] = y_values
    point_xx[extrema_indices] = x_values * x_values
    point_xy[extrema_indices] = x_values * y_values

    prefix_count = np.concatenate(([0.0], np.cumsum(point_count)))
    prefix_x = np.concatenate(([0.0], np.cumsum(point_x)))
    prefix_y = np.concatenate(([0.0], np.cumsum(point_y)))
    prefix_xx = np.concatenate(([0.0], np.cumsum(point_xx)))
    prefix_xy = np.concatenate(([0.0], np.cumsum(point_xy)))

    T = np.arange(M, n)
    if len(T) == 0:
        return channel

    lo = T - M
    hi = T - x
    valid_range = hi >= lo
    if not valid_range.any():
        return channel

    T_valid = T[valid_range]
    lo = lo[valid_range]
    hi = hi[valid_range]
    hi_plus_one = hi + 1

    counts = prefix_count[hi_plus_one] - prefix_count[lo]
    sum_x = prefix_x[hi_plus_one] - prefix_x[lo]
    sum_y = prefix_y[hi_plus_one] - prefix_y[lo]
    sum_xx = prefix_xx[hi_plus_one] - prefix_xx[lo]
    sum_xy = prefix_xy[hi_plus_one] - prefix_xy[lo]

    denom = counts * sum_xx - sum_x * sum_x
    valid_fit = (counts >= 2) & (denom != 0)
    if not valid_fit.any():
        return channel

    slope = (counts[valid_fit] * sum_xy[valid_fit] - sum_x[valid_fit] * sum_y[valid_fit]) / denom[valid_fit]
    intercept = (sum_y[valid_fit] - slope * sum_x[valid_fit]) / counts[valid_fit]
    channel[T_valid[valid_fit]] = intercept + slope * T_valid[valid_fit]

    return channel


def calculate_dow_channels(df, M, x):
    if M < 1:
        raise ValueError("M must be at least 1")

    out = df.copy()
    if "is_peak" not in out.columns or "is_trough" not in out.columns:
        out = detect_local_extrema(out, x)

    close = out["close"].to_numpy(dtype=float)
    out["upper_channel"] = calculate_regression_channel(
        close,
        out["is_peak"].to_numpy(dtype=bool),
        M,
        x,
    )
    out["lower_channel"] = calculate_regression_channel(
        close,
        out["is_trough"].to_numpy(dtype=bool),
        M,
        x,
    )

    return out


def generate_signals(
    df,
    initial_position=initial_position,
    stop_loss_pct=STOP_LOSS_PCT,
):
    out = df.copy()
    signals = np.zeros(len(out), dtype=int)
    previous_signal = initial_position
    entry_price = (
        out["close"].iloc[0]
        if initial_position != 0 and not out.empty
        else np.nan
    )

    close_values = out["close"].to_numpy(dtype=float)
    upper_values = out["upper_channel"].to_numpy(dtype=float)
    lower_values = out["lower_channel"].to_numpy(dtype=float)

    for i in range(len(out)):
        close = close_values[i]
        upper = upper_values[i]
        lower = lower_values[i]
        stop_loss_triggered = False

        if stop_loss_pct is not None and not np.isnan(entry_price):
            if previous_signal == 1 and close <= entry_price * (1 - stop_loss_pct):
                previous_signal = 0
                entry_price = np.nan
                stop_loss_triggered = True
            elif previous_signal == -1 and close >= entry_price * (1 + stop_loss_pct):
                previous_signal = 0
                entry_price = np.nan
                stop_loss_triggered = True

        if not stop_loss_triggered and not np.isnan(upper) and not np.isnan(lower):
            if close > upper:
                if previous_signal != 1:
                    entry_price = close
                previous_signal = 1
            elif close < lower:
                if previous_signal != -1:
                    entry_price = close
                previous_signal = -1

        signals[i] = previous_signal

    out["signal"] = signals
    out["position"] = out["signal"]

    return out


def backtest(df, initial_position=initial_position):
    out = df.copy()
    if "position" not in out.columns:
        out["position"] = out.get("signal", initial_position)

    out["bar_return"] = out["close"].pct_change().fillna(0.0)
    previous_position = out["position"].shift(1).fillna(initial_position)
    out["strategy_return"] = previous_position * out["bar_return"]
    out["strategy_nav"] = (1 + out["strategy_return"]).cumprod()
    out["benchmark_nav"] = out["close"] / out["close"].iloc[0]

    return out


def calculate_report_win_rates(df, initial_position=initial_position):
    if df.empty:
        return {
            "psychological_win_rate": np.nan,
            "bar_win_rate": np.nan,
            "traditional_win_rate": np.nan,
            "num_completed_trades": 0,
        }

    position = df["position"].astype(int).reset_index(drop=True)
    previous_position = position.shift(1).fillna(initial_position).astype(int)
    close = df["close"].reset_index(drop=True)
    strategy_nav = df["strategy_nav"].reset_index(drop=True)
    strategy_return = df["strategy_return"].reset_index(drop=True)

    # 15 分钟胜率：实际持仓后，当前 bar 收盘净值高于上一根 bar 收盘净值。
    held_previous_bar = previous_position != 0
    bar_win_rate = (
        (strategy_return.loc[held_previous_bar] > 0).mean()
        if held_previous_bar.any()
        else np.nan
    )

    # 心理胜率：持仓期间，当前 bar 净值高于最近一次调仓时的净值。
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
        "bar_win_rate": bar_win_rate,
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
            "bar_win_rate": np.nan,
            "traditional_win_rate": np.nan,
            "num_bars": 0,
            "num_long_bars": 0,
            "num_short_bars": 0,
            "num_flat_bars": 0,
            "num_trades": 0,
            "num_completed_trades": 0,
            "final_nav": np.nan,
            "benchmark_final_nav": np.nan,
        }])[SUMMARY_COLUMNS]

    num_bars = len(df)
    final_nav = df["strategy_nav"].iloc[-1]
    annual_return = (
        final_nav ** (annual_days / num_bars) - 1
        if final_nav > 0 and num_bars > 0
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
        # 兼容旧字段：win_rate 指向报告口径中的 15 分钟 bar 胜率。
        "win_rate": win_rates["bar_win_rate"],
        "psychological_win_rate": win_rates["psychological_win_rate"],
        "bar_win_rate": win_rates["bar_win_rate"],
        "traditional_win_rate": win_rates["traditional_win_rate"],
        "num_bars": num_bars,
        "num_long_bars": int((position == 1).sum()),
        "num_short_bars": int((position == -1).sum()),
        "num_flat_bars": int((position == 0).sum()),
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


def plot_nav(df, output_path=None):
    output_path = (
        Path(output_path)
        if output_path is not None
        else (
            PROJECT_ROOT
            / "results"
            / "figures"
            / "backtest"
            / STRATEGY_NAME
            / f"{STRATEGY_NAME}_nav.png"
        )
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
    ax.set_title("Dow Version 2 15m CTA NAV")
    ax.set_xlabel("Date")
    ax.set_ylabel("NAV")
    ax.grid(True, linestyle="--", alpha=0.3)
    ax.legend()
    fig.autofmt_xdate()
    fig.tight_layout()
    fig.savefig(output_path, dpi=150)
    plt.close(fig)

    return output_path


def plot_channels(df, output_path=None, initial_position=initial_position):
    output_path = (
        Path(output_path)
        if output_path is not None
        else (
            PROJECT_ROOT
            / "results"
            / "figures"
            / "backtest"
            / STRATEGY_NAME
            / f"{STRATEGY_NAME}_channels.png"
        )
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
    ax.set_title("Dow Version 2 15m CTA Channels")
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
        out["date"] = pd.to_datetime(out["date"]).dt.strftime(
            OUTPUT_DATETIME_FORMAT
        )
    for column in ["start_date", "end_date"]:
        if column in out.columns:
            out[column] = pd.to_datetime(
                out[column],
                errors="coerce",
            ).dt.strftime(OUTPUT_DATETIME_FORMAT)
    return out


def build_output_paths(output_root, figure_root, symbol=None, symbol_prefix=False):
    stem = STRATEGY_NAME
    if symbol_prefix and symbol:
        stem = f"{symbol}_{stem}"

    output_root = Path(output_root)
    figure_root = Path(figure_root)

    return {
        "daily": output_root / "backtest" / f"{stem}_15m.csv",
        "summary": output_root / "summary" / f"{stem}_summary.csv",
        "nav": figure_root / f"{stem}_nav.png",
        "channels": figure_root / f"{stem}_channels.png",
    }


def build_sweep_output_path(output_root, symbol):
    output_root = Path(output_root)
    symbol_text = symbol.upper() if symbol else "UNKNOWN"
    return output_root / "summary" / f"{symbol_text}_{STRATEGY_NAME}_param_compare.csv"


def save_outputs(daily, summary, output_paths):
    for key in ["daily", "summary"]:
        output_paths[key].parent.mkdir(parents=True, exist_ok=True)

    normalize_dates_for_output(daily[BAR_COLUMNS]).to_csv(
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
    plot_nav(daily, output_paths["nav"])
    plot_channels(
        daily,
        output_paths["channels"],
        initial_position=initial_position_value,
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

        print("Dow Version 2 15m CTA parameter sweep complete.")
        print(f"symbol: {symbol}")
        print(f"M values: {M_values}")
        print(f"x values: {x_values}")
        print(f"comparison file: {output_path}")
        print(f"best annual return params: M={best_M}, x={best_x}")
        print(f"best 15m rows: {len(best_daily)}")
        print(f"best 15m file: {best_output_paths['daily']}")
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

    print("Dow Version 2 15m CTA backtest complete.")
    print(f"symbol: {symbol}")
    print(f"15m rows: {len(daily)}")
    print(f"15m file: {output_paths['daily']}")
    print(f"summary file: {output_paths['summary']}")
    print(f"NAV figure: {output_paths['nav']}")
    print(f"channels figure: {output_paths['channels']}")
    print(summary.to_string(index=False))


if __name__ == "__main__":
    main()
