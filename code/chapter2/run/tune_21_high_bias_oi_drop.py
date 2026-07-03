import argparse
import itertools
from pathlib import Path

import numpy as np
import pandas as pd


RUN_DIR = Path(__file__).resolve().parent
CHAPTER_DIR = RUN_DIR.parent
PROJECT_ROOT = CHAPTER_DIR.parents[1]
CHAPTER_RESULTS_DIR = PROJECT_ROOT / "results" / "chapter2"
DEFAULT_BAR_DIR = CHAPTER_RESULTS_DIR / "tables" / "daily"
DEFAULT_OUTPUT_DIR = CHAPTER_RESULTS_DIR / "parameter_sweep"

FACTOR_ID = "21"
FACTOR_NAME = "high_bias_oi_drop"
TRADING_DAYS_PER_YEAR = 252


def parse_int_list(value):
    return [int(item) for item in value.split(",") if item.strip()]


def parse_float_list(value):
    return [float(item) for item in value.split(",") if item.strip()]


def parse_args():
    parser = argparse.ArgumentParser(
        description="Fast parameter sweep for factor 21 without writing plots."
    )
    parser.add_argument(
        "--bar-dir",
        type=Path,
        default=DEFAULT_BAR_DIR,
        help=f"Input bar directory. Default: {DEFAULT_BAR_DIR}",
    )
    parser.add_argument(
        "--suffix",
        default="_daily.csv",
        help="File suffix used to discover symbols. Default: _daily.csv",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=DEFAULT_OUTPUT_DIR,
        help=f"Sweep output directory. Default: {DEFAULT_OUTPUT_DIR}",
    )
    parser.add_argument("--ma-short-windows", default="5")
    parser.add_argument("--ma-long-windows", default="20")
    parser.add_argument("--ma-trend-windows", default="60")
    parser.add_argument("--ma-bias-thresholds", default="0.02,0.03,0.04")
    parser.add_argument("--ma-long-bias-thresholds", default="0.03,0.05,0.07")
    parser.add_argument("--slope-windows", default="3,4,5")
    parser.add_argument("--volatility-windows", default="10")
    parser.add_argument("--trailing-multipliers", default="2,3,4,5")
    parser.add_argument("--stop-buffers", default="0,0.005,0.01")
    parser.add_argument("--oi-slope-rate-thresholds", default="0,-0.0005")
    parser.add_argument("--close-slope-rate-thresholds", default="0,-0.0005")
    parser.add_argument(
        "--valid-start",
        default="2023-01-01",
        help="Validation period start date. Empty string disables split metrics.",
    )
    parser.add_argument(
        "--bars-per-year",
        type=float,
        default=TRADING_DAYS_PER_YEAR,
        help="Annualization factor for Sharpe. Use a larger value for hourly bars.",
    )
    parser.add_argument(
        "--min-full-trades",
        type=int,
        default=250,
        help="Minimum full-sample trades required in the ranked table.",
    )
    parser.add_argument(
        "--min-valid-trades",
        type=int,
        default=50,
        help="Minimum validation trades required in the ranked table.",
    )
    parser.add_argument(
        "--max-combos",
        type=int,
        default=0,
        help="Optional limit for quick smoke tests. 0 means no limit.",
    )
    return parser.parse_args()


def linear_regression_slope(values):
    values = np.asarray(values, dtype=float)
    if np.isnan(values).any():
        return np.nan

    x = np.arange(len(values), dtype=float)
    x = x - x.mean()
    denominator = np.square(x).sum()
    if denominator == 0:
        return np.nan

    return np.dot(x, values - values.mean()) / denominator


def annualized_sharpe(returns, bars_per_year):
    returns = pd.Series(returns).fillna(0)
    std = returns.std(ddof=1)
    if pd.isna(std) or std == 0:
        return np.nan
    return returns.mean() / std * np.sqrt(bars_per_year)


def safe_ratio(numerator, denominator):
    if denominator == 0:
        return np.nan
    return numerator / denominator


def discover_bar_files(bar_dir, suffix):
    if not bar_dir.is_dir():
        raise FileNotFoundError(f"Input bar directory does not exist: {bar_dir}")

    files = sorted(path for path in bar_dir.glob(f"*{suffix}") if path.is_file())
    if not files:
        raise FileNotFoundError(f"No *{suffix} files found in {bar_dir}")
    return files


def symbol_from_path(path, suffix):
    return path.name[: -len(suffix)].upper()


def load_bar_file(path):
    frame = pd.read_csv(path)
    required_columns = {
        "date",
        "open",
        "close",
        "high",
        "low",
        "volume",
        "total_turnover",
        "open_interest",
    }
    missing = required_columns - set(frame.columns)
    if missing:
        raise ValueError(f"{path} missing columns: {','.join(sorted(missing))}")

    frame["date"] = pd.to_datetime(frame["date"])
    numeric_columns = [
        "open",
        "close",
        "high",
        "low",
        "volume",
        "total_turnover",
        "open_interest",
    ]
    for column in numeric_columns:
        frame[column] = pd.to_numeric(frame[column], errors="coerce")
    return frame.sort_values("date").reset_index(drop=True)


def precompute_symbol(path, suffix, params):
    symbol = symbol_from_path(path, suffix)
    base = load_bar_file(path)

    ma_windows = sorted(
        set(params["ma_short_windows"])
        | set(params["ma_long_windows"])
        | set(params["ma_trend_windows"])
    )
    slope_windows = sorted(set(params["slope_windows"]))
    volatility_windows = sorted(set(params["volatility_windows"]))

    features = {
        "symbol": symbol,
        "date": base["date"].to_numpy(),
        "open": base["open"].to_numpy(dtype=float),
        "close": base["close"].to_numpy(dtype=float),
        "high": base["high"].to_numpy(dtype=float),
        "low": base["low"].to_numpy(dtype=float),
        "ma": {},
        "oi_slope_rate": {},
        "close_slope_rate": {},
        "avg_volatility_rate": {},
    }

    close = base["close"]
    open_interest = base["open_interest"]
    price_range_rate = (
        (base["high"] - base["low"]) / base["close"].replace(0, np.nan)
    )

    for window in ma_windows:
        features["ma"][window] = (
            close.rolling(window=window, min_periods=window)
            .mean()
            .to_numpy(dtype=float)
        )

    for window in slope_windows:
        oi_slope = (
            open_interest.rolling(window=window, min_periods=window)
            .apply(linear_regression_slope, raw=True)
        )
        oi_mean = open_interest.rolling(window=window, min_periods=window).mean()
        close_slope = (
            close.rolling(window=window, min_periods=window)
            .apply(linear_regression_slope, raw=True)
        )
        close_mean = close.rolling(window=window, min_periods=window).mean()
        features["oi_slope_rate"][window] = (
            oi_slope / oi_mean.replace(0, np.nan)
        ).to_numpy(dtype=float)
        features["close_slope_rate"][window] = (
            close_slope / close_mean.replace(0, np.nan)
        ).to_numpy(dtype=float)

    for window in volatility_windows:
        features["avg_volatility_rate"][window] = (
            price_range_rate.shift(1)
            .rolling(window=window, min_periods=window)
            .mean()
            .to_numpy(dtype=float)
        )

    return features


def build_open_signal(features, combo):
    close = features["close"]
    ma_short = features["ma"][combo["ma_short_window"]]
    ma_long = features["ma"][combo["ma_long_window"]]
    ma_trend = features["ma"][combo["ma_trend_window"]]

    close_ma_short_bias = close / ma_short - 1
    close_ma_long_bias = close / ma_long - 1
    close_ma_trend_bias = close / ma_trend - 1
    ma_bias_spread = close_ma_long_bias - close_ma_short_bias
    ma_long_bias_spread = close_ma_trend_bias - close_ma_long_bias

    oi_slope_rate = features["oi_slope_rate"][combo["slope_window"]]
    close_slope_rate = features["close_slope_rate"][combo["slope_window"]]

    return (
        (ma_bias_spread >= combo["ma_bias_threshold"])
        & (ma_long_bias_spread >= combo["ma_long_bias_threshold"])
        & (oi_slope_rate <= combo["oi_slope_rate_threshold"])
        & (close_slope_rate <= combo["close_slope_rate_threshold"])
    )


def simulate_short_state(features, open_signal, combo):
    open_price = features["open"]
    close = features["close"]
    low = features["low"]
    avg_volatility_rate = features["avg_volatility_rate"][
        combo["volatility_window"]
    ]
    n = len(close)

    position = np.zeros(n, dtype=np.int8)
    trade_signal = np.zeros(n, dtype=np.int8)
    entry_price_series = np.full(n, np.nan)
    exit_price_series = np.full(n, np.nan)
    exit_reason_series = np.full(n, "", dtype=object)

    current_position = 0
    entry_price = np.nan
    low_since_entry = np.nan
    pending_cover = False
    pending_exit_reason = ""

    for index in range(n):
        actual_open = bool(open_signal[index - 1]) if index > 0 else False
        actual_cover = pending_cover
        actual_exit_reason = pending_exit_reason
        pending_cover = False
        pending_exit_reason = ""

        opened_today = False
        if current_position == -1 and actual_cover and not np.isnan(open_price[index]):
            trade_signal[index] = 1
            exit_price_series[index] = open_price[index]
            entry_price_series[index] = entry_price
            exit_reason_series[index] = actual_exit_reason or "cover_short"
            current_position = 0
            entry_price = np.nan
            low_since_entry = np.nan
        elif current_position == 0 and actual_open and not np.isnan(open_price[index]):
            trade_signal[index] = -1
            entry_price = open_price[index]
            low_since_entry = open_price[index]
            entry_price_series[index] = entry_price
            current_position = -1
            opened_today = True

        if current_position == -1:
            low_candidate = low[index]
            if np.isnan(low_candidate):
                low_candidate = open_price[index] if opened_today else close[index]
            if not np.isnan(low_candidate):
                low_since_entry = min(low_since_entry, low_candidate)
            entry_price_series[index] = entry_price

            price_above_entry = (
                not np.isnan(close[index])
                and not np.isnan(entry_price)
                and close[index] > entry_price * (1 + combo["stop_buffer"])
            )
            trailing_rebound = False
            if (
                not np.isnan(close[index])
                and not np.isnan(low_since_entry)
                and not np.isnan(avg_volatility_rate[index])
            ):
                trailing_stop_price = low_since_entry * (
                    1
                    + avg_volatility_rate[index]
                    * combo["trailing_multiplier"]
                )
                trailing_rebound = close[index] > trailing_stop_price

            pending_cover = bool(price_above_entry or trailing_rebound)
            if price_above_entry and trailing_rebound:
                pending_exit_reason = "price_above_entry_and_trailing_rebound"
            elif price_above_entry:
                pending_exit_reason = "price_above_entry"
            elif trailing_rebound:
                pending_exit_reason = "trailing_rebound"

        position[index] = current_position

    return position, trade_signal, entry_price_series, exit_price_series, exit_reason_series


def evaluate_symbol(features, combo):
    open_signal = build_open_signal(features, combo)
    (
        position,
        trade_signal,
        entry_price_series,
        exit_price_series,
        exit_reason_series,
    ) = simulate_short_state(features, open_signal, combo)

    open_price = features["open"]
    close = features["close"]
    previous_close = np.roll(close, 1)
    previous_close[0] = np.nan
    previous_position = np.roll(position, 1)
    previous_position[0] = 0

    benchmark_return = np.divide(
        close,
        previous_close,
        out=np.full_like(close, np.nan, dtype=float),
        where=previous_close != 0,
    ) - 1
    gap_return = np.divide(
        open_price,
        previous_close,
        out=np.full_like(close, np.nan, dtype=float),
        where=previous_close != 0,
    ) - 1
    intrabar_return = np.divide(
        close,
        open_price,
        out=np.full_like(close, np.nan, dtype=float),
        where=open_price != 0,
    ) - 1

    benchmark_return = np.nan_to_num(benchmark_return, nan=0.0)
    gap_return = np.nan_to_num(gap_return, nan=0.0)
    intrabar_return = np.nan_to_num(intrabar_return, nan=0.0)
    overlay_return = 2 * (
        previous_position * gap_return + position * intrabar_return
    )
    simple_return = benchmark_return + overlay_return

    daily = pd.DataFrame(
        {
            "symbol": features["symbol"],
            "date": features["date"],
            "benchmark_return": benchmark_return,
            "overlay_return": overlay_return,
            "simple_return": simple_return,
            "position": position,
            "trade_signal": trade_signal,
        }
    )

    trades = build_trade_rows(
        symbol=features["symbol"],
        dates=features["date"],
        trade_signal=trade_signal,
        entry_price_series=entry_price_series,
        exit_price_series=exit_price_series,
        exit_reason_series=exit_reason_series,
        benchmark_return=benchmark_return,
        overlay_return=overlay_return,
        simple_return=simple_return,
        close=close,
    )
    return daily, trades


def build_trade_rows(
    symbol,
    dates,
    trade_signal,
    entry_price_series,
    exit_price_series,
    exit_reason_series,
    benchmark_return,
    overlay_return,
    simple_return,
    close,
):
    rows = []
    open_index = None
    for index, signal in enumerate(trade_signal):
        if signal == -1 and open_index is None:
            open_index = index
            continue

        if signal == 1 and open_index is not None:
            rows.append(
                build_trade_row(
                    symbol,
                    dates,
                    open_index,
                    index,
                    "closed",
                    entry_price_series,
                    exit_price_series,
                    exit_reason_series,
                    benchmark_return,
                    overlay_return,
                    simple_return,
                    close,
                )
            )
            open_index = None

    if open_index is not None:
        rows.append(
            build_trade_row(
                symbol,
                dates,
                open_index,
                len(dates) - 1,
                "open",
                entry_price_series,
                exit_price_series,
                exit_reason_series,
                benchmark_return,
                overlay_return,
                simple_return,
                close,
            )
        )
    return rows


def build_trade_row(
    symbol,
    dates,
    open_index,
    exit_index,
    status,
    entry_price_series,
    exit_price_series,
    exit_reason_series,
    benchmark_return,
    overlay_return,
    simple_return,
    close,
):
    entry_price = entry_price_series[open_index]
    exit_price = (
        close[exit_index] if status == "open" else exit_price_series[exit_index]
    )
    price_return = np.nan
    if not np.isnan(entry_price) and not np.isnan(exit_price) and exit_price != 0:
        price_return = entry_price / exit_price - 1
    trade_slice = slice(open_index, exit_index + 1)
    return {
        "symbol": symbol,
        "status": status,
        "entry_date": dates[open_index],
        "exit_date": dates[exit_index],
        "holding_bars": max(exit_index - open_index, 0),
        "trade_benchmark_return": benchmark_return[trade_slice].sum(),
        "trade_overlay_return": overlay_return[trade_slice].sum(),
        "trade_simple_return": simple_return[trade_slice].sum(),
        "trade_price_return": price_return,
        "exit_reason": (
            "open_position"
            if status == "open"
            else exit_reason_series[exit_index]
        ),
    }


def summarize_returns(portfolio, trades, prefix, bars_per_year):
    if portfolio.empty:
        return {
            f"{prefix}_bars": 0,
            f"{prefix}_overlay_total_return": np.nan,
            f"{prefix}_overlay_sharpe": np.nan,
            f"{prefix}_simple_total_return": np.nan,
            f"{prefix}_benchmark_total_return": np.nan,
            f"{prefix}_trade_count": 0,
            f"{prefix}_trade_win_rate": np.nan,
            f"{prefix}_mean_trade_overlay_return": np.nan,
            f"{prefix}_mean_holding_bars": np.nan,
        }

    period_trades = trades.copy()
    closed_trades = period_trades[period_trades["status"] == "closed"]
    return {
        f"{prefix}_bars": len(portfolio),
        f"{prefix}_overlay_total_return": portfolio["overlay_return"].sum(),
        f"{prefix}_overlay_sharpe": annualized_sharpe(
            portfolio["overlay_return"],
            bars_per_year,
        ),
        f"{prefix}_simple_total_return": portfolio["simple_return"].sum(),
        f"{prefix}_benchmark_total_return": portfolio["benchmark_return"].sum(),
        f"{prefix}_trade_count": len(period_trades),
        f"{prefix}_closed_trade_count": len(closed_trades),
        f"{prefix}_trade_win_rate": safe_ratio(
            int((closed_trades["trade_overlay_return"] > 0).sum()),
            len(closed_trades),
        ),
        f"{prefix}_mean_trade_overlay_return": closed_trades[
            "trade_overlay_return"
        ].mean(),
        f"{prefix}_mean_holding_bars": closed_trades["holding_bars"].mean(),
    }


def evaluate_combo(symbol_features, combo, valid_start, bars_per_year):
    all_daily = []
    all_trades = []
    symbol_rows = []

    for features in symbol_features:
        daily, trades = evaluate_symbol(features, combo)
        all_daily.append(daily)
        all_trades.extend(trades)
        symbol_rows.append(
            {
                "symbol": features["symbol"],
                "overlay_total_return": daily["overlay_return"].sum(),
                "entry_count": int((daily["trade_signal"] == -1).sum()),
            }
        )

    merged = pd.concat(all_daily, ignore_index=True)
    portfolio = (
        merged.groupby("date", sort=True)
        .agg(
            symbol_count=("symbol", "nunique"),
            holding_symbol_count=("position", lambda values: int((values != 0).sum())),
            benchmark_return=("benchmark_return", "mean"),
            overlay_return=("overlay_return", "mean"),
            simple_return=("simple_return", "mean"),
        )
        .reset_index()
    )
    trades = pd.DataFrame(all_trades)
    if trades.empty:
        trades = pd.DataFrame(
            columns=[
                "symbol",
                "status",
                "entry_date",
                "exit_date",
                "holding_bars",
                "trade_benchmark_return",
                "trade_overlay_return",
                "trade_simple_return",
                "trade_price_return",
                "exit_reason",
            ]
        )
    trades["entry_date"] = pd.to_datetime(trades["entry_date"])
    trades["exit_date"] = pd.to_datetime(trades["exit_date"])

    row = dict(combo)
    row.update(summarize_returns(portfolio, trades, "full", bars_per_year))
    symbol_metrics = pd.DataFrame(symbol_rows)
    row["full_positive_symbol_count"] = int(
        (symbol_metrics["overlay_total_return"] > 0).sum()
    )
    row["full_symbol_count"] = len(symbol_metrics)
    row["full_entry_count"] = int(symbol_metrics["entry_count"].sum())
    row["full_mean_holding_symbol_count"] = portfolio[
        "holding_symbol_count"
    ].mean()

    if valid_start is not None:
        valid_mask = portfolio["date"] >= valid_start
        train_mask = portfolio["date"] < valid_start
        valid_trades = trades[trades["entry_date"] >= valid_start]
        train_trades = trades[trades["entry_date"] < valid_start]
        row.update(
            summarize_returns(
                portfolio.loc[train_mask],
                train_trades,
                "train",
                bars_per_year,
            )
        )
        row.update(
            summarize_returns(
                portfolio.loc[valid_mask],
                valid_trades,
                "valid",
                bars_per_year,
            )
        )

    return row


def build_param_grid(args):
    params = {
        "ma_short_windows": parse_int_list(args.ma_short_windows),
        "ma_long_windows": parse_int_list(args.ma_long_windows),
        "ma_trend_windows": parse_int_list(args.ma_trend_windows),
        "ma_bias_thresholds": parse_float_list(args.ma_bias_thresholds),
        "ma_long_bias_thresholds": parse_float_list(args.ma_long_bias_thresholds),
        "slope_windows": parse_int_list(args.slope_windows),
        "volatility_windows": parse_int_list(args.volatility_windows),
        "trailing_multipliers": parse_float_list(args.trailing_multipliers),
        "stop_buffers": parse_float_list(args.stop_buffers),
        "oi_slope_rate_thresholds": parse_float_list(
            args.oi_slope_rate_thresholds
        ),
        "close_slope_rate_thresholds": parse_float_list(
            args.close_slope_rate_thresholds
        ),
    }

    grid = itertools.product(
        params["ma_short_windows"],
        params["ma_long_windows"],
        params["ma_trend_windows"],
        params["ma_bias_thresholds"],
        params["ma_long_bias_thresholds"],
        params["slope_windows"],
        params["volatility_windows"],
        params["trailing_multipliers"],
        params["stop_buffers"],
        params["oi_slope_rate_thresholds"],
        params["close_slope_rate_thresholds"],
    )
    combos = []
    for values in grid:
        combo = {
            "ma_short_window": values[0],
            "ma_long_window": values[1],
            "ma_trend_window": values[2],
            "ma_bias_threshold": values[3],
            "ma_long_bias_threshold": values[4],
            "slope_window": values[5],
            "volatility_window": values[6],
            "trailing_multiplier": values[7],
            "stop_buffer": values[8],
            "oi_slope_rate_threshold": values[9],
            "close_slope_rate_threshold": values[10],
        }
        if (
            combo["ma_short_window"] < combo["ma_long_window"]
            and combo["ma_long_window"] < combo["ma_trend_window"]
        ):
            combos.append(combo)

    if args.max_combos and args.max_combos > 0:
        combos = combos[: args.max_combos]
    return params, combos


def main():
    args = parse_args()
    params, combos = build_param_grid(args)
    valid_start = pd.Timestamp(args.valid_start) if args.valid_start else None

    bar_files = discover_bar_files(args.bar_dir.resolve(), args.suffix)
    print(f"Input bar directory: {args.bar_dir.resolve()}", flush=True)
    print(f"Symbols: {len(bar_files)}", flush=True)
    print(f"Parameter combinations: {len(combos)}", flush=True)

    symbol_features = [
        precompute_symbol(path, args.suffix, params)
        for path in bar_files
    ]

    rows = []
    for index, combo in enumerate(combos, start=1):
        rows.append(
            evaluate_combo(
                symbol_features=symbol_features,
                combo=combo,
                valid_start=valid_start,
                bars_per_year=args.bars_per_year,
            )
        )
        if index == 1 or index % 50 == 0 or index == len(combos):
            print(f"Evaluated {index}/{len(combos)}", flush=True)

    results = pd.DataFrame(rows)
    results["factor_id"] = FACTOR_ID
    results["factor_name"] = FACTOR_NAME

    ranked = results.copy()
    if "valid_trade_count" in ranked.columns:
        ranked = ranked[
            (ranked["full_trade_count"] >= args.min_full_trades)
            & (ranked["valid_trade_count"] >= args.min_valid_trades)
        ].copy()
        ranked["rank_score"] = (
            ranked["valid_overlay_sharpe"].fillna(-999)
            + 0.35 * ranked["full_overlay_sharpe"].fillna(-999)
            + 0.02 * ranked["valid_overlay_total_return"].fillna(-999)
            + 0.01 * ranked["full_overlay_total_return"].fillna(-999)
            + 0.005 * ranked["full_positive_symbol_count"].fillna(0)
        )
    else:
        ranked = ranked[ranked["full_trade_count"] >= args.min_full_trades].copy()
        ranked["rank_score"] = (
            ranked["full_overlay_sharpe"].fillna(-999)
            + 0.01 * ranked["full_overlay_total_return"].fillna(-999)
            + 0.005 * ranked["full_positive_symbol_count"].fillna(0)
        )
    ranked = ranked.sort_values("rank_score", ascending=False)

    args.output_dir.mkdir(parents=True, exist_ok=True)
    suffix_label = args.suffix.replace(".", "_").replace("*", "").strip("_")
    all_path = args.output_dir / f"{FACTOR_ID}_{FACTOR_NAME}_{suffix_label}_all.csv"
    ranked_path = (
        args.output_dir / f"{FACTOR_ID}_{FACTOR_NAME}_{suffix_label}_ranked.csv"
    )
    results.to_csv(all_path, index=False)
    ranked.to_csv(ranked_path, index=False)

    print(f"All sweep results: {all_path}", flush=True)
    print(f"Ranked sweep results: {ranked_path}", flush=True)
    if ranked.empty:
        print("No ranked rows after trade-count filters.", flush=True)
        return

    display_columns = [
        "rank_score",
        "full_overlay_total_return",
        "full_overlay_sharpe",
        "valid_overlay_total_return",
        "valid_overlay_sharpe",
        "full_trade_count",
        "valid_trade_count",
        "full_positive_symbol_count",
        "ma_bias_threshold",
        "ma_long_bias_threshold",
        "slope_window",
        "trailing_multiplier",
        "stop_buffer",
        "oi_slope_rate_threshold",
        "close_slope_rate_threshold",
    ]
    display_columns = [col for col in display_columns if col in ranked.columns]
    print(ranked[display_columns].head(20).to_string(index=False), flush=True)


if __name__ == "__main__":
    main()
