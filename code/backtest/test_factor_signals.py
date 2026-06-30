import argparse
import os
import re
from pathlib import Path


BACKTEST_DIR = Path(__file__).resolve().parent
CODE_DIR = BACKTEST_DIR.parent
PROJECT_ROOT = CODE_DIR.parent
RESULTS_DIR = PROJECT_ROOT / "results"
DEFAULT_RUNS_DIR = RESULTS_DIR
DEFAULT_OUTPUT_DIR = RESULTS_DIR / "test"
DEFAULT_FACTOR_IDS = ("11", "12", "13", "14")
DEFAULT_FORWARD_DAYS = (3, 5, 10)
DEFAULT_VOL_WINDOW = 20
DEFAULT_MIN_VOL_PERIODS = DEFAULT_VOL_WINDOW
DEFAULT_VOLATILITY_METHOD = "abs_return"
SKIPPED_RESULT_DIRS = {"combined", "test"}
FACTOR_FILE_PATTERN = re.compile(r"^(.+?)_(\d+)_(.+)\.csv$")

os.environ.setdefault("MPLCONFIGDIR", str(PROJECT_ROOT / ".matplotlib"))
os.environ.setdefault("XDG_CACHE_HOME", str(PROJECT_ROOT / ".cache"))
(PROJECT_ROOT / ".matplotlib").mkdir(exist_ok=True)
(PROJECT_ROOT / ".cache").mkdir(exist_ok=True)

import matplotlib
import numpy as np
import pandas as pd


matplotlib.use("Agg")
import matplotlib.dates as mdates
import matplotlib.pyplot as plt


def parse_symbols(raw_value):
    if raw_value is None:
        return None

    symbols = [
        item.strip().upper()
        for item in raw_value.split(",")
        if item.strip()
    ]
    return set(symbols) or None


def parse_factor_ids(raw_value):
    if raw_value is None:
        return set(DEFAULT_FACTOR_IDS)
    if raw_value.strip().upper() == "ALL":
        return None

    factor_ids = {
        item.strip()
        for item in raw_value.split(",")
        if item.strip()
    }
    return factor_ids or set(DEFAULT_FACTOR_IDS)


def parse_forward_days(raw_value):
    if raw_value is None:
        values = list(DEFAULT_FORWARD_DAYS)
    else:
        values = []
        for item in raw_value.split(","):
            item = item.strip()
            if not item:
                continue
            try:
                values.append(int(item))
            except ValueError as exc:
                raise ValueError(
                    f"--forward-days must be comma-separated integers: {raw_value}"
                ) from exc

    forward_days = []
    for value in values:
        if value < 1:
            raise ValueError("--forward-days values must be at least 1")
        if value not in forward_days:
            forward_days.append(value)

    if not forward_days:
        raise ValueError("--forward-days must include at least one value")

    return tuple(forward_days)


def parse_factor_file(path):
    match = FACTOR_FILE_PATTERN.match(path.name)
    if match is None:
        return None

    return {
        "symbol": match.group(1).upper(),
        "factor_id": match.group(2),
        "factor_name": match.group(3),
    }


def discover_factor_files(runs_dir, symbol_filter=None, factor_id_filter=None):
    factor_files = []

    for run_dir in sorted(runs_dir.iterdir()):
        if not run_dir.is_dir():
            continue
        if run_dir.name in SKIPPED_RESULT_DIRS:
            continue

        factors_dir = run_dir / "tables" / "factors"
        if not factors_dir.is_dir():
            continue

        for path in sorted(factors_dir.glob("*.csv")):
            metadata = parse_factor_file(path)
            if metadata is None:
                continue
            if symbol_filter is not None and metadata["symbol"] not in symbol_filter:
                continue
            if (
                factor_id_filter is not None
                and metadata["factor_id"] not in factor_id_filter
            ):
                continue

            factor_files.append({
                **metadata,
                "run_dir": run_dir.name,
                "path": path,
            })

    if not factor_files:
        raise FileNotFoundError(f"no factor csv found under {runs_dir}")

    return factor_files


def numeric_series(frame, column):
    return pd.to_numeric(frame[column], errors="coerce")


def compute_rolling_volatility(
    daily,
    method,
    window,
    min_periods,
    include_signal_day,
):
    close = daily["close"]
    close_return = close.pct_change()
    daily["daily_abs_return"] = close_return.abs()

    if method == "abs_return":
        daily["volatility_source"] = daily["daily_abs_return"]
        rolling = daily["volatility_source"].rolling(
            window=window,
            min_periods=min_periods,
        ).mean()
    elif method == "range":
        missing_columns = {"high", "low"} - set(daily.columns)
        if missing_columns:
            raise ValueError(
                "range volatility requires columns: "
                f"{','.join(sorted(missing_columns))}"
            )
        denominator = close.abs().replace(0, np.nan)
        daily["daily_range_return"] = (
            (daily["high"] - daily["low"]).abs() / denominator
        )
        daily["volatility_source"] = daily["daily_range_return"]
        rolling = daily["volatility_source"].rolling(
            window=window,
            min_periods=min_periods,
        ).mean()
    elif method == "close_to_close_std":
        daily["volatility_source"] = close_return
        rolling = daily["volatility_source"].rolling(
            window=window,
            min_periods=min_periods,
        ).std()
    else:
        raise ValueError(f"unsupported volatility method: {method}")

    if not include_signal_day:
        rolling = rolling.shift(1)

    daily["rolling_volatility"] = rolling
    daily["volatility_threshold"] = rolling
    return daily


def load_factor_daily(
    file_info,
    test_price_column,
    volatility_method,
    vol_window,
    min_vol_periods,
    include_signal_day_volatility,
):
    columns = {
        "date",
        "open",
        "close",
        "high",
        "low",
        "factor_id",
        "factor_name",
        "factor_value",
        "signal",
        test_price_column,
    }
    daily = pd.read_csv(
        file_info["path"],
        usecols=lambda col: col in columns,
    )

    missing_columns = {"date", "close", "signal", test_price_column} - set(
        daily.columns
    )
    if missing_columns:
        raise ValueError(
            f"{file_info['path']} missing columns: "
            f"{','.join(sorted(missing_columns))}"
        )

    daily["date"] = pd.to_datetime(daily["date"])
    for column in ["open", "close", "high", "low", test_price_column]:
        if column in daily.columns:
            daily[column] = numeric_series(daily, column)

    daily["signal"] = (
        numeric_series(daily, "signal")
        .fillna(0)
        .astype(int)
    )

    if "factor_value" in daily.columns:
        daily["factor_value"] = numeric_series(daily, "factor_value")
    else:
        daily["factor_value"] = np.nan

    if "factor_id" in daily.columns and daily["factor_id"].notna().any():
        factor_id = str(daily["factor_id"].dropna().iloc[0])
    else:
        factor_id = file_info["factor_id"]

    if "factor_name" in daily.columns and daily["factor_name"].notna().any():
        factor_name = str(daily["factor_name"].dropna().iloc[0])
    else:
        factor_name = file_info["factor_name"]

    daily = daily.sort_values("date").reset_index(drop=True)
    daily = compute_rolling_volatility(
        daily=daily,
        method=volatility_method,
        window=vol_window,
        min_periods=min_vol_periods,
        include_signal_day=include_signal_day_volatility,
    )
    daily["symbol"] = file_info["symbol"]
    daily["factor_id"] = factor_id
    daily["factor_name"] = factor_name
    daily["run_dir"] = file_info["run_dir"]
    daily["factor_label"] = f"{factor_id}_{factor_name}"
    daily["volatility_method"] = volatility_method
    daily["volatility_window"] = vol_window
    daily["volatility_min_periods"] = min_vol_periods
    daily["volatility_includes_signal_day"] = include_signal_day_volatility

    return daily


def load_all_factor_data(
    factor_files,
    test_price_column,
    volatility_method,
    vol_window,
    min_vol_periods,
    include_signal_day_volatility,
):
    frames = []
    errors = []

    for file_info in factor_files:
        try:
            frames.append(
                load_factor_daily(
                    file_info=file_info,
                    test_price_column=test_price_column,
                    volatility_method=volatility_method,
                    vol_window=vol_window,
                    min_vol_periods=min_vol_periods,
                    include_signal_day_volatility=include_signal_day_volatility,
                )
            )
        except Exception as exc:
            errors.append((file_info["path"], exc))

    if not frames:
        error_text = "\n".join(f"- {path}: {exc}" for path, exc in errors)
        raise RuntimeError(f"no factor data could be loaded\n{error_text}")

    return frames, errors


def empty_events_table(forward_days):
    columns = [
        "run_dir",
        "symbol",
        "factor_id",
        "factor_name",
        "factor_label",
        "signal_date",
        "signal_close",
        "factor_value",
        "test_price_column",
        "forward_days",
        "volatility_method",
        "volatility_window",
        "volatility_min_periods",
        "volatility_includes_signal_day",
        "volatility_threshold",
        "required_drop_pct",
        "required_drop_price",
        "target_price",
        "available_forward_days",
        "has_full_forward_window",
        "is_tested",
        "correct",
        "unknown_reason",
        "hit_day",
        "hit_date",
        "hit_price",
        "hit_return",
        "worst_forward_return",
        "worst_forward_date",
        "worst_forward_price",
    ]
    for step in range(1, forward_days + 1):
        columns.extend([
            f"future_date_t{step}",
            f"future_price_t{step}",
            f"return_t{step}",
        ])
    return pd.DataFrame(columns=columns)


def evaluate_frame(daily, forward_days, test_price_column):
    records = []
    signal_points = daily.index[daily["signal"] == 1].tolist()

    for index in signal_points:
        row = daily.loc[index]
        signal_close = row["close"]
        threshold = row["volatility_threshold"]
        future = daily.iloc[index + 1:index + 1 + forward_days]
        threshold_valid = (
            pd.notna(threshold)
            and threshold >= 0
            and pd.notna(signal_close)
            and signal_close != 0
        )

        required_drop_price = (
            abs(signal_close) * threshold
            if threshold_valid
            else np.nan
        )
        target_price = (
            signal_close * (1 - threshold)
            if threshold_valid
            else np.nan
        )

        record = {
            "run_dir": row["run_dir"],
            "symbol": row["symbol"],
            "factor_id": row["factor_id"],
            "factor_name": row["factor_name"],
            "factor_label": row["factor_label"],
            "signal_date": row["date"],
            "signal_close": signal_close,
            "factor_value": row["factor_value"],
            "test_price_column": test_price_column,
            "forward_days": forward_days,
            "volatility_method": row["volatility_method"],
            "volatility_window": row["volatility_window"],
            "volatility_min_periods": row["volatility_min_periods"],
            "volatility_includes_signal_day": row[
                "volatility_includes_signal_day"
            ],
            "volatility_threshold": threshold,
            "required_drop_pct": threshold,
            "required_drop_price": required_drop_price,
            "target_price": target_price,
            "available_forward_days": int(len(future)),
            "has_full_forward_window": bool(len(future) >= forward_days),
        }

        returns = []
        for step in range(1, forward_days + 1):
            future_key = step - 1
            if future_key < len(future):
                future_row = future.iloc[future_key]
                future_price = future_row[test_price_column]
                if (
                    pd.notna(signal_close)
                    and signal_close != 0
                    and pd.notna(future_price)
                ):
                    forward_return = future_price / signal_close - 1
                else:
                    forward_return = np.nan

                record[f"future_date_t{step}"] = future_row["date"]
                record[f"future_price_t{step}"] = future_price
                record[f"return_t{step}"] = forward_return
                if pd.notna(forward_return):
                    returns.append(
                        (step, future_row["date"], future_price, forward_return)
                    )
            else:
                record[f"future_date_t{step}"] = pd.NaT
                record[f"future_price_t{step}"] = np.nan
                record[f"return_t{step}"] = np.nan

        hits = []
        if threshold_valid:
            hits = [
                (step, date_value, price_value, return_value)
                for step, date_value, price_value, return_value in returns
                if return_value <= -threshold
            ]
        worst = min(returns, key=lambda item: item[3]) if returns else None
        hit = hits[0] if hits else None

        correct = bool(hit is not None)
        is_tested = bool(
            threshold_valid
            and (correct or record["has_full_forward_window"])
        )
        unknown_reason = ""
        if not is_tested:
            if not threshold_valid:
                unknown_reason = "missing_volatility_threshold_or_signal_price"
            elif not record["has_full_forward_window"]:
                unknown_reason = "incomplete_forward_window"

        record["is_tested"] = is_tested
        record["correct"] = correct
        record["unknown_reason"] = unknown_reason
        record["hit_day"] = int(hit[0]) if hit is not None else pd.NA
        record["hit_date"] = hit[1] if hit is not None else pd.NaT
        record["hit_price"] = hit[2] if hit is not None else np.nan
        record["hit_return"] = hit[3] if hit is not None else np.nan
        record["worst_forward_return"] = (
            worst[3] if worst is not None else np.nan
        )
        record["worst_forward_date"] = (
            worst[1] if worst is not None else pd.NaT
        )
        record["worst_forward_price"] = (
            worst[2] if worst is not None else np.nan
        )
        records.append(record)

    if not records:
        return empty_events_table(forward_days)

    return pd.DataFrame(records)


def evaluate_all_frames(frames, forward_days_list, test_price_column):
    event_frames = []
    for forward_days in forward_days_list:
        event_frames.extend(
            evaluate_frame(
                daily=frame,
                forward_days=forward_days,
                test_price_column=test_price_column,
            )
            for frame in frames
        )

    event_frames = [frame for frame in event_frames if not frame.empty]
    if not event_frames:
        return empty_events_table(max(forward_days_list))

    events = pd.concat(event_frames, ignore_index=True)
    for column in ["correct", "is_tested", "has_full_forward_window"]:
        events[column] = events[column].fillna(False).astype(bool)
    return events


def summarize_events(events, group_columns):
    base_columns = list(group_columns) + [
        "total_signals",
        "tested_signals",
        "correct_signals",
        "incorrect_signals",
        "unknown_signals",
        "win_rate",
        "first_signal_date",
        "last_signal_date",
        "mean_volatility_threshold",
        "median_volatility_threshold",
        "mean_worst_forward_return",
        "min_worst_forward_return",
        "mean_factor_value",
    ]

    if events.empty:
        return pd.DataFrame(columns=base_columns)

    working = events.copy()
    working["tested_int"] = working["is_tested"].astype(int)
    working["correct_int"] = working["correct"].astype(int)
    working["incorrect_int"] = (
        working["is_tested"] & ~working["correct"]
    ).astype(int)
    working["unknown_int"] = (~working["is_tested"]).astype(int)

    summary = (
        working.groupby(list(group_columns), dropna=False)
        .agg(
            total_signals=("signal_date", "size"),
            tested_signals=("tested_int", "sum"),
            correct_signals=("correct_int", "sum"),
            incorrect_signals=("incorrect_int", "sum"),
            unknown_signals=("unknown_int", "sum"),
            first_signal_date=("signal_date", "min"),
            last_signal_date=("signal_date", "max"),
            mean_volatility_threshold=("volatility_threshold", "mean"),
            median_volatility_threshold=("volatility_threshold", "median"),
            mean_worst_forward_return=("worst_forward_return", "mean"),
            min_worst_forward_return=("worst_forward_return", "min"),
            mean_factor_value=("factor_value", "mean"),
        )
        .reset_index()
    )
    summary["win_rate"] = np.where(
        summary["tested_signals"] > 0,
        summary["correct_signals"] / summary["tested_signals"],
        np.nan,
    )
    return summary[base_columns]


def summarize_overall(events):
    return summarize_events(events, ["forward_days"])


def build_symbol_price_frames(frames):
    price_frames = {}

    for frame in frames:
        symbol = frame["symbol"].iloc[0]
        candidate_columns = [
            "date",
            "close",
            "rolling_volatility",
            "volatility_threshold",
        ]
        candidate = (
            frame[candidate_columns]
            .dropna(subset=["date", "close"])
            .drop_duplicates(subset=["date"])
            .sort_values("date")
            .reset_index(drop=True)
        )
        if symbol not in price_frames or len(candidate) > len(price_frames[symbol]):
            price_frames[symbol] = candidate

    return price_frames


def plot_symbol_test(
    symbol,
    price_frame,
    events,
    figures_dir,
    forward_days,
    volatility_method,
    vol_window,
    test_price_column,
    dpi,
):
    symbol_events = events[events["symbol"] == symbol].copy()
    tested = symbol_events["is_tested"].fillna(False).astype(bool)
    correct = symbol_events["correct"].fillna(False).astype(bool)
    correct_events = symbol_events[tested & correct]
    incorrect_events = symbol_events[tested & ~correct]
    unknown_events = symbol_events[~tested]

    plt.figure(figsize=(12, 6))
    plt.plot(
        price_frame["date"],
        price_frame["close"],
        color="#1f2937",
        linewidth=1.2,
        label="close",
    )

    marker_specs = [
        (correct_events, "#2a9d8f", "v", "valid"),
        (incorrect_events, "#d62828", "x", "invalid"),
        (unknown_events, "#6b7280", "o", "unknown"),
    ]
    for marker_events, color, marker, label in marker_specs:
        if marker_events.empty:
            continue
        plt.scatter(
            marker_events["signal_date"],
            marker_events["signal_close"],
            s=26,
            color=color,
            marker=marker,
            label=f"{label} ({len(marker_events)})",
            alpha=0.85,
        )

    if symbol_events.empty:
        y_min, y_max = plt.ylim()
        x_min, x_max = price_frame["date"].min(), price_frame["date"].max()
        plt.text(
            x_min + (x_max - x_min) / 30,
            y_min + (y_max - y_min) * 0.85,
            "no signals",
            fontsize=11,
            color="#6b7280",
        )

    plt.title(
        f"{symbol} signal test: next {forward_days}d "
        f"{test_price_column} drop >= rolling {vol_window}d {volatility_method}"
    )
    plt.xlabel("Date")
    plt.ylabel("Close Price")
    plt.legend(loc="best")
    ax = plt.gca()
    ax.xaxis.set_major_locator(mdates.AutoDateLocator())
    ax.xaxis.set_major_formatter(
        mdates.ConciseDateFormatter(ax.xaxis.get_major_locator())
    )
    plt.tight_layout()

    figures_dir.mkdir(parents=True, exist_ok=True)
    figure_path = figures_dir / f"{symbol}_factor_signal_test_{forward_days}d.png"
    plt.savefig(figure_path, dpi=dpi)
    plt.close()

    return figure_path


def save_outputs(
    frames,
    events,
    output_dir,
    forward_days_list,
    volatility_method,
    vol_window,
    test_price_column,
    dpi,
):
    tables_dir = output_dir / "tables"
    figures_dir = output_dir / "figures"
    tables_dir.mkdir(parents=True, exist_ok=True)
    figures_dir.mkdir(parents=True, exist_ok=True)

    events_output = events.sort_values([
        "symbol",
        "factor_id",
        "factor_name",
        "signal_date",
        "forward_days",
    ]).copy()
    ordered_columns = [
        column
        for column in empty_events_table(max(forward_days_list)).columns
        if column in events_output.columns
    ]
    extra_columns = [
        column
        for column in events_output.columns
        if column not in ordered_columns
    ]
    events_output = events_output[ordered_columns + extra_columns]
    event_path = tables_dir / "factor_signal_test_events.csv"
    events_output.to_csv(event_path, index=False)

    symbol_factor_summary = summarize_events(
        events,
        [
            "symbol",
            "forward_days",
            "run_dir",
            "factor_id",
            "factor_name",
            "factor_label",
        ],
    )
    symbol_factor_summary_path = (
        tables_dir / "factor_signal_test_summary_by_symbol_factor.csv"
    )
    symbol_factor_summary.to_csv(symbol_factor_summary_path, index=False)

    symbol_summary = summarize_events(events, ["symbol", "forward_days"])
    symbol_summary_path = tables_dir / "factor_signal_test_summary_by_symbol.csv"
    symbol_summary.to_csv(symbol_summary_path, index=False)

    factor_summary = summarize_events(
        events,
        ["forward_days", "run_dir", "factor_id", "factor_name", "factor_label"],
    )
    factor_summary_path = tables_dir / "factor_signal_test_summary_by_factor.csv"
    factor_summary.to_csv(factor_summary_path, index=False)

    overall_summary = summarize_overall(events)
    overall_summary_path = tables_dir / "factor_signal_test_overall.csv"
    overall_summary.to_csv(overall_summary_path, index=False)

    price_frames = build_symbol_price_frames(frames)
    figure_paths = []
    for forward_days in forward_days_list:
        horizon_events = events[events["forward_days"] == forward_days]
        for symbol in sorted(price_frames):
            figure_paths.append(
                plot_symbol_test(
                    symbol=symbol,
                    price_frame=price_frames[symbol],
                    events=horizon_events,
                    figures_dir=figures_dir,
                    forward_days=forward_days,
                    volatility_method=volatility_method,
                    vol_window=vol_window,
                    test_price_column=test_price_column,
                    dpi=dpi,
                )
            )

    return {
        "event_path": event_path,
        "symbol_factor_summary_path": symbol_factor_summary_path,
        "symbol_summary_path": symbol_summary_path,
        "factor_summary_path": factor_summary_path,
        "overall_summary_path": overall_summary_path,
        "figure_paths": figure_paths,
    }


def main():
    parser = argparse.ArgumentParser(
        description=(
            "Test factor signals with a dynamic threshold. A signal is valid "
            "if the selected future price drops by at least the symbol's "
            "rolling one-month average price volatility within selected "
            "future trading-day windows."
        )
    )
    parser.add_argument(
        "--runs-dir",
        type=Path,
        default=DEFAULT_RUNS_DIR,
        help=f"directory containing factor result folders, default: {DEFAULT_RUNS_DIR}",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=DEFAULT_OUTPUT_DIR,
        help=f"output directory, default: {DEFAULT_OUTPUT_DIR}",
    )
    parser.add_argument(
        "--symbols",
        "--symbol",
        dest="symbols",
        help="optional comma-separated symbols, for example: JD,CU",
    )
    parser.add_argument(
        "--factor-ids",
        default=",".join(DEFAULT_FACTOR_IDS),
        help=(
            "comma-separated factor ids to test, default: "
            f"{','.join(DEFAULT_FACTOR_IDS)}; use ALL to test every factor result"
        ),
    )
    parser.add_argument(
        "--forward-days",
        default=",".join(str(value) for value in DEFAULT_FORWARD_DAYS),
        help=(
            "comma-separated future trading-day windows after each signal, "
            "default: "
            f"{','.join(str(value) for value in DEFAULT_FORWARD_DAYS)}"
        ),
    )
    parser.add_argument(
        "--vol-window",
        type=int,
        default=DEFAULT_VOL_WINDOW,
        help=f"rolling volatility window in trading days, default: {DEFAULT_VOL_WINDOW}",
    )
    parser.add_argument(
        "--min-vol-periods",
        type=int,
        default=DEFAULT_MIN_VOL_PERIODS,
        help=(
            "minimum observations required for rolling volatility, default: "
            f"{DEFAULT_MIN_VOL_PERIODS}"
        ),
    )
    parser.add_argument(
        "--volatility-method",
        choices=["abs_return", "range", "close_to_close_std"],
        default=DEFAULT_VOLATILITY_METHOD,
        help=(
            "volatility threshold source, default: abs_return. "
            "abs_return is the rolling mean of absolute close-to-close returns; "
            "range is the rolling mean of high-low range divided by close."
        ),
    )
    parser.add_argument(
        "--include-signal-day-volatility",
        action="store_true",
        help=(
            "include the signal day in the rolling volatility window; by "
            "default the threshold only uses data before the signal day"
        ),
    )
    parser.add_argument(
        "--test-price-column",
        choices=["close", "low"],
        default="close",
        help="future price column used for the drop test, default: close",
    )
    parser.add_argument(
        "--dpi",
        type=int,
        default=300,
        help="figure dpi, default: 300",
    )

    args = parser.parse_args()
    runs_dir = args.runs_dir.resolve()
    output_dir = args.output_dir.resolve()
    forward_days_list = parse_forward_days(args.forward_days)

    if args.vol_window < 1:
        raise ValueError("--vol-window must be at least 1")
    if args.min_vol_periods < 1:
        raise ValueError("--min-vol-periods must be at least 1")
    if args.min_vol_periods > args.vol_window:
        raise ValueError("--min-vol-periods cannot exceed --vol-window")
    if not runs_dir.is_dir():
        raise FileNotFoundError(f"factor results directory not found: {runs_dir}")

    factor_files = discover_factor_files(
        runs_dir=runs_dir,
        symbol_filter=parse_symbols(args.symbols),
        factor_id_filter=parse_factor_ids(args.factor_ids),
    )
    factor_frames, errors = load_all_factor_data(
        factor_files=factor_files,
        test_price_column=args.test_price_column,
        volatility_method=args.volatility_method,
        vol_window=args.vol_window,
        min_vol_periods=args.min_vol_periods,
        include_signal_day_volatility=args.include_signal_day_volatility,
    )
    events = evaluate_all_frames(
        frames=factor_frames,
        forward_days_list=forward_days_list,
        test_price_column=args.test_price_column,
    )
    outputs = save_outputs(
        frames=factor_frames,
        events=events,
        output_dir=output_dir,
        forward_days_list=forward_days_list,
        volatility_method=args.volatility_method,
        vol_window=args.vol_window,
        test_price_column=args.test_price_column,
        dpi=args.dpi,
    )

    overall_summary = summarize_overall(events)

    print("factor signal test complete.")
    print(f"factor files loaded: {len(factor_frames)}")
    print(f"signal event rows: {len(events)}")
    print(f"volatility method: {args.volatility_method}")
    print(f"volatility window: {args.vol_window}")
    print(f"future windows: {','.join(str(value) for value in forward_days_list)}")
    if not overall_summary.empty:
        for _, row in overall_summary.sort_values("forward_days").iterrows():
            win_rate = row["win_rate"]
            win_rate_text = f"{win_rate:.4f}" if pd.notna(win_rate) else "nan"
            print(
                f"{int(row['forward_days'])}d: "
                f"signals={int(row['total_signals'])}, "
                f"tested={int(row['tested_signals'])}, "
                f"valid={int(row['correct_signals'])}, "
                f"win rate={win_rate_text}"
            )
    print(f"figures saved: {len(outputs['figure_paths'])}")
    print(f"event table: {outputs['event_path']}")
    print(f"summary by symbol/factor: {outputs['symbol_factor_summary_path']}")
    print(f"summary by symbol: {outputs['symbol_summary_path']}")
    print(f"summary by factor: {outputs['factor_summary_path']}")
    print(f"overall summary: {outputs['overall_summary_path']}")
    print(f"figures dir: {output_dir / 'figures'}")

    if errors:
        print("\nskipped files:")
        for path, exc in errors:
            print(f"- {path}: {exc}")


if __name__ == "__main__":
    main()
