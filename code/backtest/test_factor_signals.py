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


def load_factor_daily(file_info, test_price_column):
    columns = {
        "date",
        "close",
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
    daily["close"] = pd.to_numeric(daily["close"], errors="coerce")
    daily[test_price_column] = pd.to_numeric(
        daily[test_price_column],
        errors="coerce",
    )
    daily["signal"] = (
        pd.to_numeric(daily["signal"], errors="coerce")
        .fillna(0)
        .astype(int)
    )

    if "factor_value" in daily.columns:
        daily["factor_value"] = pd.to_numeric(
            daily["factor_value"],
            errors="coerce",
        )
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
    daily["symbol"] = file_info["symbol"]
    daily["factor_id"] = factor_id
    daily["factor_name"] = factor_name
    daily["run_dir"] = file_info["run_dir"]
    daily["factor_label"] = f"{factor_id}_{factor_name}"

    return daily


def load_all_factor_data(factor_files, test_price_column):
    frames = []
    errors = []

    for file_info in factor_files:
        try:
            frames.append(load_factor_daily(file_info, test_price_column))
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
        "drop_threshold",
        "available_forward_days",
        "has_full_forward_window",
        "is_tested",
        "correct",
        "hit_day",
        "hit_date",
        "worst_forward_return",
        "worst_forward_date",
    ]
    for step in range(1, forward_days + 1):
        columns.extend([
            f"future_date_t{step}",
            f"future_price_t{step}",
            f"return_t{step}",
        ])
    return pd.DataFrame(columns=columns)


def evaluate_frame(daily, forward_days, drop_threshold, test_price_column):
    records = []
    signal_points = daily.index[daily["signal"] == 1].tolist()

    for index in signal_points:
        row = daily.loc[index]
        signal_close = row["close"]
        future = daily.iloc[index + 1:index + 1 + forward_days]

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
            "drop_threshold": drop_threshold,
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
                    returns.append((step, future_row["date"], forward_return))
            else:
                record[f"future_date_t{step}"] = pd.NaT
                record[f"future_price_t{step}"] = np.nan
                record[f"return_t{step}"] = np.nan

        hits = [
            (step, date_value, return_value)
            for step, date_value, return_value in returns
            if return_value <= -drop_threshold
        ]
        worst = min(returns, key=lambda item: item[2]) if returns else None
        hit = hits[0] if hits else None

        record["correct"] = bool(hit is not None)
        record["hit_day"] = int(hit[0]) if hit is not None else pd.NA
        record["hit_date"] = hit[1] if hit is not None else pd.NaT
        record["worst_forward_return"] = worst[2] if worst is not None else np.nan
        record["worst_forward_date"] = worst[1] if worst is not None else pd.NaT
        record["is_tested"] = bool(
            record["correct"] or record["has_full_forward_window"]
        )
        records.append(record)

    if not records:
        return empty_events_table(forward_days)

    return pd.DataFrame(records)


def evaluate_all_frames(frames, forward_days, drop_threshold, test_price_column):
    event_frames = [
        evaluate_frame(
            daily=frame,
            forward_days=forward_days,
            drop_threshold=drop_threshold,
            test_price_column=test_price_column,
        )
        for frame in frames
    ]
    event_frames = [frame for frame in event_frames if not frame.empty]
    if not event_frames:
        return empty_events_table(forward_days)

    events = pd.concat(event_frames, ignore_index=True)
    events["correct"] = events["correct"].fillna(False).astype(bool)
    events["is_tested"] = events["is_tested"].fillna(False).astype(bool)
    events["has_full_forward_window"] = (
        events["has_full_forward_window"].fillna(False).astype(bool)
    )
    return events


def summarize_events(events, group_columns):
    base_columns = list(group_columns) + [
        "total_signals",
        "tested_signals",
        "correct_signals",
        "incorrect_signals",
        "unknown_signals",
        "accuracy",
        "first_signal_date",
        "last_signal_date",
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
            mean_worst_forward_return=("worst_forward_return", "mean"),
            min_worst_forward_return=("worst_forward_return", "min"),
            mean_factor_value=("factor_value", "mean"),
        )
        .reset_index()
    )
    summary["accuracy"] = np.where(
        summary["tested_signals"] > 0,
        summary["correct_signals"] / summary["tested_signals"],
        np.nan,
    )
    return summary[base_columns]


def summarize_overall(events):
    columns = [
        "total_signals",
        "tested_signals",
        "correct_signals",
        "incorrect_signals",
        "unknown_signals",
        "accuracy",
        "mean_worst_forward_return",
        "min_worst_forward_return",
    ]
    if events.empty:
        return pd.DataFrame(columns=columns)

    tested = events["is_tested"].astype(bool)
    correct = events["correct"].astype(bool)
    tested_count = int(tested.sum())
    correct_count = int(correct.sum())
    incorrect_count = int((tested & ~correct).sum())
    unknown_count = int((~tested).sum())

    return pd.DataFrame([{
        "total_signals": int(len(events)),
        "tested_signals": tested_count,
        "correct_signals": correct_count,
        "incorrect_signals": incorrect_count,
        "unknown_signals": unknown_count,
        "accuracy": (
            correct_count / tested_count
            if tested_count > 0
            else np.nan
        ),
        "mean_worst_forward_return": events["worst_forward_return"].mean(),
        "min_worst_forward_return": events["worst_forward_return"].min(),
    }])


def build_symbol_price_frames(frames):
    price_frames = {}

    for frame in frames:
        symbol = frame["symbol"].iloc[0]
        candidate = (
            frame[["date", "close"]]
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
    drop_threshold,
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
        (correct_events, "#2a9d8f", "v", "correct"),
        (incorrect_events, "#d62828", "x", "incorrect"),
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

    threshold_pct = drop_threshold * 100
    plt.title(
        f"{symbol} factor signal test "
        f"({forward_days}d {test_price_column} drop >= {threshold_pct:.1f}%)"
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
    figure_path = figures_dir / f"{symbol}_factor_signal_test.png"
    plt.savefig(figure_path, dpi=dpi)
    plt.close()

    return figure_path


def save_outputs(
    frames,
    events,
    output_dir,
    forward_days,
    drop_threshold,
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
    ]).copy()
    event_path = tables_dir / "factor_signal_test_events.csv"
    events_output.to_csv(event_path, index=False)

    symbol_factor_summary = summarize_events(
        events,
        ["symbol", "run_dir", "factor_id", "factor_name", "factor_label"],
    )
    symbol_factor_summary_path = (
        tables_dir / "factor_signal_test_summary_by_symbol_factor.csv"
    )
    symbol_factor_summary.to_csv(symbol_factor_summary_path, index=False)

    symbol_summary = summarize_events(events, ["symbol"])
    symbol_summary_path = tables_dir / "factor_signal_test_summary_by_symbol.csv"
    symbol_summary.to_csv(symbol_summary_path, index=False)

    factor_summary = summarize_events(
        events,
        ["run_dir", "factor_id", "factor_name", "factor_label"],
    )
    factor_summary_path = tables_dir / "factor_signal_test_summary_by_factor.csv"
    factor_summary.to_csv(factor_summary_path, index=False)

    overall_summary = summarize_overall(events)
    overall_summary_path = tables_dir / "factor_signal_test_overall.csv"
    overall_summary.to_csv(overall_summary_path, index=False)

    price_frames = build_symbol_price_frames(frames)
    figure_paths = []
    for symbol in sorted(price_frames):
        figure_paths.append(
            plot_symbol_test(
                symbol=symbol,
                price_frame=price_frames[symbol],
                events=events,
                figures_dir=figures_dir,
                forward_days=forward_days,
                drop_threshold=drop_threshold,
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
            "Test factor signals. A signal is correct if any selected future "
            "price within the next N trading days drops by at least the "
            "threshold."
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
        type=int,
        default=3,
        help="future trading-day window after each signal, default: 3",
    )
    parser.add_argument(
        "--drop-threshold",
        type=float,
        default=0.03,
        help="drop threshold for a correct signal, default: 0.03",
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
    drop_threshold = abs(args.drop_threshold)

    if args.forward_days < 1:
        raise ValueError("--forward-days must be at least 1")
    if not runs_dir.is_dir():
        raise FileNotFoundError(f"factor results directory not found: {runs_dir}")

    factor_files = discover_factor_files(
        runs_dir=runs_dir,
        symbol_filter=parse_symbols(args.symbols),
        factor_id_filter=parse_factor_ids(args.factor_ids),
    )
    factor_frames, errors = load_all_factor_data(
        factor_files,
        test_price_column=args.test_price_column,
    )
    events = evaluate_all_frames(
        frames=factor_frames,
        forward_days=args.forward_days,
        drop_threshold=drop_threshold,
        test_price_column=args.test_price_column,
    )
    outputs = save_outputs(
        frames=factor_frames,
        events=events,
        output_dir=output_dir,
        forward_days=args.forward_days,
        drop_threshold=drop_threshold,
        test_price_column=args.test_price_column,
        dpi=args.dpi,
    )

    tested_count = int(events["is_tested"].sum()) if not events.empty else 0
    correct_count = int(events["correct"].sum()) if not events.empty else 0

    print("factor signal test complete.")
    print(f"factor files loaded: {len(factor_frames)}")
    print(f"signal events: {len(events)}")
    print(f"tested signals: {tested_count}")
    print(f"correct signals: {correct_count}")
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
