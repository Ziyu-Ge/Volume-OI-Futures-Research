import argparse
import re
from pathlib import Path

import numpy as np
import pandas as pd


BACKTEST_DIR = Path(__file__).resolve().parent
CODE_DIR = BACKTEST_DIR.parent
PROJECT_ROOT = CODE_DIR.parent
RESULTS_DIR = PROJECT_ROOT / "results"
DEFAULT_RUNS_DIR = RESULTS_DIR
DEFAULT_OUTPUT_DIR = RESULTS_DIR / "test"
DEFAULT_FACTOR_IDS = ("11", "12", "13", "14")
DEFAULT_FORWARD_DAYS = (3, 5, 10)
DEFAULT_VOL_WINDOW = 20
SKIPPED_RESULT_DIRS = {"combined", "test"}
FACTOR_FILE_PATTERN = re.compile(r"^(.+?)_(\d+)_(.+)\.csv$")

EVENT_COLUMNS = [
    "symbol",
    "forward_days",
    "factor_id",
    "factor_name",
    "signal_date",
    "signal_close",
    "future_date",
    "future_close",
    "avg_volatility_1m",
    "total_return",
    "max_drawdown",
    "win",
    "is_tested",
]


def parse_symbols(raw_value):
    if raw_value is None:
        return None
    symbols = [item.strip().upper() for item in raw_value.split(",") if item.strip()]
    return set(symbols) or None


def parse_factor_ids(raw_value):
    if raw_value is None:
        return set(DEFAULT_FACTOR_IDS)
    if raw_value.strip().upper() == "ALL":
        return None
    factor_ids = {item.strip() for item in raw_value.split(",") if item.strip()}
    return factor_ids or set(DEFAULT_FACTOR_IDS)


def parse_forward_days(raw_value):
    if raw_value is None:
        return DEFAULT_FORWARD_DAYS

    forward_days = []
    for item in raw_value.split(","):
        item = item.strip()
        if not item:
            continue
        value = int(item)
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
        "path": path,
    }


def discover_factor_files(runs_dir, symbol_filter, factor_id_filter):
    factor_files = []
    for run_dir in sorted(runs_dir.iterdir()):
        if not run_dir.is_dir() or run_dir.name in SKIPPED_RESULT_DIRS:
            continue

        factors_dir = run_dir / "tables" / "factors"
        if not factors_dir.is_dir():
            continue

        for path in sorted(factors_dir.glob("*.csv")):
            info = parse_factor_file(path)
            if info is None:
                continue
            if symbol_filter is not None and info["symbol"] not in symbol_filter:
                continue
            if factor_id_filter is not None and info["factor_id"] not in factor_id_filter:
                continue
            factor_files.append(info)

    if not factor_files:
        raise FileNotFoundError(f"no factor csv found under {runs_dir}")
    return factor_files


def load_factor_frame(file_info, vol_window):
    daily = pd.read_csv(
        file_info["path"],
        usecols=lambda col: col in {"date", "close", "signal"},
    )
    missing = {"date", "close", "signal"} - set(daily.columns)
    if missing:
        raise ValueError(
            f"{file_info['path']} missing columns: {','.join(sorted(missing))}"
        )

    daily["date"] = pd.to_datetime(daily["date"])
    daily["close"] = pd.to_numeric(daily["close"], errors="coerce")
    daily["signal"] = (
        pd.to_numeric(daily["signal"], errors="coerce")
        .fillna(0)
        .astype(int)
    )
    daily = daily.sort_values("date").reset_index(drop=True)

    daily["avg_volatility_1m"] = (
        daily["close"]
        .pct_change()
        .abs()
        .rolling(vol_window, min_periods=vol_window)
        .mean()
        .shift(1)
        .abs()
    )
    daily["symbol"] = file_info["symbol"]
    daily["factor_id"] = file_info["factor_id"]
    daily["factor_name"] = file_info["factor_name"]
    return daily


def evaluate_frame(daily, forward_days_list):
    records = []
    signal_indexes = daily.index[daily["signal"] == 1].tolist()

    for index in signal_indexes:
        signal = daily.loc[index]
        signal_close = signal["close"]
        threshold = signal["avg_volatility_1m"]

        for forward_days in forward_days_list:
            future_index = index + forward_days
            record = {
                "symbol": signal["symbol"],
                "forward_days": forward_days,
                "factor_id": signal["factor_id"],
                "factor_name": signal["factor_name"],
                "signal_date": signal["date"],
                "signal_close": signal_close,
                "future_date": pd.NaT,
                "future_close": np.nan,
                "avg_volatility_1m": threshold,
                "total_return": np.nan,
                "max_drawdown": np.nan,
                "win": False,
                "is_tested": False,
            }

            can_test = (
                future_index < len(daily)
                and pd.notna(signal_close)
                and signal_close != 0
                and pd.notna(threshold)
            )
            if can_test:
                future = daily.iloc[index + 1:future_index + 1]
                future_close = future.iloc[-1]["close"]
                future_returns = future["close"] / signal_close - 1
                total_return = future_close / signal_close - 1
                max_drawdown = max(0, -future_returns.min())

                record.update({
                    "future_date": future.iloc[-1]["date"],
                    "future_close": future_close,
                    "total_return": total_return,
                    "max_drawdown": max_drawdown,
                    "win": bool(total_return < -abs(threshold)),
                    "is_tested": pd.notna(total_return),
                })

            records.append(record)

    return pd.DataFrame(records, columns=EVENT_COLUMNS)


def evaluate_all(frames, forward_days_list):
    events = [evaluate_frame(frame, forward_days_list) for frame in frames]
    events = [frame for frame in events if not frame.empty]
    if not events:
        return pd.DataFrame(columns=EVENT_COLUMNS)
    return pd.concat(events, ignore_index=True)


def summarize(events, group_columns):
    columns = list(group_columns) + [
        "total_signals",
        "tested_signals",
        "win_signals",
        "win_rate",
        "mean_total_return",
        "mean_max_drawdown",
        "max_drawdown",
    ]
    if events.empty:
        return pd.DataFrame(columns=columns)

    working = events.copy()
    working["tested_int"] = working["is_tested"].astype(int)
    working["win_int"] = working["win"].astype(int)

    summary = (
        working.groupby(list(group_columns), dropna=False)
        .agg(
            total_signals=("signal_date", "size"),
            tested_signals=("tested_int", "sum"),
            win_signals=("win_int", "sum"),
            mean_total_return=("total_return", "mean"),
            mean_max_drawdown=("max_drawdown", "mean"),
            max_drawdown=("max_drawdown", "max"),
        )
        .reset_index()
    )
    summary["win_rate"] = np.where(
        summary["tested_signals"] > 0,
        summary["win_signals"] / summary["tested_signals"],
        np.nan,
    )
    return summary[columns]


def save_outputs(events, output_dir):
    tables_dir = output_dir / "tables"
    tables_dir.mkdir(parents=True, exist_ok=True)

    event_path = tables_dir / "factor_signal_test_events.csv"
    events.sort_values([
        "symbol",
        "factor_id",
        "signal_date",
        "forward_days",
    ]).to_csv(event_path, index=False)

    outputs = {"event_path": event_path}
    summary_specs = {
        "overall_summary_path": (
            "factor_signal_test_overall.csv",
            ["forward_days"],
        ),
        "symbol_summary_path": (
            "factor_signal_test_summary_by_symbol.csv",
            ["symbol", "forward_days"],
        ),
        "factor_summary_path": (
            "factor_signal_test_summary_by_factor.csv",
            ["forward_days", "factor_id", "factor_name"],
        ),
        "symbol_factor_summary_path": (
            "factor_signal_test_summary_by_symbol_factor.csv",
            ["symbol", "forward_days", "factor_id", "factor_name"],
        ),
    }

    for output_key, (filename, group_columns) in summary_specs.items():
        path = tables_dir / filename
        summarize(events, group_columns).to_csv(path, index=False)
        outputs[output_key] = path

    return outputs


def main():
    parser = argparse.ArgumentParser(
        description=(
            "Test factor signals by simple total return over future horizons. "
            "A signal wins when the horizon total return is below negative "
            "one-month average volatility."
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
            f"default: {','.join(str(value) for value in DEFAULT_FORWARD_DAYS)}"
        ),
    )
    parser.add_argument(
        "--vol-window",
        type=int,
        default=DEFAULT_VOL_WINDOW,
        help=f"rolling volatility window in trading days, default: {DEFAULT_VOL_WINDOW}",
    )

    args = parser.parse_args()
    runs_dir = args.runs_dir.resolve()
    output_dir = args.output_dir.resolve()
    forward_days_list = parse_forward_days(args.forward_days)

    if args.vol_window < 1:
        raise ValueError("--vol-window must be at least 1")
    if not runs_dir.is_dir():
        raise FileNotFoundError(f"factor results directory not found: {runs_dir}")

    factor_files = discover_factor_files(
        runs_dir=runs_dir,
        symbol_filter=parse_symbols(args.symbols),
        factor_id_filter=parse_factor_ids(args.factor_ids),
    )
    frames = [load_factor_frame(file_info, args.vol_window) for file_info in factor_files]
    events = evaluate_all(frames, forward_days_list)
    outputs = save_outputs(events, output_dir)
    overall = summarize(events, ["forward_days"]).sort_values("forward_days")

    print("factor signal test complete.")
    print(f"factor files loaded: {len(frames)}")
    print(f"signal event rows: {len(events)}")
    print(f"future windows: {','.join(str(value) for value in forward_days_list)}")
    for _, row in overall.iterrows():
        win_rate = row["win_rate"]
        win_rate_text = f"{win_rate:.4f}" if pd.notna(win_rate) else "nan"
        print(
            f"{int(row['forward_days'])}d: "
            f"signals={int(row['total_signals'])}, "
            f"tested={int(row['tested_signals'])}, "
            f"wins={int(row['win_signals'])}, "
            f"win rate={win_rate_text}"
        )
    print(f"event table: {outputs['event_path']}")
    print(f"overall summary: {outputs['overall_summary_path']}")
    print(f"summary by symbol: {outputs['symbol_summary_path']}")
    print(f"summary by factor: {outputs['factor_summary_path']}")
    print(f"summary by symbol/factor: {outputs['symbol_factor_summary_path']}")


if __name__ == "__main__":
    main()
