import re

import pandas as pd

# 读取因子结果，并生成 combined 统计表。
from plot_combined_signals_config import DEFAULT_FACTOR_IDS


FACTOR_FILE_PATTERN = re.compile(r"^(.+?)_(\d+)_(.+)\.csv$")


def parse_factor_ids(raw_value):
    if raw_value is None:
        return set(DEFAULT_FACTOR_IDS)

    raw_value = raw_value.strip()
    if raw_value == "" or raw_value.upper() in {"ALL", "*"}:
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


def discover_factor_files(runs_dir, factor_id_filter=None):
    factor_files = []

    for run_dir in sorted(runs_dir.iterdir()):
        if not run_dir.is_dir():
            continue
        if run_dir.name == "combined":
            continue

        factors_dir = run_dir / "factors"
        if not factors_dir.is_dir():
            continue

        for path in sorted(factors_dir.glob("*.csv")):
            metadata = parse_factor_file(path)
            if metadata is None:
                continue
            if (
                factor_id_filter is not None and
                metadata["factor_id"] not in factor_id_filter
            ):
                continue

            factor_files.append({
                **metadata,
                "run_dir": run_dir.name,
                "path": path,
            })

    if not factor_files:
        raise FileNotFoundError(
            f"no factor csv found under {runs_dir}"
        )

    return factor_files


def load_factor_daily(file_info):
    columns = [
        "date",
        "close",
        "factor_id",
        "factor_name",
        "factor_value",
        "signal",
    ]
    daily = pd.read_csv(
        file_info["path"],
        usecols=lambda col: col in columns,
    )

    missing_columns = {"date", "close", "signal"} - set(daily.columns)
    if missing_columns:
        raise ValueError(
            f"{file_info['path']} missing columns: "
            f"{','.join(sorted(missing_columns))}"
        )

    daily["date"] = pd.to_datetime(daily["date"])
    daily["close"] = pd.to_numeric(daily["close"], errors="coerce")
    daily["signal"] = (
        pd.to_numeric(daily["signal"], errors="coerce")
        .fillna(0)
        .astype(int)
    )

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


def load_all_factor_data(factor_files):
    frames = []
    errors = []

    for file_info in factor_files:
        try:
            frames.append(load_factor_daily(file_info))
        except Exception as exc:
            errors.append((file_info["path"], exc))

    if not frames:
        error_text = "\n".join(f"- {path}: {exc}" for path, exc in errors)
        raise RuntimeError(f"no factor data could be loaded\n{error_text}")

    return frames, errors


def build_signal_stats(factor_frames):
    rows = []

    for daily in factor_frames:
        signal_points = daily[daily["signal"] == 1]
        signal_factor_values = (
            pd.to_numeric(signal_points["factor_value"], errors="coerce")
            if "factor_value" in signal_points.columns
            else pd.Series(dtype="float64")
        )

        first_signal = None
        last_signal = None
        if len(signal_points) > 0:
            first_signal = signal_points.iloc[0]
            last_signal = signal_points.iloc[-1]

        rows.append({
            "symbol": daily["symbol"].iloc[0],
            "factor_id": daily["factor_id"].iloc[0],
            "factor_name": daily["factor_name"].iloc[0],
            "factor_label": daily["factor_label"].iloc[0],
            "run_dir": daily["run_dir"].iloc[0],
            "total_days": len(daily),
            "valid_close_days": int(daily["close"].notna().sum()),
            "valid_factor_days": (
                int(pd.to_numeric(daily["factor_value"], errors="coerce").notna().sum())
                if "factor_value" in daily.columns
                else 0
            ),
            "signal_days": int(daily["signal"].sum()),
            "signal_ratio": daily["signal"].mean(),
            "first_signal_date": (
                first_signal["date"].date().isoformat()
                if first_signal is not None
                else ""
            ),
            "last_signal_date": (
                last_signal["date"].date().isoformat()
                if last_signal is not None
                else ""
            ),
            "first_signal_close": (
                first_signal["close"] if first_signal is not None else pd.NA
            ),
            "last_signal_close": (
                last_signal["close"] if last_signal is not None else pd.NA
            ),
            "mean_signal_close": signal_points["close"].mean(),
            "min_signal_close": signal_points["close"].min(),
            "max_signal_close": signal_points["close"].max(),
            "mean_signal_factor_value": signal_factor_values.mean(),
            "min_signal_factor_value": signal_factor_values.min(),
            "max_signal_factor_value": signal_factor_values.max(),
        })

    stats = pd.DataFrame(rows)
    stats = stats.sort_values(["symbol", "factor_id", "factor_name"])

    aggregate = (
        stats
        .groupby(["factor_id", "factor_name", "factor_label"], as_index=False)
        .agg(
            run_dir=("run_dir", "first"),
            total_days=("total_days", "sum"),
            valid_close_days=("valid_close_days", "sum"),
            valid_factor_days=("valid_factor_days", "sum"),
            signal_days=("signal_days", "sum"),
            first_signal_date=("first_signal_date", _min_nonempty),
            last_signal_date=("last_signal_date", _max_nonempty),
            mean_signal_close=("mean_signal_close", "mean"),
            mean_signal_factor_value=("mean_signal_factor_value", "mean"),
        )
    )
    aggregate.insert(0, "symbol", "ALL_SYMBOLS")
    aggregate["signal_ratio"] = (
        aggregate["signal_days"] / aggregate["total_days"]
    )

    aggregate = aggregate[
        [
            "symbol",
            "factor_id",
            "factor_name",
            "factor_label",
            "run_dir",
            "total_days",
            "valid_close_days",
            "valid_factor_days",
            "signal_days",
            "signal_ratio",
            "first_signal_date",
            "last_signal_date",
            "mean_signal_close",
            "mean_signal_factor_value",
        ]
    ]

    stats["row_type"] = "symbol_factor"
    aggregate["row_type"] = "all_symbols_factor"

    ordered_columns = ["row_type"] + [
        col for col in stats.columns if col != "row_type"
    ]
    output = pd.concat([stats, aggregate], ignore_index=True, sort=False)
    return output[ordered_columns]


def _min_nonempty(values):
    values = [value for value in values if isinstance(value, str) and value]
    return min(values) if values else ""


def _max_nonempty(values):
    values = [value for value in values if isinstance(value, str) and value]
    return max(values) if values else ""
