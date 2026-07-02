import argparse
import os
from pathlib import Path

import numpy as np
import pandas as pd


BACKTEST_DIR = Path(__file__).resolve().parent
CODE_DIR = BACKTEST_DIR.parent
PROJECT_ROOT = CODE_DIR.parent
DEFAULT_RUNS_DIR = PROJECT_ROOT / "results"
DEFAULT_OUTPUT_DIR = PROJECT_ROOT / "results" / "evaluation"
DEFAULT_FACTOR_IDS = "11,12,13,14"
DEFAULT_LOOKAHEAD_DAYS_LIST = "3,5,10"
DEFAULT_CLUSTER_MAX_GAP = 10
DEFAULT_VOLATILITY_WINDOW = 20
DEFAULT_VOLATILITY_MULTIPLIER = 1.0
DEFAULT_VOLATILITY_MIN_HISTORY_DAYS = 20
DEFAULT_CONFIDENCE_WINDOW_DAYS = 10
CONFIDENCE_FULL_SIGNAL_DAYS = 4
ALL_CONFIDENCE_LEVEL = "ALL_CONFIDENCE"
NO_CONFIDENCE_LEVEL = "none"
UNKNOWN_CONFIDENCE_LEVEL = "unknown"
CONFIDENCE_LEVELS = ("low", "medium", "high")
CONFIDENCE_LEVEL_ORDER = {
    level: index
    for index, level in enumerate(CONFIDENCE_LEVELS)
}
CONFIDENCE_LEVEL_ORDER[NO_CONFIDENCE_LEVEL] = len(CONFIDENCE_LEVEL_ORDER)
CONFIDENCE_LEVEL_ORDER[UNKNOWN_CONFIDENCE_LEVEL] = len(CONFIDENCE_LEVEL_ORDER)
CONFIDENCE_LEVEL_ALIASES = {
    "none": NO_CONFIDENCE_LEVEL,
    "no": NO_CONFIDENCE_LEVEL,
    "0": NO_CONFIDENCE_LEVEL,
    "l": "low",
    "low": "low",
    "低": "low",
    "低置信度": "low",
    "m": "medium",
    "mid": "medium",
    "med": "medium",
    "medium": "medium",
    "中": "medium",
    "中置信度": "medium",
    "h": "high",
    "high": "high",
    "高": "high",
    "高置信度": "high",
}
CONFIDENCE_LEVEL_SOURCE_COLUMNS = ("confidence_level",)
CONFIDENCE_SCORE_SOURCE_COLUMNS = ("confidence_score", "confidence")
CONFIDENCE_INPUT_COLUMNS = (
    "confidence_level",
    "confidence",
    "confidence_score",
    "confidence_signal_days",
    "confidence_recent_dates",
)
SUMMARY_TABLE_SPECS = (
    (
        "overall",
        "all_symbols_all_factors",
        "overall_by_lookahead.csv",
    ),
    (
        "factor",
        "all_symbols_factor",
        "factor_by_lookahead.csv",
    ),
    (
        "overall_confidence",
        "all_symbols_all_factors_confidence",
        "overall_by_confidence.csv",
    ),
    (
        "factor_confidence",
        "all_symbols_factor_confidence",
        "factor_by_confidence.csv",
    ),
)
SUMMARY_GROUP_COLUMNS = [
    "symbol",
    "factor_id",
    "factor_name",
    "lookahead_days",
    "confidence_level",
]
SUMMARY_GROUP_SPECS = (
    (
        "all_symbols_factor",
        ["factor_id", "factor_name", "lookahead_days"],
        {"symbol": "ALL_SYMBOLS", "confidence_level": ALL_CONFIDENCE_LEVEL},
    ),
    (
        "all_symbols_all_factors",
        ["lookahead_days"],
        {
            "symbol": "ALL_SYMBOLS",
            "factor_id": "ALL_FACTORS",
            "factor_name": "all_factors",
            "confidence_level": ALL_CONFIDENCE_LEVEL,
        },
    ),
    (
        "all_symbols_factor_confidence",
        ["factor_id", "factor_name", "lookahead_days", "confidence_level"],
        {"symbol": "ALL_SYMBOLS"},
    ),
    (
        "all_symbols_all_factors_confidence",
        ["lookahead_days", "confidence_level"],
        {
            "symbol": "ALL_SYMBOLS",
            "factor_id": "ALL_FACTORS",
            "factor_name": "all_factors",
        },
    ),
)
SUMMARY_OUTPUT_COLUMNS = [
    "factor_id",
    "factor_name",
    "lookahead_days",
    "confidence_level",
    "threshold",
    "strategy_win_rate",
    "baseline_win_rate",
    "win_rate_diff",
    "strategy_observation_max_drawdown",
]

os.environ.setdefault("MPLCONFIGDIR", str(PROJECT_ROOT / ".matplotlib"))
os.environ.setdefault("XDG_CACHE_HOME", str(PROJECT_ROOT / ".cache"))
(PROJECT_ROOT / ".matplotlib").mkdir(exist_ok=True)
(PROJECT_ROOT / ".cache").mkdir(exist_ok=True)

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt


def parse_csv_list(raw_value):
    if raw_value is None:
        return None

    values = [
        item.strip().upper()
        for item in str(raw_value).split(",")
        if item.strip()
    ]
    return values or None


def parse_factor_ids(raw_value):
    values = [
        item.strip()
        for item in str(raw_value).split(",")
        if item.strip()
    ]
    return values or None


def parse_int_list(raw_value):
    values = [
        int(item.strip())
        for item in str(raw_value).split(",")
        if item.strip()
    ]
    return values or None


def first_existing_column(frame, columns):
    return next((column for column in columns if column in frame.columns), None)


def normalize_confidence_text(value):
    if pd.isna(value):
        return UNKNOWN_CONFIDENCE_LEVEL

    text = str(value).strip().lower()
    text = text.replace("-", "_").replace(" ", "_")
    return CONFIDENCE_LEVEL_ALIASES.get(text, UNKNOWN_CONFIDENCE_LEVEL)


def confidence_score_to_level(value):
    score = pd.to_numeric(value, errors="coerce")
    if pd.isna(score) or score < 0:
        return UNKNOWN_CONFIDENCE_LEVEL

    if score <= 1:
        normalized_score = score
    elif score <= 100:
        normalized_score = score / 100
    else:
        return UNKNOWN_CONFIDENCE_LEVEL

    if normalized_score <= 0:
        return NO_CONFIDENCE_LEVEL
    if normalized_score < 0.5:
        return "low"
    if normalized_score < 1:
        return "medium"
    return "high"


def normalize_confidence_level(value):
    text_level = normalize_confidence_text(value)
    if text_level != UNKNOWN_CONFIDENCE_LEVEL:
        return text_level

    return confidence_score_to_level(value)


def confidence_level_sort_value(value):
    return CONFIDENCE_LEVEL_ORDER.get(
        normalize_confidence_level(value),
        len(CONFIDENCE_LEVEL_ORDER),
    )


def ordered_confidence_levels(values):
    levels = {
        normalize_confidence_level(value)
        for value in values
    }
    return sorted(levels, key=confidence_level_sort_value)


def confidence_level_from_signal_days(signal_day_count):
    if signal_day_count >= CONFIDENCE_FULL_SIGNAL_DAYS:
        return "high"
    if signal_day_count >= 2:
        return "medium"
    if signal_day_count >= 1:
        return "low"

    return NO_CONFIDENCE_LEVEL


def format_date_list(date_values):
    return ",".join(
        pd.Timestamp(value).strftime("%Y-%m-%d")
        for value in date_values
    )


def load_confidence_source_frame(file_info):
    columns = {"date", "signal"}
    daily = pd.read_csv(
        file_info["path"],
        usecols=lambda col: col in columns,
    )
    if not columns.issubset(daily.columns):
        return None

    daily = daily[["date", "signal"]].copy()
    daily["date"] = pd.to_datetime(daily["date"])
    daily["signal"] = (
        pd.to_numeric(daily["signal"], errors="coerce")
        .fillna(0)
        .astype(int)
    )
    return daily.dropna(subset=["date"]).sort_values("date")


def build_symbol_confidence_by_date(symbol_frames, lookback_days):
    all_dates = sorted({
        value
        for frame in symbol_frames
        for value in frame["date"].dropna()
    })
    if not all_dates:
        return {}

    signal_dates = {
        value
        for frame in symbol_frames
        for value in frame.loc[frame["signal"] == 1, "date"].dropna()
    }
    lookback_days = max(int(lookback_days), 1)
    confidence_by_date = {}

    for index, current_date in enumerate(all_dates):
        window_start = max(0, index - lookback_days + 1)
        window_dates = all_dates[window_start:index + 1]
        recent_signal_dates = [
            value
            for value in window_dates
            if value in signal_dates
        ]
        signal_day_count = len(recent_signal_dates)
        confidence_score = min(
            signal_day_count / CONFIDENCE_FULL_SIGNAL_DAYS,
            1.0,
        )

        confidence_by_date[current_date] = {
            "confidence_score": confidence_score,
            "confidence_signal_days": signal_day_count,
            "confidence_level": confidence_level_from_signal_days(
                signal_day_count
            ),
            "confidence_recent_dates": format_date_list(recent_signal_dates),
        }

    return confidence_by_date


def build_confidence_by_symbol_date(factor_files, lookback_days):
    frames_by_symbol = {}

    for file_info in factor_files:
        try:
            daily = load_confidence_source_frame(file_info)
        except Exception:
            continue
        if daily is None:
            continue

        frames_by_symbol.setdefault(file_info["symbol"], []).append(daily)

    return {
        symbol: build_symbol_confidence_by_date(
            symbol_frames=frames,
            lookback_days=lookback_days,
        )
        for symbol, frames in frames_by_symbol.items()
    }


def parse_factor_file(path):
    parts = path.stem.split("_", 2)
    if len(parts) != 3:
        return None

    symbol, factor_id, factor_name = parts
    return {
        "symbol": symbol.upper(),
        "factor_id": factor_id,
        "factor_name": factor_name,
        "factor_label": f"{factor_id}_{factor_name}",
        "run_dir": path.parents[2].name,
        "path": path,
    }


def discover_factor_files(runs_dir, symbols=None, factor_ids=None):
    symbols = set(symbols) if symbols is not None else None
    factor_ids = set(factor_ids) if factor_ids is not None else None
    factor_files = []

    for run_dir in sorted(runs_dir.iterdir()):
        if not run_dir.is_dir() or run_dir.name == "combined":
            continue

        factors_dir = run_dir / "tables" / "factors"
        if not factors_dir.is_dir():
            continue

        for path in sorted(factors_dir.glob("*.csv")):
            metadata = parse_factor_file(path)
            if metadata is None:
                continue
            if symbols is not None and metadata["symbol"] not in symbols:
                continue
            if factor_ids is not None and metadata["factor_id"] not in factor_ids:
                continue
            factor_files.append(metadata)

    if not factor_files:
        raise FileNotFoundError(f"没有找到因子日频表：{runs_dir}")

    return factor_files


def load_signal_confidence(file_info):
    signal_path = (
        file_info["path"].parents[1]
        / "signals"
        / f"{file_info['symbol']}_{file_info['factor_id']}_"
        f"{file_info['factor_name']}_signals.csv"
    )
    if not signal_path.is_file():
        return None

    columns = {"date", "signal_date", *CONFIDENCE_INPUT_COLUMNS}
    signals = pd.read_csv(
        signal_path,
        usecols=lambda col: col in columns,
    )
    date_column = first_existing_column(signals, ("signal_date", "date"))
    confidence_columns = [
        column
        for column in CONFIDENCE_INPUT_COLUMNS
        if column in signals.columns
    ]
    if date_column is None or not confidence_columns:
        return None

    signals = signals[[date_column] + confidence_columns].copy()
    signals = signals.rename(columns={date_column: "date"})
    signals["date"] = pd.to_datetime(signals["date"])
    return signals.drop_duplicates("date", keep="last")


def merge_signal_confidence(daily, file_info):
    signal_confidence = load_signal_confidence(file_info)
    if signal_confidence is None:
        return daily

    daily = daily.merge(
        signal_confidence,
        on="date",
        how="left",
        suffixes=("", "_signal"),
    )
    for column in CONFIDENCE_INPUT_COLUMNS:
        signal_column = f"{column}_signal"
        if signal_column not in daily.columns:
            continue

        if column in daily.columns:
            daily[column] = daily[column].where(
                daily[column].notna(),
                daily[signal_column],
            )
        else:
            daily[column] = daily[signal_column]
        daily = daily.drop(columns=[signal_column])

    return daily


def apply_computed_signal_confidence(daily, confidence_by_date):
    if not confidence_by_date:
        return daily

    daily = daily.copy()
    confidence_values = daily["date"].map(
        lambda value: confidence_by_date.get(pd.Timestamp(value), {})
    )
    computed_level = confidence_values.map(
        lambda item: item.get("confidence_level", UNKNOWN_CONFIDENCE_LEVEL)
    ).map(normalize_confidence_level)
    computed_score = confidence_values.map(
        lambda item: item.get("confidence_score", np.nan)
    )
    computed_signal_days = confidence_values.map(
        lambda item: item.get("confidence_signal_days", np.nan)
    )
    computed_recent_dates = confidence_values.map(
        lambda item: item.get("confidence_recent_dates", "")
    )
    computed_level_available = computed_level.ne(UNKNOWN_CONFIDENCE_LEVEL)
    computed_score = pd.to_numeric(computed_score, errors="coerce")
    computed_signal_days = pd.to_numeric(computed_signal_days, errors="coerce")

    if "confidence_level" in daily.columns:
        existing_level = daily["confidence_level"].map(normalize_confidence_level)
        daily["confidence_level"] = computed_level.where(
            computed_level_available,
            existing_level,
        )
    else:
        daily["confidence_level"] = computed_level

    if "confidence_score" in daily.columns:
        existing_score = pd.to_numeric(
            daily["confidence_score"],
            errors="coerce",
        )
        daily["confidence_score"] = computed_score.where(
            computed_score.notna(),
            existing_score,
        )
    else:
        daily["confidence_score"] = computed_score

    if "confidence_signal_days" in daily.columns:
        existing_signal_days = pd.to_numeric(
            daily["confidence_signal_days"],
            errors="coerce",
        )
        daily["confidence_signal_days"] = computed_signal_days.where(
            computed_signal_days.notna(),
            existing_signal_days,
        )
    else:
        daily["confidence_signal_days"] = computed_signal_days

    if "confidence_recent_dates" in daily.columns:
        existing_recent_dates = (
            daily["confidence_recent_dates"].fillna("").astype(str)
        )
        daily["confidence_recent_dates"] = computed_recent_dates.where(
            computed_recent_dates.ne(""),
            existing_recent_dates,
        )
    else:
        daily["confidence_recent_dates"] = computed_recent_dates

    return daily


def load_factor_daily(file_info, price_column, confidence_by_date=None):
    columns = {
        "date",
        price_column,
        "factor_id",
        "factor_name",
        "signal",
        *CONFIDENCE_INPUT_COLUMNS,
    }
    daily = pd.read_csv(
        file_info["path"],
        usecols=lambda col: col in columns,
    )

    missing_columns = {"date", price_column, "signal"} - set(daily.columns)
    if missing_columns:
        raise ValueError(
            f"{file_info['path']} 缺少字段："
            f"{','.join(sorted(missing_columns))}"
        )

    daily["date"] = pd.to_datetime(daily["date"])
    daily[price_column] = pd.to_numeric(daily[price_column], errors="coerce")
    daily = merge_signal_confidence(daily, file_info)
    daily["signal"] = (
        pd.to_numeric(daily["signal"], errors="coerce")
        .fillna(0)
        .astype(int)
    )

    confidence_level_column = first_existing_column(
        daily,
        CONFIDENCE_LEVEL_SOURCE_COLUMNS,
    )
    confidence_score_column = first_existing_column(
        daily,
        CONFIDENCE_SCORE_SOURCE_COLUMNS,
    )
    if confidence_level_column is not None:
        daily["confidence_level"] = daily[confidence_level_column].map(
            normalize_confidence_level
        )
    elif confidence_score_column is not None:
        daily["confidence_level"] = daily[confidence_score_column].map(
            normalize_confidence_level
        )
    else:
        daily["confidence_level"] = UNKNOWN_CONFIDENCE_LEVEL

    if confidence_score_column is not None:
        daily["confidence_score"] = pd.to_numeric(
            daily[confidence_score_column],
            errors="coerce",
        )
    else:
        daily["confidence_score"] = np.nan

    if "confidence_signal_days" in daily.columns:
        daily["confidence_signal_days"] = pd.to_numeric(
            daily["confidence_signal_days"],
            errors="coerce",
        )
    else:
        daily["confidence_signal_days"] = np.nan

    if "confidence_recent_dates" in daily.columns:
        daily["confidence_recent_dates"] = (
            daily["confidence_recent_dates"].fillna("").astype(str)
        )
    else:
        daily["confidence_recent_dates"] = ""

    daily = apply_computed_signal_confidence(daily, confidence_by_date)
    daily = daily.sort_values("date").reset_index(drop=True)

    daily["symbol"] = file_info["symbol"]
    daily["factor_id"] = file_info["factor_id"]
    daily["factor_name"] = file_info["factor_name"]
    daily["factor_label"] = file_info["factor_label"]
    daily["run_dir"] = file_info["run_dir"]
    return daily


def add_threshold_features(daily, args):
    daily = daily.copy()
    price_return = daily[args.price_column].pct_change()
    daily["past_abs_return_mean"] = (
        price_return
        .abs()
        .rolling(
            window=args.volatility_window,
            min_periods=args.volatility_min_history_days,
        )
        .mean()
        .shift(1)
    )
    daily["dynamic_drawdown_threshold"] = (
        daily["past_abs_return_mean"] * DEFAULT_VOLATILITY_MULTIPLIER
    )
    return daily


def get_threshold(daily, position):
    return daily.loc[position, "dynamic_drawdown_threshold"]


def iter_signal_clusters(daily, cluster_max_gap):
    signal_positions = list(daily.index[daily["signal"] == 1])
    if not signal_positions:
        return []

    clusters = []
    current_positions = [signal_positions[0]]

    for position in signal_positions[1:]:
        previous_position = current_positions[-1]
        no_signal_gap = position - previous_position - 1
        if no_signal_gap <= cluster_max_gap:
            current_positions.append(position)
        else:
            clusters.append(current_positions)
            current_positions = [position]

    clusters.append(current_positions)
    return clusters


def evaluate_cluster(daily, cluster_positions, args, lookahead_days):
    start_position = cluster_positions[0]
    event_position = cluster_positions[-1]
    event_row = daily.loc[event_position]
    event_price = event_row[args.price_column]
    drawdown_threshold = get_threshold(daily, event_position)

    lookahead = daily.iloc[
        event_position + 1:event_position + 1 + lookahead_days
    ].copy()
    future_prices = lookahead[args.price_column]
    is_evaluable = (
        pd.notna(event_price) and
        pd.notna(drawdown_threshold) and
        drawdown_threshold > 0 and
        future_prices.notna().sum() >= lookahead_days
    )

    row = {
        "symbol": event_row["symbol"],
        "factor_id": event_row["factor_id"],
        "factor_name": event_row["factor_name"],
        "factor_label": event_row["factor_label"],
        "cluster_start_date": daily.loc[start_position, "date"].date().isoformat(),
        "event_date": event_row["date"].date().isoformat(),
        "event_price": event_price,
        "cluster_signal_days": len(cluster_positions),
        "confidence_level": normalize_confidence_level(
            event_row.get("confidence_level", UNKNOWN_CONFIDENCE_LEVEL)
        ),
        "confidence_score": event_row.get("confidence_score", np.nan),
        "confidence_signal_days": event_row.get(
            "confidence_signal_days",
            np.nan,
        ),
        "confidence_recent_dates": event_row.get("confidence_recent_dates", ""),
        "lookahead_days": lookahead_days,
        "drawdown_threshold": drawdown_threshold,
        "is_evaluable": bool(is_evaluable),
        "win": pd.NA,
        "observation_end_drawdown": np.nan,
        "max_drawdown": np.nan,
    }

    if not is_evaluable:
        return row

    min_index = future_prices.idxmin()

    future_min_price = daily.loc[min_index, args.price_column]
    future_end_price = future_prices.iloc[-1]
    observation_end_drawdown = max(event_price - future_end_price, 0) / event_price
    max_drawdown = max(event_price - future_min_price, 0) / event_price

    win = observation_end_drawdown >= drawdown_threshold

    row.update({
        "win": bool(win),
        "observation_end_drawdown": observation_end_drawdown,
        "max_drawdown": max_drawdown,
    })
    return row


def plot_signal_clusters(daily, event_rows, args, lookahead_days):
    if not event_rows:
        return None

    events = pd.DataFrame(event_rows)
    if events.empty:
        return None

    events["event_date"] = pd.to_datetime(events["event_date"])
    events["event_price"] = pd.to_numeric(
        events["event_price"],
        errors="coerce",
    )

    symbol = daily["symbol"].iloc[0]
    factor_id = daily["factor_id"].iloc[0]
    factor_name = daily["factor_name"].iloc[0]
    figures_dir = (
        args.output_dir
        / "figures"
        / "signal_clusters"
        / f"{lookahead_days}d"
    )
    figures_dir.mkdir(parents=True, exist_ok=True)
    figure_path = (
        figures_dir
        / f"{symbol}_{factor_id}_{factor_name}_signal_clusters_{lookahead_days}d.png"
    )

    fig, ax = plt.subplots(figsize=(14, 7))
    ax.plot(daily["date"], daily[args.price_column], color="#333333", linewidth=1.1)

    marker_specs = (
        (events["win"].eq(True), "win", "#2f9e44", "^"),
        (
            events["is_evaluable"].eq(True) & ~events["win"].eq(True),
            "miss",
            "#c92a2a",
            "x",
        ),
        (~events["is_evaluable"].eq(True), "not evaluable", "#868e96", "o"),
    )
    for mask, label, color, marker in marker_specs:
        points = events[mask].dropna(subset=["event_date", "event_price"])
        if points.empty:
            continue
        ax.scatter(
            points["event_date"],
            points["event_price"],
            s=34,
            marker=marker,
            color=color,
            label=label,
            zorder=4,
        )

    evaluable_count = int(events["is_evaluable"].sum())
    useful_count = int(events["win"].eq(True).sum())
    win_rate = useful_count / evaluable_count if evaluable_count else np.nan
    title_rate = "NA" if pd.isna(win_rate) else f"{win_rate:.1%}"
    ax.set_title(
        (
            f"{symbol} factor {factor_id}: {factor_name} | "
            f"lookahead {lookahead_days}d | "
            f"useful {useful_count}/{evaluable_count} ({title_rate})"
        ),
        fontsize=12,
    )
    ax.set_xlabel("date")
    ax.set_ylabel(args.price_column)
    ax.grid(True, alpha=0.22)
    if ax.get_legend_handles_labels()[0]:
        ax.legend(loc="best", fontsize=8)
    fig.autofmt_xdate()
    fig.tight_layout()
    fig.savefig(figure_path, dpi=args.plot_dpi)
    plt.close(fig)

    return figure_path


def evaluate_factor_file(file_info, args):
    confidence_by_symbol_date = getattr(args, "confidence_by_symbol_date", {})
    daily = load_factor_daily(
        file_info,
        args.price_column,
        confidence_by_date=confidence_by_symbol_date.get(file_info["symbol"]),
    )
    daily = add_threshold_features(daily, args)
    clusters = iter_signal_clusters(daily, args.cluster_max_gap)
    event_rows = []
    baseline_rows = []
    plot_paths = []

    for lookahead_days in args.lookahead_days_list:
        lookahead_event_rows = list(
            evaluate_cluster(daily, cluster_positions, args, lookahead_days)
            for cluster_positions in clusters
        )
        event_rows.extend(lookahead_event_rows)
        baseline_rows.append(evaluate_baseline(daily, args, lookahead_days))

        if not args.skip_plots:
            plot_path = plot_signal_clusters(
                daily=daily,
                event_rows=lookahead_event_rows,
                args=args,
                lookahead_days=lookahead_days,
            )
            if plot_path is not None:
                plot_paths.append(plot_path)

    return event_rows, baseline_rows, plot_paths


def evaluate_baseline(daily, args, lookahead_days):
    prices = daily[args.price_column]
    future_price_frames = [
        prices.shift(-offset)
        for offset in range(1, lookahead_days + 1)
    ]
    future_prices = pd.concat(future_price_frames, axis=1)
    future_counts = future_prices.notna().sum(axis=1)

    future_end_price = prices.shift(-lookahead_days)
    observation_end_drawdown = (1 - future_end_price / prices).clip(lower=0)
    drawdown_threshold = daily["dynamic_drawdown_threshold"]

    evaluable = (
        prices.notna() &
        (future_counts >= lookahead_days) &
        future_end_price.notna() &
        drawdown_threshold.notna() &
        (drawdown_threshold > 0)
    )
    win = observation_end_drawdown >= drawdown_threshold

    evaluable_count = int(evaluable.sum())
    return {
        "symbol": daily["symbol"].iloc[0],
        "factor_id": daily["factor_id"].iloc[0],
        "factor_name": daily["factor_name"].iloc[0],
        "lookahead_days": lookahead_days,
        "baseline_events": evaluable_count,
        "baseline_win_events": int(win[evaluable].sum()),
    }


def summarize_events(events, baselines):
    if events.empty:
        return pd.DataFrame()

    if "confidence_level" not in events.columns:
        events = events.copy()
        events["confidence_level"] = UNKNOWN_CONFIDENCE_LEVEL

    confidence_levels = ordered_confidence_levels(
        events["confidence_level"]
        .dropna()
        .astype(str)
        .unique()
    )
    rows = []

    for row_type, group_columns, fixed_values in SUMMARY_GROUP_SPECS:
        for values, group in iter_group_values(events, group_columns):
            values.update(fixed_values)
            rows.append(build_summary_row(values, group, row_type))

    summary = pd.DataFrame(rows)
    baseline_summary = summarize_baselines(baselines, confidence_levels)
    summary = summary.merge(
        baseline_summary,
        on=["row_type"] + SUMMARY_GROUP_COLUMNS,
        how="left",
    )
    summary["win_rate_lift"] = (
        summary["win_rate"] - summary["baseline_win_rate"]
    )
    summary["_confidence_sort"] = summary["confidence_level"].map(
        confidence_level_sort_value
    )
    summary = summary.sort_values(
        [
            "row_type",
            "symbol",
            "factor_id",
            "lookahead_days",
            "_confidence_sort",
        ]
    )
    return summary.drop(columns=["_confidence_sort"])


def iter_group_values(frame, group_columns):
    for key, group in frame.groupby(group_columns, dropna=False):
        if len(group_columns) == 1 and not isinstance(key, tuple):
            key = (key,)
        elif not isinstance(key, tuple):
            key = (key,)
        yield dict(zip(group_columns, key)), group


def summarize_baselines(baselines, confidence_levels):
    baseline_columns = [
        "baseline_events",
        "baseline_win_events",
        "baseline_win_rate",
    ]
    if baselines.empty:
        return pd.DataFrame(
            columns=["row_type"] + SUMMARY_GROUP_COLUMNS + baseline_columns
        )

    rows = []

    for row_type, group_columns, fixed_values in SUMMARY_GROUP_SPECS:
        baseline_group_columns = [
            column
            for column in group_columns
            if column != "confidence_level"
        ]
        confidence_values = (
            confidence_levels
            if "confidence_level" in group_columns
            else [ALL_CONFIDENCE_LEVEL]
        )
        for values, group in iter_group_values(baselines, baseline_group_columns):
            values.update(fixed_values)
            for confidence_level in confidence_values:
                values = values.copy()
                values["confidence_level"] = confidence_level
                rows.append(build_baseline_summary_row(values, group, row_type))

    return pd.DataFrame(rows)


def build_baseline_summary_row(values, group, row_type):
    baseline_events = int(group["baseline_events"].sum())
    baseline_win_events = int(group["baseline_win_events"].sum())

    if baseline_events:
        baseline_win_rate = baseline_win_events / baseline_events
    else:
        baseline_win_rate = np.nan

    row = {column: values.get(column) for column in SUMMARY_GROUP_COLUMNS}
    row.update({
        "row_type": row_type,
        "baseline_events": baseline_events,
        "baseline_win_events": baseline_win_events,
        "baseline_win_rate": baseline_win_rate,
    })
    return row


def build_summary_row(values, group, row_type):
    evaluable = group[group["is_evaluable"]].copy()
    event_count = len(group)
    evaluable_count = len(evaluable)

    if evaluable_count:
        win_count = int(evaluable["win"].sum())
        win_rate = win_count / evaluable_count
    else:
        win_count = 0
        win_rate = np.nan

    row = {column: values.get(column) for column in SUMMARY_GROUP_COLUMNS}
    row.update({
        "row_type": row_type,
        "cluster_events": event_count,
        "evaluable_events": evaluable_count,
        "win_events": win_count,
        "win_rate": win_rate,
        "mean_drawdown_threshold": evaluable["drawdown_threshold"].mean(),
        "strategy_observation_end_drawdown": (
            evaluable["observation_end_drawdown"].max()
        ),
        "strategy_observation_max_drawdown": evaluable["max_drawdown"].max(),
    })
    return row


def build_output_events(events):
    output = events.copy()
    output = output.rename(
        columns={
            "cluster_start_date": "signal_cluster_start_date",
            "event_date": "signal_date",
            "event_price": "signal_close",
            "drawdown_threshold": "threshold",
            "observation_end_drawdown": "strategy_observation_end_drawdown",
            "max_drawdown": "strategy_observation_max_drawdown",
            "win": "strategy_win",
        }
    )
    columns = [
        "symbol",
        "factor_id",
        "factor_name",
        "lookahead_days",
        "signal_cluster_start_date",
        "signal_date",
        "signal_close",
        "cluster_signal_days",
        "confidence_level",
        "confidence_score",
        "confidence_signal_days",
        "confidence_recent_dates",
        "threshold",
        "strategy_observation_end_drawdown",
        "strategy_observation_max_drawdown",
        "strategy_win",
        "is_evaluable",
    ]
    return output[[col for col in columns if col in output.columns]]


def build_output_summary(summary):
    output = summary.copy()
    output = output.rename(
        columns={
            "cluster_events": "signal_clusters",
            "evaluable_events": "strategy_samples",
            "win_events": "strategy_wins",
            "win_rate": "strategy_win_rate",
            "win_rate_lift": "win_rate_diff",
            "mean_drawdown_threshold": "threshold",
            "baseline_events": "baseline_samples",
            "baseline_win_events": "baseline_wins",
        }
    )
    return output[SUMMARY_OUTPUT_COLUMNS]


def save_outputs(events, summary, output_dir):
    tables_dir = output_dir / "tables"
    events_dir = tables_dir / "events"
    summary_dir = tables_dir / "summary"
    events_dir.mkdir(parents=True, exist_ok=True)
    summary_dir.mkdir(parents=True, exist_ok=True)

    events_path = events_dir / "signal_cluster_events.csv"
    summary_output = build_output_summary(summary)
    summary_paths = {}

    build_output_events(events).to_csv(events_path, index=False)
    for old_summary_path in summary_dir.glob("*.csv"):
        old_summary_path.unlink()

    for table_name, row_type, filename in SUMMARY_TABLE_SPECS:
        table_path = summary_dir / filename
        summary_table = summary_output[
            summary["row_type"].eq(row_type).to_numpy()
        ].copy()
        summary_table.to_csv(table_path, index=False)
        summary_paths[table_name] = table_path

    return {
        "events": events_path,
        "summary": summary_paths,
    }


def main():
    parser = argparse.ArgumentParser(
        description=(
            "用信号簇最后一天评估信号胜率："
            "未来窗口最后一天相对事件日的回撤达到"
            "前 20 日平均波动率一倍即算胜，"
            "并按置信度分档输出胜率。"
        )
    )
    parser.add_argument(
        "--runs-dir",
        type=Path,
        default=DEFAULT_RUNS_DIR,
        help=f"因子结果根目录，默认：{DEFAULT_RUNS_DIR}",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=DEFAULT_OUTPUT_DIR,
        help=f"评估结果输出目录，默认：{DEFAULT_OUTPUT_DIR}",
    )
    parser.add_argument(
        "--factor-ids",
        default=DEFAULT_FACTOR_IDS,
        help=f"要评估的因子 ID，逗号分隔，默认：{DEFAULT_FACTOR_IDS}",
    )
    parser.add_argument(
        "--symbols",
        "--symbol",
        dest="symbols",
        help="只评估指定品种，多个品种用逗号分隔，例如：JD,CU。",
    )
    parser.add_argument(
        "--price-column",
        default="close",
        help="用于评估的价格字段，默认：close。",
    )
    parser.add_argument(
        "--lookahead-days",
        type=int,
        default=None,
        help=(
            "只评估一个观察窗口，例如：10。"
            "如果没有设置该参数或 --lookahead-days-list，默认评估 3,5,10。"
        ),
    )
    parser.add_argument(
        "--lookahead-days-list",
        help="一次评估多个观察窗口，逗号分隔，例如：3,5,10。",
    )
    parser.add_argument(
        "--volatility-window",
        type=int,
        default=DEFAULT_VOLATILITY_WINDOW,
        help=(
            "计算前 N 日平均绝对日收益率，默认：20；"
            "胜负阈值固定为该均值的 1 倍。"
        ),
    )
    parser.add_argument(
        "--volatility-min-history-days",
        type=int,
        default=DEFAULT_VOLATILITY_MIN_HISTORY_DAYS,
        help="计算平均波动率时最少需要多少个历史日收益，默认：20。",
    )
    parser.add_argument(
        "--cluster-max-gap",
        type=int,
        default=DEFAULT_CLUSTER_MAX_GAP,
        help=(
            "两个信号之间最多允许隔多少个无信号交易日仍算同一簇，"
            f"默认：{DEFAULT_CLUSTER_MAX_GAP}；设为 -1 表示每个信号单独计票。"
        ),
    )
    parser.add_argument(
        "--confidence-window-days",
        type=int,
        default=DEFAULT_CONFIDENCE_WINDOW_DAYS,
        help=(
            "按同一品种所有选中因子的信号日期计算置信度时，"
            "回看多少个交易日，默认："
            f"{DEFAULT_CONFIDENCE_WINDOW_DAYS}。"
        ),
    )
    parser.add_argument(
        "--skip-plots",
        action="store_true",
        help="只输出胜率表，不生成价格信号图。",
    )
    parser.add_argument(
        "--plot-dpi",
        type=int,
        default=160,
        help="输出 PNG 图片的 dpi，默认：160。",
    )

    args = parser.parse_args()
    if args.lookahead_days_list:
        args.lookahead_days_list = parse_int_list(args.lookahead_days_list)
    elif args.lookahead_days is not None:
        args.lookahead_days_list = [args.lookahead_days]
    else:
        args.lookahead_days_list = parse_int_list(DEFAULT_LOOKAHEAD_DAYS_LIST)
    args.lookahead_days_list = sorted(set(args.lookahead_days_list))

    if any(value <= 0 for value in args.lookahead_days_list):
        raise ValueError("lookahead-days 必须为正整数。")
    if args.volatility_window <= 0:
        raise ValueError("volatility-window 必须为正整数。")
    if args.volatility_min_history_days <= 0:
        raise ValueError("volatility-min-history-days 必须为正整数。")
    if args.confidence_window_days <= 0:
        raise ValueError("confidence-window-days 必须为正整数。")

    runs_dir = args.runs_dir.resolve()
    output_dir = args.output_dir.resolve()
    args.output_dir = output_dir
    symbols = parse_csv_list(args.symbols)
    factor_ids = parse_factor_ids(args.factor_ids)

    factor_files = discover_factor_files(
        runs_dir=runs_dir,
        symbols=symbols,
        factor_ids=factor_ids,
    )
    args.confidence_by_symbol_date = build_confidence_by_symbol_date(
        factor_files=factor_files,
        lookback_days=args.confidence_window_days,
    )

    rows = []
    baseline_rows = []
    plot_paths = []
    errors = []
    for file_info in factor_files:
        try:
            event_rows, baseline_row, factor_plot_paths = evaluate_factor_file(
                file_info,
                args,
            )
            rows.extend(event_rows)
            baseline_rows.extend(baseline_row)
            plot_paths.extend(factor_plot_paths)
        except Exception as exc:
            errors.append((file_info["path"], exc))

    if not rows:
        error_text = "\n".join(f"- {path}: {exc}" for path, exc in errors)
        raise RuntimeError(f"没有可评估的信号事件。\n{error_text}")

    events = pd.DataFrame(rows)
    baselines = pd.DataFrame(baseline_rows)
    summary = summarize_events(events, baselines)
    output_paths = save_outputs(events, summary, output_dir)

    aggregate = (
        summary[summary["row_type"] == "all_symbols_all_factors"]
        .sort_values("lookahead_days")
    )
    print("信号胜率评估完成。", flush=True)
    print(f"读取因子文件数量：{len(factor_files)}", flush=True)
    for _, row in aggregate.iterrows():
        print(
            "观察窗口："
            f"{int(row['lookahead_days'])} 日；"
            f"信号簇事件：{int(row['cluster_events'])}；"
            f"可评价：{int(row['evaluable_events'])}；"
            f"阈值：{row['mean_drawdown_threshold']:.2%}；"
            f"策略胜率：{row['win_rate']:.2%}；"
            f"基准胜率：{row['baseline_win_rate']:.2%}；"
            f"胜率差：{row['win_rate_lift']:.2%}；"
            f"策略观察窗口末日回撤："
            f"{row['strategy_observation_end_drawdown']:.2%}；"
            f"策略观察窗口最大回撤："
            f"{row['strategy_observation_max_drawdown']:.2%}",
            flush=True,
        )

    confidence_aggregate = (
        summary[summary["row_type"] == "all_symbols_all_factors_confidence"]
        .copy()
    )
    if not confidence_aggregate.empty:
        confidence_aggregate["_confidence_sort"] = (
            confidence_aggregate["confidence_level"]
            .map(confidence_level_sort_value)
        )
        confidence_aggregate = confidence_aggregate.sort_values(
            ["lookahead_days", "_confidence_sort"]
        )
        print("按置信度汇总：", flush=True)
        for _, row in confidence_aggregate.iterrows():
            print(
                "观察窗口："
                f"{int(row['lookahead_days'])} 日；"
                f"置信度：{row['confidence_level']}；"
                f"信号簇事件：{int(row['cluster_events'])}；"
                f"可评价：{int(row['evaluable_events'])}；"
                f"策略胜率：{row['win_rate']:.2%}；"
                f"基准胜率：{row['baseline_win_rate']:.2%}；"
                f"胜率差：{row['win_rate_lift']:.2%}",
                flush=True,
            )
    print(f"事件明细：{output_paths['events']}", flush=True)
    print(f"总览汇总：{output_paths['summary']['overall']}", flush=True)
    print(f"因子汇总：{output_paths['summary']['factor']}", flush=True)
    print(
        f"总览-置信度汇总：{output_paths['summary']['overall_confidence']}",
        flush=True,
    )
    print(
        f"因子-置信度汇总：{output_paths['summary']['factor_confidence']}",
        flush=True,
    )
    print(f"置信度回看窗口：{args.confidence_window_days} 个交易日", flush=True)
    if not args.skip_plots:
        print(f"信号图数量：{len(plot_paths)}", flush=True)
        print(
            f"信号图目录：{output_dir / 'figures' / 'signal_clusters'}",
            flush=True,
        )

    if errors:
        print("\n以下文件读取失败，已跳过：", flush=True)
        for path, exc in errors:
            print(f"- {path}: {exc}", flush=True)


if __name__ == "__main__":
    main()