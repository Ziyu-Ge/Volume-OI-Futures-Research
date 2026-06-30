import os
import re
import sys

project_root = os.path.dirname(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
)
code_dir = os.path.join(project_root, "code")

if code_dir not in sys.path:
    sys.path.insert(0, code_dir)

from config import SYMBOL

os.environ.setdefault(
    "MPLCONFIGDIR",
    os.path.join(project_root, ".matplotlib")
)
os.environ.setdefault(
    "XDG_CACHE_HOME",
    os.path.join(project_root, ".cache")
)

import numpy as np
import pandas as pd
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt


MAD_SCALE = 1.4826
MAD_EPSILON = 1e-12
RESULTS_OUTPUT_ENV_VAR = "RESULTS_OUTPUT_DIR"


def get_results_dir(results_dir=None):
    if results_dir is None:
        results_dir = os.environ.get(RESULTS_OUTPUT_ENV_VAR)

    if results_dir is None:
        results_dir = os.path.join(project_root, "results")

    return os.path.abspath(os.path.expanduser(str(results_dir)))


def parse_factor_script_metadata(file_path):
    stem = os.path.splitext(os.path.basename(file_path))[0]
    match = re.match(r"^(\d+)_?(.+)$", stem)
    if match is None:
        raise ValueError(f"factor script filename must start with an id: {stem}")

    return match.group(1), match.group(2)


def load_daily(symbol, results_dir=None):
    tables_dir = os.path.join(get_results_dir(results_dir), "tables")
    daily_filename = f"{symbol}_daily.csv"
    daily_input_candidates = [
        os.path.join(tables_dir, "daily", daily_filename),
        os.path.join(tables_dir, daily_filename),
    ]

    default_tables_dir = os.path.join(project_root, "results", "tables")
    if os.path.abspath(tables_dir) != os.path.abspath(default_tables_dir):
        daily_input_candidates.extend([
            os.path.join(default_tables_dir, "daily", daily_filename),
            os.path.join(default_tables_dir, daily_filename),
        ])

    daily_input_path = next(
        (
            candidate_path
            for candidate_path in daily_input_candidates
            if os.path.exists(candidate_path)
        ),
        daily_input_candidates[0],
    )

    daily = pd.read_csv(daily_input_path)

    daily["date"] = pd.to_datetime(daily["date"])
    daily = daily.sort_values("date").reset_index(drop=True)

    return daily


def past_rank(series, window, min_history_days):
    ranks = []

    for i, current_value in enumerate(series):
        history = series.iloc[max(0, i - window):i].dropna()

        if len(history) < min_history_days or pd.isna(current_value):
            ranks.append(np.nan)
            continue

        ranks.append((history <= current_value).mean())

    return pd.Series(ranks, index=series.index)


def mad_score(
    series,
    window,
    min_history_days,
    mad_scale=MAD_SCALE,
    mad_epsilon=MAD_EPSILON,
):
    median_past = (
        series
        .rolling(window=window, min_periods=min_history_days)
        .median()
        .shift(1)
    )

    mad_values = []

    for i in range(len(series)):
        history = series.iloc[max(0, i - window):i].dropna()

        if len(history) < min_history_days:
            mad_values.append(np.nan)
            continue

        history_median = history.median()
        history_mad = (history - history_median).abs().median()
        mad_values.append(history_mad)

    mad_past = pd.Series(mad_values, index=series.index)
    score = (
        (series - median_past) /
        (mad_scale * mad_past + mad_epsilon)
    )
    score[mad_past <= 0] = np.nan

    return median_past, mad_past, score


def past_mad_score(
    series,
    window,
    min_history_days,
    mad_scale=MAD_SCALE,
    mad_epsilon=MAD_EPSILON,
):
    median_values = []
    mad_values = []
    score_values = []

    for i, current_value in enumerate(series):
        history = series.iloc[max(0, i - window):i].dropna()

        if len(history) < min_history_days or pd.isna(current_value):
            median_values.append(np.nan)
            mad_values.append(np.nan)
            score_values.append(np.nan)
            continue

        history_median = history.median()
        history_mad = (history - history_median).abs().median()

        if abs(history_mad) <= mad_epsilon:
            median_values.append(np.nan)
            mad_values.append(np.nan)
            score_values.append(np.nan)
            continue

        median_values.append(history_median)
        mad_values.append(history_mad)
        score_values.append(
            (current_value - history_median) / (mad_scale * history_mad)
        )

    return (
        pd.Series(median_values, index=series.index),
        pd.Series(mad_values, index=series.index),
        pd.Series(score_values, index=series.index),
    )


def positive_part(series):
    return series.clip(lower=0).fillna(0)


def add_price_ma_features(
    daily,
    ma_gap_threshold,
    short_window=5,
    mid_window=10,
    long_window=20,
    trend_window=120,
):
    daily = daily.copy()

    for window in [short_window, mid_window, long_window, trend_window]:
        daily[f"ma{window}"] = (
            daily["close"]
            .rolling(window=window, min_periods=window)
            .mean()
        )

    daily["is_ma_bullish"] = (
        (daily[f"ma{short_window}"] > daily[f"ma{mid_window}"]) &
        (daily[f"ma{mid_window}"] > daily[f"ma{long_window}"])
    ).astype(int)

    daily["ma5_ma10_bias"] = (
        daily[f"ma{short_window}"] / daily[f"ma{mid_window}"] - 1
    )
    daily["ma10_ma20_bias"] = (
        daily[f"ma{mid_window}"] / daily[f"ma{long_window}"] - 1
    )
    daily["close_ma20_bias"] = (
        daily["close"] / daily[f"ma{long_window}"] - 1
    )
    daily["ma_bull_stack_filter"] = (
        (daily[f"ma{short_window}"] > daily[f"ma{mid_window}"]) &
        (daily[f"ma{mid_window}"] > daily[f"ma{long_window}"]) &
        (daily["ma5_ma10_bias"] >= ma_gap_threshold) &
        (daily["ma10_ma20_bias"] >= ma_gap_threshold)
    )

    return daily


def add_volume_price_features(
    daily,
    price_rank_window=20,
    mad_window=10,
    min_rank_days=8,
    min_mad_days=5,
):
    daily = daily.copy()
    daily["daily_return"] = daily["close"].pct_change()

    for window in [1, 3, 5, 10]:
        daily[f"ret_{window}"] = daily["close"].pct_change(window)

    safe_open_interest = daily["open_interest"].where(
        daily["open_interest"] > 0,
        np.nan
    )
    safe_volume = daily["volume"].where(daily["volume"] > 0, np.nan)

    daily["log_open_interest"] = np.log(safe_open_interest)
    daily["log_volume"] = np.log(safe_volume)

    for window in [1, 3, 5, 10]:
        daily[f"oi_ret_{window}"] = (
            daily["log_open_interest"] -
            daily["log_open_interest"].shift(window)
        )
        daily[f"volume_ret_{window}"] = (
            daily["log_volume"] -
            daily["log_volume"].shift(window)
        )

    daily["range_pct"] = (
        (daily["high"] - daily["low"]) /
        daily["close"].replace(0, np.nan)
    )

    daily["close_location"] = (
        (daily["close"] - daily["low"]) /
        (daily["high"] - daily["low"]).replace(0, np.nan)
    )

    daily["volume_to_open_interest"] = safe_volume / safe_open_interest
    daily["price_efficiency_5"] = (
        daily["ret_5"].abs() /
        daily["volume_to_open_interest"].replace(0, np.nan)
    )

    daily["close_rank_20"] = past_rank(
        daily["close"],
        window=price_rank_window,
        min_history_days=min_rank_days,
    )
    daily["close_rank_60"] = past_rank(
        daily["close"],
        window=60,
        min_history_days=20,
    )
    daily["price_efficiency_rank_20"] = past_rank(
        daily["price_efficiency_5"],
        window=price_rank_window,
        min_history_days=min_rank_days,
    )
    daily["price_efficiency_low_rank_20"] = (
        1 - daily["price_efficiency_rank_20"]
    )

    (
        daily["log_volume_median_past"],
        daily["log_volume_mad_past"],
        daily["volume_mad_score"],
    ) = mad_score(
        daily["log_volume"],
        window=mad_window,
        min_history_days=min_mad_days,
    )

    (
        daily["range_pct_median_past"],
        daily["range_pct_mad_past"],
        daily["range_mad_score"],
    ) = mad_score(
        daily["range_pct"],
        window=mad_window,
        min_history_days=min_mad_days,
    )

    (
        daily["speculation_median_past"],
        daily["speculation_mad_past"],
        daily["speculation_mad_score"],
    ) = mad_score(
        daily["speculation"],
        window=mad_window,
        min_history_days=min_mad_days,
    )

    (
        daily["log_open_interest_median_past"],
        daily["log_open_interest_mad_past"],
        daily["open_interest_mad_score"],
    ) = mad_score(
        daily["log_open_interest"],
        window=mad_window,
        min_history_days=min_mad_days,
    )

    return daily


def build_signal_table(daily, factor_id, factor_name, feature_columns):
    base_columns = [
        "factor_id",
        "factor_name",
        "signal_date",
        "signal_close",
        "factor_value",
    ]
    output_columns = base_columns.copy()

    for col in feature_columns:
        if col in daily.columns and col not in output_columns:
            output_columns.append(col)

    signal_points = daily[daily["signal"] == 1].copy()
    if len(signal_points) == 0:
        return pd.DataFrame(columns=output_columns)

    signal_table = signal_points.rename(
        columns={
            "date": "signal_date",
            "close": "signal_close",
        }
    )

    return signal_table[output_columns].copy()


def build_summary_table(daily, factor_id, factor_name, feature_columns):
    signal_points = daily[daily["signal"] == 1].copy()

    if len(signal_points) > 0:
        first_signal_date = signal_points["date"].min()
        first_signal_close = signal_points.loc[
            signal_points["date"].idxmin(),
            "close"
        ]
    else:
        first_signal_date = pd.NaT
        first_signal_close = np.nan

    row = {
        "factor_id": factor_id,
        "factor_name": factor_name,
        "total_days": len(daily),
        "valid_factor_days": int(daily["factor_value"].notna().sum()),
        "signal_days": int(daily["signal"].sum()),
        "signal_ratio": daily["signal"].mean(),
        "first_signal_date": first_signal_date,
        "first_signal_close": first_signal_close,
        "mean_factor_value": daily["factor_value"].mean(),
        "max_factor_value": daily["factor_value"].max(),
        "min_factor_value": daily["factor_value"].min(),
    }

    for col in feature_columns:
        if col not in daily.columns:
            continue

        if not pd.api.types.is_numeric_dtype(daily[col]):
            continue

        row[f"mean_{col}"] = daily[col].mean()

    return pd.DataFrame([row])


def save_factor_outputs(
    daily,
    symbol,
    factor_id,
    factor_name,
    factor_value_column,
    signal_column,
    feature_columns,
    figure_feature_columns=None,
    results_dir=None,
):
    daily = daily.copy()

    daily["factor_id"] = factor_id
    daily["factor_name"] = factor_name
    daily["factor_value"] = daily[factor_value_column]
    daily["signal"] = daily[signal_column].fillna(0).astype(int)

    output_root = get_results_dir(results_dir)
    tables_dir = os.path.join(output_root, "tables")
    factor_figures_dir = os.path.join(output_root, "figures", "factors")

    factor_output_path = os.path.join(
        tables_dir,
        "factors",
        f"{symbol}_{factor_id}_{factor_name}.csv"
    )
    signal_output_path = os.path.join(
        tables_dir,
        "signals",
        f"{symbol}_{factor_id}_{factor_name}_signals.csv"
    )
    summary_output_path = os.path.join(
        tables_dir,
        "summary",
        f"{symbol}_{factor_id}_{factor_name}_summary.csv"
    )
    price_figure_path = os.path.join(
        factor_figures_dir,
        f"{symbol}_{factor_id}_{factor_name}_signal_on_price.png"
    )
    factor_figure_path = os.path.join(
        factor_figures_dir,
        f"{symbol}_{factor_id}_{factor_name}_factor_value.png"
    )

    base_columns = [
        "date",
        "open",
        "close",
        "high",
        "low",
        "volume",
        "total_turnover",
        "open_interest",
        "speculation",
        "threshold",
        "factor_id",
        "factor_name",
        "factor_value",
    ]

    output_columns = []

    for col in base_columns + feature_columns + [
        "signal",
    ]:
        if col in daily.columns and col not in output_columns:
            output_columns.append(col)

    factor_daily = daily[output_columns].copy()
    signal_table = build_signal_table(
        daily,
        factor_id,
        factor_name,
        feature_columns,
    )
    summary_table = build_summary_table(
        daily,
        factor_id,
        factor_name,
        feature_columns,
    )

    os.makedirs(os.path.dirname(factor_output_path), exist_ok=True)
    os.makedirs(os.path.dirname(signal_output_path), exist_ok=True)
    os.makedirs(os.path.dirname(summary_output_path), exist_ok=True)
    os.makedirs(os.path.dirname(price_figure_path), exist_ok=True)

    factor_daily.to_csv(factor_output_path, index=False)
    signal_table.to_csv(signal_output_path, index=False)
    summary_table.to_csv(summary_output_path, index=False)

    plt.figure(figsize=(12, 6))
    plt.plot(daily["date"], daily["close"], label="close")

    signal_points = daily[daily["signal"] == 1]

    plt.scatter(
        signal_points["date"],
        signal_points["close"],
        s=22,
        color="red",
        label="signal",
    )
    plt.title(f"{symbol} Factor {factor_id}: {factor_name}")
    plt.xlabel("Date")
    plt.ylabel("Close Price")
    plt.legend()
    plt.tight_layout()
    plt.savefig(price_figure_path, dpi=300)
    plt.close()

    plt.figure(figsize=(12, 5))
    plt.plot(
        daily["date"],
        daily["factor_value"],
        label="factor value",
    )

    if figure_feature_columns is not None:
        for col in figure_feature_columns:
            if col not in daily.columns:
                continue

            plt.plot(daily["date"], daily[col], label=col, alpha=0.75)

    plt.title(f"{symbol} Factor {factor_id}: Factor Value")
    plt.xlabel("Date")
    plt.ylabel("Value")
    plt.legend()
    plt.tight_layout()
    plt.savefig(factor_figure_path, dpi=300)
    plt.close()

    print(f"factor {factor_id}: {factor_name} complete.")
    print(f"factor daily table: {factor_output_path}")
    print(f"signal table: {signal_output_path}")
    print(f"summary table: {summary_output_path}")
    print(f"price figure: {price_figure_path}")
    print(f"factor figure: {factor_figure_path}")
    print(
        "signal days:",
        int(daily["signal"].sum()),
    )

    return {
        "factor_daily": factor_daily,
        "signal_table": signal_table,
        "summary_table": summary_table,
        "factor_output_path": factor_output_path,
        "signal_output_path": signal_output_path,
        "summary_output_path": summary_output_path,
        "price_figure_path": price_figure_path,
        "factor_figure_path": factor_figure_path,
    }
