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
FACTOR_DATA_FREQUENCY_ENV = "FACTOR_DATA_FREQUENCY"
INTRADAY_FREQUENCIES = {"15min", "15m", "15"}


def parse_factor_script_metadata(file_path):
    stem = os.path.splitext(os.path.basename(file_path))[0]
    match = re.match(r"^(\d+)_?(.+)$", stem)
    if match is None:
        raise ValueError(f"factor script filename must start with an id: {stem}")

    return match.group(1), match.group(2)


def get_factor_data_frequency():
    return os.environ.get(FACTOR_DATA_FREQUENCY_ENV, "daily").lower()


def is_intraday_frequency():
    return get_factor_data_frequency() in INTRADAY_FREQUENCIES


def load_daily(symbol):
    if is_intraday_frequency():
        return load_15min(symbol)

    tables_dir = os.path.join(project_root, "results", "tables")

    daily_input_path = os.path.join(
        tables_dir,
        "daily",
        f"{symbol}_daily.csv"
    )

    if not os.path.exists(daily_input_path):
        daily_input_path = os.path.join(
            tables_dir,
            f"{symbol}_daily.csv"
        )

    daily = pd.read_csv(daily_input_path)

    daily["date"] = pd.to_datetime(daily["date"])
    daily = daily.sort_values("date").reset_index(drop=True)

    return daily


def load_15min(symbol):
    input_path = os.path.join(
        project_root,
        "results",
        "tables",
        "15min",
        f"{symbol}_15min.csv",
    )
    if not os.path.exists(input_path):
        raise FileNotFoundError(
            f"15min table not found: {input_path}. "
            "Please run code/01_prepare_data.py first."
        )

    data = pd.read_csv(input_path)

    datetime_columns = [
        "date",
        "trading_day",
        "trading_day_start_datetime",
        "bar_start_datetime",
        "bar_end_datetime",
    ]
    for column in datetime_columns:
        if column in data.columns:
            data[column] = pd.to_datetime(data[column])

    data = data.sort_values("date").reset_index(drop=True)
    return data


def is_intraday_data(data):
    required_columns = {
        "trading_day_id",
        "bar_index_in_trading_day",
    }
    return required_columns.issubset(data.columns)


def _bar_time_key(data):
    if "bar_end_datetime" in data.columns:
        return pd.to_datetime(data["bar_end_datetime"]).dt.strftime("%H:%M:%S")

    return pd.to_datetime(data["date"]).dt.strftime("%H:%M:%S")


def _same_bar_time_transform(data, column, transform_func):
    result = pd.Series(np.nan, index=data.index, dtype="float64")
    bar_time = _bar_time_key(data)

    for _, index in data.groupby(bar_time, sort=False).groups.items():
        group_series = data.loc[index, column]
        result.loc[index] = transform_func(group_series).to_numpy()

    return result


def trading_day_shift(data, column, days):
    if not is_intraday_data(data):
        return data[column].shift(days)

    return _same_bar_time_transform(
        data,
        column,
        lambda series: series.shift(days),
    )


def trading_day_pct_change(data, column, days):
    previous = trading_day_shift(data, column, days)
    return data[column] / previous - 1


def trading_day_rolling_mean(
    data,
    column,
    window_days,
    min_history_days=None,
):
    if min_history_days is None:
        min_history_days = window_days

    if not is_intraday_data(data):
        return (
            data[column]
            .rolling(window_days, min_periods=min_history_days)
            .mean()
        )

    return _same_bar_time_transform(
        data,
        column,
        lambda series: (
            series
            .rolling(window_days, min_periods=min_history_days)
            .mean()
        ),
    )


def trading_day_past_rank(data, column, window_days, min_history_days):
    if not is_intraday_data(data):
        return past_rank(
            data[column],
            window=window_days,
            min_history_days=min_history_days,
        )

    return _same_bar_time_transform(
        data,
        column,
        lambda series: past_rank(
            series,
            window=window_days,
            min_history_days=min_history_days,
        ),
    )


def trading_day_mad_score(
    data,
    column,
    window_days,
    min_history_days,
    mad_scale=MAD_SCALE,
    mad_epsilon=MAD_EPSILON,
):
    if not is_intraday_data(data):
        return mad_score(
            data[column],
            window=window_days,
            min_history_days=min_history_days,
            mad_scale=mad_scale,
            mad_epsilon=mad_epsilon,
        )

    median_past = pd.Series(np.nan, index=data.index, dtype="float64")
    mad_past = pd.Series(np.nan, index=data.index, dtype="float64")
    score = pd.Series(np.nan, index=data.index, dtype="float64")
    bar_time = _bar_time_key(data)

    for _, index in data.groupby(bar_time, sort=False).groups.items():
        group_median, group_mad, group_score = mad_score(
            data.loc[index, column],
            window=window_days,
            min_history_days=min_history_days,
            mad_scale=mad_scale,
            mad_epsilon=mad_epsilon,
        )
        median_past.loc[index] = group_median.to_numpy()
        mad_past.loc[index] = group_mad.to_numpy()
        score.loc[index] = group_score.to_numpy()

    return median_past, mad_past, score


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


def add_volume_price_features(
    daily,
    price_rank_window=20,
    mad_window=10,
    min_rank_days=8,
    min_mad_days=5,
):
    daily = daily.copy()
    daily["daily_return"] = trading_day_pct_change(daily, "close", 1)

    for window in [1, 3, 5, 10]:
        daily[f"ret_{window}"] = trading_day_pct_change(
            daily,
            "close",
            window,
        )

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
            trading_day_shift(daily, "log_open_interest", window)
        )
        daily[f"volume_ret_{window}"] = (
            daily["log_volume"] -
            trading_day_shift(daily, "log_volume", window)
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

    daily["close_rank_20"] = trading_day_past_rank(
        daily,
        "close",
        window_days=price_rank_window,
        min_history_days=min_rank_days,
    )
    daily["close_rank_60"] = trading_day_past_rank(
        daily,
        "close",
        window_days=60,
        min_history_days=20,
    )
    daily["price_efficiency_rank_20"] = trading_day_past_rank(
        daily,
        "price_efficiency_5",
        window_days=price_rank_window,
        min_history_days=min_rank_days,
    )
    daily["price_efficiency_low_rank_20"] = (
        1 - daily["price_efficiency_rank_20"]
    )

    (
        daily["log_volume_median_past"],
        daily["log_volume_mad_past"],
        daily["volume_mad_score"],
    ) = trading_day_mad_score(
        daily,
        "log_volume",
        window_days=mad_window,
        min_history_days=min_mad_days,
    )

    (
        daily["range_pct_median_past"],
        daily["range_pct_mad_past"],
        daily["range_mad_score"],
    ) = trading_day_mad_score(
        daily,
        "range_pct",
        window_days=mad_window,
        min_history_days=min_mad_days,
    )

    (
        daily["speculation_median_past"],
        daily["speculation_mad_past"],
        daily["speculation_mad_score"],
    ) = trading_day_mad_score(
        daily,
        "speculation",
        window_days=mad_window,
        min_history_days=min_mad_days,
    )

    (
        daily["log_open_interest_median_past"],
        daily["log_open_interest_mad_past"],
        daily["open_interest_mad_score"],
    ) = trading_day_mad_score(
        daily,
        "log_open_interest",
        window_days=mad_window,
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
        "position",
        "position_scale",
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
    position_scale_on_signal,
    feature_columns,
    figure_feature_columns=None,
    signal_holding_days=1,
):
    daily = daily.copy()

    daily["factor_id"] = factor_id
    daily["factor_name"] = factor_name
    daily["factor_value"] = daily[factor_value_column]
    daily["signal"] = daily[signal_column].fillna(0).astype(int)

    signal_holding_days = max(int(signal_holding_days), 1)
    position_mask = daily["signal"] == 1
    if signal_holding_days > 1:
        position_mask = (
            daily["signal"]
            .rolling(window=signal_holding_days, min_periods=1)
            .max()
            .fillna(0)
            .astype(int)
            == 1
        )

    daily["position"] = 1.0
    daily.loc[position_mask, "position"] = position_scale_on_signal

    # Backward-compatible alias for older backtest/output code.
    daily["position_scale"] = daily["position"]

    tables_dir = os.path.join(project_root, "results", "tables")
    figures_dir = os.path.join(project_root, "results", "figures")

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
        figures_dir,
        f"{symbol}_{factor_id}_{factor_name}_signal_on_price.png"
    )
    factor_figure_path = os.path.join(
        figures_dir,
        f"{symbol}_{factor_id}_{factor_name}_factor_value.png"
    )

    base_columns = [
        "date",
        "trading_day",
        "trading_day_id",
        "bar_index_in_trading_day",
        "bars_in_trading_day",
        "trading_day_start_datetime",
        "bar_start_datetime",
        "bar_end_datetime",
        "source_minutes",
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
        "position",
        "position_scale",
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
    }
