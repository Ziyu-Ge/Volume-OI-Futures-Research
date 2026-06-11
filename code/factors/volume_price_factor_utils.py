import os

project_root = os.path.dirname(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
)

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


def load_daily_and_segments(symbol):
    tables_dir = os.path.join(project_root, "results", "tables")

    daily_input_path = os.path.join(
        tables_dir,
        "daily",
        f"{symbol}_daily_with_trend.csv"
    )
    segments_input_path = os.path.join(
        tables_dir,
        "daily",
        f"{symbol}_trend_segments.csv"
    )

    if not os.path.exists(daily_input_path):
        daily_input_path = os.path.join(
            tables_dir,
            f"{symbol}_daily_with_trend.csv"
        )

    if not os.path.exists(segments_input_path):
        segments_input_path = os.path.join(
            tables_dir,
            f"{symbol}_trend_segments.csv"
        )

    daily = pd.read_csv(daily_input_path)
    segments = pd.read_csv(segments_input_path)

    daily["date"] = pd.to_datetime(daily["date"])
    daily = daily.sort_values("date").reset_index(drop=True)

    date_columns = [
        "start_date",
        "end_date",
        "end_signal_date",
        "reversal_start_date",
        "reversal_end_date",
    ]

    for col in date_columns:
        if col in segments.columns:
            segments[col] = pd.to_datetime(segments[col])

    if "is_reversal_window" not in daily.columns:
        daily["is_reversal_window"] = 0

    if "reversal_segment_id" not in daily.columns:
        daily["reversal_segment_id"] = np.nan

    return daily, segments


def add_reversal_trend_labels(daily, segments):
    trend_map = segments.set_index("segment_id")["trend"].to_dict()

    daily["reversal_trend"] = daily["reversal_segment_id"].map(trend_map)
    daily["is_uptrend_reversal_window"] = (
        (daily["is_reversal_window"] == 1) &
        (daily["reversal_trend"] == "up_trend")
    ).astype(int)
    daily["is_downtrend_reversal_window"] = (
        (daily["is_reversal_window"] == 1) &
        (daily["reversal_trend"] == "down_trend")
    ).astype(int)

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


def mad_score(series, window, min_history_days):
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
        (MAD_SCALE * mad_past + MAD_EPSILON)
    )
    score[mad_past <= 0] = np.nan

    return median_past, mad_past, score


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
    trend_group = (
        daily["realtime_trend"].fillna("no_trend") !=
        daily["realtime_trend"].fillna("no_trend").shift(1)
    ).cumsum()
    daily["realtime_trend_age"] = (
        daily
        .groupby(trend_group)
        .cumcount()
        .add(1)
    )
    daily.loc[
        daily["realtime_trend"].fillna("no_trend") == "no_trend",
        "realtime_trend_age"
    ] = 0

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


def build_signal_table(daily, segments, factor_id, factor_name):
    signal_columns = [
        "factor_id",
        "factor_name",
        "segment_id",
        "trend",
        "reversal_segment_id",
        "reversal_trend",
        "realtime_trend",
        "matched_segment_id",
        "matched_segment_trend",
        "matched_segment_source",
        "signal_date",
        "signal_close",
        "factor_value",
        "position",
        "position_scale",
        "is_reversal_window",
        "is_effective_signal",
        "end_date",
        "end_close",
        "end_signal_date",
        "end_signal_close",
        "reversal_start_date",
        "reversal_end_date",
        "days_to_trend_end",
    ]

    signal_rows = []
    signal_points = daily[daily["signal"] == 1].copy()

    for _, row in signal_points.iterrows():
        segment_id = row.get("segment_id", np.nan)
        reversal_segment_id = row.get("reversal_segment_id", np.nan)
        matched_segment = pd.DataFrame()
        matched_segment_source = ""

        if (
            row.get("is_reversal_window", 0) == 1 and
            pd.notna(reversal_segment_id)
        ):
            matched_segment = segments[
                segments["segment_id"] == reversal_segment_id
            ]
            matched_segment_source = "reversal_segment_id"

        if len(matched_segment) == 0 and pd.notna(segment_id):
            matched_segment = segments[
                segments["segment_id"] == segment_id
            ]
            matched_segment_source = "segment_id"

        if len(matched_segment) > 0:
            seg = matched_segment.iloc[0]
            matched_segment_id = seg["segment_id"]
            matched_segment_trend = seg["trend"]
            end_date = seg["end_date"]
            days_to_trend_end = (end_date - row["date"]).days
            end_close = seg["end_close"]
            end_signal_date = seg.get("end_signal_date", pd.NaT)
            end_signal_close = seg.get("end_signal_close", np.nan)
            reversal_start_date = seg.get("reversal_start_date", pd.NaT)
            reversal_end_date = seg.get("reversal_end_date", pd.NaT)
        else:
            matched_segment_id = np.nan
            matched_segment_trend = np.nan
            end_date = pd.NaT
            days_to_trend_end = np.nan
            end_close = np.nan
            end_signal_date = pd.NaT
            end_signal_close = np.nan
            reversal_start_date = pd.NaT
            reversal_end_date = pd.NaT

        signal_rows.append({
            "factor_id": factor_id,
            "factor_name": factor_name,
            "segment_id": segment_id,
            "trend": row.get("trend", np.nan),
            "reversal_segment_id": reversal_segment_id,
            "reversal_trend": row.get("reversal_trend", np.nan),
            "realtime_trend": row.get("realtime_trend", np.nan),
            "matched_segment_id": matched_segment_id,
            "matched_segment_trend": matched_segment_trend,
            "matched_segment_source": matched_segment_source,
            "signal_date": row["date"],
            "signal_close": row["close"],
            "factor_value": row["factor_value"],
            "position": row["position"],
            "position_scale": row["position_scale"],
            "is_reversal_window": row["is_reversal_window"],
            "is_effective_signal": row["is_effective_signal"],
            "end_date": end_date,
            "end_close": end_close,
            "end_signal_date": end_signal_date,
            "end_signal_close": end_signal_close,
            "reversal_start_date": reversal_start_date,
            "reversal_end_date": reversal_end_date,
            "days_to_trend_end": days_to_trend_end,
        })

    return pd.DataFrame(signal_rows, columns=signal_columns)


def build_summary_table(daily, segments, factor_id, factor_name,
                        feature_columns, target_trend=None):
    result_rows = []

    if target_trend is not None:
        evaluated_segments = segments[segments["trend"] == target_trend]
    else:
        evaluated_segments = segments

    for _, seg in evaluated_segments.iterrows():
        segment_id = seg["segment_id"]
        part = daily[daily["segment_id"] == segment_id].copy()

        if len(part) == 0:
            continue

        reversal_part = daily[
            daily["reversal_segment_id"] == segment_id
        ].copy()

        signal_part = part[part["signal"] == 1].copy()
        effective_signal_part = reversal_part[
            reversal_part["is_effective_signal"] == 1
        ].copy()

        if len(signal_part) > 0:
            first_signal_date = signal_part["date"].min()
            first_signal_close = signal_part.loc[
                signal_part["date"].idxmin(),
                "close"
            ]
            days_to_end_first_signal = (
                seg["end_date"] - first_signal_date
            ).days
        else:
            first_signal_date = pd.NaT
            first_signal_close = np.nan
            days_to_end_first_signal = np.nan

        if len(effective_signal_part) > 0:
            first_effective_signal_date = (
                effective_signal_part["date"].min()
            )
            first_effective_signal_close = effective_signal_part.loc[
                effective_signal_part["date"].idxmin(),
                "close"
            ]
            days_to_end_first_effective_signal = (
                seg["end_date"] - first_effective_signal_date
            ).days
        else:
            first_effective_signal_date = pd.NaT
            first_effective_signal_close = np.nan
            days_to_end_first_effective_signal = np.nan

        row = {
            "factor_id": factor_id,
            "factor_name": factor_name,
            "target_trend": target_trend,
            "segment_id": segment_id,
            "trend": seg["trend"],
            "start_date": seg["start_date"],
            "end_date": seg["end_date"],
            "end_signal_date": seg.get("end_signal_date", pd.NaT),
            "reversal_start_date": seg.get("reversal_start_date", pd.NaT),
            "reversal_end_date": seg.get("reversal_end_date", pd.NaT),
            "days": seg["days"],
            "return": seg["return"],
            "signal_days_in_trend": part["signal"].sum(),
            "signal_ratio_in_trend": part["signal"].mean(),
            "signal_days_in_reversal_window": (
                reversal_part["signal"].sum()
            ),
            "signal_ratio_in_reversal_window": (
                reversal_part["signal"].mean()
            ),
            "has_signal": int(part["signal"].sum() > 0),
            "has_effective_signal": int(
                reversal_part["is_effective_signal"].sum() > 0
            ),
            "first_signal_date": first_signal_date,
            "first_signal_close": first_signal_close,
            "days_to_end_first_signal": days_to_end_first_signal,
            "first_effective_signal_date": first_effective_signal_date,
            "first_effective_signal_close": first_effective_signal_close,
            "days_to_end_first_effective_signal": (
                days_to_end_first_effective_signal
            ),
            "max_factor_value_in_trend": part["factor_value"].max(),
            "mean_factor_value_in_trend": part["factor_value"].mean(),
            "max_factor_value_in_reversal_window": (
                reversal_part["factor_value"].max()
            ),
            "mean_factor_value_in_reversal_window": (
                reversal_part["factor_value"].mean()
            ),
        }

        for col in feature_columns:
            if col not in daily.columns:
                continue

            if not pd.api.types.is_numeric_dtype(daily[col]):
                continue

            row[f"mean_{col}_in_trend"] = part[col].mean()
            row[f"mean_{col}_in_reversal_window"] = (
                reversal_part[col].mean()
            )

        result_rows.append(row)

    return pd.DataFrame(result_rows)


def save_factor_outputs(
    daily,
    segments,
    symbol,
    factor_id,
    factor_name,
    factor_value_column,
    signal_column,
    position_scale_on_signal,
    feature_columns,
    figure_feature_columns=None,
    target_trend=None,
    signal_holding_days=1,
):
    daily = daily.copy()
    daily = add_reversal_trend_labels(daily, segments)

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

    effective_mask = (
        (daily["signal"] == 1) &
        (daily["is_reversal_window"] == 1)
    )

    if target_trend is not None:
        effective_mask = effective_mask & (
            daily["reversal_trend"] == target_trend
        )

    daily["is_effective_signal"] = effective_mask.astype(int)

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
        "open",
        "close",
        "high",
        "low",
        "volume",
        "total_turnover",
        "open_interest",
        "speculation",
        "threshold",
        "realtime_trend",
        "realtime_position",
        "realtime_segment_id",
        "trend",
        "segment_id",
        "is_reversal_window",
        "reversal_segment_id",
        "reversal_trend",
        "is_uptrend_reversal_window",
        "is_downtrend_reversal_window",
        "factor_id",
        "factor_name",
        "factor_value",
    ]

    output_columns = []

    for col in base_columns + feature_columns + [
        "signal",
        "position",
        "position_scale",
        "is_effective_signal",
    ]:
        if col in daily.columns and col not in output_columns:
            output_columns.append(col)

    factor_daily = daily[output_columns].copy()
    signal_table = build_signal_table(
        daily,
        segments,
        factor_id,
        factor_name,
    )
    summary_table = build_summary_table(
        daily,
        segments,
        factor_id,
        factor_name,
        feature_columns,
        target_trend=target_trend,
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
    effective_points = daily[daily["is_effective_signal"] == 1]

    plt.scatter(
        signal_points["date"],
        signal_points["close"],
        s=22,
        label="signal",
    )
    plt.scatter(
        effective_points["date"],
        effective_points["close"],
        s=48,
        marker="X",
        label="effective signal",
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
        "effective days:",
        int(daily["is_effective_signal"].sum()),
    )

    return {
        "factor_daily": factor_daily,
        "signal_table": signal_table,
        "summary_table": summary_table,
        "factor_output_path": factor_output_path,
        "signal_output_path": signal_output_path,
        "summary_output_path": summary_output_path,
    }
