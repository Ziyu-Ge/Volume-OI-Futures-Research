import os

import numpy as np
import pandas as pd

from volume_price_factor_core import get_results_dir


def build_signal_table(daily, factor_id, factor_name, feature_columns):
    """生成只包含信号日的明细表。"""
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
    """生成单品种单因子的汇总表。"""
    signal_points = daily[daily["signal"] == 1].copy()

    if len(signal_points) > 0:
        first_signal_date = signal_points["date"].min()
        first_signal_close = signal_points.loc[
            signal_points["date"].idxmin(),
            "close",
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
    results_dir=None,
):
    """保存因子日频表、信号表和汇总表。"""
    daily = daily.copy()

    daily["factor_id"] = factor_id
    daily["factor_name"] = factor_name
    daily["factor_value"] = daily[factor_value_column]
    daily["signal"] = daily[signal_column].fillna(0).astype(int)

    output_root = get_results_dir(results_dir)

    factor_output_path = os.path.join(
        output_root,
        "factors",
        f"{symbol}_{factor_id}_{factor_name}.csv",
    )
    signal_output_path = os.path.join(
        output_root,
        "signals",
        f"{symbol}_{factor_id}_{factor_name}_signals.csv",
    )
    summary_output_path = os.path.join(
        output_root,
        "summary",
        f"{symbol}_{factor_id}_{factor_name}_summary.csv",
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
    for col in base_columns + feature_columns + ["signal"]:
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

    factor_daily.to_csv(factor_output_path, index=False)
    signal_table.to_csv(signal_output_path, index=False)
    summary_table.to_csv(summary_output_path, index=False)

    print(f"factor {factor_id}: {factor_name} complete.")
    print(f"factor daily table: {factor_output_path}")
    print(f"signal table: {signal_output_path}")
    print(f"summary table: {summary_output_path}")
    print("signal days:", int(daily["signal"].sum()))

    return {
        "factor_daily": factor_daily,
        "signal_table": signal_table,
        "summary_table": summary_table,
        "factor_output_path": factor_output_path,
        "signal_output_path": signal_output_path,
        "summary_output_path": summary_output_path,
    }
