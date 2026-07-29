from pathlib import Path

import numpy as np
import pandas as pd


BACKTEST_DIR = Path(__file__).resolve().parent
CODE_DIR = BACKTEST_DIR.parent
PROJECT_ROOT = CODE_DIR.parents[1]

DEFAULT_RUNS_DIR = PROJECT_ROOT / "results" / "chapter1"
DEFAULT_OUTPUT_DIR = DEFAULT_RUNS_DIR / "evaluation"
DEFAULT_FACTOR_IDS = "11,12,13,14"
DEFAULT_LOOKAHEAD_DAYS_LIST = "3,5,10"
DEFAULT_CLUSTER_MAX_GAP = 10
DEFAULT_VOLATILITY_WINDOW = 20
DEFAULT_VOLATILITY_MULTIPLIER = 1.0
DEFAULT_VOLATILITY_MIN_HISTORY_DAYS = 20

SUMMARY_GROUP_COLUMNS = [
    "symbol",
    "factor_id",
    "factor_name",
    "lookahead_days",
]
SUMMARY_GROUP_SPECS = (
    (
        "all_symbols_factor",
        ["factor_id", "factor_name", "lookahead_days"],
        {"symbol": "ALL_SYMBOLS"},
    ),
    (
        "all_symbols_all_factors",
        ["lookahead_days"],
        {
            "symbol": "ALL_SYMBOLS",
            "factor_id": "ALL_FACTORS",
            "factor_name": "all_factors",
        },
    ),
)


def parse_factor_ids(raw_value):
    """把 11,12 这种参数转成列表。"""
    values = [
        item.strip()
        for item in str(raw_value).split(",")
        if item.strip()
    ]
    return values or None


def parse_int_list(raw_value):
    """把 3,5,10 这种参数转成整数列表。"""
    values = [
        int(item.strip())
        for item in str(raw_value).split(",")
        if item.strip()
    ]
    return values or None


def parse_factor_file(path):
    """从因子日频文件名解析品种、因子编号和因子名。"""
    parts = path.stem.split("_", 2)
    if len(parts) != 3:
        return None

    symbol, factor_id, factor_name = parts
    return {
        "symbol": symbol.upper(),
        "factor_id": factor_id,
        "factor_name": factor_name,
        "factor_label": f"{factor_id}_{factor_name}",
        "run_dir": path.parents[1].name,
        "path": path,
    }


def discover_factor_files(runs_dir, factor_ids=None):
    """在每个因子结果目录的 factors 子目录中寻找日频表。"""
    factor_ids = set(factor_ids) if factor_ids is not None else None
    factor_files = []

    for run_dir in sorted(runs_dir.iterdir()):
        if not run_dir.is_dir() or run_dir.name == "combined":
            continue

        factors_dir = run_dir / "factors"
        if not factors_dir.is_dir():
            continue

        for path in sorted(factors_dir.glob("*.csv")):
            metadata = parse_factor_file(path)
            if metadata is None:
                continue
            if factor_ids is not None and metadata["factor_id"] not in factor_ids:
                continue
            factor_files.append(metadata)

    if not factor_files:
        raise FileNotFoundError(f"没有找到因子日频表：{runs_dir}")

    return factor_files


def load_factor_daily(file_info, price_column):
    """读取单个因子的日频表，只保留评估需要的字段。"""
    columns = {"date", price_column, "signal"}
    daily = pd.read_csv(file_info["path"], usecols=lambda col: col in columns)

    missing_columns = {"date", price_column, "signal"} - set(daily.columns)
    if missing_columns:
        raise ValueError(
            f"{file_info['path']} 缺少字段："
            f"{','.join(sorted(missing_columns))}"
        )

    daily["date"] = pd.to_datetime(daily["date"])
    daily[price_column] = pd.to_numeric(daily[price_column], errors="coerce")
    daily["signal"] = (
        pd.to_numeric(daily["signal"], errors="coerce")
        .fillna(0)
        .astype(int)
    )

    daily = daily.sort_values("date").reset_index(drop=True)
    daily["symbol"] = file_info["symbol"]
    daily["factor_id"] = file_info["factor_id"]
    daily["factor_name"] = file_info["factor_name"]
    daily["factor_label"] = file_info["factor_label"]
    daily["run_dir"] = file_info["run_dir"]
    return daily


def add_threshold_features(daily, args):
    """用过去平均绝对收益率生成动态回撤阈值。"""
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


def iter_signal_clusters(daily, cluster_max_gap):
    """把距离较近的信号合并成一簇。"""
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


def prepare_factor_daily(file_info, args):
    """读取日频表并补好回测阈值。"""
    daily = load_factor_daily(file_info, args.price_column)
    return add_threshold_features(daily, args)


def evaluate_cluster(daily, cluster_positions, args, lookahead_days):
    """评估一个信号簇，事件日固定为簇内最后一个信号日。"""
    start_position = cluster_positions[0]
    event_position = cluster_positions[-1]
    event_row = daily.loc[event_position]
    event_price = event_row[args.price_column]
    drawdown_threshold = daily.loc[event_position, "dynamic_drawdown_threshold"]

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
        "lookahead_days": lookahead_days,
        "drawdown_threshold": drawdown_threshold,
        "is_evaluable": bool(is_evaluable),
        "win": pd.NA,
        "observation_end_drawdown": np.nan,
        "max_drawdown": np.nan,
    }

    if not is_evaluable:
        return row

    future_min_price = daily.loc[future_prices.idxmin(), args.price_column]
    future_end_price = future_prices.iloc[-1]
    row.update({
        "win": bool(max(event_price - future_end_price, 0) / event_price >= drawdown_threshold),
        "observation_end_drawdown": max(event_price - future_end_price, 0) / event_price,
        "max_drawdown": max(event_price - future_min_price, 0) / event_price,
    })
    return row


def evaluate_baseline(daily, args, lookahead_days):
    """不看信号，对所有可评价交易日计算同一胜负规则，作为基准。"""
    prices = daily[args.price_column]
    future_prices = pd.concat(
        [prices.shift(-offset) for offset in range(1, lookahead_days + 1)],
        axis=1,
    )
    future_end_price = prices.shift(-lookahead_days)
    observation_end_drawdown = (1 - future_end_price / prices).clip(lower=0)
    drawdown_threshold = daily["dynamic_drawdown_threshold"]
    evaluable = (
        prices.notna() &
        (future_prices.notna().sum(axis=1) >= lookahead_days) &
        future_end_price.notna() &
        drawdown_threshold.notna() &
        (drawdown_threshold > 0)
    )
    win = observation_end_drawdown >= drawdown_threshold

    return {
        "symbol": daily["symbol"].iloc[0],
        "factor_id": daily["factor_id"].iloc[0],
        "factor_name": daily["factor_name"].iloc[0],
        "lookahead_days": lookahead_days,
        "baseline_events": int(evaluable.sum()),
        "baseline_win_events": int(win[evaluable].sum()),
    }


def evaluate_factor_file(file_info, args):
    """评估单个因子文件，返回日频表、事件和基准结果。"""
    daily = prepare_factor_daily(file_info, args)
    clusters = iter_signal_clusters(daily, args.cluster_max_gap)
    event_rows = []
    baseline_rows = []
    rows_by_lookahead = {}

    for lookahead_days in args.lookahead_days_list:
        lookahead_rows = [
            evaluate_cluster(daily, positions, args, lookahead_days)
            for positions in clusters
        ]
        rows_by_lookahead[lookahead_days] = lookahead_rows
        event_rows.extend(lookahead_rows)
        baseline_rows.append(evaluate_baseline(daily, args, lookahead_days))

    return daily, event_rows, baseline_rows, rows_by_lookahead


def summarize_events(events, baselines):
    """把事件明细汇总成总体和分因子胜率表。"""
    if events.empty:
        return pd.DataFrame()

    rows = []
    for row_type, group_columns, fixed_values in SUMMARY_GROUP_SPECS:
        for values, group in iter_group_values(events, group_columns):
            values.update(fixed_values)
            rows.append(build_summary_row(values, group, row_type))

    summary = pd.DataFrame(rows)
    baseline_summary = summarize_baselines(baselines)
    summary = summary.merge(
        baseline_summary,
        on=["row_type"] + SUMMARY_GROUP_COLUMNS,
        how="left",
    )
    summary["win_rate_lift"] = summary["win_rate"] - summary["baseline_win_rate"]
    return summary.sort_values(
        ["row_type", "symbol", "factor_id", "lookahead_days"]
    )


def iter_group_values(frame, group_columns):
    for key, group in frame.groupby(group_columns, dropna=False):
        if not isinstance(key, tuple):
            key = (key,)
        yield dict(zip(group_columns, key)), group


def summarize_baselines(baselines):
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
        for values, group in iter_group_values(baselines, group_columns):
            values.update(fixed_values)
            rows.append(build_baseline_summary_row(values, group, row_type))

    return pd.DataFrame(rows)


def build_baseline_summary_row(values, group, row_type):
    baseline_events = int(group["baseline_events"].sum())
    baseline_win_events = int(group["baseline_win_events"].sum())
    baseline_win_rate = baseline_win_events / baseline_events if baseline_events else np.nan

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
    win_count = int(evaluable["win"].sum()) if evaluable_count else 0
    win_rate = win_count / evaluable_count if evaluable_count else np.nan

    row = {column: values.get(column) for column in SUMMARY_GROUP_COLUMNS}
    row.update({
        "row_type": row_type,
        "cluster_events": event_count,
        "evaluable_events": evaluable_count,
        "win_events": win_count,
        "win_rate": win_rate,
        "mean_drawdown_threshold": evaluable["drawdown_threshold"].mean(),
        "strategy_observation_end_drawdown": evaluable[
            "observation_end_drawdown"
        ].max(),
        "strategy_observation_max_drawdown": evaluable["max_drawdown"].max(),
    })
    return row
