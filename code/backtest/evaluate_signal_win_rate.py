import argparse
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
DEFAULT_VOLATILITY_MULTIPLIER = 2.0
DEFAULT_VOLATILITY_MIN_HISTORY_DAYS = 20


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


def load_factor_daily(file_info, price_column):
    columns = ["date", price_column, "factor_id", "factor_name", "signal"]
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


def get_min_lookahead_days(args, lookahead_days):
    if args.min_lookahead_days is None:
        return lookahead_days
    return args.min_lookahead_days


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
    min_lookahead_days = get_min_lookahead_days(args, lookahead_days)
    is_evaluable = (
        pd.notna(event_price) and
        pd.notna(drawdown_threshold) and
        drawdown_threshold > 0 and
        len(lookahead) >= min_lookahead_days and
        lookahead[args.price_column].notna().any()
    )

    row = {
        "symbol": event_row["symbol"],
        "factor_id": event_row["factor_id"],
        "factor_name": event_row["factor_name"],
        "factor_label": event_row["factor_label"],
        "run_dir": event_row["run_dir"],
        "cluster_start_date": daily.loc[start_position, "date"].date().isoformat(),
        "event_date": event_row["date"].date().isoformat(),
        "event_price": event_price,
        "cluster_signal_days": len(cluster_positions),
        "cluster_trading_span": event_position - start_position + 1,
        "lookahead_days": lookahead_days,
        "volatility_window": args.volatility_window,
        "volatility_multiplier": DEFAULT_VOLATILITY_MULTIPLIER,
        "drawdown_threshold": drawdown_threshold,
        "is_evaluable": bool(is_evaluable),
        "win": pd.NA,
        "future_min_date": "",
        "future_min_price": np.nan,
        "future_min_return": np.nan,
        "future_end_date": "",
        "future_end_price": np.nan,
        "future_end_return": np.nan,
        "days_to_future_min": np.nan,
    }

    if not is_evaluable:
        return row

    future_prices = lookahead[args.price_column]
    min_index = future_prices.idxmin()
    end_row = lookahead.iloc[-1]

    future_min_price = daily.loc[min_index, args.price_column]
    future_end_price = end_row[args.price_column]

    future_min_return = future_min_price / event_price - 1
    future_end_return = future_end_price / event_price - 1

    win = future_min_return <= -drawdown_threshold

    row.update({
        "win": bool(win),
        "future_min_date": daily.loc[min_index, "date"].date().isoformat(),
        "future_min_price": future_min_price,
        "future_min_return": future_min_return,
        "future_end_date": end_row["date"].date().isoformat(),
        "future_end_price": future_end_price,
        "future_end_return": future_end_return,
        "days_to_future_min": int(min_index - event_position),
    })
    return row


def evaluate_factor_file(file_info, args):
    daily = load_factor_daily(file_info, args.price_column)
    daily = add_threshold_features(daily, args)
    clusters = iter_signal_clusters(daily, args.cluster_max_gap)
    event_rows = []
    baseline_rows = []

    for lookahead_days in args.lookahead_days_list:
        event_rows.extend(
            evaluate_cluster(daily, cluster_positions, args, lookahead_days)
            for cluster_positions in clusters
        )
        baseline_rows.append(evaluate_baseline(daily, args, lookahead_days))

    return event_rows, baseline_rows


def evaluate_baseline(daily, args, lookahead_days):
    prices = daily[args.price_column]
    future_price_frames = [
        prices.shift(-offset)
        for offset in range(1, lookahead_days + 1)
    ]
    future_prices = pd.concat(future_price_frames, axis=1)
    future_counts = future_prices.notna().sum(axis=1)

    future_min = future_prices.min(axis=1)
    future_end = prices.shift(-lookahead_days)

    future_min_return = future_min / prices - 1
    future_end_return = future_end / prices - 1
    min_lookahead_days = get_min_lookahead_days(args, lookahead_days)
    drawdown_threshold = daily["dynamic_drawdown_threshold"]

    evaluable = (
        prices.notna() &
        (future_counts >= min_lookahead_days) &
        future_min.notna() &
        drawdown_threshold.notna() &
        (drawdown_threshold > 0)
    )
    win = future_min_return <= -drawdown_threshold

    evaluable_count = int(evaluable.sum())
    return {
        "symbol": daily["symbol"].iloc[0],
        "factor_id": daily["factor_id"].iloc[0],
        "factor_name": daily["factor_name"].iloc[0],
        "factor_label": daily["factor_label"].iloc[0],
        "run_dir": daily["run_dir"].iloc[0],
        "lookahead_days": lookahead_days,
        "baseline_events": evaluable_count,
        "baseline_win_events": int(win[evaluable].sum()),
        "baseline_future_min_return_sum": future_min_return[evaluable].sum(),
        "baseline_future_end_return_sum": future_end_return[evaluable].sum(),
        "baseline_drawdown_threshold_sum": drawdown_threshold[evaluable].sum(),
    }


def summarize_events(events, baselines):
    if events.empty:
        return pd.DataFrame()

    group_columns = [
        "symbol",
        "factor_id",
        "factor_name",
        "factor_label",
        "run_dir",
        "lookahead_days",
    ]
    rows = []

    for keys, group in events.groupby(group_columns, dropna=False):
        rows.append(build_summary_row(keys, group, group_columns, "symbol_factor"))

    for keys, group in events.groupby(
        [
            "factor_id",
            "factor_name",
            "factor_label",
            "run_dir",
            "lookahead_days",
        ],
        dropna=False,
    ):
        full_keys = ("ALL_SYMBOLS",) + tuple(keys)
        rows.append(
            build_summary_row(
                full_keys,
                group,
                group_columns,
                "all_symbols_factor",
            )
        )

    for lookahead_days, group in events.groupby("lookahead_days", dropna=False):
        full_keys = (
            "ALL_SYMBOLS",
            "ALL_FACTORS",
            "all_factors",
            "ALL_FACTORS",
            "ALL_RUNS",
            lookahead_days,
        )
        rows.append(
            build_summary_row(
                full_keys,
                group,
                group_columns,
                "all_symbols_all_factors",
            )
        )

    summary = pd.DataFrame(rows)
    baseline_summary = summarize_baselines(baselines, group_columns)
    summary = summary.merge(
        baseline_summary,
        on=["row_type"] + group_columns,
        how="left",
    )
    summary["win_rate_lift"] = (
        summary["win_rate"] - summary["baseline_win_rate"]
    )
    return summary.sort_values(
        ["row_type", "symbol", "factor_id", "lookahead_days"]
    )


def summarize_baselines(baselines, group_columns):
    rows = []

    for keys, group in baselines.groupby(group_columns, dropna=False):
        rows.append(
            build_baseline_summary_row(
                keys,
                group,
                group_columns,
                "symbol_factor",
            )
        )

    for keys, group in baselines.groupby(
        [
            "factor_id",
            "factor_name",
            "factor_label",
            "run_dir",
            "lookahead_days",
        ],
        dropna=False,
    ):
        full_keys = ("ALL_SYMBOLS",) + tuple(keys)
        rows.append(
            build_baseline_summary_row(
                full_keys,
                group,
                group_columns,
                "all_symbols_factor",
            )
        )

    for lookahead_days, group in baselines.groupby(
        "lookahead_days",
        dropna=False,
    ):
        full_keys = (
            "ALL_SYMBOLS",
            "ALL_FACTORS",
            "all_factors",
            "ALL_FACTORS",
            "ALL_RUNS",
            lookahead_days,
        )
        rows.append(
            build_baseline_summary_row(
                full_keys,
                group,
                group_columns,
                "all_symbols_all_factors",
            )
        )

    return pd.DataFrame(rows)


def build_baseline_summary_row(keys, group, group_columns, row_type):
    baseline_events = int(group["baseline_events"].sum())
    baseline_win_events = int(group["baseline_win_events"].sum())

    if baseline_events:
        baseline_win_rate = baseline_win_events / baseline_events
        baseline_mean_future_min_return = (
            group["baseline_future_min_return_sum"].sum() /
            baseline_events
        )
        baseline_mean_future_end_return = (
            group["baseline_future_end_return_sum"].sum() /
            baseline_events
        )
        baseline_mean_drawdown_threshold = (
            group["baseline_drawdown_threshold_sum"].sum() /
            baseline_events
        )
    else:
        baseline_win_rate = np.nan
        baseline_mean_future_min_return = np.nan
        baseline_mean_future_end_return = np.nan
        baseline_mean_drawdown_threshold = np.nan

    row = dict(zip(group_columns, keys))
    row.update({
        "row_type": row_type,
        "baseline_events": baseline_events,
        "baseline_win_events": baseline_win_events,
        "baseline_win_rate": baseline_win_rate,
        "baseline_mean_future_min_return": baseline_mean_future_min_return,
        "baseline_mean_future_end_return": baseline_mean_future_end_return,
        "baseline_mean_drawdown_threshold": baseline_mean_drawdown_threshold,
    })
    return row


def build_summary_row(keys, group, group_columns, row_type):
    evaluable = group[group["is_evaluable"]].copy()
    event_count = len(group)
    evaluable_count = len(evaluable)

    if evaluable_count:
        win_count = int(evaluable["win"].sum())
        win_rate = win_count / evaluable_count
    else:
        win_count = 0
        win_rate = np.nan

    row = dict(zip(group_columns, keys))
    row.update({
        "row_type": row_type,
        "signal_days": int(group["cluster_signal_days"].sum()),
        "cluster_events": event_count,
        "evaluable_events": evaluable_count,
        "win_events": win_count,
        "win_rate": win_rate,
        "mean_cluster_signal_days": group["cluster_signal_days"].mean(),
        "mean_drawdown_threshold": evaluable["drawdown_threshold"].mean(),
        "mean_future_min_return": evaluable["future_min_return"].mean(),
        "median_future_min_return": evaluable["future_min_return"].median(),
        "mean_future_end_return": evaluable["future_end_return"].mean(),
        "mean_days_to_future_min": evaluable["days_to_future_min"].mean(),
    })
    return row


def save_outputs(events, summary, output_dir):
    tables_dir = output_dir / "tables"
    tables_dir.mkdir(parents=True, exist_ok=True)

    events_path = tables_dir / "signal_cluster_events.csv"
    summary_path = tables_dir / "signal_cluster_win_rate_summary.csv"

    events.to_csv(events_path, index=False)
    summary.to_csv(summary_path, index=False)

    return events_path, summary_path


def main():
    parser = argparse.ArgumentParser(
        description=(
            "用信号簇最后一天评估信号胜率："
            "未来窗口内最大回撤达到前 20 日平均波动率两倍即算胜。"
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
        "--min-lookahead-days",
        type=int,
        default=None,
        help="最少需要多少个未来交易日才纳入胜率。默认等于各自观察窗口。",
    )
    parser.add_argument(
        "--volatility-window",
        type=int,
        default=DEFAULT_VOLATILITY_WINDOW,
        help=(
            "计算前 N 日平均绝对日收益率，默认：20；"
            "胜负阈值固定为该均值的 2 倍。"
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

    runs_dir = args.runs_dir.resolve()
    output_dir = args.output_dir.resolve()
    symbols = parse_csv_list(args.symbols)
    factor_ids = parse_factor_ids(args.factor_ids)

    factor_files = discover_factor_files(
        runs_dir=runs_dir,
        symbols=symbols,
        factor_ids=factor_ids,
    )

    rows = []
    baseline_rows = []
    errors = []
    for file_info in factor_files:
        try:
            event_rows, baseline_row = evaluate_factor_file(file_info, args)
            rows.extend(event_rows)
            baseline_rows.extend(baseline_row)
        except Exception as exc:
            errors.append((file_info["path"], exc))

    if not rows:
        error_text = "\n".join(f"- {path}: {exc}" for path, exc in errors)
        raise RuntimeError(f"没有可评估的信号事件。\n{error_text}")

    events = pd.DataFrame(rows)
    baselines = pd.DataFrame(baseline_rows)
    summary = summarize_events(events, baselines)
    events_path, summary_path = save_outputs(events, summary, output_dir)

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
            f"胜率：{row['win_rate']:.2%}；"
            f"基准：{row['baseline_win_rate']:.2%}；"
            f"提升：{row['win_rate_lift']:.2%}",
            flush=True,
        )
    print(f"事件明细：{events_path}", flush=True)
    print(f"胜率汇总：{summary_path}", flush=True)

    if errors:
        print("\n以下文件读取失败，已跳过：", flush=True)
        for path, exc in errors:
            print(f"- {path}: {exc}", flush=True)


if __name__ == "__main__":
    main()
