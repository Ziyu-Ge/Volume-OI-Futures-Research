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
DEFAULT_VOLATILITY_MULTIPLIER = 2.0
DEFAULT_VOLATILITY_MIN_HISTORY_DAYS = 20
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
        "symbol_factor",
        "symbol_factor",
        "symbol_factor_by_lookahead.csv",
    ),
)

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


def dedupe_legend(ax):
    handles, labels = ax.get_legend_handles_labels()
    seen = set()
    unique_handles = []
    unique_labels = []

    for handle, label in zip(handles, labels):
        if label in seen:
            continue
        seen.add(label)
        unique_handles.append(handle)
        unique_labels.append(label)

    ax.legend(unique_handles, unique_labels, loc="best", fontsize=8)


def is_truthy(value):
    return bool(pd.notna(value) and bool(value))


def plot_signal_clusters(daily, clusters, event_rows, args, lookahead_days):
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
    ax.plot(
        daily["date"],
        daily[args.price_column],
        color="#333333",
        linewidth=1.2,
        label="close",
    )

    signal_points = daily[daily["signal"] == 1].dropna(
        subset=["date", args.price_column]
    )
    if not signal_points.empty:
        ax.scatter(
            signal_points["date"],
            signal_points[args.price_column],
            s=14,
            color="#f59f00",
            alpha=0.45,
            label="raw signal",
            zorder=3,
        )

    event_by_date = {
        pd.Timestamp(row["event_date"]).normalize(): row
        for _, row in events.iterrows()
    }

    for cluster_positions in clusters:
        start_date = daily.loc[cluster_positions[0], "date"]
        event_date = daily.loc[cluster_positions[-1], "date"]
        event_key = pd.Timestamp(event_date).normalize()
        event = event_by_date.get(event_key)

        if event is None or not is_truthy(event.get("is_evaluable")):
            span_color = "#adb5bd"
        elif is_truthy(event.get("win")):
            span_color = "#2f9e44"
        else:
            span_color = "#e03131"

        if start_date == event_date:
            ax.axvline(
                event_date,
                color=span_color,
                alpha=0.16,
                linewidth=1.0,
                zorder=1,
            )
        else:
            ax.axvspan(
                start_date,
                event_date,
                color=span_color,
                alpha=0.08,
                zorder=1,
            )

    useful_events = events[events["win"].eq(True)].dropna(
        subset=["event_date", "event_price"]
    )
    other_events = events[
        events["is_evaluable"].eq(True) & ~events["win"].eq(True)
    ].dropna(subset=["event_date", "event_price"])
    pending_events = events[~events["is_evaluable"].eq(True)].dropna(
        subset=["event_date", "event_price"]
    )

    if not other_events.empty:
        ax.scatter(
            other_events["event_date"],
            other_events["event_price"],
            s=34,
            marker="x",
            color="#c92a2a",
            linewidths=1.2,
            label="cluster end, not useful",
            zorder=5,
        )

    if not useful_events.empty:
        ax.scatter(
            useful_events["event_date"],
            useful_events["event_price"],
            s=78,
            marker="*",
            color="#2f9e44",
            edgecolors="#ffffff",
            linewidths=0.6,
            label="useful cluster end",
            zorder=6,
        )

    if not pending_events.empty:
        ax.scatter(
            pending_events["event_date"],
            pending_events["event_price"],
            s=34,
            marker="D",
            facecolors="none",
            edgecolors="#868e96",
            linewidths=1.0,
            label="cluster end, not evaluable",
            zorder=5,
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
    dedupe_legend(ax)
    fig.autofmt_xdate()
    fig.tight_layout()
    fig.savefig(figure_path, dpi=args.plot_dpi)
    plt.close(fig)

    return figure_path


def evaluate_factor_file(file_info, args):
    daily = load_factor_daily(file_info, args.price_column)
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
                clusters=clusters,
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

    group_columns = [
        "symbol",
        "factor_id",
        "factor_name",
        "lookahead_days",
    ]
    rows = []

    for keys, group in events.groupby(group_columns, dropna=False):
        rows.append(build_summary_row(keys, group, group_columns, "symbol_factor"))

    for keys, group in events.groupby(
        [
            "factor_id",
            "factor_name",
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
    else:
        baseline_win_rate = np.nan

    row = dict(zip(group_columns, keys))
    row.update({
        "row_type": row_type,
        "baseline_events": baseline_events,
        "baseline_win_events": baseline_win_events,
        "baseline_win_rate": baseline_win_rate,
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
    columns = [
        "row_type",
        "symbol",
        "factor_id",
        "factor_name",
        "lookahead_days",
        "signal_clusters",
        "strategy_samples",
        "strategy_wins",
        "threshold",
        "strategy_win_rate",
        "baseline_samples",
        "baseline_wins",
        "baseline_win_rate",
        "win_rate_diff",
        "strategy_observation_end_drawdown",
        "strategy_observation_max_drawdown",
    ]
    return output[[col for col in columns if col in output.columns]]


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

    for table_name, row_type, filename in SUMMARY_TABLE_SPECS:
        table_path = summary_dir / filename
        summary_table = summary_output[
            summary_output["row_type"].eq(row_type)
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
            "前 20 日平均波动率两倍即算胜。"
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
    print(f"事件明细：{output_paths['events']}", flush=True)
    print(f"总览汇总：{output_paths['summary']['overall']}", flush=True)
    print(f"因子汇总：{output_paths['summary']['factor']}", flush=True)
    print(
        f"品种-因子汇总：{output_paths['summary']['symbol_factor']}",
        flush=True,
    )
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
