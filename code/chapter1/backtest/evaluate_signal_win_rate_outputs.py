import os

import pandas as pd

import evaluate_signal_win_rate_core as core


os.environ.setdefault("MPLCONFIGDIR", str(core.PROJECT_ROOT / ".matplotlib"))
os.environ.setdefault("XDG_CACHE_HOME", str(core.PROJECT_ROOT / ".cache"))
(core.PROJECT_ROOT / ".matplotlib").mkdir(exist_ok=True)
(core.PROJECT_ROOT / ".cache").mkdir(exist_ok=True)

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt


SUMMARY_TABLE_SPECS = (
    ("overall", "all_symbols_all_factors", "overall_by_lookahead.csv"),
    ("factor", "all_symbols_factor", "factor_by_lookahead.csv"),
)
SUMMARY_OUTPUT_COLUMNS = [
    "factor_id",
    "factor_name",
    "lookahead_days",
    "threshold",
    "strategy_win_rate",
    "baseline_win_rate",
    "win_rate_diff",
    "strategy_observation_max_drawdown",
]


def plot_signal_clusters(daily, event_rows, args, lookahead_days):
    """保存单个品种、单个因子的信号簇评估图。"""
    if not event_rows:
        return None

    events = pd.DataFrame(event_rows)
    if events.empty:
        return None

    events["event_date"] = pd.to_datetime(events["event_date"])
    events["event_price"] = pd.to_numeric(events["event_price"], errors="coerce")

    symbol = daily["symbol"].iloc[0]
    factor_id = daily["factor_id"].iloc[0]
    factor_name = daily["factor_name"].iloc[0]
    figures_dir = args.output_dir / "figures" / "signal_clusters" / f"{lookahead_days}d"
    figures_dir.mkdir(parents=True, exist_ok=True)
    figure_path = (
        figures_dir
        / f"{symbol}_{factor_id}_{factor_name}_signal_clusters_{lookahead_days}d.png"
    )

    fig, ax = plt.subplots(figsize=(14, 7))
    ax.plot(daily["date"], daily[args.price_column], color="#333333", linewidth=1.1)

    marker_specs = (
        (events["win"].eq(True), "win", "#2f9e44", "^"),
        (events["is_evaluable"].eq(True) & ~events["win"].eq(True), "miss", "#c92a2a", "x"),
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
    win_count = int(events["win"].eq(True).sum())
    win_rate = win_count / evaluable_count if evaluable_count else pd.NA
    title_rate = "NA" if pd.isna(win_rate) else f"{win_rate:.1%}"
    ax.set_title(
        (
            f"{symbol} factor {factor_id}: {factor_name} | "
            f"lookahead {lookahead_days}d | "
            f"useful {win_count}/{evaluable_count} ({title_rate})"
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


def build_output_events(events):
    """把内部事件字段改成最终 CSV 字段名。"""
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
    """把内部汇总字段改成最终 CSV 字段名。"""
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
    """保存事件明细和两张汇总表。"""
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
        summary_table = summary_output[summary["row_type"].eq(row_type).to_numpy()]
        summary_table.to_csv(table_path, index=False)
        summary_paths[table_name] = table_path

    return {"events": events_path, "summary": summary_paths}


def print_report(factor_files, summary, output_paths, plot_paths, args, errors):
    """在终端打印一段简洁的评估报告。"""
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
    if not args.skip_plots:
        print(f"信号图数量：{len(plot_paths)}", flush=True)
        print(f"信号图目录：{args.output_dir / 'figures' / 'signal_clusters'}", flush=True)

    if errors:
        print("\n以下文件读取失败，已跳过：", flush=True)
        for path, exc in errors:
            print(f"- {path}: {exc}", flush=True)
