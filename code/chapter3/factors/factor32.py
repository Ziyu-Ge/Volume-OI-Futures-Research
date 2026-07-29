"""32 号因子：23 号信号识别龙头，做空高关系度跟随品种。

默认开仓条件沿用 23 号因子；区别是实际开仓品种不直接用触发信号的
品种，而是选择同板块、过去 20 日收益率相关系数大于阈值的跟随品种。
平仓逻辑沿用 chapter2 的空头退出规则。
"""

import argparse
import sys
from dataclasses import asdict, fields
from pathlib import Path

import numpy as np
import pandas as pd

CHAPTER3_DIR = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(CHAPTER3_DIR))

from common.config import FLOAT_FORMAT, GROUP, OUTPUT_ENCODING  # noqa: E402


ROOT = Path(__file__).resolve().parents[3]
CHAPTER2_DIR = ROOT / "code" / "chapter2"
DAILY_DIR = ROOT / "results" / "chapter2" / "tables" / "daily"
OUTPUT_DIR = ROOT / "results" / "chapter3" / "factor32"

sys.path.insert(0, str(CHAPTER2_DIR))
from rules.entry_rules import EntryConfig  # noqa: E402
from rules.entry_rules import add_entry_signals  # noqa: E402
from rules.exit_rules import ExitConfig  # noqa: E402
from rules.exit_rules import check_short_exit  # noqa: E402


FACTOR_ID = "32"
FACTOR_NAME = "high_bias_oi_speculation_drop_high_relation"
ENGINE = "daily"
USE_SPECULATION = True

# 32 号：23 号开仓条件 + 高关系度跟随品种过滤，日频执行。
ENTRY_CONFIG = EntryConfig(
    ma_short=5,
    ma_long=20,
    ma_trend=60,
    ma_bias_threshold=0.00,
    ma_long_bias_threshold=0.02,
    oi_slope_window=5,
    close_slope_window=7,
    speculation_slope_window=3,
    speculation_slope_threshold=-0.005,
    volatility_window=7,
)
EXIT_CONFIG = ExitConfig(
    trailing_multiplier=2.5,
    entry_loss_volatility_multiplier=0.0,
)

LOOKBACK_DAYS = 20
RELATION_THRESHOLD = 0.35


def parse_args():
    parser = argparse.ArgumentParser(
        description="factor32：用 23 号式信号识别龙头，做空高关系度跟随品种"
    )
    parser.add_argument("--daily-dir", type=Path, default=DAILY_DIR)
    parser.add_argument("--output-dir", type=Path, default=OUTPUT_DIR)
    parser.add_argument("--start", default=None, help="开始日期 YYYY-MM-DD")
    parser.add_argument("--end", default=None, help="结束日期 YYYY-MM-DD")
    parser.add_argument("--symbols", default=None, help="只处理指定品种，逗号分隔")
    parser.add_argument(
        "--relation",
        type=float,
        default=RELATION_THRESHOLD,
        help="过去 20 日收益率相关系数阈值，开仓要求严格大于该值",
    )
    add_config_arguments(parser, EntryConfig, ENTRY_CONFIG, "entry")
    add_config_arguments(parser, ExitConfig, EXIT_CONFIG, "exit")
    return parser.parse_args()


def add_config_arguments(parser, config_type, default_config, prefix):
    defaults = asdict(default_config)
    for field in fields(config_type):
        option = f"--{prefix}-{field.name.replace('_', '-')}"
        parser.add_argument(
            option,
            type=field.type,
            default=defaults[field.name],
            help=f"{config_type.__name__}.{field.name}，默认 {defaults[field.name]}",
        )


def config_from_args(config_type, args, prefix):
    values = {}
    for field in fields(config_type):
        arg_name = f"{prefix}_{field.name}"
        values[field.name] = getattr(args, arg_name)
    return config_type(**values)


def wanted_symbols(symbols_text):
    if not symbols_text:
        return None
    return {item.strip().upper() for item in symbols_text.split(",") if item.strip()}


def load_one_daily(path):
    symbol = path.name.replace("_daily.csv", "").upper()
    data = pd.read_csv(path)
    data["date"] = pd.to_datetime(data["date"])
    data = data.sort_values("date", kind="mergesort").reset_index(drop=True)
    data["symbol"] = symbol
    data["group"] = GROUP.get(symbol, "未分类")
    data["daily_return"] = data["close"].pct_change()
    return data


def load_all_daily(daily_dir, symbols):
    files = sorted(Path(daily_dir).glob("*_daily.csv"))
    if symbols is not None:
        files = [
            path
            for path in files
            if path.name.replace("_daily.csv", "").upper() in symbols
        ]
    if not files:
        raise FileNotFoundError(f"{daily_dir} 中没有可用的日频数据")

    data = {}
    for path in files:
        frame = load_one_daily(path)
        data[frame["symbol"].iloc[0]] = frame
    return data


def add_configured_entry_signals(data_by_symbol, entry_config):
    result = {}
    for symbol, daily in data_by_symbol.items():
        result[symbol] = add_entry_signals(
            daily,
            entry_config,
            USE_SPECULATION,
        )
    return result


def between_dates(frame, start, end):
    result = frame
    if start is not None:
        result = result.loc[result["date"].ge(start)]
    if end is not None:
        result = result.loc[result["date"].le(end)]
    return result


def find_leaders(data_by_symbol, start, end):
    rows = []
    for _, frame in data_by_symbol.items():
        signals = frame.loc[
            frame["open_short_signal"].eq(1) & frame["group"].ne("未分类")
        ].copy()
        signals = between_dates(signals, start, end)
        if signals.empty:
            continue
        rows.append(signals[["date", "symbol", "group", "factor_value", "daily_return"]])

    if not rows:
        return pd.DataFrame(
            columns=["信号日", "板块", "龙头品种", "龙头得分", "龙头当日收益率"]
        )

    signals = pd.concat(rows, ignore_index=True)
    signals = signals.sort_values(
        ["date", "group", "factor_value", "symbol"],
        ascending=[True, True, False, True],
        kind="mergesort",
    )
    leaders = signals.drop_duplicates(["date", "group"], keep="first")
    leaders = leaders.rename(
        columns={
            "date": "信号日",
            "group": "板块",
            "symbol": "龙头品种",
            "factor_value": "龙头得分",
            "daily_return": "龙头当日收益率",
        }
    )
    return leaders.reset_index(drop=True)


def prior_returns(frame, signal_date):
    return frame.loc[
        frame["date"].lt(signal_date),
        ["date", "daily_return"],
    ].tail(LOOKBACK_DAYS)


def past_relation(data_by_symbol, leader, follower, signal_date):
    leader_returns = prior_returns(data_by_symbol[leader], signal_date)
    follower_returns = prior_returns(data_by_symbol[follower], signal_date)
    paired = leader_returns.merge(
        follower_returns,
        on="date",
        suffixes=("_leader", "_follower"),
    )
    if len(paired) < LOOKBACK_DAYS:
        return np.nan
    relation = paired["daily_return_leader"].corr(paired["daily_return_follower"])
    return float(relation) if np.isfinite(relation) else np.nan


def current_return(data_by_symbol, symbol, signal_date):
    frame = data_by_symbol[symbol]
    value = frame.loc[frame["date"].eq(signal_date), "daily_return"]
    if value.empty:
        return np.nan
    return value.iloc[0]


def find_high_relation_followers(leaders, data_by_symbol, relation_limit):
    rows = []
    for _, leader in leaders.iterrows():
        leader_symbol = leader["龙头品种"]
        signal_date = leader["信号日"]

        for follower, frame in data_by_symbol.items():
            if follower == leader_symbol:
                continue
            if frame["group"].iloc[0] != leader["板块"]:
                continue
            if frame.loc[frame["date"].eq(signal_date)].empty:
                continue

            relation = past_relation(data_by_symbol, leader_symbol, follower, signal_date)
            if pd.isna(relation) or relation <= relation_limit:
                continue

            rows.append(
                {
                    "信号日": signal_date,
                    "板块": leader["板块"],
                    "龙头品种": leader_symbol,
                    "跟随品种": follower,
                    "龙头得分": leader["龙头得分"],
                    "20日关系度": relation,
                    "龙头当日收益率": leader["龙头当日收益率"],
                    "跟随当日收益率": current_return(data_by_symbol, follower, signal_date),
                }
            )

    columns = [
        "信号日", "板块", "龙头品种", "跟随品种", "龙头得分", "20日关系度",
        "龙头当日收益率", "跟随当日收益率",
    ]
    if not rows:
        return pd.DataFrame(columns=columns)
    return pd.DataFrame(rows).sort_values(
        ["信号日", "板块", "龙头品种", "20日关系度", "跟随品种"],
        ascending=[True, True, True, False, True],
        kind="mergesort",
    ).reset_index(drop=True)


def follower_signal_map(followers):
    signal_map = {}
    for _, row in followers.iterrows():
        follower = row["跟随品种"]
        signal_date = row["信号日"]
        signal_map.setdefault(follower, {})[signal_date] = row.to_dict()
    return signal_map


def make_trades(followers, data_by_symbol, exit_config):
    signal_map = follower_signal_map(followers)
    rows = []
    for follower, signals in signal_map.items():
        frame = data_by_symbol[follower]
        rows.extend(run_one_follower(frame, signals, exit_config))

    columns = [
        "信号日", "开空日", "平空日", "板块", "龙头品种", "跟随品种",
        "开空价", "平空价", "收益率", "平仓原因", "20日关系度",
    ]
    if not rows:
        return pd.DataFrame(columns=columns)
    return (
        pd.DataFrame(rows)
        .sort_values(["开空日", "跟随品种"])
        .reset_index(drop=True)
        .reindex(columns=columns)
    )


def run_one_follower(frame, signals, exit_config):
    position = 0
    entry_price = np.nan
    low_since_entry = np.nan
    pending_open = None
    pending_cover = False
    pending_exit_reason = ""
    open_trade = None
    trades = []

    for _, row in frame.iterrows():
        open_price = row["open"]
        close = row["close"]
        low = row["low"]

        if position == -1 and pending_cover and pd.notna(open_price):
            open_trade["平空日"] = row["date"]
            open_trade["平空价"] = open_price
            open_trade["收益率"] = entry_price / open_price - 1.0
            open_trade["平仓原因"] = pending_exit_reason or "cover_short"
            trades.append(open_trade)
            position = 0
            entry_price = np.nan
            low_since_entry = np.nan
            open_trade = None

        elif position == 0 and pending_open is not None and pd.notna(open_price):
            position = -1
            entry_price = open_price
            low_since_entry = open_price
            open_trade = {
                "信号日": pending_open["信号日"],
                "开空日": row["date"],
                "板块": pending_open["板块"],
                "龙头品种": pending_open["龙头品种"],
                "跟随品种": pending_open["跟随品种"],
                "开空价": entry_price,
                "20日关系度": pending_open["20日关系度"],
            }

        pending_open = None
        pending_cover = False
        pending_exit_reason = ""

        if position == -1:
            low_candidate = low if pd.notna(low) else close
            if pd.notna(low_candidate):
                low_since_entry = min(low_since_entry, low_candidate)

            exit_info = check_short_exit(
                close,
                entry_price,
                low_since_entry,
                row["avg_volatility_rate"],
                exit_config,
            )
            pending_cover = bool(exit_info["cover_signal"])
            pending_exit_reason = exit_info["exit_reason"]

        if row["date"] in signals:
            pending_open = signals[row["date"]]

    return trades


def summarize(trades):
    if trades.empty:
        empty_metrics = pd.DataFrame(
            {"指标": ["交易次数", "胜率", "累计收益率"], "数值": [0, np.nan, np.nan]}
        )
        return pd.DataFrame(columns=["交易日", "当日收益率", "累计净值", "回撤"]), empty_metrics

    daily = trades.groupby("平空日", as_index=False)["收益率"].mean()
    daily = daily.rename(columns={"平空日": "交易日", "收益率": "当日收益率"})
    daily["累计净值"] = 1.0 + daily["当日收益率"].cumsum()
    daily["回撤"] = daily["累计净值"] / daily["累计净值"].cummax() - 1.0

    metrics = pd.DataFrame(
        {
            "指标": ["交易次数", "胜率", "累计收益率", "最大回撤"],
            "数值": [
                len(trades),
                trades["收益率"].gt(0).mean(),
                daily["当日收益率"].sum(),
                -daily["回撤"].min(),
            ],
        }
    )
    return daily, metrics


def write_csv(frame, path):
    path.parent.mkdir(parents=True, exist_ok=True)
    frame.to_csv(
        path,
        index=False,
        encoding=OUTPUT_ENCODING,
        float_format=FLOAT_FORMAT,
    )


def write_parameters(entry_config, exit_config, relation_limit, output_dir):
    rows = [
        {"参数": "FACTOR_ID", "数值": FACTOR_ID},
        {"参数": "FACTOR_NAME", "数值": FACTOR_NAME},
        {"参数": "ENGINE", "数值": ENGINE},
        {"参数": "USE_SPECULATION", "数值": USE_SPECULATION},
        {"参数": "RELATION_THRESHOLD", "数值": relation_limit},
    ]
    rows.extend(
        {"参数": f"ENTRY_CONFIG.{key}", "数值": value}
        for key, value in asdict(entry_config).items()
    )
    rows.extend(
        {"参数": f"EXIT_CONFIG.{key}", "数值": value}
        for key, value in asdict(exit_config).items()
    )
    write_csv(pd.DataFrame(rows), output_dir / "parameters.csv")


def main():
    args = parse_args()
    start = pd.Timestamp(args.start) if args.start else None
    end = pd.Timestamp(args.end) if args.end else None
    entry_config = config_from_args(EntryConfig, args, "entry")
    exit_config = config_from_args(ExitConfig, args, "exit")

    data = load_all_daily(args.daily_dir, wanted_symbols(args.symbols))
    data = add_configured_entry_signals(data, entry_config)

    leaders = find_leaders(data, start, end)
    followers = find_high_relation_followers(leaders, data, args.relation)
    trades = make_trades(followers, data, exit_config)
    daily_returns, metrics = summarize(trades)

    write_csv(leaders, args.output_dir / "leader_signals.csv")
    write_csv(followers, args.output_dir / "follower_signals.csv")
    write_csv(trades, args.output_dir / "trades.csv")
    write_csv(daily_returns, args.output_dir / "daily_returns.csv")
    write_csv(metrics, args.output_dir / "metrics.csv")
    write_parameters(entry_config, exit_config, args.relation, args.output_dir)

    print(f"参数表: {args.output_dir / 'parameters.csv'}")
    print(f"龙头信号: {args.output_dir / 'leader_signals.csv'}")
    print(f"跟随信号: {args.output_dir / 'follower_signals.csv'}")
    print(f"交易明细: {args.output_dir / 'trades.csv'}")
    print(f"绩效指标: {args.output_dir / 'metrics.csv'}")
    print(f"关系度阈值: > {args.relation:g}")
    print(f"龙头数量: {len(leaders)}")
    print(f"跟随信号数量: {len(followers)}")
    print(f"交易数量: {len(trades)}")


if __name__ == "__main__":
    main()
