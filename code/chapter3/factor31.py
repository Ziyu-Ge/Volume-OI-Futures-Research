"""31 号因子：用 23 号因子识别龙头，做空同板块跟随品种。

逻辑尽量写直白：
1. 每个品种先计算 factor_23 的开空信号。
2. 同一天、同板块里，有 23 号信号的品种按 factor_value 最大选一个龙头。
3. 龙头出现后，找同板块、过去 20 日收益相关性足够高的跟随品种。
4. 下一交易日开盘做空跟随品种，平仓逻辑和 factor_23 完全一样。
"""

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd

from config import FLOAT_FORMAT, GROUP, OUTPUT_ENCODING


ROOT = Path(__file__).resolve().parents[2]
CHAPTER2_DIR = ROOT / "code" / "chapter2"
DAILY_DIR = ROOT / "results" / "chapter2" / "tables" / "daily"
OUTPUT_DIR = ROOT / "results" / "chapter3" / "factor31"

sys.path.insert(0, str(CHAPTER2_DIR))
from factors import factor_23  # noqa: E402
from rules.entry_rules import add_entry_signals  # noqa: E402
from rules.exit_rules import check_short_exit  # noqa: E402


LOOKBACK_DAYS = 20
CORRELATION_THRESHOLD = 0.5


def parse_args():
    parser = argparse.ArgumentParser(
        description="factor31：用 factor23 识别龙头，做空跟随品种"
    )
    parser.add_argument("--daily-dir", type=Path, default=DAILY_DIR)
    parser.add_argument("--output-dir", type=Path, default=OUTPUT_DIR)
    parser.add_argument("--start", default=None, help="开始日期 YYYY-MM-DD")
    parser.add_argument("--end", default=None, help="结束日期 YYYY-MM-DD")
    parser.add_argument("--symbols", default=None, help="只处理指定品种，逗号分隔")
    parser.add_argument(
        "--corr",
        type=float,
        default=CORRELATION_THRESHOLD,
        help="过去 20 日收益率相关系数阈值",
    )
    return parser.parse_args()


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


def add_factor23(data_by_symbol):
    result = {}
    for symbol, daily in data_by_symbol.items():
        frame = add_entry_signals(
            daily,
            factor_23.ENTRY_CONFIG,
            factor_23.USE_SPECULATION,
        )
        result[symbol] = frame
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
    for symbol, frame in data_by_symbol.items():
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


def past_correlation(data_by_symbol, leader, follower, signal_date):
    leader_returns = prior_returns(data_by_symbol[leader], signal_date)
    follower_returns = prior_returns(data_by_symbol[follower], signal_date)
    paired = leader_returns.merge(
        follower_returns,
        on="date",
        suffixes=("_leader", "_follower"),
    )
    if len(paired) < LOOKBACK_DAYS:
        return np.nan
    return paired["daily_return_leader"].corr(paired["daily_return_follower"])


def prior_returns(frame, signal_date):
    return frame.loc[
        frame["date"].lt(signal_date),
        ["date", "daily_return"],
    ].tail(LOOKBACK_DAYS)


def current_return(data_by_symbol, symbol, signal_date):
    frame = data_by_symbol[symbol]
    value = frame.loc[frame["date"].eq(signal_date), "daily_return"]
    if value.empty:
        return np.nan
    return value.iloc[0]


def find_followers(leaders, data_by_symbol, corr_limit):
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

            corr = past_correlation(data_by_symbol, leader_symbol, follower, signal_date)
            if pd.isna(corr) or corr < corr_limit:
                continue

            rows.append(
                {
                    "信号日": signal_date,
                    "板块": leader["板块"],
                    "龙头品种": leader_symbol,
                    "跟随品种": follower,
                    "龙头得分": leader["龙头得分"],
                    "20日相关系数": corr,
                    "龙头当日收益率": leader["龙头当日收益率"],
                    "跟随当日收益率": current_return(data_by_symbol, follower, signal_date),
                }
            )

    columns = [
        "信号日", "板块", "龙头品种", "跟随品种", "龙头得分", "20日相关系数",
        "龙头当日收益率", "跟随当日收益率",
    ]
    if not rows:
        return pd.DataFrame(columns=columns)
    return pd.DataFrame(rows).sort_values(
        ["信号日", "板块", "龙头品种", "20日相关系数", "跟随品种"],
        ascending=[True, True, True, False, True],
        kind="mergesort",
    ).reset_index(drop=True)


def make_trades(followers, data_by_symbol):
    signal_map = follower_signal_map(followers)
    rows = []
    for follower, signals in signal_map.items():
        frame = data_by_symbol[follower]
        rows.extend(run_one_follower(frame, signals))

    columns = [
        "信号日", "开空日", "平空日", "板块", "龙头品种", "跟随品种",
        "开空价", "平空价", "收益率", "平仓原因", "20日相关系数",
    ]
    if not rows:
        return pd.DataFrame(columns=columns)
    return (
        pd.DataFrame(rows)
        .sort_values(["开空日", "跟随品种"])
        .reset_index(drop=True)
        .reindex(columns=columns)
    )


def follower_signal_map(followers):
    signal_map = {}
    for _, row in followers.iterrows():
        follower = row["跟随品种"]
        signal_date = row["信号日"]
        signal_map.setdefault(follower, {})[signal_date] = row.to_dict()
    return signal_map


def run_one_follower(frame, signals):
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
                "20日相关系数": pending_open["20日相关系数"],
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
                factor_23.EXIT_CONFIG,
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
    daily = daily.rename(columns={"平空日": "交易日"})
    daily = daily.rename(columns={"收益率": "当日收益率"})
    daily["累计净值"] = (1.0 + daily["当日收益率"]).cumprod()
    daily["回撤"] = daily["累计净值"] / daily["累计净值"].cummax() - 1.0

    metrics = pd.DataFrame(
        {
            "指标": ["交易次数", "胜率", "累计收益率", "最大回撤"],
            "数值": [
                len(trades),
                trades["收益率"].gt(0).mean(),
                daily["累计净值"].iloc[-1] - 1.0,
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


def main():
    args = parse_args()
    start = pd.Timestamp(args.start) if args.start else None
    end = pd.Timestamp(args.end) if args.end else None

    data = load_all_daily(args.daily_dir, wanted_symbols(args.symbols))
    data = add_factor23(data)

    leaders = find_leaders(data, start, end)
    followers = find_followers(leaders, data, args.corr)
    trades = make_trades(followers, data)
    daily_returns, metrics = summarize(trades)

    write_csv(leaders, args.output_dir / "leader_signals.csv")
    write_csv(followers, args.output_dir / "follower_signals.csv")
    write_csv(trades, args.output_dir / "trades.csv")
    write_csv(daily_returns, args.output_dir / "daily_returns.csv")
    write_csv(metrics, args.output_dir / "metrics.csv")

    print(f"龙头信号: {args.output_dir / 'leader_signals.csv'}")
    print(f"跟随信号: {args.output_dir / 'follower_signals.csv'}")
    print(f"交易明细: {args.output_dir / 'trades.csv'}")
    print(f"绩效指标: {args.output_dir / 'metrics.csv'}")
    print(f"龙头数量: {len(leaders)}")
    print(f"跟随信号数量: {len(followers)}")
    print(f"交易数量: {len(trades)}")


if __name__ == "__main__":
    main()
