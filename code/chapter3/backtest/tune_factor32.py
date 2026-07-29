"""factor32 参数粗调脚本。

搜索目标是同时观察累计收益率和最大回撤：
- ranked_by_return.csv 按累计收益率优先排序；
- ranked_by_score.csv 用 累计收益率 - 2 * 最大回撤 排序，避免只追高收益。
- ranked_by_return_drawdown_ratio.csv 按 累计收益率 / 最大回撤 排序。
"""

import sys
from dataclasses import asdict, replace
from itertools import product
from pathlib import Path

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[3]
CHAPTER3_DIR = ROOT / "code" / "chapter3"
sys.path.insert(0, str(CHAPTER3_DIR))

from factors import factor32  # noqa: E402


OUT_DIR = ROOT / "results" / "chapter3" / "factor32_tuning"


def signal_map(followers):
    grouped = {}
    for _, row in followers.iterrows():
        grouped.setdefault(row["跟随品种"], {})[row["信号日"]] = row.to_dict()
    return grouped


def fast_trades(followers, data_by_symbol, exit_config):
    grouped = signal_map(followers)
    trade_rows = []
    trailing = exit_config.trailing_multiplier
    loss = exit_config.entry_loss_volatility_multiplier

    for follower, signals in grouped.items():
        frame = data_by_symbol[follower]
        dates = frame["date"].to_numpy()
        opens = frame["open"].to_numpy(dtype=float)
        closes = frame["close"].to_numpy(dtype=float)
        lows = frame["low"].to_numpy(dtype=float)
        volatilities = frame["avg_volatility_rate"].to_numpy(dtype=float)

        position = 0
        entry_price = np.nan
        low_since_entry = np.nan
        pending_open = None
        pending_cover = False
        pending_exit_reason = ""
        open_trade = None

        for i, date in enumerate(dates):
            open_price = opens[i]
            close = closes[i]
            low = lows[i]
            volatility = volatilities[i]

            if position == -1 and pending_cover and np.isfinite(open_price):
                open_trade["平空日"] = date
                open_trade["平空价"] = open_price
                open_trade["收益率"] = entry_price / open_price - 1.0
                open_trade["平仓原因"] = pending_exit_reason or "cover_short"
                trade_rows.append(open_trade)
                position = 0
                entry_price = np.nan
                low_since_entry = np.nan
                open_trade = None

            elif position == 0 and pending_open is not None and np.isfinite(open_price):
                position = -1
                entry_price = open_price
                low_since_entry = open_price
                open_trade = {
                    "信号日": pending_open["信号日"],
                    "开空日": date,
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
                low_candidate = low if np.isfinite(low) else close
                if np.isfinite(low_candidate):
                    low_since_entry = min(low_since_entry, low_candidate)

                entry_loss_stop_price = np.nan
                if np.isfinite(entry_price):
                    if loss == 0:
                        entry_loss_stop_price = entry_price
                    elif np.isfinite(volatility):
                        entry_loss_stop_price = entry_price * (1 + volatility * loss)

                price_above_entry = (
                    np.isfinite(close)
                    and np.isfinite(entry_loss_stop_price)
                    and close > entry_loss_stop_price
                )
                trailing_rebound = False
                if (
                    np.isfinite(close)
                    and np.isfinite(low_since_entry)
                    and np.isfinite(volatility)
                ):
                    trailing_stop_price = low_since_entry * (1 + volatility * trailing)
                    trailing_rebound = close > trailing_stop_price

                pending_cover = bool(price_above_entry or trailing_rebound)
                if price_above_entry and trailing_rebound:
                    pending_exit_reason = "price_above_entry_and_trailing_rebound"
                elif price_above_entry:
                    pending_exit_reason = "price_above_entry"
                elif trailing_rebound:
                    pending_exit_reason = "trailing_rebound"

            if date in signals:
                pending_open = signals[date]

    if not trade_rows:
        return pd.DataFrame(
            columns=["信号日", "开空日", "平空日", "跟随品种", "收益率"]
        )
    return pd.DataFrame(trade_rows)


def metrics(trades):
    if trades.empty:
        return {
            "交易次数": 0,
            "胜率": np.nan,
            "累计收益率": np.nan,
            "最大回撤": np.nan,
        }
    daily = trades.groupby("平空日", as_index=False)["收益率"].mean()
    daily = daily.rename(columns={"平空日": "交易日", "收益率": "当日收益率"})
    daily["累计净值"] = 1.0 + daily["当日收益率"].cumsum()
    daily["回撤"] = daily["累计净值"] / daily["累计净值"].cummax() - 1.0
    return {
        "交易次数": len(trades),
        "胜率": trades["收益率"].gt(0).mean(),
        "累计收益率": daily["当日收益率"].sum(),
        "最大回撤": -daily["回撤"].min(),
    }


def positive_part(series):
    return series.where(series > 0, 0)


def rethreshold_features(featured_data, entry_config):
    result = {}
    for symbol, frame in featured_data.items():
        data = frame.copy()
        data["ma_bias_spread_signal"] = (
            data["ma_bias_spread"] >= entry_config.ma_bias_threshold
        ).astype(int)
        data["ma_long_bias_spread_signal"] = (
            data["ma_long_bias_spread"] >= entry_config.ma_long_bias_threshold
        ).astype(int)
        data["oi_slope_signal"] = (
            data["oi_slope_rate"] <= entry_config.oi_slope_threshold
        ).astype(int)
        data["close_slope_signal"] = (
            data["close_slope_rate"] <= entry_config.close_slope_threshold
        ).astype(int)
        data["speculation_slope_signal"] = (
            data["speculation_slope"] <= entry_config.speculation_slope_threshold
        ).astype(int)
        data["open_short_signal"] = (
            (data["ma_bias_spread_signal"] == 1)
            & (data["ma_long_bias_spread_signal"] == 1)
            & (data["oi_slope_signal"] == 1)
            & (data["close_slope_signal"] == 1)
            & (data["speculation_slope_signal"] == 1)
        ).astype(int)
        data["factor_value"] = (
            positive_part(data["ma_bias_spread"])
            + positive_part(data["ma_long_bias_spread"])
            + positive_part(-data["oi_slope_rate"])
            + positive_part(-data["close_slope_rate"])
            + positive_part(-data["speculation_slope"])
        )
        result[symbol] = data
    return result


def build_context(data, entry_config, relation_floor, featured_data=None):
    if featured_data is None:
        signaled = factor32.add_configured_entry_signals(data, entry_config)
    else:
        signaled = rethreshold_features(featured_data, entry_config)
    leaders = factor32.find_leaders(signaled, None, None)
    followers = factor32.find_high_relation_followers(leaders, signaled, relation_floor)
    return signaled, leaders, followers


def evaluate_context(signaled, leaders, all_followers, relation, exit_config):
    followers = all_followers.loc[all_followers["20日关系度"].gt(relation)].copy()
    trades = fast_trades(followers, signaled, exit_config)
    row = metrics(trades)
    row["龙头数量"] = len(leaders)
    row["跟随信号数量"] = len(followers)
    return row


def row_with_params(entry_config, relation, exit_config, stage, context):
    signaled, leaders, all_followers = context
    row = evaluate_context(signaled, leaders, all_followers, relation, exit_config)
    row.update({f"entry_{key}": value for key, value in asdict(entry_config).items()})
    row.update({f"exit_{key}": value for key, value in asdict(exit_config).items()})
    row["relation"] = relation
    row["stage"] = stage
    row["score"] = row["累计收益率"] - 2.0 * row["最大回撤"]
    if row["最大回撤"] and np.isfinite(row["最大回撤"]):
        row["return_drawdown_ratio"] = row["累计收益率"] / row["最大回撤"]
    else:
        row["return_drawdown_ratio"] = np.nan
    return row


def save_rankings(results):
    frame = pd.DataFrame(results)
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    frame.to_csv(OUT_DIR / "all_results.csv", index=False, encoding="utf-8-sig")
    frame.sort_values(
        ["累计收益率", "最大回撤", "交易次数"],
        ascending=[False, True, False],
    ).to_csv(OUT_DIR / "ranked_by_return.csv", index=False, encoding="utf-8-sig")
    frame.sort_values(
        ["score", "累计收益率", "交易次数"],
        ascending=[False, False, False],
    ).to_csv(OUT_DIR / "ranked_by_score.csv", index=False, encoding="utf-8-sig")
    frame.sort_values(
        ["return_drawdown_ratio", "累计收益率", "交易次数"],
        ascending=[False, False, False],
    ).to_csv(
        OUT_DIR / "ranked_by_return_drawdown_ratio.csv",
        index=False,
        encoding="utf-8-sig",
    )
    return frame


def best_by_return_drawdown_ratio(frame):
    return frame.sort_values(
        ["return_drawdown_ratio", "累计收益率", "交易次数"],
        ascending=[False, False, False],
    ).iloc[0]


def main():
    data = factor32.load_all_daily(factor32.DAILY_DIR, None)
    base_entry = factor32.ENTRY_CONFIG
    base_exit = factor32.EXIT_CONFIG
    results = []
    relation_floor = 0.35
    base_context = build_context(data, base_entry, relation_floor)
    base_featured = base_context[0]

    print("stage 1: relation / exit", flush=True)
    for relation, trailing, loss in product(
        [0.35, 0.45, 0.5, 0.55, 0.65, 0.75],
        [1.5, 2.0, 2.5, 3.0, 3.5, 4.0],
        [0.0, 0.5, 1.0, 1.5],
    ):
        exit_config = replace(
            base_exit,
            trailing_multiplier=trailing,
            entry_loss_volatility_multiplier=loss,
        )
        results.append(
            row_with_params(base_entry, relation, exit_config, "exit", base_context)
        )

    interim = save_rankings(results)
    best_exit = best_by_return_drawdown_ratio(interim)
    tuned_exit = replace(
        base_exit,
        trailing_multiplier=float(best_exit["exit_trailing_multiplier"]),
        entry_loss_volatility_multiplier=float(
            best_exit["exit_entry_loss_volatility_multiplier"]
        ),
    )
    tuned_relation = float(best_exit["relation"])

    print("stage 2: entry thresholds", flush=True)
    for ma_bias, long_bias, spec_threshold in product(
        [0.0, 0.01, 0.02, 0.03, 0.04],
        [0.02, 0.04, 0.05, 0.07, 0.10],
        [-0.02, -0.015, -0.01, -0.005],
    ):
        entry_config = replace(
            base_entry,
            ma_bias_threshold=ma_bias,
            ma_long_bias_threshold=long_bias,
            speculation_slope_threshold=spec_threshold,
        )
        context = build_context(
            data, entry_config, min(relation_floor, tuned_relation), base_featured
        )
        results.append(
            row_with_params(entry_config, tuned_relation, tuned_exit, "threshold", context)
        )

    interim = save_rankings(results)
    best_threshold = best_by_return_drawdown_ratio(interim)
    tuned_entry = replace(
        base_entry,
        ma_bias_threshold=float(best_threshold["entry_ma_bias_threshold"]),
        ma_long_bias_threshold=float(best_threshold["entry_ma_long_bias_threshold"]),
        speculation_slope_threshold=float(
            best_threshold["entry_speculation_slope_threshold"]
        ),
    )

    print("stage 3: windows", flush=True)
    for oi_window, close_window, spec_window, vol_window in product(
        [3, 5, 7],
        [5, 7, 10],
        [3, 5, 7],
        [7, 10],
    ):
        entry_config = replace(
            tuned_entry,
            oi_slope_window=oi_window,
            close_slope_window=close_window,
            speculation_slope_window=spec_window,
            volatility_window=vol_window,
        )
        context = build_context(data, entry_config, min(relation_floor, tuned_relation))
        results.append(
            row_with_params(entry_config, tuned_relation, tuned_exit, "window", context)
        )

    final = save_rankings(results)
    print("top by return/drawdown ratio")
    columns = [
        "stage", "return_drawdown_ratio", "累计收益率", "最大回撤", "score",
        "交易次数", "胜率", "relation", "entry_ma_bias_threshold",
        "entry_ma_long_bias_threshold",
        "entry_oi_slope_window", "entry_close_slope_window",
        "entry_speculation_slope_window", "entry_speculation_slope_threshold",
        "entry_volatility_window", "exit_trailing_multiplier",
        "exit_entry_loss_volatility_multiplier",
    ]
    print(
        final.sort_values(
            ["return_drawdown_ratio", "累计收益率", "交易次数"],
            ascending=[False, False, False],
        )[columns].head(20).to_string(index=False)
    )
    print("top by return")
    print(
        final.sort_values(
            ["累计收益率", "最大回撤"], ascending=[False, True]
        )[columns].head(20).to_string(index=False)
    )
    print("top by score")
    print(
        final.sort_values(
            ["score", "累计收益率"], ascending=[False, False]
        )[columns].head(20).to_string(index=False)
    )


if __name__ == "__main__":
    main()
