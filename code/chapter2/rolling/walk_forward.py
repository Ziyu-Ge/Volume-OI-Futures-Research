from dataclasses import asdict

import numpy as np
import pandas as pd

from core import io
from core.reports import empty_trade_table
from engines import backtest
from engines.hourly_exit_engine import (
    attach_daily_entry,
    build_intraday_frame,
    run_hourly_state_machine,
)
from rules.entry_rules import add_entry_signals


FACTOR_ID = "24_ROLLING"
FACTOR_NAME = "factor_24_walk_forward"


def make_windows(first_date, last_date, train_years=3, test_months=6):
    """生成滚动窗口：过去 3 年训练，未来 6 个月样本外。"""
    windows = []
    train_start = pd.Timestamp(first_date).normalize()
    last_date = pd.Timestamp(last_date).normalize()

    while True:
        train_end = train_start + pd.DateOffset(years=train_years)
        test_start = train_end
        test_end = test_start + pd.DateOffset(months=test_months)
        if test_start > last_date:
            break
        windows.append(
            {
                "window_id": len(windows) + 1,
                "train_start": train_start,
                "train_end": train_end,
                "test_start": test_start,
                "test_end": min(test_end, last_date + pd.Timedelta(days=1)),
            }
        )
        train_start = train_start + pd.DateOffset(months=test_months)

    return windows


def load_symbol_data(symbols, daily_dir, hourly_dir):
    """一次性加载数据，避免每个候选参数重复读 CSV。"""
    data = {}
    for symbol in symbols:
        data[symbol] = {
            "daily": io.load_daily(symbol, daily_dir),
            "hourly": io.load_hourly(symbol, hourly_dir),
        }
    return data


def run_walk_forward(
    symbol_data,
    param_candidates,
    train_years=3,
    test_months=6,
    drawdown_penalty=2.0,
):
    """执行 walk-forward，并只返回样本外结果。"""
    first_date, last_date = data_date_range(symbol_data)
    windows = make_windows(first_date, last_date, train_years, test_months)

    selected_rows = []
    oos_curve_parts = []
    oos_trade_parts = []

    for window_index, window in enumerate(windows, start=1):
        print(
            f"[窗口 {window_index}/{len(windows)}] "
            f"训练期 {window['train_start'].date()} 至 {window['train_end'].date()}，"
            f"样本外 {window['test_start'].date()} 至 {window['test_end'].date()}",
            flush=True,
        )
        best_candidate, best_metric, best_score = choose_best_candidate(
            symbol_data, param_candidates, window, drawdown_penalty
        )
        selected_row = build_selected_param_row(
            window, best_candidate, best_metric, best_score
        )

        if best_candidate is not None:
            oos_result = evaluate_period(
                symbol_data,
                best_candidate,
                window["test_start"],
                window["test_end"],
                window["window_id"],
            )
            if oos_result is not None:
                selected_row["oos_annual_return"] = oos_result["portfolio_metric"][
                    "annual_return"
                ]
                selected_row["oos_max_drawdown"] = oos_result["portfolio_metric"][
                    "max_drawdown"
                ]
                selected_row["oos_sharpe"] = oos_result["portfolio_metric"]["sharpe"]
                oos_curve_parts.append(oos_result["curves"])
                if not oos_result["trades"].empty:
                    oos_trade_parts.append(oos_result["trades"])

        selected_rows.append(selected_row)

    if not oos_curve_parts:
        raise RuntimeError("所有样本外窗口都没有实际开仓，无法生成 rolling 结果")

    selected_params = pd.DataFrame(selected_rows)
    oos_curves = rebuild_symbol_curves(pd.concat(oos_curve_parts, ignore_index=True))
    oos_trades = (
        pd.concat(oos_trade_parts, ignore_index=True)
        if oos_trade_parts
        else empty_trade_table()
    )
    periods = max(infer_periods(item["hourly"]) for item in symbol_data.values())
    portfolio, portfolio_metric = backtest.build_portfolio(
        oos_curves, FACTOR_ID, FACTOR_NAME, periods
    )
    metrics = build_final_symbol_metrics(oos_curves, oos_trades, periods)
    metrics = pd.concat([metrics, pd.DataFrame([portfolio_metric])], ignore_index=True)

    return {
        "selected_params": selected_params,
        "metrics": metrics,
        "trades": oos_trades,
        "curves": oos_curves,
        "portfolio": portfolio,
    }


def choose_best_candidate(symbol_data, param_candidates, window, drawdown_penalty):
    """只看训练期表现，选分数最高的参数。"""
    best_candidate = None
    best_metric = None
    best_score = -np.inf

    for candidate_index, candidate in enumerate(param_candidates, start=1):
        print(
            f"  评估候选参数 {candidate_index}/{len(param_candidates)}："
            f"{candidate['changed_field']}",
            flush=True,
        )
        result = evaluate_period(
            symbol_data,
            candidate,
            window["train_start"],
            window["train_end"],
            window["window_id"],
        )
        if result is None:
            continue

        metric = result["portfolio_metric"]
        score = score_metric(metric, drawdown_penalty)
        if score > best_score:
            best_candidate = candidate
            best_metric = metric
            best_score = score

    return best_candidate, best_metric, best_score


def evaluate_period(symbol_data, candidate, start_time, end_time, window_id):
    """用一套参数跑完整时间段；没有信号时保持基准仓位。"""
    symbol_results = []
    for symbol, data in symbol_data.items():
        frame = run_hourly_segment(
            data["daily"],
            data["hourly"],
            candidate["entry_config"],
            candidate["exit_config"],
            start_time,
            end_time,
        )
        if frame.empty:
            continue

        periods = backtest.infer_periods_per_year(frame, is_hourly=True)
        trades = backtest.build_trade_table(frame, symbol, FACTOR_ID, FACTOR_NAME)
        symbol_results.append(
            {
                "symbol": symbol,
                "frame": frame,
                "trades": trades,
                "periods": periods,
            }
        )

    if not symbol_results:
        return None

    curves = []
    trades = []
    for item in symbol_results:
        frame = item["frame"].copy()
        frame = backtest.add_return_columns(frame)
        curve = backtest.build_curve_table(
            frame, item["symbol"], FACTOR_ID, FACTOR_NAME
        )
        curve.insert(3, "window_id", window_id)
        curves.append(curve)
        if not item["trades"].empty:
            trades.append(item["trades"])

    if not curves:
        return None

    curves = pd.concat(curves, ignore_index=True)
    trades = pd.concat(trades, ignore_index=True) if trades else empty_trade_table()
    periods = max(item["periods"] for item in symbol_results)
    portfolio, portfolio_metric = backtest.build_portfolio(
        curves, FACTOR_ID, FACTOR_NAME, periods
    )
    return {
        "curves": curves,
        "trades": trades,
        "portfolio_metric": portfolio_metric,
    }


def run_hourly_segment(daily, hourly, entry_config, exit_config, start_time, end_time):
    """计算指标用历史数据，交易状态机只从本窗口开始跑。"""
    start_time = pd.Timestamp(start_time)
    end_time = pd.Timestamp(end_time)

    # 日频指标只用 end_time 之前的数据，避免训练期看到样本外数据。
    daily_history = daily.loc[daily["date"] < end_time.normalize()].copy()
    hourly_segment = hourly.loc[
        (hourly["date"] >= start_time) & (hourly["date"] < end_time)
    ].copy()
    if daily_history.empty or hourly_segment.empty:
        return pd.DataFrame()

    entry_daily = add_entry_signals(daily_history, entry_config, use_speculation=True)
    frame = build_intraday_frame(hourly_segment)
    frame = attach_daily_entry(frame, entry_daily)
    return run_hourly_state_machine(frame, exit_config)


def score_metric(metric, drawdown_penalty):
    """训练期打分：Sharpe 越高越好，回撤越小越好。"""
    sharpe = metric["sharpe"]
    max_drawdown = metric["max_drawdown"]
    if pd.isna(sharpe) or pd.isna(max_drawdown):
        return -np.inf
    return sharpe - drawdown_penalty * abs(max_drawdown)


def build_selected_param_row(window, candidate, metric, score):
    row = {key: value for key, value in window.items()}
    row["score"] = score
    if candidate is None or metric is None:
        row["changed_field"] = ""
        row["train_annual_return"] = np.nan
        row["train_max_drawdown"] = np.nan
        row["train_sharpe"] = np.nan
        return row

    row["changed_field"] = candidate["changed_field"]
    row["train_annual_return"] = metric["annual_return"]
    row["train_max_drawdown"] = metric["max_drawdown"]
    row["train_sharpe"] = metric["sharpe"]
    row.update(flatten_candidate(candidate))
    return row


def flatten_candidate(candidate):
    """把 dataclass 参数展开，方便保存到 selected_params.csv。"""
    row = {}
    row.update(
        {
            f"entry_{key}": value
            for key, value in asdict(candidate["entry_config"]).items()
        }
    )
    row.update(
        {
            f"exit_{key}": value
            for key, value in asdict(candidate["exit_config"]).items()
        }
    )
    return row


def rebuild_symbol_curves(curves):
    """样本外窗口拼接后，按品种重新累计收益。"""
    rebuilt = []
    for symbol, group in curves.sort_values(["symbol", "date"]).groupby("symbol"):
        data = group[["date", "strategy_return", "benchmark_return", "excess_return"]]
        data = backtest.add_curves(data.copy())
        data.insert(0, "factor_name", FACTOR_NAME)
        data.insert(0, "factor_id", FACTOR_ID)
        data.insert(0, "symbol", symbol)
        rebuilt.append(data[[
            "symbol",
            "factor_id",
            "factor_name",
            "date",
            "strategy_return",
            "benchmark_return",
            "excess_return",
            "strategy_cumulative_return",
            "benchmark_cumulative_return",
            "excess_cumulative_return",
        ]])
    return pd.concat(rebuilt, ignore_index=True)


def build_final_symbol_metrics(curves, trades, periods):
    rows = []
    for symbol, group in curves.groupby("symbol"):
        symbol_trades = (
            trades.loc[trades["symbol"] == symbol].copy()
            if not trades.empty
            else empty_trade_table()
        )
        rows.append(
            backtest.summarize_symbol(
                group,
                symbol_trades,
                symbol,
                FACTOR_ID,
                FACTOR_NAME,
                periods,
            )
        )
    return pd.DataFrame(rows)


def data_date_range(symbol_data):
    starts = [item["hourly"]["date"].min() for item in symbol_data.values()]
    ends = [item["hourly"]["date"].max() for item in symbol_data.values()]
    return min(starts), max(ends)


def infer_periods(hourly):
    trading_days = hourly["trading_date"].nunique()
    if trading_days <= 0:
        return 252 * 9
    return 252 * len(hourly) / trading_days
