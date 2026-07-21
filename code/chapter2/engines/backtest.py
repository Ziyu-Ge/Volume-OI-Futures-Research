import numpy as np
import pandas as pd

from core.metrics import add_curve_columns, summarize_returns
from core.reports import empty_trade_table


def add_return_columns(frame):
    """计算常态做多、信号期反向做空的单期收益和单利曲线。"""
    data = frame.copy()
    open_col = "hourly_open" if "hourly_open" in data.columns else "open"
    bar_open = data[open_col].replace(0, np.nan)
    previous_close = data["close"].shift(1).replace(0, np.nan)
    previous_position = data["position"].shift(1).fillna(0)

    data["gap_return"] = (bar_open / previous_close - 1).fillna(0)
    data["intraday_return"] = (data["close"] / bar_open - 1).fillna(0)
    if not data.empty:
        data.loc[data.index[0], "gap_return"] = 0

    data["benchmark_return"] = data["gap_return"] + data["intraday_return"]
    data["strategy_net_position"] = 1 + 2 * data["position"]
    data["previous_strategy_net_position"] = 1 + 2 * previous_position
    data["strategy_return"] = (
        data["previous_strategy_net_position"] * data["gap_return"]
        + data["strategy_net_position"] * data["intraday_return"]
    )
    data["excess_return"] = data["strategy_return"] - data["benchmark_return"]
    return add_curves(data)


def add_curves(frame):
    data = add_curve_columns(frame, "strategy_return", "strategy")
    data = add_curve_columns(data, "benchmark_return", "benchmark")
    data = add_curve_columns(data, "excess_return", "excess")
    return data


def infer_periods_per_year(frame, is_hourly):
    if not is_hourly:
        return 252
    trading_days = frame["trading_date"].nunique()
    if trading_days <= 0:
        return 252 * 9
    return 252 * len(frame) / trading_days


def build_trade_table(frame, symbol, factor_id, factor_name):
    trades = []
    open_trade = None
    for _, row in frame.iterrows():
        if row["trade_signal"] == -1 and open_trade is None:
            open_trade = {
                "entry_time": row["date"],
                "entry_price": row["entry_price"],
            }
            continue
        if row["trade_signal"] == 1 and open_trade is not None:
            trades.append(
                _trade_row(
                    factor_id,
                    factor_name,
                    symbol,
                    "closed",
                    open_trade,
                    row,
                )
            )
            open_trade = None

    if open_trade is not None:
        trades.append(
            {
                "factor_id": factor_id,
                "factor_name": factor_name,
                "symbol": symbol,
                "status": "open",
                "entry_time": open_trade["entry_time"],
                "exit_time": pd.NaT,
                "entry_price": open_trade["entry_price"],
                "exit_price": np.nan,
                "trade_return": np.nan,
                "exit_reason": "",
            }
        )

    if not trades:
        return empty_trade_table()
    return pd.DataFrame(trades)


def _trade_row(factor_id, factor_name, symbol, status, open_trade, exit_row):
    entry_price = open_trade["entry_price"]
    exit_price = exit_row["exit_price"]
    trade_return = np.nan
    if pd.notna(entry_price) and pd.notna(exit_price):
        trade_return = entry_price / exit_price - 1
    return {
        "factor_id": factor_id,
        "factor_name": factor_name,
        "symbol": symbol,
        "status": status,
        "entry_time": open_trade["entry_time"],
        "exit_time": exit_row["date"],
        "entry_price": entry_price,
        "exit_price": exit_price,
        "trade_return": trade_return,
        "exit_reason": exit_row["exit_reason"],
    }


def build_curve_table(frame, symbol, factor_id, factor_name):
    columns = [
        "date",
        "strategy_return",
        "benchmark_return",
        "excess_return",
        "strategy_cumulative_return",
        "benchmark_cumulative_return",
        "excess_cumulative_return",
    ]
    data = frame[columns].copy()
    data.insert(0, "factor_name", factor_name)
    data.insert(0, "factor_id", factor_id)
    data.insert(0, "symbol", symbol)
    return data


def summarize_symbol(frame, trades, symbol, factor_id, factor_name, periods_per_year):
    strategy = summarize_returns(frame["strategy_return"], periods_per_year)
    benchmark = summarize_returns(frame["benchmark_return"], periods_per_year)
    excess = summarize_returns(frame["excess_return"], periods_per_year)
    closed = trades[trades["status"] == "closed"] if not trades.empty else trades
    return {
        "factor_id": factor_id,
        "factor_name": factor_name,
        "symbol": symbol,
        "annual_return": strategy["annual_return"],
        "max_drawdown": strategy["max_drawdown"],
        "sharpe": strategy["sharpe"],
        "benchmark_annual_return": benchmark["annual_return"],
        "benchmark_max_drawdown": benchmark["max_drawdown"],
        "benchmark_sharpe": benchmark["sharpe"],
        "excess_annual_return": excess["annual_return"],
        "trade_count": len(closed),
        "win_rate": (closed["trade_return"] > 0).mean() if len(closed) else np.nan,
    }


def build_portfolio(curves, factor_id, factor_name, periods_per_year):
    """按每个时间点的可用品种等权汇总单期收益。"""
    portfolio = (
        curves.groupby("date", sort=True)
        .agg(
            strategy_return=("strategy_return", "mean"),
            benchmark_return=("benchmark_return", "mean"),
            symbol_count=("symbol", "nunique"),
        )
        .reset_index()
    )
    portfolio["excess_return"] = (
        portfolio["strategy_return"] - portfolio["benchmark_return"]
    )
    portfolio = add_curves(portfolio)
    portfolio.insert(0, "factor_name", factor_name)
    portfolio.insert(0, "factor_id", factor_id)
    portfolio.insert(0, "symbol", "ALL_SYMBOLS_EQUAL_WEIGHT")
    metric_row = summarize_symbol(
        portfolio,
        empty_trade_table(),
        "ALL_SYMBOLS_EQUAL_WEIGHT",
        factor_id,
        factor_name,
        periods_per_year,
    )
    return portfolio, metric_row
