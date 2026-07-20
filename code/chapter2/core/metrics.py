import numpy as np
import pandas as pd


def add_curve_columns(frame, return_column, prefix):
    """按固定初始本金，把单期收益累加成单利收益曲线。"""
    data = frame.copy()
    returns = data[return_column].fillna(0)
    data[f"{prefix}_cumulative_return"] = returns.cumsum()
    data[f"{prefix}_equity"] = 1 + data[f"{prefix}_cumulative_return"]
    high = data[f"{prefix}_equity"].cummax().clip(lower=1.0)
    data[f"{prefix}_drawdown"] = data[f"{prefix}_equity"] / high - 1
    return data


def annual_return(returns, periods_per_year):
    """按固定初始本金线性年化单期收益。"""
    returns = pd.Series(returns).fillna(0)
    if returns.empty:
        return np.nan
    return returns.sum() * periods_per_year / len(returns)


def max_drawdown(returns):
    """计算单利权益曲线相对历史高点的最大回撤。"""
    returns = pd.Series(returns).fillna(0)
    if returns.empty:
        return np.nan
    equity = 1 + returns.cumsum()
    high = equity.cummax().clip(lower=1.0)
    return (equity / high - 1).min()


def sharpe(returns, periods_per_year):
    returns = pd.Series(returns).fillna(0)
    std = returns.std(ddof=1)
    if pd.isna(std) or std == 0:
        return np.nan
    return returns.mean() / std * np.sqrt(periods_per_year)


def summarize_returns(returns, periods_per_year):
    return {
        "annual_return": annual_return(returns, periods_per_year),
        "max_drawdown": max_drawdown(returns),
        "sharpe": sharpe(returns, periods_per_year),
    }
