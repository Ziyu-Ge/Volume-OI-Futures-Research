import numpy as np
import pandas as pd


def add_curve_columns(frame, return_column, prefix):
    """把单期收益转成累计收益、净值和回撤。"""
    data = frame.copy()
    returns = data[return_column].fillna(0)
    data[f"{prefix}_cumulative_return"] = (1 + returns).cumprod() - 1
    data[f"{prefix}_equity"] = 1 + data[f"{prefix}_cumulative_return"]
    high = data[f"{prefix}_equity"].cummax()
    data[f"{prefix}_drawdown"] = data[f"{prefix}_equity"] / high - 1
    return data


def annual_return(returns, periods_per_year):
    returns = pd.Series(returns).fillna(0)
    if returns.empty:
        return np.nan
    total_return = (1 + returns).prod() - 1
    if total_return <= -1:
        return -1.0
    return (1 + total_return) ** (periods_per_year / len(returns)) - 1


def max_drawdown(returns):
    returns = pd.Series(returns).fillna(0)
    if returns.empty:
        return np.nan
    equity = (1 + returns).cumprod()
    return (equity / equity.cummax() - 1).min()


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

