import numpy as np
import pandas as pd


def positive_part(series):
    return series.clip(lower=0).fillna(0)


def regression_slope(values):
    """普通一元线性回归斜率；有缺失值时返回空值。"""
    values = np.asarray(values, dtype=float)
    if np.isnan(values).any():
        return np.nan
    x = np.arange(len(values), dtype=float)
    x = x - x.mean()
    denominator = np.square(x).sum()
    if denominator == 0:
        return np.nan
    return np.dot(x, values - values.mean()) / denominator


def add_ma_features(frame, short_window, long_window, trend_window):
    """增加均线、均线比例和收盘价乖离。"""
    data = frame.copy()
    for window, name in [
        (short_window, "ma_short"),
        (long_window, "ma_long"),
        (trend_window, "ma_trend"),
    ]:
        data[name] = data["close"].rolling(window, min_periods=window).mean()

    data["ma_short_long_ratio"] = data["ma_short"] / data["ma_long"]
    data["ma_long_trend_ratio"] = data["ma_long"] / data["ma_trend"]
    data["close_ma_short_bias"] = data["close"] / data["ma_short"] - 1
    data["close_ma_long_bias"] = data["close"] / data["ma_long"] - 1
    data["close_ma_trend_bias"] = data["close"] / data["ma_trend"] - 1
    data["ma_bias_spread"] = (
        data["close_ma_long_bias"] - data["close_ma_short_bias"]
    )
    data["ma_long_bias_spread"] = (
        data["close_ma_trend_bias"] - data["close_ma_long_bias"]
    )
    return data


def add_slope_features(frame, column, prefix, window):
    """增加回归斜率、斜率相对均值和斜率向下信号。"""
    data = frame.copy()
    slope_col = f"{prefix}_slope"
    mean_col = f"{prefix}_mean"
    rate_col = f"{prefix}_slope_rate"
    down_col = f"{prefix}_slope_down"

    data[slope_col] = (
        data[column].rolling(window, min_periods=window).apply(regression_slope, raw=True)
    )
    data[mean_col] = data[column].rolling(window, min_periods=window).mean()
    data[rate_col] = data[slope_col] / data[mean_col].replace(0, np.nan)
    data[down_col] = (data[slope_col] < 0).astype(int)
    return data


def add_volatility_features(frame, window):
    """历史波动率只用前 window 根完整 bar。"""
    data = frame.copy()
    data["price_range"] = data["high"] - data["low"]
    data["price_range_rate"] = data["price_range"] / data["close"].replace(0, np.nan)
    data["avg_price_range"] = (
        data["price_range"].shift(1).rolling(window, min_periods=window).mean()
    )
    data["avg_volatility_rate"] = (
        data["price_range_rate"].shift(1).rolling(window, min_periods=window).mean()
    )
    return data


def add_speculation(frame):
    """投机度定义为 log(volume / open_interest)。"""
    data = frame.copy()
    ratio = data["volume"] / data["open_interest"].replace(0, np.nan)
    ratio = ratio.replace([np.inf, -np.inf], np.nan)
    data["speculation"] = np.nan
    valid = ratio > 0
    data.loc[valid, "speculation"] = np.log(ratio.loc[valid])
    return data
