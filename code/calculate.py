import os

import pandas as pd


symbol = os.environ.get("SYMBOL", "LC")
direction_mode = os.environ.get("DIRECTION_MODE", "open_close")

project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

daily_input_path = os.environ.get(
    "DAILY_INPUT_PATH",
    os.path.join(
        project_root,
        "results",
        "tables",
        "daily",
        f"{symbol}_daily_with_trend.csv",
    ),
)

output_path = os.path.join(
    project_root,
    "results",
    "tables",
    "daily",
    f"{symbol}_consecutive_day_direction_probability.csv",
)


def classify_direction(data):
    if direction_mode == "open_close":
        price_change = data["close"] - data["open"]
    elif direction_mode == "close_to_close":
        price_change = data["close"].diff()
    else:
        raise ValueError(
            "DIRECTION_MODE 只能是 open_close 或 close_to_close"
        )

    direction = price_change.map(
        lambda value: "up" if value > 0 else "down" if value < 0 else "flat"
    )

    return price_change, direction


daily = pd.read_csv(daily_input_path)
daily["date"] = pd.to_datetime(daily["date"])
daily = daily.sort_values("date").reset_index(drop=True)

daily["price_change"], daily["direction"] = classify_direction(daily)
daily["next_direction"] = daily["direction"].shift(-1)

# 只统计第一天和第二天都明确上涨/下跌的样本，剔除平盘和最后一天。
valid = daily[
    daily["direction"].isin(["up", "down"])
    & daily["next_direction"].isin(["up", "down"])
].copy()

rows = []
for first_direction, second_direction, label in [
    ("up", "up", "第一天涨，第二天也涨"),
    ("up", "down", "第一天涨，第二天跌"),
    ("down", "up", "第一天跌，第二天涨"),
    ("down", "down", "第一天跌，第二天也跌"),
]:
    first_day_count = int((valid["direction"] == first_direction).sum())
    second_day_count = int(
        (
            (valid["direction"] == first_direction)
            & (valid["next_direction"] == second_direction)
        ).sum()
    )
    probability = (
        second_day_count / first_day_count
        if first_day_count > 0
        else None
    )

    rows.append(
        {
            "symbol": symbol,
            "direction_mode": direction_mode,
            "event": label,
            "first_day_direction": first_direction,
            "second_day_direction": second_direction,
            "first_day_count": first_day_count,
            "second_day_count": second_day_count,
            "probability": probability,
        }
    )

summary = pd.DataFrame(rows)
summary.to_csv(output_path, index=False)

print("连续两天涨跌概率")
print("数据文件:", daily_input_path)
print("方向口径:", direction_mode)
print(summary.to_string(index=False))
print("结果已保存:", output_path)
