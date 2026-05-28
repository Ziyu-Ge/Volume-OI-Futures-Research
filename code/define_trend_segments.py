import pandas as pd
import numpy as np


# =========================
# 1. 读取分钟数据
# =========================

df = pd.read_csv("../data/LC.csv")

df["datetime"] = pd.to_datetime(df["datetime"])
df = df.sort_values("datetime")

df["date"] = df["datetime"].dt.date


# =========================
# 2. 分钟数据聚合成日频数据
# =========================

daily = df.groupby("date").agg({
    "open": "first", 
    "close": "last",
    "high": "max",
    "low": "min",
    "volume": "sum", # 当天成交量
    "total_turnover": "sum", # 当天总成交额
    "open_interest": "last" # 当天持仓量
}).reset_index()

daily["date"] = pd.to_datetime(daily["date"])


# =========================
# 3. 计算投机度
# =========================

daily.loc[daily["open_interest"] <= 0, "open_interest"] = np.nan #把小于等于0的持仓量置为NaN

daily["speculation"] = np.log(
    daily["volume"] / daily["open_interest"]
)


# =========================
# 4. 用波段高低点定义趋势段
# =========================

def find_trend_segments(daily, threshold=0.08, min_days=5):
    """
    用波段高低点定义趋势段。根据每日收盘价，找出上涨趋势段和下跌趋势段。

    threshold = 0.08 表示：
    上涨后从高点回撤超过 8%，确认上涨段结束；
    下跌后从低点反弹超过 8%，确认下跌段结束。

    min_days = 5 表示：
    少于 5 个交易日的波段不作为正式趋势段。
    """

    prices = daily["close"].values

    # 记录重要拐点的位置
    pivot_indices = [0]

    # 当前方向：
    # 0  = 还没有方向
    # 1  = 当前在上涨波段中
    # -1 = 当前在下跌波段中
    direction = 0

    # 当前参考点
    pivot_price = prices[0]

    # 当前波段中的最高点或最低点
    extreme_index = 0
    extreme_price = prices[0]

    for i in range(1, len(daily)):

        price = prices[i]

        # =========================
        # 还没有确认方向
        # =========================
        if direction == 0:

            change = (price - pivot_price) / pivot_price

            # 从起点上涨超过 threshold，确认进入上涨波段
            if change >= threshold:
                direction = 1
                extreme_index = i
                extreme_price = price

            # 从起点下跌超过 threshold，确认进入下跌波段
            elif change <= -threshold:
                direction = -1
                extreme_index = i
                extreme_price = price

        # =========================
        # 当前是上涨波段
        # =========================
        elif direction == 1:

            # 如果继续创新高，就更新最高点
            if price > extreme_price:
                extreme_price = price
                extreme_index = i

            # 如果从最高点回撤超过 threshold，
            # 说明上涨波段结束，最高点成为一个拐点
            drawdown = (extreme_price - price) / extreme_price

            if drawdown >= threshold:
                pivot_indices.append(extreme_index)

                # 之后开始观察下跌波段
                direction = -1
                extreme_index = i
                extreme_price = price
                pivot_price = price

        # =========================
        # 当前是下跌波段
        # =========================
        elif direction == -1:

            # 如果继续创新低，就更新最低点
            if price < extreme_price:
                extreme_price = price
                extreme_index = i

            # 如果从最低点反弹超过 threshold，
            # 说明下跌波段结束，最低点成为一个拐点
            rebound = (price - extreme_price) / extreme_price

            if rebound >= threshold:
                pivot_indices.append(extreme_index)

                # 之后开始观察上涨波段
                direction = 1
                extreme_index = i
                extreme_price = price
                pivot_price = price

    # 把最后一个极值点也加入
    if pivot_indices[-1] != extreme_index:
        pivot_indices.append(extreme_index)

    # =========================
    # 根据相邻拐点生成趋势段
    # =========================

    segment_rows = []

    for j in range(len(pivot_indices) - 1):

        start_i = pivot_indices[j]
        end_i = pivot_indices[j + 1]

        start_price = prices[start_i]
        end_price = prices[end_i]

        days = end_i - start_i + 1
        total_return = (end_price - start_price) / start_price

        # 太短的不算正式趋势
        if days < min_days:
            continue

        # 涨跌幅太小的不算正式趋势
        if abs(total_return) < threshold:
            continue

        if total_return > 0:
            trend = "up_trend"
        else:
            trend = "down_trend"

        part = daily.iloc[start_i:end_i + 1]

        segment_rows.append({
            "segment_id": len(segment_rows) + 1,
            "trend": trend,
            "start_date": daily.loc[start_i, "date"],
            "end_date": daily.loc[end_i, "date"],
            "start_close": start_price,
            "end_close": end_price,
            "return": total_return,
            "highest_close": part["close"].max(),
            "lowest_close": part["close"].min(),
            "days": days,
            "mean_speculation": part["speculation"].mean(),
            "max_speculation": part["speculation"].max(),
            "mean_volume": part["volume"].mean(),
            "mean_open_interest": part["open_interest"].mean()
        })

    segments = pd.DataFrame(segment_rows)

    return segments


segments = find_trend_segments(
    daily,
    threshold=0.08,
    min_days=5
)


# =========================
# 5. 给 daily 标记 trend 和 segment_id
# =========================

daily["trend"] = "no_trend"
daily["segment_id"] = np.nan

for _, row in segments.iterrows():

    mask = (
        (daily["date"] >= row["start_date"]) &
        (daily["date"] <= row["end_date"])
    )

    daily.loc[mask, "trend"] = row["trend"]
    daily.loc[mask, "segment_id"] = row["segment_id"]


# =========================
# 6. 保存结果
# =========================

daily.to_csv("../results/tables/LC_daily_with_trend.csv", index=False)
segments.to_csv("../results/tables/LC_trend_segments.csv", index=False)

print("趋势段定义完成。")
print("每日趋势结果保存为：../results/tables/LC_daily_with_trend.csv")
print("趋势段汇总保存为：../results/tables/LC_trend_segments.csv")

print("\n趋势段预览：")
print(segments.head(20))

print("\n趋势段数量：")
print(segments["trend"].value_counts())

