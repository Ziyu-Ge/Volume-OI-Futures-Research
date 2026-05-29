import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# =========================
# 参数设置
# =========================

symbol = "LC"

# =========================
# 1. 读取分钟数据
# =========================

df = pd.read_csv(f"../data/{symbol}.csv")

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
    "volume": "sum",
    "total_turnover": "sum",
    "open_interest": "last"
}).reset_index()

daily["date"] = pd.to_datetime(daily["date"])


# =========================
# 3. 计算投机度
# =========================

daily.loc[daily["open_interest"] <= 0, "open_interest"] = np.nan

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

    # 记录“确认趋势结束”的位置
    # 注意：趋势真正的终点是最高点/最低点，
    # 但确认趋势结束是在之后回撤/反弹超过 threshold 的那一天
    signal_indices = []

    direction = 0
    pivot_price = prices[0]

    extreme_index = 0
    extreme_price = prices[0]

    def start_new_direction(new_direction, i, price, update_pivot=False):
        nonlocal direction, extreme_index, extreme_price, pivot_price

        direction = new_direction
        extreme_index = i
        extreme_price = price

        if update_pivot:
            pivot_price = price

    def update_extreme(i, price):
        nonlocal extreme_index, extreme_price

        if direction == 1 and price > extreme_price:
            extreme_index = i
            extreme_price = price

        elif direction == -1 and price < extreme_price:
            extreme_index = i
            extreme_price = price

    for i in range(1, len(daily)):

        price = prices[i]

        # =========================
        # 还没有确认方向
        # =========================
        if direction == 0:

            change = (price - pivot_price) / pivot_price

            if change >= threshold:
                start_new_direction(1, i, price)

            elif change <= -threshold:
                start_new_direction(-1, i, price)

        # =========================
        # 当前是上涨波段
        # =========================
        elif direction == 1:

            update_extreme(i, price)

            # 从最高点回撤超过 threshold，确认上涨波段结束
            drawdown = (extreme_price - price) / extreme_price

            if drawdown >= threshold:
                pivot_indices.append(extreme_index)

                # i 是确认上涨趋势结束的那一天
                signal_indices.append(i)

                # 之后开始观察下跌波段
                start_new_direction(-1, i, price, update_pivot=True)

        # =========================
        # 当前是下跌波段
        # =========================
        elif direction == -1:

            update_extreme(i, price)

            # 从最低点反弹超过 threshold，确认下跌波段结束
            rebound = (price - extreme_price) / extreme_price

            if rebound >= threshold:
                pivot_indices.append(extreme_index)

                # i 是确认下跌趋势结束的那一天
                signal_indices.append(i)

                # 之后开始观察上涨波段
                start_new_direction(1, i, price, update_pivot=True)

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

        if days < min_days:
            continue

        if abs(total_return) < threshold:
            continue

        if total_return > 0:
            trend = "up_trend"
        else:
            trend = "down_trend"

        part = daily.iloc[start_i:end_i + 1]

        # 有些最后一个波段可能还没有出现确认结束信号
        if j < len(signal_indices):
            signal_i = signal_indices[j]
            signal_date = daily.loc[signal_i, "date"]
            signal_close = prices[signal_i]
        else:
            signal_i = np.nan
            signal_date = pd.NaT
            signal_close = np.nan

        segment_rows.append({
            "segment_id": len(segment_rows) + 1,
            "trend": trend,

            "start_index": start_i,
            "end_index": end_i,
            "end_signal_index": signal_i,

            "start_date": daily.loc[start_i, "date"],
            "end_date": daily.loc[end_i, "date"],

            # 这个点是“确认趋势结束”的点
            "end_signal_date": signal_date,
            "end_signal_close": signal_close,

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
# 6. 画出价格、趋势段、趋势结束确认点
# =========================

def plot_trend_segments(daily, segments, save_path=None):
    """
    在价格图上画出：
    1. 每个上涨/下跌波段；
    2. 每个波段结束的确认点。
    """

    plt.figure(figsize=(16, 7))

    # 先画完整收盘价，作为背景
    plt.plot(
        daily["date"],
        daily["close"],
        color="lightgray",
        linewidth=1,
        label="Close Price"
    )

    shown_up_label = False
    shown_down_label = False
    shown_signal_label = False

    for _, row in segments.iterrows():

        segment_data = daily[
            (daily["date"] >= row["start_date"]) &
            (daily["date"] <= row["end_date"])
        ]

        if row["trend"] == "up_trend":
            color = "red"
            label = "Up Trend" if not shown_up_label else None
            shown_up_label = True
        else:
            color = "green"
            label = "Down Trend" if not shown_down_label else None
            shown_down_label = True

        # 画趋势段
        plt.plot(
            segment_data["date"],
            segment_data["close"],
            color=color,
            linewidth=2.5,
            label=label
        )

        # 标出趋势段起点
        plt.scatter(
            row["start_date"],
            row["start_close"],
            color=color,
            s=35
        )

        # 标出趋势段真正结束点，也就是最高点/最低点
        plt.scatter(
            row["end_date"],
            row["end_close"],
            color=color,
            s=60,
            edgecolors="black",
            zorder=5
        )

        # 标出确认趋势结束的点
        if pd.notna(row["end_signal_date"]):
            plt.scatter(
                row["end_signal_date"],
                row["end_signal_close"],
                color="orange",
                marker="X",
                s=100,
                edgecolors="black",
                zorder=6,
                label="Trend End Signal" if not shown_signal_label else None
            )
            shown_signal_label = True

            # 从真正的极值点连到确认点，表示：
            # 趋势在 end_date 结束，但到 end_signal_date 才被确认
            plt.plot(
                [row["end_date"], row["end_signal_date"]],
                [row["end_close"], row["end_signal_close"]],
                color="orange",
                linestyle="--",
                linewidth=1
            )

        # 在每个波段中间标上 segment_id
        mid_i = int((row["start_index"] + row["end_index"]) / 2)
        plt.text(
            daily.loc[mid_i, "date"],
            daily.loc[mid_i, "close"],
            str(int(row["segment_id"])),
            fontsize=9,
            ha="center",
            va="bottom"
        )

    plt.title("Price Trend Segments and Trend End Signals")
    plt.xlabel("Date")
    plt.ylabel("Close Price")
    plt.legend()
    plt.grid(alpha=0.3)
    plt.tight_layout()

    if save_path is not None:
        plt.savefig(save_path, dpi=300)

    plt.close()


plot_trend_segments(
    daily,
    segments,
    save_path="../results/figures/LC_trend_segments.png"
)


# =========================
# 7. 保存结果
# =========================

daily.to_csv("../results/tables/LC_daily_with_trend.csv", index=False)
segments.to_csv("../results/tables/LC_trend_segments.csv", index=False)

print("趋势段定义完成。")
print("每日趋势结果保存为：../results/tables/LC_daily_with_trend.csv")
print("趋势段汇总保存为：../results/tables/LC_trend_segments.csv")
print("趋势段图保存为：../results/figures/LC_trend_segments.png")

print("\n趋势段预览：")
print(segments.head(20))

print("\n趋势段数量：")
print(segments["trend"].value_counts())

