import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# =========================
# 参数设置
# =========================

symbol = "LC"
data_path = f"../data/{symbol}.csv"

threshold_window = 30
threshold_quantile = 0.9
threshold_multiplier = 3
min_threshold = 0.04
max_threshold = 0.10
initial_threshold = min_threshold

min_trend_days = 5

plot_figsize = (16, 7)
plot_dpi = 300

trend_figure_path = f"../results/figures/{symbol}_trend_segments.png"
daily_output_path = f"../results/tables/{symbol}_daily_with_trend.csv"
segments_output_path = f"../results/tables/{symbol}_trend_segments.csv"

# =========================
# 1. 读取分钟数据
# =========================

df = pd.read_csv(data_path)

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
def add_dynamic_threshold(daily, window, quantile, multiplier,
                          min_threshold, max_threshold,
                          initial_threshold):
    """
    根据历史价格波动，为每天计算趋势反转阈值。

    逻辑：
    1. 先计算每日收益率，取绝对值，只看波动大小，不看涨跌方向；
    2. 用过去 window 天的 quantile 分位数代表“较大的正常波动”；
    3. shift(1)，保证今天的阈值只使用昨天及以前的数据；
    4. 再乘以 multiplier，作为趋势反转阈值；
    5. 最后限制在 min_threshold 和 max_threshold 之间。
    """

    abs_return = daily["close"].pct_change().abs()

    threshold = (
        abs_return
        .rolling(window=window) # 比如 window=30，意思是每一天都看它前后对应窗口里的 30 天数据。
        .quantile(quantile) # 在每个滚动窗口里取分位数。
        .shift(1) # 它保证今天的阈值只能使用昨天及以前的数据，不能偷看今天的数据。
        * multiplier 
    )

    initial_threshold = min_threshold
    
    # 把算好的阈值放进 daily 数据表里，新增一列 threshold。
    daily["threshold"] = (
        threshold
        .clip(lower=min_threshold, upper=max_threshold)
        .fillna(initial_threshold)
    )


def find_trend_segments(daily, min_days):
    """
    用波段高低点定义趋势段。根据每日收盘价，找出上涨趋势段和下跌趋势段。

    daily["threshold"] 表示每天使用的动态阈值：
    上涨后从高点回撤超过该阈值，确认上涨段结束；
    下跌后从低点反弹超过该阈值，确认下跌段结束。

    min_days 表示：
    少于该交易日数量的波段不作为正式趋势段。
    """

    prices = daily["close"].values
    thresholds = daily["threshold"].values

    # 记录重要拐点的位置
    pivot_indices = [0]

    # 记录“确认趋势结束”的位置
    # 注意：趋势真正的终点是最高点/最低点，
    # 但确认趋势结束是在之后回撤/反弹超过 threshold 的那一天
    signal_indices = []

    direction = 0
    pivot_price = prices[0]

    extreme_index = 0
    extreme_price = prices[0] # 一开始极值点就是第 0 天。

    def start_new_direction(new_direction, i, price, update_pivot=False):
        ### 当趋势方向发生变化，更新当前状态 ###
        nonlocal direction, extreme_index, extreme_price, pivot_price

        direction = new_direction # 1 表示上涨波段，-1 表示下跌波段
        extreme_index = i # 新趋势刚开始时，当前点暂时就是这个新趋势里的极值点。
        extreme_price = price

        if update_pivot: # 是否要更新 pivot_price，发生在趋势反转确认之后
            pivot_price = price

    def update_extreme(i, price):
        ### 作用是：在当前趋势中，更新最高点或最低点 ###
        nonlocal extreme_index, extreme_price

        if direction == 1 and price > extreme_price:
            extreme_index = i
            extreme_price = price

        elif direction == -1 and price < extreme_price:
            extreme_index = i
            extreme_price = price

    for i in range(1, len(daily)):

        price = prices[i]
        threshold = thresholds[i]

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
            segment_threshold = thresholds[signal_i]
        else:
            signal_i = np.nan
            signal_date = pd.NaT
            signal_close = np.nan
            segment_threshold = thresholds[end_i]

        if days < min_days:
            continue

        if abs(total_return) < segment_threshold:
            continue

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
            "threshold": segment_threshold,

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


add_dynamic_threshold(
    daily,
    window=threshold_window,
    quantile=threshold_quantile,
    multiplier=threshold_multiplier,
    min_threshold=min_threshold,
    max_threshold=max_threshold,
    initial_threshold=initial_threshold
)

print(
    "本次使用动态阈值，范围为："
    f"{daily['threshold'].min():.2%} - {daily['threshold'].max():.2%}"
)

segments = find_trend_segments(
    daily,
    min_days=min_trend_days
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

    plt.figure(figsize=plot_figsize)

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
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path, dpi=plot_dpi)

    plt.close()


plot_trend_segments(
    daily,
    segments,
    save_path=trend_figure_path
)


# =========================
# 7. 保存结果
# =========================

os.makedirs(os.path.dirname(daily_output_path), exist_ok=True)
os.makedirs(os.path.dirname(segments_output_path), exist_ok=True)

daily.to_csv(daily_output_path, index=False)
segments.to_csv(segments_output_path, index=False)

print("趋势段定义完成。")
print(f"每日趋势结果保存为：{daily_output_path}")
print(f"趋势段汇总保存为：{segments_output_path}")
print(f"趋势段图保存为：{trend_figure_path}")

print("\n趋势段预览：")
print(segments.head(20))

print("\n趋势段数量：")
print(segments["trend"].value_counts())
