import os

project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
os.environ.setdefault("MPLCONFIGDIR", os.path.join(project_root, ".matplotlib"))

import pandas as pd
import numpy as np
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

# =========================
# 参数设置
# =========================

symbol = "LC"

price_column = "close"  # 用哪个价格列判断趋势内价格创新高
trend_for_divergence = "up_trend"  # 只在上涨趋势中判断价格创新高但投机度不创新高
min_segment_days = 2  # 趋势段内至少第几天以后才判断背离，避免趋势起点误判
price_new_high_epsilon = 1e-12  # 判断价格创新高时的容忍误差
speculation_new_high_epsilon = 1e-12  # 判断投机度是否创新高时的容忍误差

before_end_days = 5  # 趋势结束前观察天数

price_figsize = (12, 6)
divergence_figsize = (12, 6)
signal_point_size = 24
plot_dpi = 300

tables_dir = os.path.join(project_root, "results", "tables")
figures_dir = os.path.join(project_root, "results", "figures")

daily_input_path = os.path.join(tables_dir, f"{symbol}_daily_with_trend.csv")
segments_input_path = os.path.join(tables_dir, f"{symbol}_trend_segments.csv")
daily_output_path = os.path.join(
    tables_dir,
    f"{symbol}_daily_price_speculation_divergence_signal.csv"
)
summary_output_path = os.path.join(
    tables_dir,
    f"{symbol}_price_speculation_divergence_summary.csv"
)
price_figure_path = os.path.join(
    figures_dir,
    f"{symbol}_price_speculation_divergence_signal_on_price.png"
)
price_divergence_figure_path = os.path.join(
    figures_dir,
    f"{symbol}_price_and_price_speculation_divergence.png"
)


# =========================
# 1. 读取已经做好的趋势数据
# =========================

daily = pd.read_csv(daily_input_path)
segments = pd.read_csv(segments_input_path)

daily["date"] = pd.to_datetime(daily["date"])
segments["start_date"] = pd.to_datetime(segments["start_date"])
segments["end_date"] = pd.to_datetime(segments["end_date"])

daily = daily.sort_values("date").reset_index(drop=True)


# =========================
# 2. 计算价格创新高但投机度不创新高的背离因子
# =========================
# DivUp_t = I(P_t = max(P_i), i in [s, t]) * I(Spec_t < max(Spec_i), i in [s, t])
# 只在上涨趋势段内计算。这里 P 使用 price_column，默认 close。

daily["price_trend_high"] = np.nan
daily["speculation_trend_high"] = np.nan
daily["price_new_high_signal"] = 0
daily["speculation_new_high_signal"] = 0
daily["price_speculation_divergence_factor"] = 0
daily["speculation_gap_from_trend_high"] = np.nan
daily["segment_day_number"] = np.nan

for _, seg in segments.iterrows():

    segment_id = seg["segment_id"]

    mask = (
        (daily["segment_id"] == segment_id)
        & (daily["trend"] == trend_for_divergence)
    )

    part = daily[mask].copy()

    if len(part) == 0:
        continue

    price_high = part[price_column].cummax()
    speculation_high = part["speculation"].cummax()
    segment_day_number = np.arange(1, len(part) + 1)

    price_new_high = (
        part[price_column] >= price_high - price_new_high_epsilon
    )
    speculation_new_high = (
        part["speculation"]
        >= speculation_high - speculation_new_high_epsilon
    )
    enough_segment_days = segment_day_number >= min_segment_days

    divergence_signal = (
        price_new_high
        & (~speculation_new_high)
        & enough_segment_days
    )

    daily.loc[mask, "price_trend_high"] = price_high.values
    daily.loc[mask, "speculation_trend_high"] = speculation_high.values
    daily.loc[mask, "price_new_high_signal"] = price_new_high.astype(int).values
    daily.loc[mask, "speculation_new_high_signal"] = (
        speculation_new_high.astype(int).values
    )
    daily.loc[mask, "price_speculation_divergence_factor"] = (
        divergence_signal.astype(int).values
    )
    daily.loc[mask, "speculation_gap_from_trend_high"] = (
        speculation_high - part["speculation"]
    ).values
    daily.loc[mask, "segment_day_number"] = segment_day_number


# =========================
# 3. 按趋势段汇总
# =========================

trend_daily = daily[daily["trend"] == trend_for_divergence].copy()

result_rows = []

for _, seg in segments.iterrows():

    if seg["trend"] != trend_for_divergence:
        continue

    segment_id = seg["segment_id"]

    part = trend_daily[trend_daily["segment_id"] == segment_id].copy()

    if len(part) == 0:
        continue

    end_date = seg["end_date"]
    before_end = part[
        part["date"] >= end_date - pd.Timedelta(days=before_end_days)
    ]

    signal_part = part[
        part["price_speculation_divergence_factor"] == 1
    ]
    signal_before_end = before_end[
        before_end["price_speculation_divergence_factor"] == 1
    ]

    result_rows.append({
        "segment_id": segment_id,
        "trend": seg["trend"],
        "start_date": seg["start_date"],
        "end_date": seg["end_date"],
        "days": seg["days"],
        "return": seg["return"],

        "price_new_high_days_in_trend": part[
            "price_new_high_signal"
        ].sum(),
        "speculation_new_high_days_in_trend": part[
            "speculation_new_high_signal"
        ].sum(),
        "divergence_days_in_trend": part[
            "price_speculation_divergence_factor"
        ].sum(),
        "divergence_ratio_in_trend": part[
            "price_speculation_divergence_factor"
        ].mean(),
        "max_speculation_gap_in_trend": part[
            "speculation_gap_from_trend_high"
        ].max(),
        "mean_speculation_gap_in_trend": part[
            "speculation_gap_from_trend_high"
        ].mean(),
        "first_divergence_date": (
            signal_part["date"].min() if len(signal_part) > 0 else pd.NaT
        ),

        "price_new_high_days_before_end": before_end[
            "price_new_high_signal"
        ].sum(),
        "speculation_new_high_days_before_end": before_end[
            "speculation_new_high_signal"
        ].sum(),
        "divergence_days_before_end": before_end[
            "price_speculation_divergence_factor"
        ].sum(),
        "divergence_ratio_before_end": before_end[
            "price_speculation_divergence_factor"
        ].mean(),
        "max_speculation_gap_before_end": before_end[
            "speculation_gap_from_trend_high"
        ].max(),
        "mean_speculation_gap_before_end": before_end[
            "speculation_gap_from_trend_high"
        ].mean(),
        "first_divergence_date_before_end": (
            signal_before_end["date"].min()
            if len(signal_before_end) > 0
            else pd.NaT
        )
    })

divergence_summary = pd.DataFrame(result_rows)


# =========================
# 4. 保存结果表格
# =========================

os.makedirs(os.path.dirname(daily_output_path), exist_ok=True)
os.makedirs(os.path.dirname(summary_output_path), exist_ok=True)

daily.to_csv(daily_output_path, index=False)
divergence_summary.to_csv(summary_output_path, index=False)

print("价格-投机度背离因子分析完成。")
print(f"每日信号保存为：{daily_output_path}")
print(f"趋势段汇总保存为：{summary_output_path}")

print("\n趋势段汇总预览：")
print(divergence_summary.head(20))


# =========================
# 5. 画图：价格和背离信号
# =========================

plt.figure(figsize=price_figsize)

plt.plot(daily["date"], daily[price_column], label=price_column, color="tab:blue")

signal_points = daily[daily["price_speculation_divergence_factor"] == 1]

plt.scatter(
    signal_points["date"],
    signal_points[price_column],
    s=signal_point_size,
    label="price high without speculation high",
    color="tab:red"
)

plt.title(f"{symbol} Price-Speculation Divergence Signal on Price")
plt.xlabel("Date")
plt.ylabel(price_column.capitalize())
plt.legend()
plt.tight_layout()

os.makedirs(os.path.dirname(price_figure_path), exist_ok=True)
plt.savefig(price_figure_path, dpi=plot_dpi)
plt.close()


# =========================
# 6. 画图：价格和背离因子
# =========================

fig, ax1 = plt.subplots(figsize=divergence_figsize)

ax1.plot(
    daily["date"],
    daily[price_column],
    label=price_column,
    color="tab:blue"
)
ax1.set_xlabel("Date")
ax1.set_ylabel(price_column.capitalize(), color="tab:blue")
ax1.tick_params(axis="y", labelcolor="tab:blue")

ax2 = ax1.twinx()
ax2.step(
    daily["date"],
    daily["price_speculation_divergence_factor"],
    where="mid",
    label="DivUp",
    color="tab:orange"
)
ax2.set_ylabel("Divergence Factor", color="tab:orange")
ax2.tick_params(axis="y", labelcolor="tab:orange")
ax2.set_ylim(-0.1, 1.2)
ax2.set_yticks([0, 1])

plt.title(f"{symbol} Price and Price-Speculation Divergence Factor")
ax1.legend(loc="upper left")
ax2.legend(loc="upper right")
fig.tight_layout()

plt.savefig(price_divergence_figure_path, dpi=plot_dpi)
plt.close()

print("\n图片保存为：")
print(price_figure_path)
print(price_divergence_figure_path)
