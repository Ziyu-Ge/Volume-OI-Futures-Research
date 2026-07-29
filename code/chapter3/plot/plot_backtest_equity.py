"""画做空回测净值曲线，并标出有交易的日期。"""

from pathlib import Path
import os

ROOT = Path(__file__).resolve().parents[3]
os.environ.setdefault(
    "MPLCONFIGDIR",
    str(ROOT / "results" / "chapter3" / ".matplotlib-cache"),
)

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import pandas as pd


CURVE_FILE = ROOT / "results" / "chapter3" / "short_backtest" / "daily_returns.csv"
TRADE_FILE = ROOT / "results" / "chapter3" / "short_backtest" / "trades.csv"
OUT_FILE = ROOT / "results" / "chapter3" / "figures" / "short_backtest_equity.png"


def setup_chinese_font():
    plt.rcParams["font.sans-serif"] = [
        "PingFang SC", "Heiti SC", "Microsoft YaHei", "SimHei", "Arial Unicode MS"
    ]
    plt.rcParams["axes.unicode_minus"] = False


def main():
    if not CURVE_FILE.exists() or not TRADE_FILE.exists():
        raise SystemExit("缺少回测结果，请先运行：python3 code/chapter3/backtest/run_short_backtest.py")

    setup_chinese_font()

    curve = pd.read_csv(CURVE_FILE, parse_dates=["交易日"])
    trades = pd.read_csv(TRADE_FILE, parse_dates=["交易日"])

    trade_days = trades["交易日"].drop_duplicates()
    trade_points = curve.loc[curve["交易日"].isin(trade_days)]

    fig, ax = plt.subplots(figsize=(11, 6))
    ax.plot(curve["交易日"], curve["累计净值"], color="#2563eb", linewidth=2, label="累计净值")
    ax.scatter(
        trade_points["交易日"],
        trade_points["累计净值"],
        color="#dc2626",
        s=24,
        label="有交易",
        zorder=3,
    )

    ax.axhline(1, color="#111827", linewidth=1, linestyle="--")
    ax.set_title("做空回测净值曲线")
    ax.set_xlabel("交易日")
    ax.set_ylabel("累计净值")
    ax.grid(True, color="#e5e7eb", linewidth=0.8)
    ax.legend()

    OUT_FILE.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(OUT_FILE, dpi=150)
    print(f"已输出: {OUT_FILE}")


if __name__ == "__main__":
    main()
