from pathlib import Path

import pandas as pd


project_root = Path(__file__).resolve().parents[2]
input_file = (
    project_root
    / "results"
    / "chapter2"
    / "factor_24_rolling"
    / "tables"
    / "trades.csv"
)
output_file = input_file.parent / "holding_time_stats.csv"

trades = pd.read_csv(input_file)
trades = trades[trades["status"] == "closed"].copy()
trades["entry_time"] = pd.to_datetime(trades["entry_time"])
trades["exit_time"] = pd.to_datetime(trades["exit_time"])
trades["holding_days"] = (
    trades["exit_time"] - trades["entry_time"]
).dt.total_seconds() / (24 * 60 * 60)

stats = trades["holding_days"].describe().round(2).reset_index()
stats.columns = ["statistic", "holding_days"]
stats.to_csv(output_file, index=False)

print(stats.to_string(index=False))
print(f"\n统计表已保存到：{output_file}")
