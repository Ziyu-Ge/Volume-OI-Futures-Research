from pathlib import Path

import pandas as pd

from core.io import write_csv


def output_paths(output_dir):
    base = Path(output_dir)
    return {
        "metrics": base / "tables" / "symbol_metrics.csv",
        "trades": base / "tables" / "trades.csv",
        "curves": base / "tables" / "curves.csv",
        "portfolio": base / "tables" / "portfolio_curve.csv",
        "signals": base / "figures" / "signals",
        "strategy": base / "figures" / "strategy",
        "return_summary": base / "figures" / "return_summary.png",
        "summary": base / "figures" / "all_symbols_summary.png",
    }


def save_tables(output_dir, metrics, trades, curves, portfolio):
    """只保留报告需要的简化表格。"""
    paths = output_paths(output_dir)
    write_csv(metrics, paths["metrics"])
    write_csv(trades, paths["trades"])
    write_csv(curves, paths["curves"])
    write_csv(portfolio, paths["portfolio"])
    return paths


def empty_trade_table():
    return pd.DataFrame(
        columns=[
            "factor_id",
            "factor_name",
            "symbol",
            "status",
            "entry_time",
            "exit_time",
            "entry_price",
            "exit_price",
            "trade_return",
            "exit_reason",
        ]
    )
