from pathlib import Path

import pandas as pd


DAILY_COLUMNS = {
    "date",
    "open",
    "close",
    "high",
    "low",
    "volume",
    "total_turnover",
    "open_interest",
}
HOURLY_COLUMNS = {"date", "trading_date", "open", "close", "high", "low"}


def _symbols_from_files(data_dir, suffix):
    if not data_dir.is_dir():
        raise FileNotFoundError(f"目录不存在：{data_dir}")
    symbols = sorted(
        path.name[: -len(suffix)].upper()
        for path in data_dir.glob(f"*{suffix}")
        if path.is_file()
    )
    if not symbols:
        raise FileNotFoundError(f"{data_dir} 中没有 *{suffix} 文件")
    return symbols


def discover_daily_symbols(daily_dir):
    return _symbols_from_files(Path(daily_dir), "_daily.csv")


def discover_hourly_symbols(hourly_dir):
    return _symbols_from_files(Path(hourly_dir), "_hourly.csv")


def discover_common_symbols(daily_dir, hourly_dir):
    daily_symbols = set(discover_daily_symbols(daily_dir))
    hourly_symbols = set(discover_hourly_symbols(hourly_dir))
    symbols = sorted(daily_symbols & hourly_symbols)
    if not symbols:
        raise FileNotFoundError("没有同时具备日频和小时频率缓存的品种")
    return symbols


def select_symbols(symbols, selected):
    if selected is None:
        return list(symbols)
    wanted = {symbol.strip().upper() for symbol in selected}
    picked = [symbol for symbol in symbols if symbol in wanted]
    if not picked:
        raise FileNotFoundError(f"未找到指定品种：{','.join(sorted(wanted))}")
    return picked


def _to_numeric(frame, columns):
    for column in columns:
        if column in frame.columns:
            frame[column] = pd.to_numeric(frame[column], errors="coerce")


def load_daily(symbol, daily_dir):
    path = Path(daily_dir) / f"{symbol}_daily.csv"
    if not path.exists():
        raise FileNotFoundError(f"缺少日频数据：{path}")
    frame = pd.read_csv(path)
    missing = DAILY_COLUMNS - set(frame.columns)
    if missing:
        raise ValueError(f"{path} 缺少字段：{','.join(sorted(missing))}")
    frame["date"] = pd.to_datetime(frame["date"])
    _to_numeric(
        frame,
        ["open", "close", "high", "low", "volume", "total_turnover", "open_interest", "speculation"],
    )
    return frame.sort_values("date").reset_index(drop=True)


def load_hourly(symbol, hourly_dir):
    path = Path(hourly_dir) / f"{symbol}_hourly.csv"
    if not path.exists():
        raise FileNotFoundError(f"缺少小时频率数据：{path}")
    frame = pd.read_csv(path)
    missing = HOURLY_COLUMNS - set(frame.columns)
    if missing:
        raise ValueError(f"{path} 缺少字段：{','.join(sorted(missing))}")
    frame["date"] = pd.to_datetime(frame["date"])
    frame["trading_date"] = pd.to_datetime(frame["trading_date"])
    _to_numeric(
        frame,
        ["open", "close", "high", "low", "volume", "total_turnover", "open_interest", "speculation"],
    )
    return frame.sort_values(["trading_date", "date"]).reset_index(drop=True)


def write_csv(frame, path):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    frame.to_csv(path, index=False)
    return path

