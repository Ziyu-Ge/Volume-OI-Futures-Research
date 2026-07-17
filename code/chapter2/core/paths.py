import os
from pathlib import Path


CHAPTER_DIR = Path(__file__).resolve().parents[1]
PROJECT_ROOT = CHAPTER_DIR.parents[1]
DATA_DIR = PROJECT_ROOT / "data"
RESULTS_DIR = PROJECT_ROOT / "results" / "chapter2"
DAILY_DIR = RESULTS_DIR / "tables" / "daily"
HOURLY_DIR = RESULTS_DIR / "tables" / "hourly"


def factor_output_dir(factor_id):
    return RESULTS_DIR / f"factor_{factor_id}"


def setup_runtime_dirs():
    """把 matplotlib/cache 放到项目内，避免写用户全局目录。"""
    matplotlib_dir = PROJECT_ROOT / ".matplotlib"
    cache_dir = PROJECT_ROOT / ".cache"
    os.environ.setdefault("MPLCONFIGDIR", str(matplotlib_dir))
    os.environ.setdefault("XDG_CACHE_HOME", str(cache_dir))
    matplotlib_dir.mkdir(exist_ok=True)
    cache_dir.mkdir(exist_ok=True)

