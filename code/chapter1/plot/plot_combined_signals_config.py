import os
from pathlib import Path


# 路径、默认参数和运行缓存目录。
PLOT_DIR = Path(__file__).resolve().parent
CODE_DIR = PLOT_DIR.parent
PROJECT_ROOT = CODE_DIR.parents[1]
RESULTS_DIR = PROJECT_ROOT / "results" / "chapter1"
DEFAULT_RUNS_DIR = RESULTS_DIR
DEFAULT_OUTPUT_DIR = RESULTS_DIR / "combined"
DEFAULT_FACTOR_IDS = ("11", "12", "13", "14")


def setup_cache_dirs():
    """设置 matplotlib/plotly 运行时缓存目录。"""
    os.environ.setdefault("MPLCONFIGDIR", str(PROJECT_ROOT / ".matplotlib"))
    os.environ.setdefault("XDG_CACHE_HOME", str(PROJECT_ROOT / ".cache"))
    (PROJECT_ROOT / ".matplotlib").mkdir(exist_ok=True)
    (PROJECT_ROOT / ".cache").mkdir(exist_ok=True)
