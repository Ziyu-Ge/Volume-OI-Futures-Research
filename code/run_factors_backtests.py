import os
import subprocess
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
FACTOR_DIR = ROOT / "code" / "factors"
BACKTEST_DIR = ROOT / "code" / "backtest"
BASE_ENV = os.environ.copy()
BASE_ENV["MPLCONFIGDIR"] = str(ROOT / ".matplotlib")
BASE_ENV["XDG_CACHE_HOME"] = str(ROOT / ".cache")

(ROOT / ".matplotlib").mkdir(exist_ok=True)
(ROOT / ".cache").mkdir(exist_ok=True)

FACTORS = [
    ("11", "high_speculation", "11high_speculation.py"),
    ("12", "speculation_mad", "12speculation_mad.py"),
    ("21", "speculation_change_rate", "21speculation_change_rate.py"),
    ("22", "speculation_first_difference", "22speculation_first_difference.py"),
    ("23", "speculation_continuous_drop", "23speculation_continuous_drop.py"),
]


for factor_id, factor_name, factor_file in FACTORS:
    print(f"\n=== factor {factor_id}: {factor_name} ===", flush=True)
    subprocess.run(
        [sys.executable, factor_file],
        cwd=FACTOR_DIR,
        env=BASE_ENV,
        check=True,
    )

    env = BASE_ENV.copy()
    env["FACTOR_ID"] = factor_id
    env["FACTOR_NAME"] = factor_name

    print(f"\n=== backtest {factor_id}: {factor_name} ===", flush=True)
    subprocess.run(
        [sys.executable, "backtest_single_factor.py"],
        cwd=BACKTEST_DIR,
        env=env,
        check=True,
    )
