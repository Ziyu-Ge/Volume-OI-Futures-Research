import subprocess
import sys
from pathlib import Path


CODE_DIR = Path(__file__).resolve().parent
SCRIPTS = [
    "run_11_price_up_volume_oi_surge_all.py",
    "run_12_price_up_speculation_up_all.py",
    "run_13_price_up_oi_down_all.py",
    "run_14_uptrend_crowded_chase_all.py",
    "plot_combined_signals.py",
]


for script in SCRIPTS:
    subprocess.run([sys.executable, str(CODE_DIR / script)], check=True)
