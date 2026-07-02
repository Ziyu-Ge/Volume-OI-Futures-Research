import subprocess
import sys
from pathlib import Path


RUN_DIR = Path(__file__).resolve().parent
CODE_DIR = RUN_DIR.parent
PROJECT_ROOT = CODE_DIR.parents[1]
SCRIPTS = [
    CODE_DIR / "00_prepare_data.py",
    RUN_DIR / "run_11_price_up_volume_oi_surge_all.py",
    RUN_DIR / "run_12_price_up_speculation_up_all.py",
    RUN_DIR / "run_13_price_up_oi_down_all.py",
    RUN_DIR / "run_14_uptrend_crowded_chase_all.py",
    CODE_DIR / "plot" / "plot_combined_signals.py",
]


def main():
    for script in SCRIPTS:
        subprocess.run(
            [sys.executable, str(script)],
            cwd=PROJECT_ROOT,
            check=True,
        )


if __name__ == "__main__":
    main()
