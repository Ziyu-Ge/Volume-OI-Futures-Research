import subprocess
import sys
from pathlib import Path


RUN_DIR = Path(__file__).resolve().parent
CHAPTER_DIR = RUN_DIR.parent
PROJECT_ROOT = CHAPTER_DIR.parents[1]
FACTOR_SCRIPT = (
    CHAPTER_DIR / "factors" / "23_high_bias_oi_speculation_drop.py"
)


def main():
    subprocess.run(
        [sys.executable, str(FACTOR_SCRIPT), "--keep-going"],
        cwd=PROJECT_ROOT,
        check=True,
    )


if __name__ == "__main__":
    main()
