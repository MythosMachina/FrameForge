import subprocess
import sys
import time
from pathlib import Path

BUNDLE_ROOT = Path(__file__).resolve().parent
IMPORT_ROOT = BUNDLE_ROOT / "00_import_queue"
INPUT_ROOT = BUNDLE_ROOT / "10_input"
TRAIN_FLAG = BUNDLE_ROOT / "TRAINING_RUNNING"
TAG_FLAG = BUNDLE_ROOT / "TAGGING_RUNNING"
SLEEP_SECONDS = 300  # 5 minutes
# Command to run the full pipeline when a folder is pulled in
PIPELINE_CMD = [sys.executable, str(BUNDLE_ROOT / "workflow.py"), "--autotag", "--gpu", "--autochar", "--train"]


def log(msg: str) -> None:
    print(f"[queue] {msg}")


def is_training() -> bool:
    return TRAIN_FLAG.exists()


def is_tagging() -> bool:
    return TAG_FLAG.exists()


def pop_one_from_import() -> None:
    candidates = [p for p in sorted(IMPORT_ROOT.iterdir()) if p.is_dir()]
    if not candidates:
        return
    src = candidates[0]
    dst = INPUT_ROOT / src.name
    if dst.exists():
        log(f"Destination exists, skipping: {dst}")
        return
    log(f"Move {src} -> {dst}")
    dst.parent.mkdir(parents=True, exist_ok=True)
    src.rename(dst)
    # Trigger pipeline
    log("Start pipeline run")
    try:
        subprocess.run(PIPELINE_CMD, cwd=BUNDLE_ROOT, check=True)
    except subprocess.CalledProcessError as e:
        log(f"Pipeline run failed: {e}")


def main() -> None:
    IMPORT_ROOT.mkdir(parents=True, exist_ok=True)
    INPUT_ROOT.mkdir(parents=True, exist_ok=True)
    while True:
        if not is_training() and not is_tagging():
            pop_one_from_import()
        time.sleep(SLEEP_SECONDS)


if __name__ == "__main__":
    main()
