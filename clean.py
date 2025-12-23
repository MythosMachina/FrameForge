import argparse
import shutil
from pathlib import Path
from typing import Iterable

# Bundle-local paths (aligned with workflow.py)
BUNDLE_ROOT = Path(__file__).resolve().parent
SYSTEM_ROOT = BUNDLE_ROOT / "_system"
INPUT_ROOT = BUNDLE_ROOT / "INBOX"
FRAMES_ROOT = SYSTEM_ROOT / "workflow" / "capped"
WORKSPACE_ROOT = SYSTEM_ROOT / "workflow" / "work"
WORKSPACE_RAW = SYSTEM_ROOT / "workflow" / "raw"
READY_AUTOTAG = SYSTEM_ROOT / "workflow" / "ready"
FINAL_OUTPUT = BUNDLE_ROOT / "OUTPUTS" / "datasets"
ARCHIVE_MP4 = BUNDLE_ROOT / "ARCHIVE" / "mp4"
ARCHIVE_ZIPS = BUNDLE_ROOT / "ARCHIVE" / "zips"
TRAIN_JOBS = BUNDLE_ROOT / "trainer" / "jobs"
TRAIN_OUTPUT = SYSTEM_ROOT / "trainer" / "output"
TRAIN_DATASET = SYSTEM_ROOT / "trainer" / "dataset"
TRAIN_LOGS = SYSTEM_ROOT / "trainer" / "logs"
FINAL_LORA = BUNDLE_ROOT / "OUTPUTS" / "loras"
TRAIN_FLAG = SYSTEM_ROOT / "flags" / "TRAINING_RUNNING"
TAG_FLAG = SYSTEM_ROOT / "flags" / "TAGGING_RUNNING"


def log(msg: str) -> None:
    print(f"[info] {msg}")


def ensure_dir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


def empty_directory(path: Path) -> None:
    if not path.exists():
        return
    for child in path.iterdir():
        if child.is_dir():
            shutil.rmtree(child)
        else:
            child.unlink()


def clean_group(paths: Iterable[Path], label: str) -> None:
    log(f"Cleaning {label}")
    for p in paths:
        empty_directory(p)
        ensure_dir(p)


def remove_flags() -> None:
    for flag in [TRAIN_FLAG, TAG_FLAG]:
        if flag.exists():
            flag.unlink()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Prune FrameForge folders")
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--import", dest="clean_import", action="store_true", help="Clean INBOX")
    group.add_argument("--work", dest="clean_work", action="store_true", help="Clean workflow work/capped/ready")
    group.add_argument("--output", dest="clean_output", action="store_true", help="Clean OUTPUTS + ARCHIVE/mp4 + workflow ready")
    group.add_argument("--train", dest="clean_train", action="store_true", help="Clean trainer jobs/output/logs (keeps sample job)")
    group.add_argument("--all", dest="clean_all", action="store_true", help="Clean import, work, output, and training folders")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.clean_all:
        # Clean 10_input but keep preset subfolders
        log("Cleaning INBOX (preserving preset folders)")
        if INPUT_ROOT.exists():
            for child in INPUT_ROOT.iterdir():
                if child.is_dir() and child.name in {"furry", "human", "dragon", "daemon"}:
                    empty_directory(child)
                else:
                    if child.is_dir():
                        shutil.rmtree(child)
                    else:
                        child.unlink()
        ensure_dir(INPUT_ROOT)
        for name in ["furry", "human", "dragon", "daemon"]:
            ensure_dir(INPUT_ROOT / name)
        clean_group([FRAMES_ROOT, WORKSPACE_ROOT, WORKSPACE_RAW], "workspace")
        clean_group([FINAL_OUTPUT, FINAL_LORA, READY_AUTOTAG, ARCHIVE_MP4, ARCHIVE_ZIPS], "output")
        clean_group([TRAIN_OUTPUT, TRAIN_LOGS, TRAIN_DATASET], "trainer outputs")
        remove_flags()
        # jobs: keep sample
        log("Cleaning trainer jobs (preserving sample_job.json)")
        ensure_dir(TRAIN_JOBS)
        for child in TRAIN_JOBS.iterdir():
            if child.name == "sample_job.json":
                continue
            if child.is_dir():
                shutil.rmtree(child)
            else:
                child.unlink()
        return

    if args.clean_import:
        clean_group([INPUT_ROOT], "INBOX")
    elif args.clean_work:
        clean_group([FRAMES_ROOT, WORKSPACE_ROOT, WORKSPACE_RAW], "work")
    elif args.clean_output:
        clean_group([FINAL_OUTPUT, FINAL_LORA, READY_AUTOTAG, ARCHIVE_MP4, ARCHIVE_ZIPS], "output")
    elif args.clean_train:
        clean_group([TRAIN_OUTPUT, TRAIN_LOGS, TRAIN_DATASET], "trainer outputs")
        log("Cleaning trainer jobs (preserving sample_job.json)")
        ensure_dir(TRAIN_JOBS)
        for child in TRAIN_JOBS.iterdir():
            if child.name == "sample_job.json":
                continue
            if child.is_dir():
                shutil.rmtree(child)
            else:
                child.unlink()
    remove_flags()


if __name__ == "__main__":
    main()
