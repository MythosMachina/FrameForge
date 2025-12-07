import argparse
import shutil
from pathlib import Path
from typing import Iterable

# Bundle-local paths (aligned with workflow.py)
BUNDLE_ROOT = Path(__file__).resolve().parent
INPUT_ROOT = BUNDLE_ROOT / "10_input"
FRAMES_ROOT = BUNDLE_ROOT / "20_capped_frames"
WORKSPACE_ROOT = BUNDLE_ROOT / "30_work"
READY_AUTOTAG = BUNDLE_ROOT / "50_ready_autotag"
FINAL_OUTPUT = BUNDLE_ROOT / "60_final_output"
ARCHIVE_MP4 = BUNDLE_ROOT / "70_archive_mp4"
TRAIN_JOBS = BUNDLE_ROOT / "trainer" / "jobs"
TRAIN_OUTPUT = BUNDLE_ROOT / "trainer" / "output"
TRAIN_MODELS = BUNDLE_ROOT / "trainer" / "models"
TRAIN_LOGS = BUNDLE_ROOT / "trainer" / "logs"
FINAL_LORA = BUNDLE_ROOT / "90_final_lora"
TRAIN_FLAG = BUNDLE_ROOT / "TRAINING_RUNNING"
TAG_FLAG = BUNDLE_ROOT / "TAGGING_RUNNING"


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
    group.add_argument("--import", dest="clean_import", action="store_true", help="Clean 10_input")
    group.add_argument("--work", dest="clean_work", action="store_true", help="Clean 30_work + 20_capped_frames")
    group.add_argument("--output", dest="clean_output", action="store_true", help="Clean 60_final_output + 50_ready_autotag + 70_archive_mp4")
    group.add_argument("--train", dest="clean_train", action="store_true", help="Clean trainer jobs/output/logs (keeps sample job)")
    group.add_argument("--all", dest="clean_all", action="store_true", help="Clean import, work, output, and training folders")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.clean_all:
        # Clean 10_input but keep preset subfolders
        log("Cleaning 10_input (preserving preset folders)")
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
        clean_group([FRAMES_ROOT, WORKSPACE_ROOT], "workspace")
        clean_group([FINAL_OUTPUT, READY_AUTOTAG, ARCHIVE_MP4], "output")
        clean_group([TRAIN_OUTPUT, TRAIN_LOGS, FINAL_LORA], "trainer outputs")
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
        clean_group([INPUT_ROOT], "10_input")
    elif args.clean_work:
        clean_group([FRAMES_ROOT, WORKSPACE_ROOT], "work")
    elif args.clean_output:
        clean_group([FINAL_OUTPUT, READY_AUTOTAG, ARCHIVE_MP4], "output")
    elif args.clean_train:
        clean_group([TRAIN_OUTPUT, TRAIN_LOGS, FINAL_LORA], "trainer outputs")
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
