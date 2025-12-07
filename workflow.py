import argparse
import shutil
import subprocess
import sys
import threading
import time
import os
import sqlite3
from pathlib import Path
from typing import Dict, Iterable, List, Optional

from watchdog.events import FileSystemEvent, FileSystemEventHandler
from watchdog.observers import Observer

import autotag
import capping
import select_caps
from QuickRename import rename_files_in_hierarchy


# Bundle-local paths (portable, descriptive)
BUNDLE_ROOT = Path(__file__).resolve().parent
INPUT_ROOT = BUNDLE_ROOT / "10_input"                # place source videos/folders here
IMPORT_ROOT = BUNDLE_ROOT / "00_import_queue"        # queue for auto-import
FRAMES_ROOT = BUNDLE_ROOT / "20_capped_frames"       # ffmpeg output
WORKSPACE_ROOT = BUNDLE_ROOT / "30_work"             # selection + crops live here
WORKSPACE_RAW = WORKSPACE_ROOT / "raw"               # capped frames before selection
ARCHIVE_MP4 = BUNDLE_ROOT / "70_archive_mp4"         # MP4 archive (flat)
FINAL_OUTPUT = BUNDLE_ROOT / "60_final_output"       # optional final move target
READY_AUTOTAG = BUNDLE_ROOT / "50_ready_autotag"     # staging area for autotag before final_ready
TRAIN_OUTPUT = BUNDLE_ROOT / "trainer" / "output"    # staging for training artifacts
FINAL_LORA = BUNDLE_ROOT / "90_final_lora"           # destination for finished LoRAs

WORKING_SELECTED = WORKSPACE_ROOT  # selected images live directly in WORKSPACE_ROOT
PRESET_DIRS = {"furry", "human", "dragon", "daemon"}  # preset category folders

# Workflow switches
ENABLE_FINAL_MOVE = True  # move selected+cropped output to FINAL_OUTPUT at the end
QUIET_SECONDS = 120  # inactivity window before moving tagged folders
TRAIN_FLAG = BUNDLE_ROOT / "TRAINING_RUNNING"
TAG_FLAG = BUNDLE_ROOT / "TAGGING_RUNNING"

# Scripts
CROP_SCRIPT = Path(__file__).resolve().parent / "crop_and_flip.Bulk.py"

# Optional integration with webapp for step status
RUN_ID = os.environ.get("RUN_ID")
RUN_DB = os.environ.get("RUN_DB")


def update_run_step(step: str) -> None:
    """
    Write current step into webapp SQLite if RUN_ID and RUN_DB are provided.
    Silently ignore errors to avoid breaking pipeline.
    """
    if not RUN_ID or not RUN_DB:
        return
    try:
        conn = sqlite3.connect(RUN_DB)
        cur = conn.cursor()
        cur.execute("UPDATE Run SET lastStep=?, status='running' WHERE runId=?", (step, RUN_ID))
        conn.commit()
        conn.close()
    except Exception:
        pass


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Dataset workflow bundle")
    parser.add_argument("--autotag", action="store_true", help="Run autotagging after processing (CPU default)")
    parser.add_argument("--autotag-config", type=Path, help="Path to autotag.config.json (optional override)")
    parser.add_argument("--autotag-threshold", type=float, help="Override general tag threshold")
    parser.add_argument("--autotag-character-threshold", type=float, help="Override character tag threshold")
    parser.add_argument("--autotag-max-tags", type=int, help="Limit number of tags per image (including trigger)")
    parser.add_argument("--autotag-model", help="Override the tagger model id")
    parser.add_argument("--autochar", action="store_true", help="Filter hair/eye tags after autotag (regex-based)")
    parser.add_argument("--autochar-preset", type=str, help="Comma-separated autochar presets to apply from DB/legacy files")
    parser.add_argument("--gpu", action="store_true", help="Use GPU for autotagging (training always uses GPU)")
    parser.add_argument("--train", action="store_true", help="Run auto-training on final_ready datasets (GPU only)")
    parser.add_argument("--tagverify", action="store_true", help="Verify color tags via HSV mask to drop false positives")
    parser.add_argument("--facecap", action="store_true", help="Create face crops during capping for face quota")
    return parser.parse_args()


def log(msg: str) -> None:
    print(f"[info] {msg}")


def ensure_dir_empty(path: Path) -> None:
    if path.exists():
        shutil.rmtree(path)
    path.mkdir(parents=True, exist_ok=True)


def move_videos_flat(paths: Iterable[Path], src_root: Path, dst_root: Path) -> List[Path]:
    """
    Move only files (e.g., MP4) into dst_root without recreating source folders.
    Names are prefixed with the relative path to avoid collisions.
    """
    moved: List[Path] = []
    dst_root.mkdir(parents=True, exist_ok=True)
    for src in paths:
        if not src.is_file():
            continue
        rel = src.relative_to(src_root)
        name_parts = list(rel.parts)
        base_name = "_".join(name_parts)
        dst = dst_root / base_name
        # ensure unique
        if dst.exists():
            stem, suffix = dst.stem, dst.suffix
            counter = 1
            while dst.exists():
                dst = dst_root / f"{stem}_{counter}{suffix}"
                counter += 1
        log(f"Move {src} -> {dst}")
        shutil.move(str(src), dst)
        moved.append(dst)
    return moved


def move_trained_folder(src: Path, dst_root: Path) -> None:
    """
    Move a trained dataset folder out of final_ready into an archive target.
    """
    if not src.exists() or not src.is_dir():
        return
    dst_root.mkdir(parents=True, exist_ok=True)
    dst = dst_root / src.name
    if dst.exists():
        counter = 1
        while dst.exists():
            dst = dst_root / f"{src.name}_{counter}"
            counter += 1
    log(f"Archive trained folder {src} -> {dst}")
    shutil.move(str(src), dst)


def move_capping_to_raw() -> List[Path]:
    moved: List[Path] = []
    if not FRAMES_ROOT.exists():
        return moved
    for entry in sorted(FRAMES_ROOT.iterdir()):
        if not entry.is_dir():
            continue
        rel = entry.name
        dst = WORKSPACE_RAW / rel
        if dst.exists():
            shutil.rmtree(dst)
        log(f"Move capped frames {entry} -> {dst}")
        shutil.move(str(entry), dst)
        moved.append(dst)
    # Clean up empty folders left behind
    for entry in sorted(FRAMES_ROOT.glob("**/*"), reverse=True):
        if entry.is_dir():
            try:
                entry.rmdir()
            except OSError:
                pass
    return moved


def move_source_dirs_to_working() -> List[Path]:
    """
    Move remaining folders from INPUT_ROOT into WORKSPACE_ROOT (after MP4 removal).
    If the target exists, merge contents and ensure unique filenames.
    """
    moved: List[Path] = []
    if not INPUT_ROOT.exists():
        return moved

    def move_item(child: Path, dest_dir: Path) -> None:
        dest_dir.mkdir(parents=True, exist_ok=True)
        target = dest_dir / child.name
        if target.exists():
            stem, suffix = target.stem, target.suffix
            counter = 1
            while target.exists():
                target = dest_dir / f"{stem}_{counter}{suffix}"
                counter += 1
        log(f"Move {child} -> {target}")
        shutil.move(str(child), target)

    for folder in sorted(INPUT_ROOT.iterdir()):
        if not folder.is_dir():
            continue
        dest = WORKSPACE_ROOT / folder.name
        if dest.exists():
            for child in sorted(folder.iterdir()):
                move_item(child, dest)
            try:
                folder.rmdir()
            except OSError:
                pass
        else:
            log(f"Move {folder} -> {dest}")
            shutil.move(str(folder), dest)
        moved.append(dest)
    return moved


def get_top_level_folder(path: Path, root: Path) -> Optional[Path]:
    try:
        rel = path.relative_to(root)
    except ValueError:
        return None
    if not rel.parts:
        return None
    return root / rel.parts[0]


class FolderActivityHandler(FileSystemEventHandler):
    def __init__(self, root: Path, touches: Dict[Path, float]) -> None:
        super().__init__()
        self.root = root
        self.touches = touches

    def on_any_event(self, event: FileSystemEvent) -> None:
        path = Path(event.src_path)
        folder = get_top_level_folder(path, self.root)
        if folder:
            self.touches[folder] = time.time()


def wait_for_training_and_promote(
    output_root: Path,
    final_root: Path,
    stop_event: threading.Event,
    expected_checkpoints: Dict[str, int],
    done_marker: str = "TRAIN_DONE",
) -> None:
    """
    Watch training output folders; once a folder has expected_checkpoints safetensors
    (or a done_marker file) and has been idle for QUIET_SECONDS, move to final_root.
    """
    final_root.mkdir(parents=True, exist_ok=True)
    pending = {p for p in output_root.iterdir() if p.is_dir() and p.name != "raw"}
    touches: Dict[Path, float] = {p: time.time() for p in pending}

    observer = Observer()
    handler = FolderActivityHandler(output_root, touches)
    observer.schedule(handler, str(output_root), recursive=True)
    observer.start()
    try:
        while not stop_event.is_set() or pending:
            for p in output_root.iterdir():
                if p.is_dir() and p.name != "raw" and p not in pending:
                    pending.add(p)
                    touches[p] = time.time()
            now = time.time()
            finished: List[Path] = []
            for folder in list(pending):
                if not folder.exists():
                    pending.discard(folder)
                    continue
                checkpoint_count = len(list(folder.glob("*.safetensors")))
                marker = folder / done_marker
                target_epochs = expected_checkpoints.get(folder.name, 10)
                if checkpoint_count < target_epochs and not marker.exists():
                    continue
                last_touch = touches.get(folder, now)
                if now - last_touch < QUIET_SECONDS:
                    continue
                dst = final_root / folder.name
                if dst.exists():
                    raise RuntimeError(f"Destination already exists, refusing to overwrite: {dst}")
                log(f"Training complete, move {folder} -> {dst}")
                shutil.move(str(folder), dst)
                finished.append(folder)
            for f in finished:
                pending.discard(f)
            if pending:
                time.sleep(2)
    finally:
        observer.stop()
        observer.join()


def wait_for_autotag_and_promote(tag_root: Path, final_root: Path, stop_event: threading.Event) -> None:
    """
    Watch tag_root for activity; once a folder has TXT files and has been idle
    for QUIET_SECONDS, move it into final_root.
    """
    final_root.mkdir(parents=True, exist_ok=True)
    pending = {p for p in tag_root.iterdir() if p.is_dir() and p.name != "raw"}
    touches: Dict[Path, float] = {p: time.time() for p in pending}

    observer = Observer()
    handler = FolderActivityHandler(tag_root, touches)
    observer.schedule(handler, str(tag_root), recursive=True)
    observer.start()
    try:
        while not stop_event.is_set() or pending:
            # pick up new folders created during tagging
            for p in tag_root.iterdir():
                if p.is_dir() and p.name != "raw" and p not in pending:
                    pending.add(p)
                    touches[p] = time.time()
            now = time.time()
            finished: List[Path] = []
            for folder in list(pending):
                if not folder.exists():
                    pending.discard(folder)
                    continue
                has_txt = any(file.suffix.lower() == ".txt" for file in folder.rglob("*.txt"))
                if not has_txt:
                    continue
                last_touch = touches.get(folder, now)
                if now - last_touch < QUIET_SECONDS:
                    continue
                dst = final_root / folder.name
                if dst.exists():
                    raise RuntimeError(f"Destination already exists, refusing to overwrite: {dst}")
                log(f"Autotag complete, move {folder} -> {dst}")
                shutil.move(str(folder), dst)
                finished.append(folder)
            for f in finished:
                pending.discard(f)
            if pending:
                time.sleep(2)
    finally:
        observer.stop()
        observer.join()


def promote_tagged_now(tag_root: Path, final_root: Path) -> None:
    """
    Fallback promotion: move any tagged folders (with TXT files) immediately.
    """
    final_root.mkdir(parents=True, exist_ok=True)
    for folder in tag_root.iterdir():
        if not folder.is_dir() or folder.name == "raw":
            continue
        has_txt = any(file.suffix.lower() == ".txt" for file in folder.rglob("*.txt"))
        if not has_txt:
            continue
        dst = final_root / folder.name
        if dst.exists():
            # avoid overwrite; skip if already moved
            continue
        log(f"[promote] move tagged {folder} -> {dst}")
        shutil.move(str(folder), dst)


def run_crop_and_flip(target_dirs: Iterable[Path]) -> None:
    for folder in target_dirs:
        if not folder.is_dir():
            continue
        log(f"Crop+Flip in {folder}")
        subprocess.run(
            [sys.executable, str(CROP_SCRIPT), str(folder)],
            check=True,
        )


def main(args: argparse.Namespace) -> None:
    update_run_step("init")
    if not INPUT_ROOT.exists():
        raise FileNotFoundError(f"Source root not found: {INPUT_ROOT}")
    FRAMES_ROOT.mkdir(parents=True, exist_ok=True)
    ARCHIVE_MP4.mkdir(parents=True, exist_ok=True)
    WORKSPACE_ROOT.mkdir(parents=True, exist_ok=True)
    READY_AUTOTAG.mkdir(parents=True, exist_ok=True)
    IMPORT_ROOT.mkdir(parents=True, exist_ok=True)
    if args.autotag:
        ensure_dir_empty(READY_AUTOTAG)
    FINAL_LORA.mkdir(parents=True, exist_ok=True)
    TRAIN_OUTPUT.mkdir(parents=True, exist_ok=True)

    log("Step 1: QuickRename in 10_input")
    update_run_step("rename")
    rename_files_in_hierarchy(INPUT_ROOT)

    log("Prepare raw folder in Working")
    ensure_dir_empty(WORKSPACE_RAW)
    old_selected = WORKSPACE_ROOT / "selected"
    if old_selected.exists():
        shutil.rmtree(old_selected)

    log("Step 2: Capping videos (12 fps)")
    update_run_step("cap")
    produced = capping.cap_all(INPUT_ROOT, FRAMES_ROOT, facecap=args.facecap)
    log(f"Capping complete: {len(produced)} video folders")

    log("Step 2.5: Archive MP4s to archive_mp4 (files only, flat)")
    update_run_step("archive")
    videos = list(capping.iter_videos(INPUT_ROOT))
    move_videos_flat(videos, INPUT_ROOT, ARCHIVE_MP4)

    log("Step 2.5: Move capped frames into workspace/raw")
    update_run_step("move_capped")
    move_capping_to_raw()

    log("Step 2.5: Move remaining source folders into workspace (merge)")
    update_run_step("merge_inputs")
    move_source_dirs_to_working()

    log("Step 3: Select caps (40 per character, 10 face quota)")
    update_run_step("select")
    select_caps.run_selection(
        src_root=WORKSPACE_RAW,
        dest_root=WORKING_SELECTED,
    )

    log("Step 4: Crop and Flip selected images")
    update_run_step("cropflip")
    target_dirs = [
        p for p in WORKING_SELECTED.iterdir()
        if p.is_dir()
        and not p.name.startswith("_")
        and p.name not in {"selected", "raw"}
        and p.name not in PRESET_DIRS
    ]
    run_crop_and_flip(target_dirs)

    if ENABLE_FINAL_MOVE:
        dest_root = READY_AUTOTAG if args.autotag else FINAL_OUTPUT
        log(f"Step 5: Move finished set to {dest_root.name}")
        update_run_step("move_final")
        dest_root.mkdir(parents=True, exist_ok=True)
        for folder in target_dirs:
            if folder.name in PRESET_DIRS:
                continue
            preset_marker = folder / ".autochar_preset"
            if preset_marker.exists() and not args.autotag:
                try:
                    preset_marker.unlink()
                except OSError:
                    pass
            dst = dest_root / folder.name
            if dst.exists():
                raise RuntimeError(f"Destination already exists, refusing to overwrite: {dst}")
            log(f"Move {folder} -> {dst}")
            shutil.move(str(folder), dst)

    if args.autotag:
        TAG_FLAG.write_text("running", encoding="utf-8")
        tag_root = READY_AUTOTAG if ENABLE_FINAL_MOVE else WORKING_SELECTED
        device = "cuda" if args.gpu else "cpu"
        log(f"Step 6: Autotag images in {tag_root} (device={device})")
        update_run_step("autotag")
        promoter_stop = threading.Event()
        promoter_thread: Optional[threading.Thread] = None
        if ENABLE_FINAL_MOVE:
            log(f"Step 6.5: Start watcher (quiet {QUIET_SECONDS}s) to move finished folders to {FINAL_OUTPUT}")
            update_run_step("autotag_watch")
            promoter_thread = threading.Thread(
                target=wait_for_autotag_and_promote,
                args=(tag_root, FINAL_OUTPUT, promoter_stop),
                daemon=True,
            )
            promoter_thread.start()
        autotag.tag_folder(
            root=tag_root,
            model_id=args.autotag_model,
            general_threshold=args.autotag_threshold,
            character_threshold=args.autotag_character_threshold,
            max_tags=args.autotag_max_tags,
            config_path=args.autotag_config,
            device=device,
            autochar_enabled=args.autochar,
            verify_colors=args.tagverify,
        )
        if ENABLE_FINAL_MOVE and promoter_thread:
            promoter_stop.set()
            promoter_thread.join()
            # ensure any remaining tagged folders are promoted immediately
            promote_tagged_now(tag_root, FINAL_OUTPUT)
        if TAG_FLAG.exists():
            TAG_FLAG.unlink()
    if args.train:
        import torch
        from trainer import auto_trainer
        from trainer import train_runner

        if not torch.cuda.is_available():
            raise RuntimeError("Training requires CUDA GPU; none detected.")

        TRAIN_FLAG.write_text("running", encoding="utf-8")
        log("Step 7a: Analyze datasets and build training plans (Pony SDXL LoRA, <=12GB VRAM)")
        update_run_step("train_plan")
        plans = auto_trainer.run_planning(FINAL_OUTPUT)
        if plans:
            for p in plans:
                log(f"Training plan written: {p}")
        else:
            log("No datasets found in final_ready; skipping training plan.")

        log("Step 7b: Prepare training outputs (staging in trainer/output)")
        TRAIN_OUTPUT.mkdir(parents=True, exist_ok=True)
        ensure_dir_empty(TRAIN_OUTPUT)

        log("Step 7c: Start watcher to promote finished LoRAs to final_lora")
        update_run_step("train_watch")
        trainer_stop = threading.Event()
        expected_counts = {Path(p).stem: train_runner.load_plan(p).epochs for p in plans}
        trainer_thread = threading.Thread(
            target=wait_for_training_and_promote,
            args=(TRAIN_OUTPUT, FINAL_LORA, trainer_stop, expected_counts),
            daemon=True,
        )
        trainer_thread.start()

        log("Step 7d: Training runs (one per dataset)")
        update_run_step("train_run")
        for plan_path in plans:
            plan = train_runner.load_plan(plan_path)
            log(f"Training {plan.name}: {plan.dataset_path}")
            train_runner.train_one(plan, device="cuda")

        trainer_stop.set()
        trainer_thread.join()
        # Archive trained dataset folders to avoid reprocessing
        for plan_path in plans:
            name = Path(plan_path).stem
            trained_folder = FINAL_OUTPUT / name
            move_trained_folder(trained_folder, ARCHIVE_MP4)
        log("Training done.")
        if TRAIN_FLAG.exists():
            TRAIN_FLAG.unlink()
    update_run_step("done")
    log("Workflow done.")


if __name__ == "__main__":
    main(parse_args())
