import argparse
import json
import os
import re
import shutil
import sqlite3
import subprocess
import sys
import time
from pathlib import Path
from typing import Dict, Iterable, List, Optional

import autotag
import capping
import select_caps
from QuickRename import rename_files_in_hierarchy
from kohya_wrapper import KohyaPlan, build_kohya_plans, run_kohya_training
from db_broker_client import broker_enabled, broker_exec, broker_query


# Bundle-local paths (portable, descriptive)
BUNDLE_ROOT = Path(__file__).resolve().parent
SYSTEM_ROOT = BUNDLE_ROOT / "_system"
INPUT_ROOT = Path(os.environ.get("FRAMEFORGE_INPUT_ROOT", str(BUNDLE_ROOT / "INBOX")))  # place source videos/folders here
FRAMES_ROOT = SYSTEM_ROOT / "workflow" / "capped"    # ffmpeg output
WORKSPACE_ROOT = SYSTEM_ROOT / "workflow" / "work"   # selection + crops live here
WORKSPACE_RAW = SYSTEM_ROOT / "workflow" / "raw"     # capped frames before selection
ARCHIVE_MP4 = BUNDLE_ROOT / "ARCHIVE" / "mp4"        # MP4 archive (flat)
FINAL_OUTPUT = BUNDLE_ROOT / "OUTPUTS" / "datasets"  # optional final move target
READY_AUTOTAG = SYSTEM_ROOT / "workflow" / "ready"   # staging area for autotag before final_ready
TRAIN_OUTPUT = SYSTEM_ROOT / "trainer" / "output"    # staging for training artifacts
TRAIN_STAGING = SYSTEM_ROOT / "trainer" / "dataset" / "images"  # kohya staging area
FINAL_LORA = BUNDLE_ROOT / "OUTPUTS" / "loras"        # destination for finished LoRAs

WORKING_SELECTED = WORKSPACE_ROOT  # selected images live directly in WORKSPACE_ROOT

# Workflow switches
ENABLE_FINAL_MOVE = True  # move selected+cropped output to FINAL_OUTPUT at the end
FLAGS_ROOT = SYSTEM_ROOT / "flags"
TRAIN_FLAG = FLAGS_ROOT / "TRAINING_RUNNING"
TAG_FLAG = FLAGS_ROOT / "TAGGING_RUNNING"

# Scripts
CROP_SCRIPT = Path(__file__).resolve().parent / "crop_and_flip.Bulk.py"

# Optional integration with webapp for step status
RUN_ID = os.environ.get("RUN_ID")
RUN_DB = os.environ.get("RUN_DB")

IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".webp", ".bmp"}
DEFAULT_SETTINGS = {
    "capping_fps": 8,
    "capping_jpeg_quality": 2,
    "selection_target_per_character": 40,
    "selection_face_quota": 10,
    "selection_hamming_threshold": 6,
    "selection_hamming_relaxed": 4,
    "output_max_images": 600,
    "autotag_general_threshold": 0.55,
    "autotag_character_threshold": 0.4,
    "autotag_max_tags": 30,
    "autotag_model_id": "SmilingWolf/wd-eva02-large-tagger-v3",
    # Trainer (kohya) defaults, aligned with webapp settings
    "trainer_base_model": "darkstorm2150/pony-diffusion-xl-base-1.0",
    "trainer_vae": "madebyollin/sdxl-vae-fp16-fix",
    "trainer_resolution": 1024,
    "trainer_batch_size": 1,
    "trainer_grad_accum": 4,
    "trainer_epochs": 10,
    "trainer_max_train_steps": 6000,
    "trainer_learning_rate": 0.0001,
    "trainer_te_learning_rate": 0.00005,
    "trainer_lr_scheduler": "cosine",
    "trainer_lr_warmup_steps": 180,
    "trainer_lora_rank": 32,
    "trainer_lora_alpha": 32,
    "trainer_te_lora_rank": 16,
    "trainer_te_lora_alpha": 16,
    "trainer_clip_skip": 2,
    "trainer_network_dropout": 0.0,
    "trainer_caption_dropout": 0.0,
    "trainer_shuffle_caption": True,
    "trainer_keep_tokens": 1,
    "trainer_min_snr_gamma": 5.0,
    "trainer_noise_offset": 0.0,
    "trainer_weight_decay": 0.01,
    "trainer_sample_prompt_1": "front view",
    "trainer_sample_prompt_2": "face close up",
    "trainer_sample_prompt_3": "sitting, smile",
    "trainer_bucket_min_reso": 768,
    "trainer_bucket_max_reso": 1024,
    "trainer_bucket_step": 64,
    "trainer_optimizer": "adamw",
    "trainer_use_8bit_adam": True,
    "trainer_gradient_checkpointing": True,
    "trainer_dataloader_workers": 1,
    "trainer_use_prodigy": False,
    "trainer_max_grad_norm": 0,
    "hf_token": "",
}


def update_run_step(step: str) -> None:
    """
    Write current step into webapp SQLite if RUN_ID and RUN_DB are provided.
    Silently ignore errors to avoid breaking pipeline.
    """
    # If orchestrator drives status, skip most internal updates except training progress,
    # so the UI can reflect live train progress even in orchestrator-driven mode.
    if os.environ.get("ORCHESTRATOR_DRIVER") == "1" and not step.startswith("train_progress"):
        return
    if not RUN_ID or not RUN_DB:
        return
    if broker_enabled():
        try:
            broker_exec("update_run_step", {"run_id": RUN_ID, "step": step})
        except Exception:
            pass
        return
    try:
        conn = sqlite3.connect(RUN_DB)
        cur = conn.cursor()
        cur.execute("UPDATE Run SET lastStep=?, status='running' WHERE runId=?", (step, RUN_ID))
        conn.commit()
        conn.close()
    except Exception:
        pass


def mark_run_done_by_name(run_name: str) -> None:
    """
    Mark a run as done in the DB by matching runName or runId prefix.
    """
    db_path = os.environ.get("RUN_DB")
    if not db_path or not Path(db_path).exists():
        return
    run_id_part = None
    if run_name and run_name.split("_", 1)[0].isdigit():
        run_id_part = run_name.split("_", 1)[0]
    if broker_enabled():
        try:
            broker_exec("mark_run_done", {"run_name": run_name})
        except Exception:
            pass
        return
    try:
        conn = sqlite3.connect(db_path)
        cur = conn.cursor()
        cur.execute(
            "UPDATE Run SET status='done', lastStep='done', finishedAt=CURRENT_TIMESTAMP WHERE runName=? OR runId=?",
            (run_name, run_id_part or run_name),
        )
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
    parser.add_argument("--images-only", action="store_true", help="Skip video capping/archive; treat input as images")
    parser.add_argument("--manual-tagging", action="store_true", help="Pause after crop/flip for manual tag editing")
    parser.add_argument("--manual-resume", action="store_true", help="Resume manual tagging flow and skip autotag")
    parser.add_argument("--train-resume", action="store_true", help="Resume training only from finalized dataset")
    parser.add_argument("--dry-run", action="store_true", help="Run a logic dry-run without writing/moving files")
    return parser.parse_args()


def log(msg: str) -> None:
    print(f"[info] {msg}")


def ensure_dir_empty(path: Path, dry_run: bool = False) -> None:
    if dry_run:
        log(f"[dry-run] Would clear directory: {path}")
        return
    try:
        if path.exists():
            shutil.rmtree(path, ignore_errors=True)
    except FileNotFoundError:
        # Another worker may have already removed it.
        return
    path.mkdir(parents=True, exist_ok=True)


def load_settings_from_db() -> Dict[str, float]:
    settings = dict(DEFAULT_SETTINGS)
    db_path = os.environ.get("RUN_DB")
    if not db_path or not Path(db_path).exists():
        return settings
    if broker_enabled():
        try:
            resp = broker_query(
                "sql_query",
                {"sql": "SELECT key, value FROM Setting", "params": []},
            )
            rows = resp.get("data") or []
            for row in rows:
                key = row.get("key")
                val = row.get("value")
                if key not in settings:
                    continue
                try:
                    num = float(val)
                    settings[key] = num
                except Exception:
                    settings[key] = val
            return settings
        except Exception:
            return settings
    try:
        conn = sqlite3.connect(db_path)
        cur = conn.cursor()
        cur.execute("SELECT name FROM sqlite_master WHERE type='table' AND name='Setting';")
        if not cur.fetchone():
            conn.close()
            return settings
        cur.execute("SELECT key, value FROM Setting;")
        rows = cur.fetchall()
        conn.close()
        for key, val in rows:
            if key not in settings:
                continue
            try:
                num = float(val)
                settings[key] = num
            except Exception:
                settings[key] = val
    except Exception:
        pass
    return settings


def _fetch_train_profile(profile_name: Optional[str]) -> Optional[dict]:
    db_path = os.environ.get("RUN_DB")
    if not db_path:
        return None
    if broker_enabled():
        try:
            if profile_name:
                resp = broker_query(
                    "sql_query",
                    {
                        "sql": "SELECT name, settings FROM TrainProfile WHERE name=? LIMIT 1",
                        "params": [profile_name],
                    },
                )
                rows = resp.get("data") or []
                if rows:
                    return rows[0]
            resp = broker_query(
                "sql_query",
                {"sql": "SELECT name, settings FROM TrainProfile WHERE isDefault=1 ORDER BY id ASC LIMIT 1", "params": []},
            )
            rows = resp.get("data") or []
            if rows:
                return rows[0]
            resp = broker_query(
                "sql_query",
                {"sql": "SELECT name, settings FROM TrainProfile ORDER BY id ASC LIMIT 1", "params": []},
            )
            rows = resp.get("data") or []
            return rows[0] if rows else None
        except Exception:
            return None
    try:
        conn = sqlite3.connect(db_path)
        cur = conn.cursor()
        if profile_name:
            cur.execute("SELECT name, settings FROM TrainProfile WHERE name=? LIMIT 1;", (profile_name,))
            row = cur.fetchone()
            if row:
                conn.close()
                return {"name": row[0], "settings": row[1]}
        cur.execute("SELECT name, settings FROM TrainProfile WHERE isDefault=1 ORDER BY id ASC LIMIT 1;")
        row = cur.fetchone()
        if row:
            conn.close()
            return {"name": row[0], "settings": row[1]}
        cur.execute("SELECT name, settings FROM TrainProfile ORDER BY id ASC LIMIT 1;")
        row = cur.fetchone()
        conn.close()
        if row:
            return {"name": row[0], "settings": row[1]}
    except Exception:
        pass
    return None


def apply_train_profile(settings: Dict[str, float]) -> Dict[str, float]:
    """
    Override trainer-relevant settings based on TRAIN_PROFILE env or default DB profile.
    """
    profile = os.environ.get("TRAIN_PROFILE", "").strip().lower() or None
    record = _fetch_train_profile(profile)
    if not record:
        return settings
    try:
        overrides = json.loads(record.get("settings") or "{}")
    except Exception:
        return settings
    if not isinstance(overrides, dict):
        return settings
    merged = dict(settings)
    merged.update(overrides)
    return merged


def move_videos_flat(paths: Iterable[Path], src_root: Path, dst_root: Path, dry_run: bool = False) -> List[Path]:
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
        if not dry_run:
            shutil.move(str(src), dst)
        moved.append(dst)
    return moved


def move_capping_to_raw(dry_run: bool = False) -> List[Path]:
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
        if not dry_run:
            shutil.move(str(entry), dst)
        moved.append(dst)
    # Clean up empty folders left behind
    for entry in sorted(FRAMES_ROOT.glob("**/*"), reverse=True):
        if entry.is_dir() and not dry_run:
            try:
                entry.rmdir()
            except OSError:
                pass
    return moved


def move_images_to_raw(src_root: Path, dst_root: Path, dry_run: bool = False) -> List[Path]:
    """
    Move existing image folders (no capping) directly into workspace/raw.
    """
    moved: List[Path] = []
    if not src_root.exists():
        return moved
    dst_root.mkdir(parents=True, exist_ok=True)
    for entry in sorted(src_root.iterdir()):
        if not entry.is_dir():
            continue
        dst = dst_root / entry.name
        if dst.exists():
            shutil.rmtree(dst)
        log(f"Move images {entry} -> {dst}")
        if not dry_run:
            shutil.move(str(entry), dst)
        moved.append(dst)
    return moved


def move_source_dirs_to_working(dry_run: bool = False) -> List[Path]:
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
        if not dry_run:
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
            if not dry_run:
                shutil.move(str(folder), dest)
        moved.append(dest)
    return moved


def cap_selected_images(folder: Path, max_images: int = 40) -> None:
    """
    Ensure numeric-named selected images in folder do not exceed max_images.
    Extra files are moved into _overflow (ignored by downstream steps).
    """
    numeric = []
    for p in sorted(folder.iterdir()):
        if not p.is_file():
            continue
        if p.suffix.lower() not in IMAGE_EXTS:
            continue
        if not re.fullmatch(r"[1-9][0-9]*", p.stem):
            continue
        numeric.append(p)
    if len(numeric) <= max_images:
        return
    overflow = folder / "_overflow"
    overflow.mkdir(exist_ok=True)
    for p in numeric[max_images:]:
        dst = overflow / p.name
        counter = 1
        while dst.exists():
            dst = overflow / f"{p.stem}_{counter}{p.suffix.lower()}"
            counter += 1
        log(f"[cap] {folder.name}: move extra {p.name} -> {dst}")
        shutil.move(str(p), dst)


def run_crop_and_flip(target_dirs: Iterable[Path], dry_run: bool = False) -> None:
    for folder in target_dirs:
        if not folder.is_dir():
            continue
        if dry_run:
            log(f"[dry-run] Would crop+flip in {folder}")
            continue
        cap_selected_images(folder, max_images=40)  # keep final outputs stable (<=600 after crop+flip)
        log(f"Crop+Flip in {folder}")
        subprocess.run(
            [sys.executable, str(CROP_SCRIPT), str(folder)],
            check=True,
        )


def cap_total_outputs(folder: Path, max_total: int = 600, dry_run: bool = False) -> None:
    """
    Ensure total image count after crop/flip stays under max_total.
    Prefer keeping non-flip variants; move extras to _overflow.
    """
    images = [p for p in sorted(folder.iterdir()) if p.is_file() and p.suffix.lower() in IMAGE_EXTS]
    if len(images) <= max_total:
        return
    overflow = folder / "_overflow"
    if not dry_run:
        overflow.mkdir(exist_ok=True)
    non_flip = [p for p in images if "flip" not in p.stem]
    flip = [p for p in images if "flip" in p.stem]
    keep: List[Path] = []
    for lst in (non_flip, flip):
        for p in lst:
            if len(keep) < max_total:
                keep.append(p)
            else:
                dst = overflow / p.name
                counter = 1
                while dst.exists():
                    dst = overflow / f"{p.stem}_{counter}{p.suffix.lower()}"
                    counter += 1
                if not dry_run:
                    shutil.move(str(p), dst)
    log(f"[cap] {folder.name}: trimmed to {len(keep)} images (moved {len(images) - len(keep)} to _overflow){' [dry-run]' if dry_run else ''}")


def main(args: argparse.Namespace) -> None:
    dry_run = args.dry_run
    manual_pause = args.manual_tagging and not args.manual_resume
    manual_resume = args.manual_resume
    train_resume = args.train_resume
    skip_prep = manual_resume or train_resume
    update_run_step("init")
    settings = apply_train_profile(load_settings_from_db())
    cap_fps = int(settings.get("capping_fps", DEFAULT_SETTINGS["capping_fps"]))
    cap_quality = int(settings.get("capping_jpeg_quality", DEFAULT_SETTINGS["capping_jpeg_quality"]))
    target_per_char = int(settings.get("selection_target_per_character", DEFAULT_SETTINGS["selection_target_per_character"]))
    face_quota = int(settings.get("selection_face_quota", DEFAULT_SETTINGS["selection_face_quota"]))
    hamming_threshold = int(settings.get("selection_hamming_threshold", DEFAULT_SETTINGS["selection_hamming_threshold"]))
    hamming_relaxed = int(settings.get("selection_hamming_relaxed", DEFAULT_SETTINGS["selection_hamming_relaxed"]))
    max_output_images = int(settings.get("output_max_images", DEFAULT_SETTINGS["output_max_images"]))
    autotag_general_default = float(settings.get("autotag_general_threshold", DEFAULT_SETTINGS["autotag_general_threshold"]))
    autotag_character_default = float(settings.get("autotag_character_threshold", DEFAULT_SETTINGS["autotag_character_threshold"]))
    autotag_max_tags_default = int(settings.get("autotag_max_tags", DEFAULT_SETTINGS["autotag_max_tags"]))
    autotag_model_default = settings.get("autotag_model_id", DEFAULT_SETTINGS["autotag_model_id"])
    if not INPUT_ROOT.exists() and not skip_prep:
        if dry_run:
            log(f"[dry-run] Input root missing: {INPUT_ROOT}. Skipping processing.")
            update_run_step("done")
            return
        raise FileNotFoundError(f"Source root not found: {INPUT_ROOT}")
    if not dry_run:
        FRAMES_ROOT.mkdir(parents=True, exist_ok=True)
        ARCHIVE_MP4.mkdir(parents=True, exist_ok=True)
        WORKSPACE_ROOT.mkdir(parents=True, exist_ok=True)
        READY_AUTOTAG.mkdir(parents=True, exist_ok=True)
        FINAL_LORA.mkdir(parents=True, exist_ok=True)
        TRAIN_OUTPUT.mkdir(parents=True, exist_ok=True)
        FLAGS_ROOT.mkdir(parents=True, exist_ok=True)
    if args.autotag and not skip_prep:
        ensure_dir_empty(READY_AUTOTAG, dry_run=dry_run)
    # Ensure training staging is clear before runs
    if not manual_pause:
        ensure_dir_empty(TRAIN_STAGING, dry_run=dry_run)

    if not skip_prep:
        log(f"Step 1: QuickRename in {INPUT_ROOT.name}")
        update_run_step("rename")
        if dry_run:
            log(f"[dry-run] Would normalize filenames under {INPUT_ROOT}")
        else:
            rename_files_in_hierarchy(INPUT_ROOT)

    target_dirs = []
    if not skip_prep:
        log("Prepare raw folder in Working")
        ensure_dir_empty(WORKSPACE_RAW, dry_run=dry_run)
        old_selected = WORKSPACE_ROOT / "selected"
        if old_selected.exists() and not dry_run:
            shutil.rmtree(old_selected)

        videos = list(capping.iter_videos(INPUT_ROOT))
        images_only = args.images_only or not videos

        if images_only:
            reason = "flag" if args.images_only else "no videos found"
            log(f"Step 2: Images-only input ({reason}), skip capping/archive")
            update_run_step("images_only")
            move_images_to_raw(INPUT_ROOT, WORKSPACE_RAW, dry_run=dry_run)
        else:
            log(f"Step 2: Capping videos ({cap_fps} fps)")
            update_run_step("cap")
            if dry_run:
                log(f"[dry-run] Would cap {len(videos)} videos to {FRAMES_ROOT} (fps={cap_fps}, jpeg={cap_quality}, facecap={args.facecap})")
            else:
                produced = capping.cap_all(INPUT_ROOT, FRAMES_ROOT, facecap=args.facecap, fps=cap_fps, jpeg_quality=cap_quality)
                log(f"Capping complete: {len(produced)} video folders")

            log("Step 2.5: Archive MP4s to archive_mp4 (files only, flat)")
            update_run_step("archive")
            move_videos_flat(videos, INPUT_ROOT, ARCHIVE_MP4, dry_run=dry_run)

            log("Step 2.5: Move capped frames into workspace/raw")
            update_run_step("move_capped")
            move_capping_to_raw(dry_run=dry_run)

        log("Step 2.5: Move remaining source folders into workspace (merge)")
        update_run_step("merge_inputs")
        move_source_dirs_to_working(dry_run=dry_run)

        log("Step 3: Select caps (40 per character, 10 face quota)")
        update_run_step("select")
        if dry_run:
            folders = [p for p in WORKSPACE_RAW.iterdir()] if WORKSPACE_RAW.exists() else []
            log(f"[dry-run] Would run selection on {len(folders)} folders from {WORKSPACE_RAW} -> {WORKING_SELECTED}")
        else:
            select_caps.run_selection(
                src_root=WORKSPACE_RAW,
                dest_root=WORKING_SELECTED,
                face_quota=face_quota,
                target_per_char=target_per_char,
                hamming_threshold=hamming_threshold,
                hamming_relaxed=hamming_relaxed,
            )

        log("Step 4: Crop and Flip selected images")
        update_run_step("cropflip")
        target_dirs = (
            [
                p
                for p in WORKING_SELECTED.iterdir()
                if p.is_dir()
                and not p.name.startswith("_")
                and p.name not in {"selected", "raw"}
            ]
            if WORKING_SELECTED.exists()
            else []
        )
        run_crop_and_flip(target_dirs, dry_run=dry_run)
        for folder in target_dirs:
            cap_total_outputs(folder, max_total=max_output_images, dry_run=dry_run)
    else:
        target_dirs = (
            [
                p
                for p in WORKING_SELECTED.iterdir()
                if p.is_dir()
                and not p.name.startswith("_")
                and p.name not in {"selected", "raw"}
            ]
            if WORKING_SELECTED.exists()
            else []
        )

    if ENABLE_FINAL_MOVE and not train_resume:
        dest_root = READY_AUTOTAG if (args.autotag and not skip_prep) else FINAL_OUTPUT
        log(f"Step 5: Move finished set to {dest_root.name}")
        update_run_step("move_final")
        if not dry_run:
            dest_root.mkdir(parents=True, exist_ok=True)
        for folder in target_dirs:
            preset_marker = folder / ".autochar_preset"
            if preset_marker.exists() and not args.autotag and not dry_run:
                try:
                    preset_marker.unlink()
                except OSError:
                    pass
            dst = dest_root / folder.name
            if dst.exists():
                if dry_run:
                    log(f"[dry-run] Would replace existing destination: {dst}")
                else:
                    log(f"Destination exists, replacing: {dst}")
                    shutil.rmtree(dst, ignore_errors=True)
            log(f"Move {folder} -> {dst}{' [dry-run]' if dry_run else ''}")
            if not dry_run:
                shutil.move(str(folder), dst)

    if args.autotag and not skip_prep:
        if not dry_run:
            TAG_FLAG.write_text("running", encoding="utf-8")
        tag_root = READY_AUTOTAG if ENABLE_FINAL_MOVE else WORKING_SELECTED
        device = "cuda" if args.gpu else "cpu"
        if args.gpu and not dry_run:
            try:
                import torch

                if not torch.cuda.is_available():
                    log("CUDA requested for autotag but unavailable; falling back to CPU.")
                    device = "cpu"
            except Exception:
                log("CUDA check failed for autotag; falling back to CPU.")
                device = "cpu"
        log(f"Step 6: Autotag images in {tag_root} (device={device})")
        update_run_step("autotag")
        if dry_run:
            log("[dry-run] Would autotag folders and promote to final_ready")
        else:
            if args.autochar_preset:
                # Keep DB autochar presets authoritative by pushing the requested preset via env
                os.environ.setdefault("AUTOCHAR_PRESET", args.autochar_preset)
            autotag.tag_folder(
                root=tag_root,
                model_id=args.autotag_model if args.autotag_model is not None else autotag_model_default,
                general_threshold=args.autotag_threshold if args.autotag_threshold is not None else autotag_general_default,
                character_threshold=args.autotag_character_threshold if args.autotag_character_threshold is not None else autotag_character_default,
                max_tags=args.autotag_max_tags if args.autotag_max_tags is not None else autotag_max_tags_default,
                config_path=args.autotag_config,
                device=device,
                autochar_enabled=args.autochar,
                verify_colors=args.tagverify,
            )
        if ENABLE_FINAL_MOVE and not dry_run:
            FINAL_OUTPUT.mkdir(parents=True, exist_ok=True)
            for folder in tag_root.iterdir():
                if not folder.is_dir() or folder.name == "raw":
                    continue
                dst = FINAL_OUTPUT / folder.name
                if dst.exists():
                    shutil.rmtree(dst, ignore_errors=True)
                log(f"Move {folder} -> {dst}")
                shutil.move(str(folder), dst)
        if TAG_FLAG.exists() and not dry_run:
            TAG_FLAG.unlink()
        if manual_pause:
            log("Step 6.5: Manual tagging pause")
            update_run_step("manual_pause")
            return
    if args.train:
        if not train_resume:
            log("Step 6.5: Ready to train")
            update_run_step("ready_to_train")
            return
        import torch

        if dry_run:
            log("[dry-run] Training requested but skipped (dry-run mode).")
        else:
            if not torch.cuda.is_available():
                raise RuntimeError("Training requires CUDA GPU; none detected.")

        if not dry_run:
            TRAIN_FLAG.write_text("running", encoding="utf-8")
            plans: List[KohyaPlan] = []
            try:
                log("Step 7a: Analyze datasets and build kohya configs")
                update_run_step("train_plan")
                run_id = os.environ.get("RUN_ID")
                allowed_run_ids = [run_id] if run_id else None
                plans = build_kohya_plans(
                    final_root=FINAL_OUTPUT,
                    settings=settings,
                    jobs_root=BUNDLE_ROOT / "trainer" / "jobs",
                    preset_override=None,
                    allowed_run_ids=allowed_run_ids,
                )
                if not plans:
                    log("No datasets found in final_ready; skipping training plan.")
                else:
                    for p in plans:
                        log(f"Training config written: {p.config_path}")

                log("Step 7b: Prepare training outputs (staging in trainer/output)")
                TRAIN_OUTPUT.mkdir(parents=True, exist_ok=True)
                ensure_dir_empty(TRAIN_OUTPUT)

                log("Step 7c: Training runs (kohya_ss, one per dataset)")
                update_run_step("train_run")
                for plan in plans:
                    log(f"Training {plan.name}: {plan.dataset_dir}")
                    run_kohya_training(
                        plan,
                        hf_token=str(settings.get("hf_token") or "") or None,
                        run_id=os.environ.get("RUN_ID"),
                    )

                log("Training done.")
            finally:
                if TRAIN_FLAG.exists():
                    TRAIN_FLAG.unlink()
    update_run_step("done")
    log("Workflow done.")


if __name__ == "__main__":
    main(parse_args())
