#!/usr/bin/env python3
import json
import os
import re
import shutil
import time
import zipfile
from pathlib import Path
from typing import Optional

from orchestration_common import (
    BUNDLE_ROOT,
    POLL_SECONDS,
    claim_role_lock,
    db_conn,
    ensure_tables,
    fetch_next_run,
    get_queue_mode,
    append_job_log,
    log_error,
    log,
    mark_run_status,
    set_run_downloads,
    set_plan_step,
    set_queue_mode,
    update_worker_status,
)

ROLE = "finisher"

WEBAPP_ROOT = BUNDLE_ROOT / "webapp"
SYSTEM_ROOT = BUNDLE_ROOT / "_system"
OUTPUT_DIR = BUNDLE_ROOT / "ARCHIVE" / "zips"
INPUT_ROOT = BUNDLE_ROOT / "INBOX"
FINAL_OUTPUT = BUNDLE_ROOT / "OUTPUTS" / "datasets"
FINAL_LORA = BUNDLE_ROOT / "OUTPUTS" / "loras"
ARCHIVE_MP4 = BUNDLE_ROOT / "ARCHIVE" / "mp4"
TRAINER_ROOT = BUNDLE_ROOT / "trainer"
TRAIN_OUTPUT = SYSTEM_ROOT / "trainer" / "output"
TRAIN_DATASET = SYSTEM_ROOT / "trainer" / "dataset" / "images"
TRAIN_JOBS = SYSTEM_ROOT / "trainer" / "jobs"
UPLOAD_DIR = SYSTEM_ROOT / "webapp" / "storage" / "uploads"
SAMPLE_EPOCH_RE = re.compile(r"_e(\d{6})_", re.IGNORECASE)


def zip_folder(src: Path, dest_zip: Path) -> None:
    dest_zip.parent.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(dest_zip, "w", zipfile.ZIP_DEFLATED) as zf:
        for root, _, files in os.walk(src):
            for f in files:
                full = Path(root) / f
                arc = full.relative_to(src)
                zf.write(full, arcname=str(arc))


def _find_sample_dir(run_name: str) -> Optional[Path]:
    candidates = [
        FINAL_LORA / run_name / "samples",
        FINAL_LORA / run_name / "sample",
        TRAIN_OUTPUT / run_name / "samples",
        TRAIN_OUTPUT / run_name / "sample",
    ]
    for cand in candidates:
        if cand.exists():
            return cand
    return None


def _group_samples_by_epoch(sample_dir: Path) -> dict[int, list[Path]]:
    grouped: dict[int, list[Path]] = {}
    for p in sorted(sample_dir.iterdir()):
        if not p.is_file() or p.suffix.lower() not in {".png", ".jpg", ".jpeg", ".webp"}:
            continue
        match = SAMPLE_EPOCH_RE.search(p.name)
        if not match:
            continue
        epoch = int(match.group(1))
        grouped.setdefault(epoch, []).append(p)
    return grouped


def zip_samples(run: dict) -> None:
    sample_dir = _find_sample_dir(run["runName"])
    if not sample_dir:
        return
    grouped = _group_samples_by_epoch(sample_dir)
    if not grouped:
        return
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    # Zip all samples
    all_zip = OUTPUT_DIR / f"{run['name']}_samples.zip"
    zip_folder(sample_dir, all_zip)
    # Zip per-epoch samples
    for epoch, files in grouped.items():
        epoch_zip = OUTPUT_DIR / f"{run['name']}_samples_e{epoch:06d}.zip"
        with zipfile.ZipFile(epoch_zip, "w", zipfile.ZIP_DEFLATED) as zf:
            for f in files:
                zf.write(f, arcname=f.name)


def collect_training_outputs(run: dict) -> None:
    src = TRAIN_OUTPUT / run["runName"]
    if not src.exists():
        return
    FINAL_LORA.mkdir(parents=True, exist_ok=True)
    dst = FINAL_LORA / run["runName"]
    if dst.exists():
        shutil.rmtree(dst, ignore_errors=True)
    shutil.move(str(src), dst)
    trigger = run["runName"].split("_", 1)[1] if "_" in run["runName"] else run["runName"]
    staged_dir = TRAIN_DATASET / f"1_{trigger}"
    if staged_dir.exists():
        shutil.rmtree(staged_dir, ignore_errors=True)
    job_toml = TRAIN_JOBS / f"{run['runName']}.toml"
    job_prompts = TRAIN_JOBS / f"{run['runName']}_sample_prompts.txt"
    for job_file in (job_toml, job_prompts):
        if job_file.exists():
            job_file.unlink(missing_ok=True)


def cleanup_run_artifacts(run: dict) -> None:
    """
    Only remove training artifacts for the run (dataset staging + output).
    Keep input/workflow files intact for safety.
    """
    trigger = run["runName"].split("_", 1)[1] if "_" in run["runName"] else run["runName"]
    targets = [
        TRAIN_OUTPUT / run["runName"],
        TRAIN_DATASET / f"1_{trigger}",
    ]
    for target in targets:
        try:
            if target.is_dir():
                shutil.rmtree(target, ignore_errors=True)
            else:
                target.unlink(missing_ok=True)
        except Exception:
            pass


def main() -> None:
    while True:
        try:
            conn = db_conn()
            ensure_tables(conn)
        except Exception as exc:
            log(ROLE, f"DB unavailable: {exc}")
            time.sleep(POLL_SECONDS)
            continue

        try:
            if not claim_role_lock(conn, ROLE):
                log(ROLE, "another finisher is active; exiting")
                if conn:
                    conn.close()
                return
            break
        except Exception:
            if conn:
                conn.close()
            time.sleep(POLL_SECONDS)

    while True:
        try:
            mode = get_queue_mode(conn)
            if mode in {"paused", "stopped"}:
                update_worker_status(conn, ROLE, mode, message=f"queue {mode}")
                time.sleep(POLL_SECONDS)
                continue

            run = fetch_next_run(conn, "ready_for_finish")
            if not run:
                update_worker_status(conn, ROLE, "idle")
                time.sleep(POLL_SECONDS)
                continue

            update_worker_status(conn, ROLE, "busy", run_id=run["runId"])
            flags = {}
            try:
                if run.get("flags"):
                    flags = json.loads(run["flags"])
            except Exception:
                flags = {}

            collect_training_outputs(run)

            dataset_dir = FINAL_OUTPUT / run["runName"]
            if not dataset_dir.exists():
                msg = "dataset output missing"
                mark_run_status(conn, run["id"], "failed_finish", "missing_dataset", error=msg, finished=True)
                log_path = SYSTEM_ROOT / "logs" / f"finisher_{run['runId']}.log"
                append_job_log(log_path, msg)
                log_error(
                    conn,
                    run_id_db=run["id"],
                    component=ROLE,
                    stage="package_dataset",
                    step="missing_dataset",
                    error_type="io_filesystem",
                    error_code="missing_dataset",
                    error_message=msg,
                    log_path=log_path,
                )
                set_queue_mode(conn, "paused")
                log(ROLE, f"{run['runId']} {msg}")
                time.sleep(POLL_SECONDS)
                continue

            if flags.get("train"):
                lora_dir = FINAL_LORA / run["runName"]
                has_lora = lora_dir.exists() and any(lora_dir.glob("*.safetensors"))
                if not has_lora:
                    msg = "lora output missing"
                    mark_run_status(conn, run["id"], "failed_finish", "missing_lora", error=msg, finished=True)
                    log_path = SYSTEM_ROOT / "logs" / f"finisher_{run['runId']}.log"
                    append_job_log(log_path, msg)
                    log_error(
                        conn,
                        run_id_db=run["id"],
                        component=ROLE,
                        stage="package_lora",
                        step="missing_lora",
                        error_type="io_filesystem",
                        error_code="missing_lora",
                        error_message=msg,
                        log_path=log_path,
                    )
                    set_queue_mode(conn, "paused")
                    log(ROLE, f"{run['runId']} {msg}")
                    time.sleep(POLL_SECONDS)
                    continue

            set_plan_step(conn, run["runId"], "package_dataset", "done")
            zip_path = OUTPUT_DIR / f"{run['name']}.zip"
            zip_folder(dataset_dir, zip_path)
            set_run_downloads(conn, run["id"], dataset=f"/api/download/{zip_path.name}")

            if flags.get("train"):
                set_plan_step(conn, run["runId"], "package_lora", "done")
                lora_dir = FINAL_LORA / run["runName"]
                if lora_dir.exists():
                    zip_path = OUTPUT_DIR / f"{run['name']}_lora.zip"
                    zip_folder(lora_dir, zip_path)
                    set_run_downloads(conn, run["id"], lora=f"/api/download/{zip_path.name}")

            if flags.get("train"):
                zip_samples(run)

            set_plan_step(conn, run["runId"], "cleanup", "done")
            cleanup_run_artifacts(run)
            mark_run_status(conn, run["id"], "done", "done", finished=True)
            log(ROLE, f"run {run['runId']} done")
        except Exception as exc:
            log(ROLE, f"error: {exc}")
            time.sleep(POLL_SECONDS)


if __name__ == "__main__":
    main()
