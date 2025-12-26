#!/usr/bin/env python3
import json
import os
import subprocess
import time
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
    has_run_with_status,
    log_error,
    log,
    mark_run_status,
    recover_inflight_runs,
    set_plan_step,
    set_queue_mode,
    update_worker_status,
)

ROLE = "orchestrator"


def detect_step(line: str) -> Optional[str]:
    l = line.lower()
    if "step 1: quickrename" in l:
        return "rename"
    if "step 2: capping" in l:
        return "cap"
    if "images-only input" in l:
        return "images_only"
    if "step 2.5: archive" in l:
        return "archive"
    if "step 2.5: move capped frames" in l:
        return "move_capped"
    if "step 2.5: move remaining source" in l:
        return "merge_inputs"
    if "step 3: select caps" in l:
        return "select"
    if "step 4: crop and flip" in l:
        return "cropflip"
    if "manual tagging pause" in l:
        return "manual_pause"
    if "ready to train" in l:
        return "ready_to_train"
    if "step 5: move finished set" in l:
        return "move_final"
    if "step 6: autotag" in l:
        return "autotag"
    if "step 7a: analyze" in l:
        return "train_plan"
    if "step 7b: prepare training outputs" in l:
        return "train_stage"
    if "step 7c: training runs" in l:
        return "train_run"
    if "workflow done" in l:
        return "done"
    if "workflow_start" in l:
        return "workflow_start"
    return None


def build_workflow_cmd(flags: dict) -> list[str]:
    args = ["python3", str(BUNDLE_ROOT / "workflow.py")]
    if flags.get("autotag"):
        args.append("--autotag")
    if flags.get("autochar"):
        args.append("--autochar")
    if flags.get("tagverify"):
        args.append("--tagverify")
    if flags.get("facecap"):
        args.append("--facecap")
    if flags.get("gpu"):
        args.append("--gpu")
    if flags.get("train"):
        args.append("--train")
    if flags.get("imagesOnly"):
        args.append("--images-only")
    if flags.get("manualTagging") and not flags.get("manualResume"):
        args.append("--manual-tagging")
    if flags.get("manualResume"):
        args.append("--manual-resume")
    if flags.get("trainResume"):
        args.append("--train-resume")
    return args


def _recover_inflight_runs(conn) -> None:
    recover_inflight_runs(conn)


def main() -> None:
    while True:
        try:
            conn = db_conn()
            ensure_tables(conn)
            _recover_inflight_runs(conn)
        except Exception as exc:
            log(ROLE, f"DB unavailable: {exc}")
            time.sleep(POLL_SECONDS)
            continue

        try:
            if not claim_role_lock(conn, ROLE):
                log(ROLE, "another orchestrator is active; exiting")
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

            if has_run_with_status(conn, "ready_for_finish"):
                update_worker_status(conn, ROLE, "blocked", message="waiting for finisher")
                time.sleep(POLL_SECONDS)
                continue

            run = fetch_next_run(conn, "ready_to_train")
            run_status = "ready_to_train"
            if not run:
                run = fetch_next_run(conn, "queued_initiated")
                run_status = "queued_initiated"
            if not run:
                update_worker_status(conn, ROLE, "idle")
                time.sleep(POLL_SECONDS)
                continue

            update_worker_status(conn, ROLE, "busy", run_id=run["runId"])
            mark_run_status(conn, run["id"], "running", "workflow_start", started=True)

            flags = {}
            try:
                if run.get("flags"):
                    flags = json.loads(run["flags"])
            except Exception:
                flags = {}
            if run_status == "ready_to_train":
                flags["trainResume"] = True
                flags["manualTagging"] = False

            env = dict(**os.environ)
            env["RUN_ID"] = run["runId"]
            env["RUN_DB"] = os.environ.get("RUN_DB", "")
            env["FRAMEFORGE_INPUT_ROOT"] = str(BUNDLE_ROOT / "INBOX")
            # Let the orchestrator own DB status updates; workflow only emits train progress.
            env["ORCHESTRATOR_DRIVER"] = "1"
            train_profile = run.get("trainProfile") or flags.get("trainProfile")
            if train_profile:
                env["TRAIN_PROFILE"] = str(train_profile)
            if flags.get("autocharPreset"):
                env["AUTOCHAR_PRESET"] = flags["autocharPreset"]

            cmd = build_workflow_cmd(flags)
            log_dir = BUNDLE_ROOT / "_system" / "logs"
            log_dir.mkdir(parents=True, exist_ok=True)
            log_path = log_dir / f"workflow_{run['runId']}.log"
            proc = subprocess.Popen(
                cmd,
                cwd=str(BUNDLE_ROOT),
                env=env,
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                text=True,
                bufsize=1,
            )
            last_step = None
            last_heartbeat = time.time()
            if proc.stdout:
                with open(log_path, "w", encoding="utf-8") as log_fp:
                    for line in proc.stdout:
                        if not line:
                            break
                        log_fp.write(line)
                        step = detect_step(line)
                        if step:
                            last_step = step
                            try:
                                set_plan_step(conn, run["runId"], step, "done")
                            except Exception:
                                pass
                        now = time.time()
                        if now - last_heartbeat >= POLL_SECONDS:
                            try:
                                update_worker_status(conn, ROLE, "busy", run_id=run["runId"])
                            except Exception:
                                pass
                            last_heartbeat = now
            ret = proc.wait()

            if ret == 0:
                if flags.get("manualTagging") and last_step == "manual_pause":
                    mark_run_status(conn, run["id"], "manual_tagging", last_step or "manual_pause")
                    log(ROLE, f"run {run['runId']} waiting manual tagging (log: {log_path})")
                elif last_step == "ready_to_train":
                    mark_run_status(conn, run["id"], "ready_to_train", last_step or "ready_to_train")
                    log(ROLE, f"run {run['runId']} ready_to_train (log: {log_path})")
                else:
                    mark_run_status(conn, run["id"], "ready_for_finish", last_step or "packaging")
                    log(ROLE, f"run {run['runId']} ready_for_finish (log: {log_path})")
            else:
                msg = f"workflow exited {ret}; log: {log_path}"
                mark_run_status(conn, run["id"], "failed_worker", last_step or "failed_worker", error=msg, finished=True)
                log_error(
                    conn,
                    run_id_db=run["id"],
                    component=ROLE,
                    stage=last_step or "workflow",
                    step=last_step or "failed_worker",
                    error_type="external_process_failed",
                    error_code=f"workflow_exit_{ret}",
                    error_message=msg,
                    log_path=log_path,
                )
                set_queue_mode(conn, "paused")
                log(ROLE, f"run {run['runId']} failed: {msg}")
                time.sleep(POLL_SECONDS)
        except Exception as exc:
            log(ROLE, f"error: {exc}")
            time.sleep(POLL_SECONDS)


if __name__ == "__main__":
    main()
