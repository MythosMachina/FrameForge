#!/usr/bin/env python3
import shutil
import time
import zipfile
from pathlib import Path

from orchestration_common import (
    BUNDLE_ROOT,
    POLL_SECONDS,
    claim_role_lock,
    db_conn,
    ensure_run_plan,
    ensure_tables,
    fetch_active_run,
    fetch_next_run,
    get_queue_mode,
    log,
    mark_run_status,
    set_plan_step,
    set_queue_mode,
    update_worker_status,
)

ROLE = "initiator"
INPUT_ROOT = BUNDLE_ROOT / "INBOX"

PLAN_STEPS = [
    "unzip",
    "rename",
    "cap",
    "archive",
    "move_capped",
    "merge_inputs",
    "select",
    "cropflip",
    "manual_pause",
    "manual_edit",
    "manual_done",
    "move_final",
    "autotag",
    "train_plan",
    "train_stage",
    "train_run",
    "collect_training",
    "package_dataset",
    "package_lora",
    "cleanup",
]


def unzip_run(run: dict) -> Path:
    dest = INPUT_ROOT / run["runName"]
    if dest.exists():
        shutil.rmtree(dest, ignore_errors=True)
    dest.mkdir(parents=True, exist_ok=True)
    zip_path = Path(run["uploadPath"])
    with zipfile.ZipFile(zip_path, "r") as zf:
        zf.extractall(dest)
    return dest


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
                log(ROLE, "another initiator is active; exiting")
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

            # Allow only one active prep pipeline at a time.
            # ready_to_train should not block prep; only running/queued_initiated/ready_for_finish.
            active = fetch_active_run(conn)
            if active and active.get("status") not in {"ready_to_train"}:
                update_worker_status(
                    conn,
                    ROLE,
                    "blocked",
                    run_id=active["runId"],
                    message=f"active run {active['runId']} ({active['status']})",
                )
                time.sleep(POLL_SECONDS)
                continue

            run = fetch_next_run(conn, "queued")
            if not run:
                update_worker_status(conn, ROLE, "idle")
                time.sleep(POLL_SECONDS)
                continue

            update_worker_status(conn, ROLE, "busy", run_id=run["runId"])
            if not run.get("uploadPath") or not Path(run["uploadPath"]).exists():
                msg = f"missing upload at {run.get('uploadPath')}"
                log(ROLE, f"{run['runId']} {msg}")
                mark_run_status(conn, run["id"], "failed_initiator", "missing_upload", error=msg, finished=True)
                set_queue_mode(conn, "paused")
                time.sleep(POLL_SECONDS)
                continue

            try:
                unzip_run(run)
                ensure_run_plan(conn, run["runId"], PLAN_STEPS)
                set_plan_step(conn, run["runId"], "unzip", "done")
                mark_run_status(conn, run["id"], "queued_initiated", "unpacked")
                log(ROLE, f"prepared {run['runId']} -> queued_initiated")
            except Exception as exc:
                msg = f"unzip failed: {exc}"
                log(ROLE, f"{run['runId']} {msg}")
                mark_run_status(conn, run["id"], "failed_initiator", "unzip_failed", error=msg, finished=True)
                set_queue_mode(conn, "paused")
                time.sleep(POLL_SECONDS)
        except Exception as exc:
            log(ROLE, f"error: {exc}")
        finally:
            try:
                update_worker_status(conn, ROLE, "idle")
            except Exception:
                pass
            time.sleep(POLL_SECONDS)


if __name__ == "__main__":
    main()
