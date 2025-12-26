#!/usr/bin/env python3
import shutil
import time
import zipfile
from pathlib import Path

from db_broker_client import broker_enabled, broker_query
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
    get_worker_status,
    pid_alive,
    append_job_log,
    log_error,
    log,
    mark_run_status,
    set_plan_step,
    set_queue_mode,
    update_worker_status,
)

ROLE = "initiator"
INPUT_ROOT = BUNDLE_ROOT / "INBOX"
LOG_PATH = BUNDLE_ROOT / "_system" / "logs" / "initiator.service.log"
ORCHESTRATOR_ROLE = "orchestrator"
ORCHESTRATOR_STALE_SECONDS = 120
ORCHESTRATOR_QUEUE_STUCK_SECONDS = 60
ORCHESTRATOR_RUNNING_STUCK_SECONDS = 300

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


def _orchestrator_down(conn) -> bool:
    status = get_worker_status(conn, ORCHESTRATOR_ROLE)
    if not status:
        return True
    pid_ok = pid_alive(status.get("pid"))
    heartbeat = status.get("heartbeat") or 0
    if pid_ok:
        return False
    return int(time.time()) - int(heartbeat) > ORCHESTRATOR_STALE_SECONDS


def _mark_orchestrator_down_runs(conn) -> None:
    if not _orchestrator_down(conn):
        return
    sql = (
        "SELECT r.id, r.runId, r.runName, r.status, "
        "CAST(strftime('%s','now') AS INTEGER) - "
        "CAST(strftime('%s', COALESCE(tp.updatedAt, rp.updatedAt, r.startedAt, r.createdAt)) AS INTEGER) AS age "
        "FROM Run r "
        "LEFT JOIN (SELECT runId, MAX(updatedAt) AS updatedAt FROM TrainProgress GROUP BY runId) tp "
        "ON tp.runId = r.runId "
        "LEFT JOIN (SELECT runId, MAX(updatedAt) AS updatedAt FROM RunPlan GROUP BY runId) rp "
        "ON rp.runId = r.runId "
        "WHERE r.status IN ('queued_initiated', 'ready_to_train', 'running')"
    )
    if broker_enabled():
        try:
            resp = broker_query("sql_query", {"sql": sql, "params": []})
            rows = resp.get("data") or []
        except Exception:
            return
    else:
        cur = conn.cursor()
        cur.execute(sql)
        rows = cur.fetchall()
    for row in rows:
        if broker_enabled():
            run_id_db = row.get("id")
            run_id = row.get("runId")
            run_name = row.get("runName") or run_id
            status = row.get("status") or ""
            age = row.get("age") or 0
        else:
            run_id_db, run_id, run_name, status, age = row
        if not run_id_db or not run_id:
            continue
        threshold = ORCHESTRATOR_QUEUE_STUCK_SECONDS
        if str(status).lower() == "running":
            threshold = ORCHESTRATOR_RUNNING_STUCK_SECONDS
        if int(age) < threshold:
            continue
        msg = "orchestrator inactive; job stuck in queue"
        error_type = "service_down"
        if str(status).lower() == "running":
            msg = "orchestrator inactive; run stalled"
            error_type = "stalled_run"
        log_path = BUNDLE_ROOT / "_system" / "logs" / f"orchestrator_{run_id}.log"
        append_job_log(log_path, msg)
        mark_run_status(conn, run_id_db, "failed_worker", "orchestrator_down", error=msg, finished=True)
        log_error(
            conn,
            run_id_db=run_id_db,
            component=ORCHESTRATOR_ROLE,
            stage="orchestrator",
            step="orchestrator_down",
            error_type=error_type,
            error_code="orchestrator_down",
            error_message=msg,
            log_path=log_path,
        )


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
            _mark_orchestrator_down_runs(conn)

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
                log_path = BUNDLE_ROOT / "_system" / "logs" / f"initiator_{run['runId']}.log"
                append_job_log(log_path, msg)
                mark_run_status(conn, run["id"], "failed_initiator", "missing_upload", error=msg, finished=True)
                log_error(
                    conn,
                    run_id_db=run["id"],
                    component=ROLE,
                    stage="unzip",
                    step="missing_upload",
                    error_type="invalid_input",
                    error_code="missing_upload",
                    error_message=msg,
                    log_path=log_path,
                )
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
                log_path = BUNDLE_ROOT / "_system" / "logs" / f"initiator_{run['runId']}.log"
                append_job_log(log_path, msg)
                mark_run_status(conn, run["id"], "failed_initiator", "unzip_failed", error=msg, finished=True)
                log_error(
                    conn,
                    run_id_db=run["id"],
                    component=ROLE,
                    stage="unzip",
                    step="unzip_failed",
                    error_type="invalid_input",
                    error_code="unzip_failed",
                    error_message=msg,
                    error_detail=str(exc),
                    log_path=log_path,
                )
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
