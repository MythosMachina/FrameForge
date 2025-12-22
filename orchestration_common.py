import json
import os
import sqlite3
import time
from pathlib import Path
from typing import Iterable, Optional

from db_broker_client import broker_enabled, broker_exec, broker_query

BUNDLE_ROOT = Path(__file__).resolve().parent
DEFAULT_DB = BUNDLE_ROOT / "_system" / "db" / "db.sqlite"
RUN_DB = os.environ.get("RUN_DB", str(DEFAULT_DB))
POLL_SECONDS = 5


def log(role: str, msg: str) -> None:
    ts = time.strftime("%Y-%m-%d %H:%M:%S")
    print(f"[{role} {ts}] {msg}", flush=True)


def db_conn() -> Optional[sqlite3.Connection]:
    if broker_enabled():
        return None
    return sqlite3.connect(RUN_DB)


def ensure_tables(conn: Optional[sqlite3.Connection]) -> None:
    if broker_enabled():
        broker_exec("ensure_tables")
        return
    cur = conn.cursor()
    cur.execute(
        """
        CREATE TABLE IF NOT EXISTS RunPlan (
            runId TEXT NOT NULL,
            step TEXT NOT NULL,
            status TEXT NOT NULL,
            meta TEXT,
            updatedAt DATETIME DEFAULT CURRENT_TIMESTAMP,
            PRIMARY KEY (runId, step)
        );
        """
    )
    cur.execute(
        """
        CREATE TABLE IF NOT EXISTS TrainProgress (
            runId TEXT PRIMARY KEY,
            epoch INTEGER,
            epochTotal INTEGER,
            step INTEGER,
            stepTotal INTEGER,
            raw TEXT,
            updatedAt DATETIME DEFAULT CURRENT_TIMESTAMP
        );
        """
    )
    cur.execute(
        """
        CREATE TABLE IF NOT EXISTS WorkerStatus (
            role TEXT PRIMARY KEY,
            pid INTEGER,
            state TEXT,
            runId TEXT,
            message TEXT,
            heartbeat INTEGER
        );
        """
    )
    conn.commit()


def get_queue_mode(conn: Optional[sqlite3.Connection]) -> str:
    if broker_enabled():
        resp = broker_query("get_queue_mode")
        return resp.get("data") or "running"
    try:
        cur = conn.cursor()
        cur.execute("SELECT value FROM Setting WHERE key='queue_mode';")
        row = cur.fetchone()
        if row and row[0]:
            val = str(row[0]).lower()
            if val in {"running", "paused", "stopped"}:
                return val
    except Exception:
        pass
    return "running"


def set_queue_mode(conn: Optional[sqlite3.Connection], mode: str) -> None:
    if broker_enabled():
        broker_exec("set_queue_mode", {"mode": mode})
        return
    try:
        cur = conn.cursor()
        cur.execute(
            "INSERT INTO Setting (key, value) VALUES ('queue_mode', ?) ON CONFLICT(key) DO UPDATE SET value=excluded.value;",
            (mode,),
        )
        conn.commit()
    except Exception:
        pass


def update_worker_status(
    conn: Optional[sqlite3.Connection],
    role: str,
    state: str,
    run_id: Optional[str] = None,
    message: str = "",
) -> None:
    if broker_enabled():
        broker_exec(
            "update_worker_status",
            {
                "role": role,
                "pid": os.getpid(),
                "state": state,
                "run_id": run_id,
                "message": message,
            },
        )
        return
    now = int(time.time())
    cur = conn.cursor()
    cur.execute(
        """
        INSERT INTO WorkerStatus (role, pid, state, runId, message, heartbeat)
        VALUES (?, ?, ?, ?, ?, ?)
        ON CONFLICT(role) DO UPDATE SET
            pid=excluded.pid,
            state=excluded.state,
            runId=excluded.runId,
            message=excluded.message,
            heartbeat=excluded.heartbeat;
        """,
        (role, os.getpid(), state, run_id, message, now),
    )
    conn.commit()


def claim_role_lock(conn: Optional[sqlite3.Connection], role: str, stale_seconds: int = 30) -> bool:
    if broker_enabled():
        resp = broker_exec(
            "claim_role_lock",
            {"role": role, "pid": os.getpid(), "stale_seconds": stale_seconds},
        )
        return bool(resp.get("data"))
    now = int(time.time())
    cur = conn.cursor()
    cur.execute("BEGIN IMMEDIATE;")
    cur.execute("SELECT pid, heartbeat FROM WorkerStatus WHERE role=?;", (role,))
    row = cur.fetchone()
    if row:
        pid, heartbeat = row[0], row[1]
        if pid and _pid_alive(int(pid)):
            conn.rollback()
            return False
    update_worker_status(conn, role, "starting", message="lock claimed")
    return True


def _pid_alive(pid: int) -> bool:
    try:
        os.kill(pid, 0)
        return True
    except OSError:
        return False


def ensure_run_plan(conn: Optional[sqlite3.Connection], run_id: str, steps: Iterable[str]) -> None:
    if broker_enabled():
        broker_exec("ensure_run_plan", {"run_id": run_id, "steps": list(steps)})
        return
    cur = conn.cursor()
    for step in steps:
        cur.execute(
            """
            INSERT INTO RunPlan (runId, step, status)
            VALUES (?, ?, 'pending')
            ON CONFLICT(runId, step) DO NOTHING;
            """,
            (run_id, step),
        )
    conn.commit()


def set_plan_step(conn: Optional[sqlite3.Connection], run_id: str, step: str, status: str, meta: Optional[dict] = None) -> None:
    if broker_enabled():
        broker_exec(
            "set_plan_step",
            {"run_id": run_id, "step": step, "status": status, "meta": meta},
        )
        return
    cur = conn.cursor()
    meta_json = json.dumps(meta) if meta is not None else None
    cur.execute(
        """
        UPDATE RunPlan
        SET status=?, meta=?, updatedAt=CURRENT_TIMESTAMP
        WHERE runId=? AND step=?;
        """,
        (status, meta_json, run_id, step),
    )
    conn.commit()


def fetch_next_run(conn: Optional[sqlite3.Connection], status: str) -> Optional[dict]:
    if broker_enabled():
        resp = broker_query("fetch_next_run", {"status": status})
        return resp.get("data")
    cur = conn.cursor()
    cur.execute(
        """
        SELECT id, runId, runName, name, type, flags, uploadPath, trainProfile
        FROM Run
        WHERE status=?
        ORDER BY createdAt ASC
        LIMIT 1;
        """,
        (status,),
    )
    row = cur.fetchone()
    if not row:
        return None
    return {
        "id": row[0],
        "runId": row[1],
        "runName": row[2],
        "name": row[3],
        "type": row[4],
        "flags": row[5],
        "uploadPath": row[6],
        "trainProfile": row[7],
    }


def has_run_with_status(conn: Optional[sqlite3.Connection], status: str) -> bool:
    if broker_enabled():
        try:
            resp = broker_query(
                "sql_query",
                {"sql": "SELECT runId FROM Run WHERE status=? LIMIT 1;", "params": [status]},
            )
            rows = resp.get("data") or []
            return bool(rows)
        except Exception:
            return False
    try:
        cur = conn.cursor()
        cur.execute("SELECT runId FROM Run WHERE status=? LIMIT 1;", (status,))
        return cur.fetchone() is not None
    except Exception:
        return False


def mark_run_status(
    conn: Optional[sqlite3.Connection],
    run_id_db: int,
    status: str,
    last_step: str = "",
    error: Optional[str] = None,
    started: bool = False,
    finished: bool = False,
) -> None:
    if broker_enabled():
        broker_exec(
            "mark_run_status",
            {
                "run_id_db": run_id_db,
                "status": status,
                "last_step": last_step,
                "error": error,
                "started": started,
                "finished": finished,
            },
        )
        return
    fields = ["status=?", "lastStep=?"]
    params = [status, last_step or status]
    if error is not None:
        fields.append("error=?")
        params.append(error)
    if started:
        fields.append("startedAt=CURRENT_TIMESTAMP")
    if finished:
        fields.append("finishedAt=CURRENT_TIMESTAMP")
    params.append(run_id_db)
    sql = f"UPDATE Run SET {', '.join(fields)} WHERE id=?;"
    cur = conn.cursor()
    cur.execute(sql, params)
    conn.commit()


def fetch_active_run(conn: Optional[sqlite3.Connection]) -> Optional[dict]:
    if broker_enabled():
        resp = broker_query("fetch_active_run")
        return resp.get("data")
    cur = conn.cursor()
    cur.execute(
        """
        SELECT runId, status FROM Run
        WHERE status IN ('running', 'queued_initiated', 'ready_for_finish')
        ORDER BY createdAt ASC
        LIMIT 1;
        """
    )
    row = cur.fetchone()
    if not row:
        return None
    return {"runId": row[0], "status": row[1]}


def recover_inflight_runs(conn: Optional[sqlite3.Connection]) -> None:
    if broker_enabled():
        broker_exec("recover_inflight_runs")
        return
    cur = conn.cursor()
    cur.execute(
        """
        SELECT runId FROM Run
        WHERE status IN ('running', 'failed_worker', 'queued_initiated')
        """
    )
    rows = cur.fetchall()
    run_ids = [r[0] for r in rows]
    if not run_ids:
        return
    for run_id in run_ids:
        cur.execute(
            """
            UPDATE Run
            SET status='queued_initiated',
                lastStep='unpacked',
                error=NULL,
                startedAt=NULL,
                finishedAt=NULL
            WHERE runId=?;
            """,
            (run_id,),
        )
        cur.execute("DELETE FROM RunPlan WHERE runId=?;", (run_id,))
        cur.execute("DELETE FROM TrainProgress WHERE runId=?;", (run_id,))
    conn.commit()


def set_run_downloads(
    conn: Optional[sqlite3.Connection],
    run_id_db: int,
    dataset: Optional[str] = None,
    lora: Optional[str] = None,
) -> None:
    if broker_enabled():
        broker_exec("set_run_downloads", {"run_id_db": run_id_db, "dataset": dataset, "lora": lora})
        return
    cur = conn.cursor()
    if dataset is not None:
        cur.execute("UPDATE Run SET datasetDownload=? WHERE id=?;", (dataset, run_id_db))
    if lora is not None:
        cur.execute("UPDATE Run SET loraDownload=? WHERE id=?;", (lora, run_id_db))
    conn.commit()
