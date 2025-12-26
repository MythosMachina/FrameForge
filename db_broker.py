#!/usr/bin/env python3
import json
import os
import sqlite3
import threading
import time
from http.server import BaseHTTPRequestHandler, HTTPServer
from pathlib import Path
from typing import Any, Dict, Optional

BUNDLE_ROOT = Path(__file__).resolve().parent
DEFAULT_DB = BUNDLE_ROOT / "_system" / "db" / "db.sqlite"

DB_PATH = Path(os.environ.get("DB_BROKER_DB", str(DEFAULT_DB)))
HOST = os.environ.get("DB_BROKER_HOST", "127.0.0.1")
PORT = int(os.environ.get("DB_BROKER_PORT", "8799"))

_LOCK = threading.Lock()
_CONN: Optional[sqlite3.Connection] = None


def _pid_alive(pid: int) -> bool:
    try:
        os.kill(pid, 0)
        return True
    except OSError:
        return False


def _conn() -> sqlite3.Connection:
    global _CONN
    if _CONN is None:
        DB_PATH.parent.mkdir(parents=True, exist_ok=True)
        _CONN = sqlite3.connect(str(DB_PATH), check_same_thread=False)
        _CONN.row_factory = sqlite3.Row
        _CONN.execute("PRAGMA journal_mode=WAL;")
        _CONN.execute("PRAGMA synchronous=NORMAL;")
        _CONN.execute("PRAGMA busy_timeout=5000;")
        _CONN.execute("PRAGMA temp_store=MEMORY;")
        _ensure_tables(_CONN)
    return _CONN


def _ensure_tables(conn: sqlite3.Connection) -> None:
    cur = conn.cursor()
    cur.execute(
        """
        CREATE TABLE IF NOT EXISTS TrainProfile (
            id INTEGER PRIMARY KEY,
            name TEXT UNIQUE NOT NULL,
            label TEXT,
            settings TEXT NOT NULL,
            isDefault BOOLEAN DEFAULT 0,
            createdAt DATETIME DEFAULT CURRENT_TIMESTAMP,
            updatedAt DATETIME DEFAULT CURRENT_TIMESTAMP
        );
        """
    )
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
    cur.execute(
        """
        CREATE TABLE IF NOT EXISTS ErrorLog (
            id INTEGER PRIMARY KEY,
            runId TEXT,
            component TEXT,
            stage TEXT,
            step TEXT,
            errorType TEXT,
            errorCode TEXT,
            errorMessage TEXT,
            errorDetail TEXT,
            logPath TEXT,
            logTail TEXT,
            logMissing BOOLEAN DEFAULT 0,
            createdAt DATETIME DEFAULT CURRENT_TIMESTAMP
        );
        """
    )
    # Ensure Run.trainProfile exists (best-effort, no-op if already present)
    try:
        cur.execute("PRAGMA table_info(Run);")
        cols = [row[1] for row in cur.fetchall()]
        if "trainProfile" not in cols:
            cur.execute("ALTER TABLE Run ADD COLUMN trainProfile TEXT;")
    except Exception:
        pass
    conn.commit()


def _op_get_queue_mode(_: Dict[str, Any]) -> str:
    cur = _conn().cursor()
    cur.execute("SELECT value FROM Setting WHERE key='queue_mode';")
    row = cur.fetchone()
    if row and row[0]:
        val = str(row[0]).lower()
        if val in {"running", "paused", "stopped"}:
            return val
    return "running"


def _op_set_queue_mode(args: Dict[str, Any]) -> None:
    mode = args.get("mode", "running")
    cur = _conn().cursor()
    cur.execute(
        "INSERT INTO Setting (key, value) VALUES ('queue_mode', ?) "
        "ON CONFLICT(key) DO UPDATE SET value=excluded.value;",
        (mode,),
    )
    _conn().commit()


def _op_update_worker_status(args: Dict[str, Any]) -> None:
    now = int(time.time())
    cur = _conn().cursor()
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
        (
            args.get("role"),
            args.get("pid"),
            args.get("state"),
            args.get("run_id"),
            args.get("message", ""),
            now,
        ),
    )
    _conn().commit()


def _op_claim_role_lock(args: Dict[str, Any]) -> bool:
    role = args.get("role")
    pid = int(args.get("pid", 0))
    cur = _conn().cursor()
    cur.execute("BEGIN IMMEDIATE;")
    cur.execute("SELECT pid, heartbeat FROM WorkerStatus WHERE role=?;", (role,))
    row = cur.fetchone()
    if row:
        existing_pid = int(row[0] or 0)
        if existing_pid and _pid_alive(existing_pid):
            _conn().rollback()
            return False
    _op_update_worker_status(
        {
            "role": role,
            "pid": pid,
            "state": "starting",
            "run_id": None,
            "message": "lock claimed",
        }
    )
    return True


def _op_ensure_run_plan(args: Dict[str, Any]) -> None:
    run_id = args.get("run_id")
    steps = args.get("steps") or []
    cur = _conn().cursor()
    for step in steps:
        cur.execute(
            """
            INSERT INTO RunPlan (runId, step, status)
            VALUES (?, ?, 'pending')
            ON CONFLICT(runId, step) DO NOTHING;
            """,
            (run_id, step),
        )
    _conn().commit()


def _op_set_plan_step(args: Dict[str, Any]) -> None:
    run_id = args.get("run_id")
    step = args.get("step")
    status = args.get("status")
    meta = json.dumps(args.get("meta")) if args.get("meta") is not None else None
    cur = _conn().cursor()
    cur.execute(
        """
        UPDATE RunPlan
        SET status=?, meta=?, updatedAt=CURRENT_TIMESTAMP
        WHERE runId=? AND step=?;
        """,
        (status, meta, run_id, step),
    )
    _conn().commit()


def _op_fetch_next_run(args: Dict[str, Any]) -> Optional[Dict[str, Any]]:
    status = args.get("status")
    cur = _conn().cursor()
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


def _op_fetch_active_run(_: Dict[str, Any]) -> Optional[Dict[str, Any]]:
    cur = _conn().cursor()
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


def _op_mark_run_status(args: Dict[str, Any]) -> None:
    run_id_db = args.get("run_id_db")
    status = args.get("status")
    last_step = args.get("last_step") or status
    error = args.get("error")
    started = bool(args.get("started"))
    finished = bool(args.get("finished"))
    fields = ["status=?", "lastStep=?"]
    params = [status, last_step]
    if error is not None:
        fields.append("error=?")
        params.append(error)
    if started:
        fields.append("startedAt=CURRENT_TIMESTAMP")
    if finished:
        fields.append("finishedAt=CURRENT_TIMESTAMP")
    params.append(run_id_db)
    sql = f"UPDATE Run SET {', '.join(fields)} WHERE id=?;"
    cur = _conn().cursor()
    cur.execute(sql, params)
    _conn().commit()


def _op_update_run_step(args: Dict[str, Any]) -> None:
    run_id = args.get("run_id")
    step = args.get("step")
    cur = _conn().cursor()
    cur.execute(
        "UPDATE Run SET lastStep=?, status='running' WHERE runId=?",
        (step, run_id),
    )
    _conn().commit()


def _op_mark_run_done(args: Dict[str, Any]) -> None:
    run_name = args.get("run_name")
    run_id = None
    if run_name and run_name.split("_", 1)[0].isdigit():
        run_id = run_name.split("_", 1)[0]
    cur = _conn().cursor()
    cur.execute(
        "UPDATE Run SET status='done', lastStep='done', finishedAt=CURRENT_TIMESTAMP WHERE runName=? OR runId=?",
        (run_name, run_id or run_name),
    )
    _conn().commit()


def _op_upsert_train_progress(args: Dict[str, Any]) -> None:
    run_id = args.get("run_id")
    cur = _conn().cursor()
    cur.execute(
        """
        INSERT INTO TrainProgress (runId, epoch, epochTotal, step, stepTotal, raw, updatedAt)
        VALUES (?, ?, ?, ?, ?, ?, CURRENT_TIMESTAMP)
        ON CONFLICT(runId) DO UPDATE SET
            epoch=excluded.epoch,
            epochTotal=excluded.epochTotal,
            step=excluded.step,
            stepTotal=excluded.stepTotal,
            raw=excluded.raw,
            updatedAt=excluded.updatedAt;
        """,
        (
            run_id,
            args.get("epoch"),
            args.get("epoch_total"),
            args.get("step"),
            args.get("step_total"),
            args.get("raw", ""),
        ),
    )
    cur.execute("UPDATE Run SET lastStep=? WHERE runId=?;", (args.get("raw", ""), run_id))
    _conn().commit()


def _op_recover_inflight_runs(_: Dict[str, Any]) -> None:
    cur = _conn().cursor()
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
    _conn().commit()


def _op_set_run_downloads(args: Dict[str, Any]) -> None:
    run_id_db = args.get("run_id_db")
    dataset = args.get("dataset")
    lora = args.get("lora")
    cur = _conn().cursor()
    if dataset is not None:
        cur.execute("UPDATE Run SET datasetDownload=? WHERE id=?;", (dataset, run_id_db))
    if lora is not None:
        cur.execute("UPDATE Run SET loraDownload=? WHERE id=?;", (lora, run_id_db))
    _conn().commit()


def _op_get_autochar_presets(args: Dict[str, Any]) -> list[Dict[str, Any]]:
    names = args.get("names") or []
    cur = _conn().cursor()
    rows = []
    if names:
        for name in names:
            cur.execute("SELECT name, blockPatterns, allowPatterns FROM AutoCharPreset WHERE name=?", (name,))
            r = cur.fetchone()
            if r:
                rows.append(r)
    else:
        cur.execute("SELECT name, blockPatterns, allowPatterns FROM AutoCharPreset ORDER BY id ASC;")
        rows = cur.fetchall()
    return [{"name": r[0], "block": r[1], "allow": r[2]} for r in rows]


def _op_sql_query(args: Dict[str, Any]) -> list[Dict[str, Any]]:
    sql = args.get("sql")
    params = args.get("params") or []
    if not sql:
        raise ValueError("sql required")
    cur = _conn().cursor()
    cur.execute(sql, params)
    rows = cur.fetchall()
    return [dict(row) for row in rows]


def _op_sql_exec(args: Dict[str, Any]) -> Dict[str, Any]:
    sql = args.get("sql")
    params = args.get("params") or []
    if not sql:
        raise ValueError("sql required")
    cur = _conn().cursor()
    cur.execute(sql, params)
    _conn().commit()
    return {"changes": cur.rowcount, "lastRowId": cur.lastrowid}


READ_OPS = {
    "get_queue_mode": _op_get_queue_mode,
    "fetch_next_run": _op_fetch_next_run,
    "fetch_active_run": _op_fetch_active_run,
    "get_autochar_presets": _op_get_autochar_presets,
    "sql_query": _op_sql_query,
}

WRITE_OPS = {
    "ensure_tables": lambda _: _ensure_tables(_conn()),
    "set_queue_mode": _op_set_queue_mode,
    "update_worker_status": _op_update_worker_status,
    "claim_role_lock": _op_claim_role_lock,
    "ensure_run_plan": _op_ensure_run_plan,
    "set_plan_step": _op_set_plan_step,
    "mark_run_status": _op_mark_run_status,
    "update_run_step": _op_update_run_step,
    "mark_run_done": _op_mark_run_done,
    "upsert_train_progress": _op_upsert_train_progress,
    "recover_inflight_runs": _op_recover_inflight_runs,
    "set_run_downloads": _op_set_run_downloads,
    "sql_exec": _op_sql_exec,
}


class Handler(BaseHTTPRequestHandler):
    def _json(self, code: int, payload: Dict[str, Any]) -> None:
        raw = json.dumps(payload).encode("utf-8")
        self.send_response(code)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(raw)))
        self.end_headers()
        self.wfile.write(raw)

    def do_GET(self) -> None:
        if self.path == "/health":
            self._json(200, {"ok": True})
            return
        self._json(404, {"ok": False, "error": "not found"})

    def do_POST(self) -> None:
        length = int(self.headers.get("Content-Length", "0"))
        raw = self.rfile.read(length).decode("utf-8") if length else "{}"
        try:
            payload = json.loads(raw)
        except Exception:
            self._json(400, {"ok": False, "error": "invalid json"})
            return

        op = payload.get("op")
        args = payload.get("args") or {}
        if self.path == "/db/query":
            fn = READ_OPS.get(op)
            if not fn:
                self._json(400, {"ok": False, "error": f"unknown op {op}"})
                return
            with _LOCK:
                try:
                    data = fn(args)
                except Exception as exc:
                    self._json(500, {"ok": False, "error": str(exc)})
                    return
            self._json(200, {"ok": True, "data": data})
            return
        if self.path == "/db/exec":
            fn = WRITE_OPS.get(op)
            if not fn:
                self._json(400, {"ok": False, "error": f"unknown op {op}"})
                return
            with _LOCK:
                try:
                    data = fn(args)
                except Exception as exc:
                    self._json(500, {"ok": False, "error": str(exc)})
                    return
            self._json(200, {"ok": True, "data": data})
            return
        self._json(404, {"ok": False, "error": "not found"})


def main() -> None:
    _conn()
    server = HTTPServer((HOST, PORT), Handler)
    print(f"[db-broker] listening on http://{HOST}:{PORT} (db={DB_PATH})", flush=True)
    server.serve_forever()


if __name__ == "__main__":
    main()
