#!/usr/bin/env python3
import json
import os
import smtplib
import urllib.request
import hmac
import hashlib
import sqlite3
import threading
import time
from email.message import EmailMessage
from email.utils import formatdate
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
_NOTIFY_STALE_SECS = 45
_NOTIFY_QUEUE_STABLE_SECS = 45


def _as_bool(val: object, default: bool = False) -> bool:
    if isinstance(val, bool):
        return val
    if val is None:
        return default
    if isinstance(val, (int, float)):
        return bool(val)
    return str(val).strip().lower() in {"1", "true", "yes", "on"}


def _as_int(val: object, default: int) -> int:
    try:
        return int(val)  # type: ignore[arg-type]
    except (TypeError, ValueError):
        return default


def _load_settings_map(conn: sqlite3.Connection) -> Dict[str, str]:
    try:
        cur = conn.cursor()
        cur.execute("SELECT key, value FROM Setting;")
        rows = cur.fetchall()
        return {str(row[0]): str(row[1]) for row in rows or [] if row and row[0] is not None}
    except Exception:
        return {}


def _get_setting(conn: sqlite3.Connection, key: str) -> Optional[str]:
    try:
        cur = conn.cursor()
        cur.execute("SELECT value FROM Setting WHERE key=?;", (key,))
        row = cur.fetchone()
        return str(row[0]) if row and row[0] is not None else None
    except Exception:
        return None


def _set_setting(conn: sqlite3.Connection, key: str, value: str) -> None:
    try:
        cur = conn.cursor()
        cur.execute(
            "INSERT INTO Setting (key, value, updatedAt) VALUES (?, ?, CURRENT_TIMESTAMP) "
            "ON CONFLICT(key) DO UPDATE SET value=excluded.value, updatedAt=CURRENT_TIMESTAMP;",
            (key, value),
        )
        conn.commit()
    except Exception:
        return


def _delete_setting(conn: sqlite3.Connection, key: str) -> None:
    try:
        cur = conn.cursor()
        cur.execute("DELETE FROM Setting WHERE key=?;", (key,))
        conn.commit()
    except Exception:
        return


def _notifications_enabled(settings: Dict[str, str]) -> bool:
    return _as_bool(settings.get("notifications_enabled"), False)


def _email_channel_enabled(settings: Dict[str, str]) -> bool:
    return _as_bool(settings.get("notify_channel_email"), False)

def _discord_channel_enabled(settings: Dict[str, str]) -> bool:
    return _as_bool(settings.get("notify_channel_discord"), False)

def _slack_channel_enabled(settings: Dict[str, str]) -> bool:
    return _as_bool(settings.get("notify_channel_slack"), False)

def _webhook_channel_enabled(settings: Dict[str, str]) -> bool:
    return _as_bool(settings.get("notify_channel_webhook"), False)


def _email_ready(settings: Dict[str, str]) -> bool:
    host = settings.get("smtp_host", "").strip()
    smtp_from = settings.get("smtp_from", "").strip()
    smtp_to = settings.get("smtp_to", "").strip()
    port = _as_int(settings.get("smtp_port"), 0)
    return bool(host and smtp_from and smtp_to and port > 0)


def _discord_ready(settings: Dict[str, str]) -> bool:
    return bool(settings.get("discord_webhook_url", "").strip())

def _slack_ready(settings: Dict[str, str]) -> bool:
    return bool(settings.get("slack_webhook_url", "").strip())

def _webhook_ready(settings: Dict[str, str]) -> bool:
    return bool(settings.get("webhook_url", "").strip())


def _join_url(base: str, path: str) -> str:
    if not path:
        return ""
    if path.startswith("http://") or path.startswith("https://"):
        return path
    if not base:
        return path
    if path.startswith("/"):
        return f"{base.rstrip('/')}{path}"
    return f"{base.rstrip('/')}/{path}"


def _post_json(
    url: str,
    payload: Dict[str, object],
    timeout: int = 5,
    retries: int = 2,
    headers: Optional[Dict[str, str]] = None,
) -> None:
    data = json.dumps(payload).encode("utf-8")
    hdrs = {
        "Content-Type": "application/json",
        "User-Agent": "FrameForge-Notifier/1.0",
    }
    if headers:
        hdrs.update(headers)
    req = urllib.request.Request(
        url,
        data=data,
        headers=hdrs,
        method="POST",
    )
    last_exc: Optional[Exception] = None
    for _ in range(retries + 1):
        try:
            with urllib.request.urlopen(req, timeout=timeout):
                return
        except Exception as exc:
            last_exc = exc
    if last_exc:
        raise last_exc


def _send_discord(settings: Dict[str, str], payload: Dict[str, object]) -> None:
    url = settings.get("discord_webhook_url", "").strip()
    if not url:
        return
    _post_json(url, payload)


def _send_slack(settings: Dict[str, str], payload: Dict[str, object]) -> None:
    url = settings.get("slack_webhook_url", "").strip()
    if not url:
        return
    _post_json(url, payload)


def _sign_webhook(secret: str, body: bytes) -> str:
    return hmac.new(secret.encode("utf-8"), body, hashlib.sha256).hexdigest()


def _send_webhook(settings: Dict[str, str], payload: Dict[str, object]) -> None:
    url = settings.get("webhook_url", "").strip()
    if not url:
        return
    secret = str(settings.get("webhook_secret", "") or "").strip()
    headers = {}
    if secret:
        body = json.dumps(payload).encode("utf-8")
        headers["X-FrameForge-Signature"] = _sign_webhook(secret, body)
        headers["X-FrameForge-Signature-Alg"] = "sha256"
        _post_json(url, payload, headers=headers)
        return
    _post_json(url, payload)


def _event_enabled(settings: Dict[str, str], event_key: str) -> bool:
    return _as_bool(settings.get(event_key), False)


def _smtp_password(settings: Dict[str, str]) -> str:
    env_override = os.environ.get("FRAMEFORGE_SMTP_PASS") or os.environ.get("SMTP_PASS")
    return env_override if env_override is not None else str(settings.get("smtp_pass", ""))


def _reserve_notification(
    conn: sqlite3.Connection,
    *,
    run_id: str,
    notif_type: str,
    status: str,
    payload_hash: str,
) -> bool:
    cur = conn.cursor()
    cur.execute(
        """
        INSERT INTO NotificationLog (runId, type, status, payloadHash)
        VALUES (?, ?, ?, ?)
        ON CONFLICT(runId, type, status, payloadHash) DO NOTHING;
        """,
        (run_id, notif_type, status, payload_hash),
    )
    conn.commit()
    return cur.rowcount > 0


def _release_notification(
    conn: sqlite3.Connection,
    *,
    run_id: str,
    notif_type: str,
    status: str,
    payload_hash: str,
) -> None:
    cur = conn.cursor()
    cur.execute(
        "DELETE FROM NotificationLog WHERE runId=? AND type=? AND status=? AND payloadHash=?;",
        (run_id, notif_type, status, payload_hash),
    )
    conn.commit()


def _send_email(settings: Dict[str, str], subject: str, body: str, html: Optional[str] = None) -> None:
    host = settings.get("smtp_host", "").strip()
    if not host:
        return
    port = _as_int(settings.get("smtp_port"), 0)
    if port <= 0:
        return
    smtp_user = settings.get("smtp_user", "").strip()
    smtp_pass = _smtp_password(settings).strip()
    smtp_from = settings.get("smtp_from", "").strip()
    smtp_to = settings.get("smtp_to", "").strip()
    if not smtp_from or not smtp_to:
        return
    use_tls = _as_bool(settings.get("smtp_tls"), False)
    use_ssl = _as_bool(settings.get("smtp_ssl"), False)
    msg = EmailMessage()
    msg["Subject"] = subject
    msg["From"] = smtp_from
    msg["To"] = smtp_to
    msg["Date"] = formatdate(localtime=True)
    msg.set_content(body)
    if html:
        msg.add_alternative(html, subtype="html")
    timeout = 5
    if use_ssl:
        with smtplib.SMTP_SSL(host, port, timeout=timeout) as server:
            if smtp_user:
                server.login(smtp_user, smtp_pass)
            server.send_message(msg)
        return
    with smtplib.SMTP(host, port, timeout=timeout) as server:
        server.ehlo()
        if use_tls:
            server.starttls()
            server.ehlo()
        if smtp_user:
            server.login(smtp_user, smtp_pass)
        server.send_message(msg)


def _is_failed_status(status: str) -> bool:
    return status == "failed" or status.startswith("failed_")


def _format_ts(val: object) -> str:
    return str(val) if val else "n/a"


def _html_escape(text: str) -> str:
    return (
        text.replace("&", "&amp;")
        .replace("<", "&lt;")
        .replace(">", "&gt;")
        .replace('"', "&quot;")
        .replace("'", "&#39;")
    )


def _render_run_email_html(
    *,
    header: str,
    run_id: str,
    run_name: str,
    status: str,
    last_step: str,
    created_at: str,
    started_at: str,
    finished_at: str,
    dataset_url: str,
    lora_url: str,
    error: str,
    instance_label: str,
    instance_url: str,
) -> str:
    def _row(label: str, value: str) -> str:
        return f"<tr><td style=\"padding:6px 10px;color:#8fa2b8;\">{_html_escape(label)}</td><td style=\"padding:6px 10px;color:#f5f7fb;\">{_html_escape(value)}</td></tr>"

    downloads = ""
    if dataset_url or lora_url:
        rows = ""
        if dataset_url:
            rows += _row("Dataset", dataset_url)
        if lora_url:
            rows += _row("LoRA", lora_url)
        downloads = f\"\"\"\n        <div style=\"margin-top:16px;padding:12px;border-radius:10px;background:#0f1622;border:1px solid #1c2a3a;\">\n          <div style=\"font-weight:700;color:#8fd3ff;margin-bottom:6px;\">Downloads</div>\n          <table style=\"width:100%;border-collapse:collapse;\">{rows}</table>\n        </div>\n        \"\"\"\n+    error_block = ""
    if error:
        error_block = f\"\"\"\n        <div style=\"margin-top:16px;padding:12px;border-radius:10px;background:#221416;border:1px solid #3a1c20;color:#ffb3a8;\">\n          <div style=\"font-weight:700;margin-bottom:6px;\">Error</div>\n          <div style=\"white-space:pre-wrap;\">{_html_escape(error)}</div>\n        </div>\n        \"\"\"\n+    instance_line = f\"<a href=\\\"{_html_escape(instance_url)}\\\" style=\\\"color:#8fd3ff;text-decoration:none;\\\">{_html_escape(instance_url)}</a>\" if instance_url else ""

    return f\"\"\"\n+<html>\n+  <body style=\"margin:0;padding:0;background:#0b0f14;font-family:Arial,sans-serif;color:#f5f7fb;\">\n+    <div style=\"max-width:640px;margin:0 auto;padding:24px;\">\n+      <div style=\"padding:18px 20px;border-radius:16px;background:linear-gradient(180deg,#141a22,#0f141b);border:1px solid #1d2a38;\">\n+        <div style=\"font-size:18px;font-weight:700;letter-spacing:0.4px;\">FrameForge</div>\n+        <div style=\"margin-top:6px;font-size:15px;color:#8fd3ff;\">{_html_escape(header)}</div>\n+      </div>\n+      <div style=\"margin-top:16px;padding:18px;border-radius:16px;background:#111720;border:1px solid #1c2a3a;\">\n+        <table style=\"width:100%;border-collapse:collapse;\">\n+          {_row(\"Run\", run_id)}\n+          {_row(\"Name\", run_name)}\n+          {_row(\"Status\", status)}\n+          {_row(\"Last step\", last_step)}\n+          {_row(\"Created\", created_at)}\n+          {_row(\"Started\", started_at)}\n+          {_row(\"Finished\", finished_at)}\n+        </table>\n+        {downloads}\n+        {error_block}\n+      </div>\n+      <div style=\"margin-top:16px;color:#8fa2b8;font-size:12px;\">\n+        <div>{_html_escape(instance_label)}</div>\n+        <div>{instance_line}</div>\n+      </div>\n+    </div>\n+  </body>\n+</html>\n+\"\"\"


def _render_queue_email_html(
    *,
    queue_mode: str,
    instance_label: str,
    instance_url: str,
) -> str:
    instance_line = f\"<a href=\\\"{_html_escape(instance_url)}\\\" style=\\\"color:#8fd3ff;text-decoration:none;\\\">{_html_escape(instance_url)}</a>\" if instance_url else ""
    return f\"\"\"\n+<html>\n+  <body style=\"margin:0;padding:0;background:#0b0f14;font-family:Arial,sans-serif;color:#f5f7fb;\">\n+    <div style=\"max-width:640px;margin:0 auto;padding:24px;\">\n+      <div style=\"padding:18px 20px;border-radius:16px;background:linear-gradient(180deg,#141a22,#0f141b);border:1px solid #1d2a38;\">\n+        <div style=\"font-size:18px;font-weight:700;letter-spacing:0.4px;\">FrameForge</div>\n+        <div style=\"margin-top:6px;font-size:15px;color:#8fd3ff;\">Queue drained</div>\n+      </div>\n+      <div style=\"margin-top:16px;padding:18px;border-radius:16px;background:#111720;border:1px solid #1c2a3a;\">\n+        <div style=\"margin-bottom:10px;color:#c8d2df;\">All workers are idle and no runs are active.</div>\n+        <table style=\"width:100%;border-collapse:collapse;\">\n+          <tr><td style=\"padding:6px 10px;color:#8fa2b8;\">Queue mode</td><td style=\"padding:6px 10px;color:#f5f7fb;\">{_html_escape(queue_mode)}</td></tr>\n+        </table>\n+      </div>\n+      <div style=\"margin-top:16px;color:#8fa2b8;font-size:12px;\">\n+        <div>{_html_escape(instance_label)}</div>\n+        <div>{instance_line}</div>\n+      </div>\n+    </div>\n+  </body>\n+</html>\n+\"\"\"


def _log_notification_error(
    conn: sqlite3.Connection,
    run_id: Optional[str],
    message: str,
    detail: str,
) -> None:
    try:
        cur = conn.cursor()
        cur.execute(
            """
            INSERT INTO ErrorLog (
                runId, component, stage, step, errorType, errorCode, errorMessage, errorDetail, createdAt
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, CURRENT_TIMESTAMP);
            """,
            (
                run_id,
                "notifications",
                "notify",
                "notify",
                "notification_error",
                "notify_failed",
                message,
                detail,
            ),
        )
        conn.commit()
    except Exception:
        return


def _fetch_run_for_notification(conn: sqlite3.Connection, run_id_db: int) -> Optional[Dict[str, object]]:
    cur = conn.cursor()
    cur.execute(
        """
        SELECT runId, runName, status, lastStep, error, createdAt, startedAt, finishedAt,
               datasetDownload, loraDownload
        FROM Run WHERE id=?;
        """,
        (run_id_db,),
    )
    row = cur.fetchone()
    if not row:
        return None
    return {
        "runId": row[0],
        "runName": row[1],
        "status": row[2],
        "lastStep": row[3],
        "error": row[4],
        "createdAt": row[5],
        "startedAt": row[6],
        "finishedAt": row[7],
        "datasetDownload": row[8],
        "loraDownload": row[9],
    }


def _queue_finish_token(conn: sqlite3.Connection) -> str:
    cur = conn.cursor()
    cur.execute("SELECT MAX(finishedAt) FROM Run WHERE finishedAt IS NOT NULL;")
    row = cur.fetchone()
    return str(row[0]) if row and row[0] is not None else "empty"


def _workers_idle(conn: sqlite3.Connection) -> bool:
    cur = conn.cursor()
    cur.execute(
        "SELECT role, state, heartbeat FROM WorkerStatus WHERE role IN ('initiator', 'orchestrator', 'finisher');"
    )
    rows = cur.fetchall() or []
    roles = {row[0]: {"state": row[1], "heartbeat": row[2]} for row in rows if row and row[0]}
    for role in ("initiator", "orchestrator", "finisher"):
        info = roles.get(role)
        if not info:
            return False
        state = str(info.get("state") or "").lower()
        if state not in {"idle", "ok"}:
            return False
        heartbeat = info.get("heartbeat")
        if heartbeat is None:
            return False
        try:
            age = time.time() - float(heartbeat)
        except (TypeError, ValueError):
            return False
        if age > _NOTIFY_STALE_SECS:
            return False
    return True


def _active_run_count(conn: sqlite3.Connection) -> int:
    active_statuses = (
        "queued",
        "queued_initiated",
        "running",
        "manual_tagging",
        "ready_to_train",
        "ready_for_finish",
    )
    placeholders = ",".join("?" for _ in active_statuses)
    cur = conn.cursor()
    cur.execute(f"SELECT COUNT(*) FROM Run WHERE status IN ({placeholders});", active_statuses)
    row = cur.fetchone()
    return int(row[0] or 0) if row else 0


def _maybe_notify_queue_finish(conn: sqlite3.Connection, settings: Dict[str, str]) -> None:
    if not _notifications_enabled(settings):
        return
    email_ready = _email_channel_enabled(settings) and _email_ready(settings)
    discord_ready = _discord_channel_enabled(settings) and _discord_ready(settings)
    slack_ready = _slack_channel_enabled(settings) and _slack_ready(settings)
    webhook_ready = _webhook_channel_enabled(settings) and _webhook_ready(settings)
    if not (email_ready or discord_ready or slack_ready or webhook_ready):
        return
    if not _event_enabled(settings, "notify_queue_finish"):
        return
    if _active_run_count(conn) > 0:
        _delete_setting(conn, "queue_finish_candidate_since")
        return
    if not _workers_idle(conn):
        _delete_setting(conn, "queue_finish_candidate_since")
        return
    now = time.time()
    candidate = _get_setting(conn, "queue_finish_candidate_since")
    if not candidate:
        _set_setting(conn, "queue_finish_candidate_since", str(int(now)))
        return
    try:
        since = float(candidate)
    except (TypeError, ValueError):
        _set_setting(conn, "queue_finish_candidate_since", str(int(now)))
        return
    if now - since < _NOTIFY_QUEUE_STABLE_SECS:
        return
    payload_hash = _queue_finish_token(conn)
    run_id = "queue"
    notif_type = "queue_finish"
    status = "queue_drain"
    if not _reserve_notification(conn, run_id=run_id, notif_type=notif_type, status=status, payload_hash=payload_hash):
        return
    queue_mode = settings.get("queue_mode") or "running"
    subject = "FrameForge: Queue is empty"
    instance_label = settings.get("instance_label", "")
    instance_url = settings.get("instance_url", "")
    body_lines = [
        "Hello,",
        "",
        "The queue is fully drained and all workers are idle.",
        "",
        "Queue",
        "- Pending runs: 0",
        f"- Queue mode: {queue_mode}",
        "",
        "Workers",
        "- Initiator: idle/ok",
        "- Orchestrator: idle/ok",
        "- Finisher: idle/ok",
        "",
        "Instance",
        f"- {instance_label}",
    ]
    if instance_url:
        body_lines.append(f"- {instance_url}")
    body_lines.extend(
        [
            "",
            "Thanks,",
            "FrameForge",
        ]
    )
    body = "\n".join(body_lines)
    success = False
    if email_ready:
    try:
        html = _render_queue_email_html(
            queue_mode=queue_mode,
            instance_label=instance_label,
            instance_url=instance_url,
        )
        _send_email(settings, subject, body, html)
        success = True
    except Exception as exc:
        _log_notification_error(conn, None, "queue finish email failed", str(exc))
    if discord_ready:
        try:
            fields = [
                {"name": "Queue mode", "value": queue_mode, "inline": True},
            ]
            if instance_label:
                fields.append({"name": "Instance", "value": instance_label, "inline": True})
            if instance_url:
                fields.append({"name": "Open", "value": instance_url, "inline": False})
            embed = {
                "title": "FrameForge • Queue drained",
                "description": "All workers are idle and no runs are active.",
                "color": 0x2DB4FF,
                "fields": fields,
            }
            _send_discord(settings, {"embeds": [embed]})
            success = True
        except Exception as exc:
            _log_notification_error(conn, None, "queue finish discord failed", str(exc))
    if slack_ready:
        try:
            text = "\n".join(
                [
                    "FrameForge: Queue drained",
                    "All workers are idle and no runs are active.",
                    f"Queue mode: {queue_mode}",
                    f"Instance: {instance_label}" if instance_label else "",
                    instance_url or "",
                ]
            ).strip()
            fields = [
                {"type": "mrkdwn", "text": f"*Queue mode:* {queue_mode}"},
            ]
            if instance_label:
                fields.append({"type": "mrkdwn", "text": f"*Instance:* {instance_label}"})
            blocks = [
                {"type": "header", "text": {"type": "plain_text", "text": "FrameForge • Queue drained"}},
                {"type": "section", "text": {"type": "mrkdwn", "text": "All workers are idle and no runs are active."}},
                {"type": "section", "fields": fields},
            ]
            if instance_url:
                blocks.append({"type": "section", "text": {"type": "mrkdwn", "text": f"*Open:* {instance_url}"}})
            payload = {
                "attachments": [
                    {
                        "color": "#2DB4FF",
                        "fallback": "FrameForge • Queue drained",
                        "blocks": blocks,
                    }
                ],
            }
            _send_slack(settings, payload)
            success = True
        except Exception as exc:
            _log_notification_error(conn, None, "queue finish slack failed", str(exc))
    if webhook_ready:
        try:
            payload = {
                "type": "queue_finish",
                "queue_count": 0,
                "queue_mode": queue_mode,
                "instance_label": instance_label,
                "instance_url": instance_url,
            }
            _send_webhook(settings, payload)
            success = True
        except Exception as exc:
            _log_notification_error(conn, None, "queue finish webhook failed", str(exc))
    if success:
        _delete_setting(conn, "queue_finish_candidate_since")
    else:
        _release_notification(conn, run_id=run_id, notif_type=notif_type, status=status, payload_hash=payload_hash)


def _maybe_notify_run_status(conn: sqlite3.Connection, run_id_db: int, status: str) -> None:
    settings = _load_settings_map(conn)
    if not _notifications_enabled(settings):
        return
    email_ready = _email_channel_enabled(settings) and _email_ready(settings)
    discord_ready = _discord_channel_enabled(settings) and _discord_ready(settings)
    slack_ready = _slack_channel_enabled(settings) and _slack_ready(settings)
    webhook_ready = _webhook_channel_enabled(settings) and _webhook_ready(settings)
    if not (email_ready or discord_ready or slack_ready or webhook_ready):
        return
    notif_type = None
    event_key = None
    if status == "done":
        notif_type = "job_finish"
        event_key = "notify_job_finish"
    elif _is_failed_status(status):
        notif_type = "job_failed"
        event_key = "notify_job_failed"
    if not notif_type or not event_key or not _event_enabled(settings, event_key):
        return
    run = _fetch_run_for_notification(conn, run_id_db)
    if not run:
        return
    run_id = str(run.get("runId") or run_id_db)
    payload_hash = f"{notif_type}:{run_id}:{status}"
    if not _reserve_notification(conn, run_id=run_id, notif_type=notif_type, status=status, payload_hash=payload_hash):
        return
    subject = f"FrameForge: Run {run_id} {'finished' if notif_type == 'job_finish' else 'failed'}"
    instance_label = settings.get("instance_label", "")
    instance_url = settings.get("instance_url", "")
    body_lines = [
        "Hello,",
        "",
        "Your FrameForge run has completed successfully."
        if notif_type == "job_finish"
        else "Your FrameForge run needs attention. It did not complete successfully.",
        "",
        "Run",
        f"- ID: {run_id}",
        f"- Name: {run.get('runName') or ''}",
        f"- Status: {run.get('status') or status}",
        f"- Last step: {run.get('lastStep') or ''}",
        f"- Created: {_format_ts(run.get('createdAt'))}",
        f"- Started: {_format_ts(run.get('startedAt'))}",
        f"- Finished: {_format_ts(run.get('finishedAt'))}",
    ]
    if notif_type == "job_finish":
        body_lines.extend(
            [
                "",
                "Downloads",
                f"- Dataset: {run.get('datasetDownload') or ''}",
                f"- LoRA: {run.get('loraDownload') or ''}",
            ]
        )
    else:
        body_lines.extend(
            [
                "",
                "Error",
                f"- {run.get('error') or ''}",
            ]
        )
    body_lines.extend(
        [
            "",
            "Instance",
            f"- {instance_label}",
        ]
    )
    if instance_url:
        body_lines.append(f"- {instance_url}")
    body_lines.extend(
        [
            "",
            "Thanks,",
            "FrameForge",
        ]
    )
    body = "\n".join(body_lines)
    success = False
    if email_ready:
    try:
        html = _render_run_email_html(
            header="Run finished" if notif_type == "job_finish" else "Run failed",
            run_id=run_id,
            run_name=str(run.get("runName") or ""),
            status=str(run.get("status") or status),
            last_step=str(run.get("lastStep") or ""),
            created_at=_format_ts(run.get("createdAt")),
            started_at=_format_ts(run.get("startedAt")),
            finished_at=_format_ts(run.get("finishedAt")),
            dataset_url=str(run.get("datasetDownload") or ""),
            lora_url=str(run.get("loraDownload") or ""),
            error=str(run.get("error") or ""),
            instance_label=instance_label,
            instance_url=instance_url,
        )
        _send_email(settings, subject, body, html)
        success = True
    except Exception as exc:
        _log_notification_error(conn, run_id, "run email failed", str(exc))
    if discord_ready:
        try:
            base_url = str(settings.get("instance_url", "") or "").strip()
            dataset_url = _join_url(base_url, str(run.get("datasetDownload") or ""))
            lora_url = _join_url(base_url, str(run.get("loraDownload") or ""))
            header = "Run finished" if notif_type == "job_finish" else "Run failed"
            status_label = str(run.get("status") or status)
            color = 0x4FE18A if notif_type == "job_finish" else 0xFF6B57
            fields = [
                {"name": "Run", "value": run_id, "inline": True},
                {"name": "Status", "value": status_label, "inline": True},
                {"name": "Name", "value": str(run.get("runName") or ""), "inline": False},
            ]
            if notif_type == "job_finish":
                if dataset_url:
                    fields.append({"name": "Dataset", "value": dataset_url, "inline": False})
                if lora_url:
                    fields.append({"name": "LoRA", "value": lora_url, "inline": False})
            else:
                fields.append({"name": "Error", "value": str(run.get("error") or ""), "inline": False})
            if instance_label:
                fields.append({"name": "Instance", "value": instance_label, "inline": True})
            if instance_url:
                fields.append({"name": "Open", "value": instance_url, "inline": False})
            embed = {
                "title": f"FrameForge • {header}",
                "color": color,
                "fields": fields,
            }
            _send_discord(settings, {"embeds": [embed]})
            success = True
        except Exception as exc:
            _log_notification_error(conn, run_id, "run discord failed", str(exc))
    if slack_ready:
        try:
            base_url = str(settings.get("instance_url", "") or "").strip()
            dataset_url = _join_url(base_url, str(run.get("datasetDownload") or ""))
            lora_url = _join_url(base_url, str(run.get("loraDownload") or ""))
            status_label = str(run.get("status") or status)
            header = f"Run {'finished' if notif_type == 'job_finish' else 'failed'}"
            text_lines = [
                f"FrameForge: {header}",
                f"Run: {run_id}",
                f"Name: {run.get('runName') or ''}",
                f"Status: {status_label}",
            ]
            if notif_type == "job_finish":
                if dataset_url:
                    text_lines.append(f"Dataset: {dataset_url}")
                if lora_url:
                    text_lines.append(f"LoRA: {lora_url}")
            else:
                text_lines.append(f"Error: {run.get('error') or ''}")
            if instance_label:
                text_lines.append(f"Instance: {instance_label}")
            if instance_url:
                text_lines.append(instance_url)
            fields = [
                {"type": "mrkdwn", "text": f"*Run:* {run_id}"},
                {"type": "mrkdwn", "text": f"*Status:* {status_label}"},
                {"type": "mrkdwn", "text": f"*Name:* {run.get('runName') or ''}"},
            ]
            blocks = [
                {"type": "header", "text": {"type": "plain_text", "text": f"FrameForge • {header}"}},
                {"type": "section", "fields": fields},
            ]
            if notif_type == "job_finish":
                if dataset_url:
                    blocks.append({"type": "section", "text": {"type": "mrkdwn", "text": f"*Dataset:* {dataset_url}"}})
                if lora_url:
                    blocks.append({"type": "section", "text": {"type": "mrkdwn", "text": f"*LoRA:* {lora_url}"}})
            else:
                blocks.append(
                    {"type": "section", "text": {"type": "mrkdwn", "text": f"*Error:* {run.get('error') or ''}"}}
                )
            if instance_label:
                blocks.append({"type": "context", "elements": [{"type": "mrkdwn", "text": f"{instance_label}"}]})
            if instance_url:
                blocks.append({"type": "section", "text": {"type": "mrkdwn", "text": f"*Open:* {instance_url}"}})
            payload = {
                "attachments": [
                    {
                        "color": "#4FE18A" if notif_type == "job_finish" else "#FF6B57",
                        "fallback": f"FrameForge • {header}",
                        "blocks": blocks,
                    }
                ],
            }
            _send_slack(settings, payload)
            success = True
        except Exception as exc:
            _log_notification_error(conn, run_id, "run slack failed", str(exc))
    if webhook_ready:
        try:
            base_url = str(settings.get("instance_url", "") or "").strip()
            dataset_url = _join_url(base_url, str(run.get("datasetDownload") or ""))
            lora_url = _join_url(base_url, str(run.get("loraDownload") or ""))
            payload = {
                "type": notif_type,
                "run_id": run_id,
                "run_name": run.get("runName") or "",
                "status": run.get("status") or status,
                "last_step": run.get("lastStep") or "",
                "error": run.get("error") or "",
                "created_at": _format_ts(run.get("createdAt")),
                "started_at": _format_ts(run.get("startedAt")),
                "finished_at": _format_ts(run.get("finishedAt")),
                "dataset_url": dataset_url,
                "lora_url": lora_url,
                "instance_label": instance_label,
                "instance_url": instance_url,
            }
            _send_webhook(settings, payload)
            success = True
        except Exception as exc:
            _log_notification_error(conn, run_id, "run webhook failed", str(exc))
    if success:
        _maybe_notify_queue_finish(conn, settings)
    else:
        _release_notification(conn, run_id=run_id, notif_type=notif_type, status=status, payload_hash=payload_hash)


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
        CREATE TABLE IF NOT EXISTS QueueItem (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            runId TEXT UNIQUE NOT NULL,
            position INTEGER NOT NULL,
            createdAt DATETIME DEFAULT CURRENT_TIMESTAMP
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
    cur.execute(
        """
        CREATE TABLE IF NOT EXISTS NotificationLog (
            id INTEGER PRIMARY KEY,
            runId TEXT NOT NULL,
            type TEXT NOT NULL,
            status TEXT NOT NULL,
            payloadHash TEXT NOT NULL,
            sentAt DATETIME DEFAULT CURRENT_TIMESTAMP,
            UNIQUE(runId, type, status, payloadHash)
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
    if status == "queued":
        cur.execute(
            """
            SELECT r.id, r.runId, r.runName, r.name, r.type, r.flags, r.uploadPath, r.trainProfile
            FROM Run r
            JOIN QueueItem q ON q.runId = r.runId
            WHERE r.status=?
            ORDER BY q.position ASC, r.createdAt ASC
            LIMIT 1;
            """,
            (status,),
        )
    else:
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
    prev_status = None
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
    if run_id_db:
        try:
            cur.execute("SELECT status FROM Run WHERE id=?;", (run_id_db,))
            row = cur.fetchone()
            prev_status = row[0] if row else None
        except Exception:
            prev_status = None
    cur.execute(sql, params)
    if status != "queued":
        cur.execute("SELECT position FROM QueueItem WHERE runId=(SELECT runId FROM Run WHERE id=?)", (run_id_db,))
        row = cur.fetchone()
        if row:
            old_pos = row[0]
            cur.execute("DELETE FROM QueueItem WHERE runId=(SELECT runId FROM Run WHERE id=?)", (run_id_db,))
            cur.execute("UPDATE QueueItem SET position=position-1 WHERE position>?", (old_pos,))
    elif status == "queued":
        cur.execute("SELECT runId FROM Run WHERE id=?", (run_id_db,))
        row = cur.fetchone()
        run_id = row[0] if row else None
        if run_id:
            cur.execute("SELECT 1 FROM QueueItem WHERE runId=?", (run_id,))
            if not cur.fetchone():
                cur.execute("SELECT COALESCE(MAX(position), 0) FROM QueueItem")
                max_pos = cur.fetchone()[0] or 0
                cur.execute(
                    "INSERT INTO QueueItem (runId, position) VALUES (?, ?)",
                    (run_id, int(max_pos) + 1),
                )
    _conn().commit()
    if run_id_db and status and status != prev_status:
        _maybe_notify_run_status(_conn(), int(run_id_db), str(status))


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


def _op_queue_enqueue(args: Dict[str, Any]) -> None:
    run_id = args.get("run_id")
    if not run_id:
        return
    cur = _conn().cursor()
    cur.execute("SELECT 1 FROM QueueItem WHERE runId=?", (run_id,))
    if cur.fetchone():
        return
    cur.execute("SELECT COALESCE(MAX(position), 0) FROM QueueItem")
    max_pos = cur.fetchone()[0] or 0
    cur.execute(
        "INSERT INTO QueueItem (runId, position) VALUES (?, ?)",
        (run_id, int(max_pos) + 1),
    )
    _conn().commit()


def _op_queue_remove(args: Dict[str, Any]) -> None:
    run_id = args.get("run_id")
    if not run_id:
        return
    cur = _conn().cursor()
    cur.execute("SELECT position FROM QueueItem WHERE runId=?", (run_id,))
    row = cur.fetchone()
    if not row:
        return
    old_pos = row[0]
    cur.execute("DELETE FROM QueueItem WHERE runId=?", (run_id,))
    cur.execute("UPDATE QueueItem SET position=position-1 WHERE position>?", (old_pos,))
    _conn().commit()


def _op_queue_reorder(args: Dict[str, Any]) -> None:
    run_id = args.get("run_id")
    position = args.get("position")
    if not run_id or position is None:
        return
    try:
        new_pos = int(position)
    except Exception:
        return
    cur = _conn().cursor()
    cur.execute("BEGIN IMMEDIATE;")
    cur.execute("SELECT position FROM QueueItem WHERE runId=?", (run_id,))
    row = cur.fetchone()
    if not row:
        _conn().rollback()
        return
    old_pos = int(row[0])
    cur.execute("SELECT COUNT(*) FROM QueueItem")
    count = int(cur.fetchone()[0] or 0)
    if count <= 0:
        _conn().rollback()
        return
    if new_pos < 1:
        new_pos = 1
    if new_pos > count:
        new_pos = count
    if new_pos == old_pos:
        _conn().commit()
        return
    if new_pos < old_pos:
        cur.execute(
            "UPDATE QueueItem SET position=position+1 WHERE position>=? AND position<?",
            (new_pos, old_pos),
        )
    else:
        cur.execute(
            "UPDATE QueueItem SET position=position-1 WHERE position<=? AND position>?",
            (new_pos, old_pos),
        )
    cur.execute("UPDATE QueueItem SET position=? WHERE runId=?", (new_pos, run_id))
    _conn().commit()


def _op_queue_backfill(_: Dict[str, Any]) -> None:
    cur = _conn().cursor()
    cur.execute(
        """
        SELECT r.runId
        FROM Run r
        LEFT JOIN QueueItem q ON q.runId = r.runId
        WHERE r.status='queued' AND q.runId IS NULL
        ORDER BY r.createdAt ASC;
        """
    )
    rows = cur.fetchall()
    if not rows:
        return
    cur.execute("SELECT COALESCE(MAX(position), 0) FROM QueueItem")
    max_pos = cur.fetchone()[0] or 0
    for row in rows:
        max_pos += 1
        cur.execute("INSERT INTO QueueItem (runId, position) VALUES (?, ?)", (row[0], max_pos))
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
    "queue_enqueue": _op_queue_enqueue,
    "queue_remove": _op_queue_remove,
    "queue_reorder": _op_queue_reorder,
    "queue_backfill": _op_queue_backfill,
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
