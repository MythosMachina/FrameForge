import json
import os
import urllib.request
from typing import Any, Dict, Optional


def _broker_url() -> str:
    return os.environ.get("DB_BROKER_URL", "").rstrip("/")


def broker_enabled() -> bool:
    return bool(_broker_url())


def _post(path: str, payload: Dict[str, Any]) -> Dict[str, Any]:
    url = f"{_broker_url()}{path}"
    data = json.dumps(payload).encode("utf-8")
    req = urllib.request.Request(
        url,
        data=data,
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    with urllib.request.urlopen(req, timeout=10) as resp:
        raw = resp.read().decode("utf-8")
    return json.loads(raw)


def broker_exec(op: str, args: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
    if not broker_enabled():
        raise RuntimeError("DB broker not configured")
    payload = {"op": op, "args": args or {}}
    return _post("/db/exec", payload)


def broker_query(op: str, args: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
    if not broker_enabled():
        raise RuntimeError("DB broker not configured")
    payload = {"op": op, "args": args or {}}
    return _post("/db/query", payload)
