#!/usr/bin/env python3
"""
Seed AutoChar presets into the webapp SQLite DB.

Usage:
  DATABASE_URL=file:./storage/db.sqlite python scripts/seed_autochar.py
"""
import json
import os
import sqlite3
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
DB_PATH = os.environ.get("DATABASE_URL", "file:./webapp/storage/db.sqlite").replace("file:", "")
DB_FILE = ROOT / DB_PATH


def main() -> None:
    DB_FILE.parent.mkdir(parents=True, exist_ok=True)

    presets = {
        "default": {"desc": "Baseline cleanup", "block": [], "allow": []},
        "human": {"desc": "Human-specific cleanup", "block": [], "allow": []},
        "furry": {"desc": "Furry cleanup", "block": [r".*fur.*", r".*paw.*", r".*claw.*", r".*snout.*", r".*tail.*"], "allow": []},
        "dragon": {"desc": "Dragon cleanup", "block": [r".*scale.*", r".*claw.*", r".*horn.*", r".*wing.*", r".*tail.*"], "allow": []},
        "daemon": {"desc": "Daemon cleanup", "block": [r".*horn.*", r".*tail.*"], "allow": []},
    }

    conn = sqlite3.connect(DB_FILE)
    cur = conn.cursor()
    cur.execute(
        """
        CREATE TABLE IF NOT EXISTS "AutoCharPreset" (
            "id" INTEGER NOT NULL PRIMARY KEY AUTOINCREMENT,
            "name" TEXT NOT NULL UNIQUE,
            "description" TEXT NOT NULL DEFAULT '',
            "blockPatterns" TEXT NOT NULL,
            "allowPatterns" TEXT NOT NULL,
            "createdAt" DATETIME NOT NULL DEFAULT CURRENT_TIMESTAMP,
            "updatedAt" DATETIME NOT NULL DEFAULT CURRENT_TIMESTAMP
        );
        """
    )
    for name, cfg in presets.items():
        cur.execute(
            """
            INSERT INTO AutoCharPreset (name, description, blockPatterns, allowPatterns, createdAt, updatedAt)
            VALUES (?, ?, ?, ?, CURRENT_TIMESTAMP, CURRENT_TIMESTAMP)
            ON CONFLICT(name) DO UPDATE SET
                description=excluded.description,
                blockPatterns=excluded.blockPatterns,
                allowPatterns=excluded.allowPatterns,
                updatedAt=CURRENT_TIMESTAMP
            """,
            (
                name,
                cfg["desc"],
                json.dumps(cfg["block"], ensure_ascii=False),
                json.dumps(cfg["allow"], ensure_ascii=False),
            ),
        )
    conn.commit()
    conn.close()
    print(f"Seeded {len(presets)} AutoChar presets into {DB_FILE}")


if __name__ == "__main__":
    main()
