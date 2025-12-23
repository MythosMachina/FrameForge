#!/usr/bin/env python3
"""Seed basic data after DB initialization (no AutoChar presets)."""

import json
import sqlite3
from pathlib import Path

BUNDLE_ROOT = Path(__file__).resolve().parents[1]
DB_PATH = BUNDLE_ROOT / "_system" / "db" / "db.sqlite"
TAGGER_MODELS_DIR = BUNDLE_ROOT / "_system" / "webapp" / "tagger_models"

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
    "hf_token": "",
    # Trainer defaults (SDXL/Pony)
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
    "queue_mode": "running",
}

DEFAULT_PROFILES = [
    {
        "name": "balanced",
        "label": "Balanced",
        "isDefault": 1,
        "settings": {
            "trainer_batch_size": 2,
            "trainer_bucket_max_reso": 640,
            "trainer_bucket_min_reso": 64,
            "trainer_bucket_step": 64,
            "trainer_dataloader_workers": 4,
            "trainer_dataset_repeats": 1,
            "trainer_grad_accum": 2,
            "trainer_gradient_checkpointing": True,
            "trainer_lora_alpha": 24,
            "trainer_lora_rank": 24,
            "trainer_max_train_steps": 900,
            "trainer_resolution": 576,
        },
    },
    {
        "name": "fast",
        "label": "Fast",
        "isDefault": 0,
        "settings": {
            "trainer_batch_size": 2,
            "trainer_bucket_max_reso": 576,
            "trainer_bucket_min_reso": 64,
            "trainer_bucket_step": 64,
            "trainer_dataloader_workers": 4,
            "trainer_dataset_repeats": 1,
            "trainer_grad_accum": 2,
            "trainer_gradient_checkpointing": True,
            "trainer_lora_alpha": 16,
            "trainer_lora_rank": 16,
            "trainer_max_train_steps": 900,
            "trainer_resolution": 512,
        },
    },
    {
        "name": "oneshot-balanced",
        "label": "One-shot Balanced",
        "isDefault": 0,
        "settings": {
            "trainer_batch_size": 4,
            "trainer_bucket_max_reso": 640,
            "trainer_bucket_min_reso": 64,
            "trainer_bucket_step": 64,
            "trainer_dataloader_workers": 4,
            "trainer_dataset_repeats": 40,
            "trainer_grad_accum": 1,
            "trainer_gradient_checkpointing": True,
            "trainer_lora_alpha": 24,
            "trainer_lora_rank": 24,
            "trainer_max_train_steps": 900,
            "trainer_resolution": 576,
        },
    },
    {
        "name": "oneshot-fast",
        "label": "One-shot Fast",
        "isDefault": 0,
        "settings": {
            "trainer_batch_size": 4,
            "trainer_bucket_max_reso": 576,
            "trainer_bucket_min_reso": 64,
            "trainer_bucket_step": 64,
            "trainer_dataloader_workers": 4,
            "trainer_dataset_repeats": 40,
            "trainer_grad_accum": 1,
            "trainer_gradient_checkpointing": True,
            "trainer_lora_alpha": 16,
            "trainer_lora_rank": 16,
            "trainer_max_train_steps": 900,
            "trainer_resolution": 512,
        },
    },
    {
        "name": "balanced-5090",
        "label": "Balanced (5090)",
        "isDefault": 0,
        "settings": {
            "trainer_batch_size": 3,
            "trainer_bucket_max_reso": 704,
            "trainer_bucket_min_reso": 64,
            "trainer_bucket_step": 64,
            "trainer_clip_skip": 2,
            "trainer_dataloader_workers": 6,
            "trainer_dataset_repeats": 1,
            "trainer_epochs": 10,
            "trainer_grad_accum": 1,
            "trainer_gradient_checkpointing": False,
            "trainer_learning_rate": 0.0001,
            "trainer_lora_alpha": 24,
            "trainer_lora_rank": 24,
            "trainer_lr_scheduler": "cosine",
            "trainer_lr_warmup_steps": 180,
            "trainer_max_grad_norm": 0,
            "trainer_max_train_steps": 900,
            "trainer_min_snr_gamma": 5.0,
            "trainer_optimizer": "adamw",
            "trainer_resolution": 640,
            "trainer_te_learning_rate": 5e-05,
            "trainer_use_8bit_adam": True,
            "trainer_weight_decay": 0.01,
        },
    },
    {
        "name": "fast-5090",
        "label": "Fast (5090)",
        "isDefault": 0,
        "settings": {
            "trainer_batch_size": 4,
            "trainer_bucket_max_reso": 640,
            "trainer_bucket_min_reso": 64,
            "trainer_bucket_step": 64,
            "trainer_clip_skip": 2,
            "trainer_dataloader_workers": 6,
            "trainer_dataset_repeats": 1,
            "trainer_epochs": 10,
            "trainer_grad_accum": 1,
            "trainer_gradient_checkpointing": False,
            "trainer_learning_rate": 0.0001,
            "trainer_lora_alpha": 16,
            "trainer_lora_rank": 16,
            "trainer_lr_scheduler": "cosine",
            "trainer_lr_warmup_steps": 180,
            "trainer_max_grad_norm": 0,
            "trainer_max_train_steps": 900,
            "trainer_min_snr_gamma": 5.0,
            "trainer_optimizer": "adamw",
            "trainer_resolution": 576,
            "trainer_te_learning_rate": 5e-05,
            "trainer_use_8bit_adam": True,
            "trainer_weight_decay": 0.01,
        },
    },
]

TAGGER_MODELS = [
    "SmilingWolf/wd-eva02-large-tagger-v3",
    "SmilingWolf/wd-swinv2-tagger-v3",
    "SmilingWolf/wd-convnext-tagger-v3",
]


def ensure_tables(conn: sqlite3.Connection) -> None:
    cur = conn.cursor()
    cur.execute(
        """
        CREATE TABLE IF NOT EXISTS Setting (
            key TEXT PRIMARY KEY,
            value TEXT NOT NULL,
            createdAt DATETIME DEFAULT CURRENT_TIMESTAMP,
            updatedAt DATETIME DEFAULT CURRENT_TIMESTAMP
        );
        """
    )
    cur.execute(
        """
        CREATE TABLE IF NOT EXISTS TaggerModel (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            repoId TEXT UNIQUE NOT NULL,
            name TEXT NOT NULL,
            folder TEXT NOT NULL,
            size INTEGER DEFAULT 0,
            status TEXT DEFAULT "ready",
            createdAt DATETIME DEFAULT CURRENT_TIMESTAMP,
            updatedAt DATETIME DEFAULT CURRENT_TIMESTAMP
        );
        """
    )
    cur.execute(
        """
        CREATE TABLE IF NOT EXISTS TrainProfile (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            name TEXT UNIQUE NOT NULL,
            label TEXT,
            settings TEXT NOT NULL,
            isDefault BOOLEAN DEFAULT 0,
            createdAt DATETIME DEFAULT CURRENT_TIMESTAMP,
            updatedAt DATETIME DEFAULT CURRENT_TIMESTAMP
        );
        """
    )
    conn.commit()


def seed_settings(conn: sqlite3.Connection) -> None:
    cur = conn.cursor()
    cur.execute("SELECT key FROM Setting;")
    existing = {row[0] for row in cur.fetchall()}
    for key, value in DEFAULT_SETTINGS.items():
        if key in existing:
            continue
        cur.execute(
            "INSERT INTO Setting (key, value, createdAt, updatedAt) VALUES (?, ?, CURRENT_TIMESTAMP, CURRENT_TIMESTAMP)",
            (key, str(value)),
        )
    conn.commit()


def seed_train_profiles(conn: sqlite3.Connection) -> None:
    cur = conn.cursor()
    cur.execute("SELECT COUNT(*) FROM TrainProfile;")
    count = cur.fetchone()[0]
    if count:
        return
    for profile in DEFAULT_PROFILES:
        cur.execute(
            "INSERT INTO TrainProfile (name, label, settings, isDefault, createdAt, updatedAt) "
            "VALUES (?, ?, ?, ?, CURRENT_TIMESTAMP, CURRENT_TIMESTAMP)",
            (
                profile["name"],
                profile["label"],
                json.dumps(profile["settings"]),
                1 if profile.get("isDefault") else 0,
            ),
        )
    conn.commit()


def seed_tagger_model(conn: sqlite3.Connection) -> None:
    cur = conn.cursor()
    TAGGER_MODELS_DIR.mkdir(parents=True, exist_ok=True)
    for repo_id in TAGGER_MODELS:
        safe_name = repo_id.replace("/", "-").replace("\\", "-")
        dest = TAGGER_MODELS_DIR / safe_name
        cur.execute("SELECT 1 FROM TaggerModel WHERE repoId=?", (repo_id,))
        if cur.fetchone():
            continue
        cur.execute(
            "INSERT INTO TaggerModel (repoId, name, folder, size, status, createdAt, updatedAt) "
            "VALUES (?, ?, ?, ?, ?, CURRENT_TIMESTAMP, CURRENT_TIMESTAMP)",
            (repo_id, safe_name, str(dest), 0, "missing"),
        )
    conn.commit()


def main() -> None:
    DB_PATH.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(str(DB_PATH))
    try:
        ensure_tables(conn)
        seed_settings(conn)
        seed_train_profiles(conn)
        seed_tagger_model(conn)
        print("[seed] basic data inserted")
    finally:
        conn.close()


if __name__ == "__main__":
    main()
