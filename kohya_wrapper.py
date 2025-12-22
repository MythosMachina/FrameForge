import math
import os
import shutil
import subprocess
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Optional
import json
import re
import sqlite3
import toml
import time
from db_broker_client import broker_enabled, broker_exec

# Bundle-local roots
BUNDLE_ROOT = Path(__file__).resolve().parent
KOHYA_ROOT = BUNDLE_ROOT / "trainer"
DEFAULT_PRESET = KOHYA_ROOT / "presets" / "lora" / "SDXL - LoRA AI_characters standard v1.1.json"
FALLBACK_PRESET = Path("/opt/ai/kohya_ss/presets/lora/SDXL - LoRA AI_characters standard v1.1.json")
SYSTEM_ROOT = BUNDLE_ROOT / "_system"
OUTPUT_ROOT = SYSTEM_ROOT / "trainer" / "output"
LOG_ROOT = SYSTEM_ROOT / "trainer" / "logs"
DATASET_STAGING_ROOT = SYSTEM_ROOT / "trainer" / "dataset" / "images"
VENV_CANDIDATES = [
    BUNDLE_ROOT / ".venv" / "bin",
    KOHYA_ROOT / ".venv" / "bin",
    KOHYA_ROOT / "venv" / "bin",
    Path("/opt/FrameForge/.venv/bin"),
]
PROGRESS_TABLE_SQL = """
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

DEFAULT_SAMPLE_PROMPTS = [
    "front view",
    "face close up",
    "sitting, smile",
]
SAMPLE_PROMPT_KEYS = [
    "trainer_sample_prompt_1",
    "trainer_sample_prompt_2",
    "trainer_sample_prompt_3",
]


def _render_sample_prompts(trigger: str, prompts: Optional[Iterable[str]] = None) -> str:
    lines = []
    items = list(prompts) if prompts else list(DEFAULT_SAMPLE_PROMPTS)
    for text in items:
        clean = str(text or "").strip()
        if not clean:
            continue
        lines.append(f"{trigger}, {clean}")
    if not lines:
        for text in DEFAULT_SAMPLE_PROMPTS:
            lines.append(f"{trigger}, {text}")
    return "\n".join(lines)


def _resolve_sample_prompts(settings: Dict[str, object]) -> List[str]:
    resolved: List[str] = []
    for key, fallback in zip(SAMPLE_PROMPT_KEYS, DEFAULT_SAMPLE_PROMPTS):
        val = settings.get(key, "")
        if isinstance(val, str):
            text = val.strip()
        else:
            text = str(val or "").strip()
        resolved.append(text or fallback)
    return resolved

@dataclass
class KohyaPlan:
    name: str
    dataset_dir: Path
    staged_dir: Path
    config_path: Path
    output_dir: Path
    epochs: int


def _as_int(val: object, default: int) -> int:
    try:
        return int(val)  # type: ignore[arg-type]
    except (TypeError, ValueError):
        return default


def _as_float(val: object, default: float) -> float:
    try:
        return float(val)  # type: ignore[arg-type]
    except (TypeError, ValueError):
        return default


def _as_bool(val: object, default: bool = False) -> bool:
    if isinstance(val, bool):
        return val
    if val is None:
        return default
    if isinstance(val, (int, float)):
        return bool(val)
    return str(val).strip().lower() in {"1", "true", "yes", "on"}


def _ensure_dir_clean(path: Path) -> None:
    if path.exists():
        shutil.rmtree(path)
    path.mkdir(parents=True, exist_ok=True)


def _detect_preset(preset_override: Optional[Path]) -> Path:
    if preset_override:
        return preset_override
    if DEFAULT_PRESET.exists():
        return DEFAULT_PRESET
    if FALLBACK_PRESET.exists():
        return FALLBACK_PRESET
    raise FileNotFoundError("kohya preset not found; expected under trainer/presets or /opt/ai/kohya_ss.")


def _pick_model_path(settings: Dict[str, object], filename: str) -> str:
    """
    Resolve model/vae path: prefer bundled safetensors, otherwise explicit setting.
    """
    local_candidates = [
        KOHYA_ROOT / "models" / filename,
        BUNDLE_ROOT / "trainer_old" / "models" / filename,
    ]
    for candidate in local_candidates:
        if candidate.exists():
            return str(candidate)
    key = "trainer_base_model" if "ponyDiffusion" in filename else "trainer_vae"
    val = settings.get(key, "")
    if isinstance(val, str) and val:
        return val
    return ""


def _build_config(dataset_dir: Path, settings: Dict[str, object], preset_path: Path, staged_dir: Optional[Path] = None) -> dict:
    def _image_bounds(root: Path) -> tuple[Optional[int], Optional[int]]:
        """
        Return (min_side, max_side) across all images in root (non-recursive).
        """
        min_side: Optional[int] = None
        max_side: Optional[int] = None
        for p in sorted(root.iterdir()):
            if p.is_file() and p.suffix.lower() in {".png", ".jpg", ".jpeg", ".webp", ".bmp"}:
                try:
                    from PIL import Image

                    with Image.open(p) as im:
                        w, h = im.size
                    side_min = min(w, h)
                    side_max = max(w, h)
                    min_side = side_min if min_side is None else min(min_side, side_min)
                    max_side = side_max if max_side is None else max(max_side, side_max)
                except Exception:
                    continue
        return min_side, max_side

    cfg = json.loads(preset_path.read_text(encoding="utf-8"))
    name = dataset_dir.name
    output_dir = OUTPUT_ROOT / name

    # Core paths: train_data_dir is the parent; subsets live in 1_<trigger>
    cfg["train_data_dir"] = str(DATASET_STAGING_ROOT)
    cfg["output_dir"] = str(output_dir)
    cfg["output_name"] = name
    cfg["caption_extension"] = ".txt"
    cfg["train_data_dir_repeats"] = _as_int(settings.get("trainer_dataset_repeats", 1), 1)
    cfg["cache_latents"] = False
    cfg["cache_latents_to_disk"] = False
    # Avoid external dataset configs; keep args-driven config
    if "dataset_config" in cfg:
        cfg["dataset_config"] = None
    # Models
    cfg["pretrained_model_name_or_path"] = _pick_model_path(settings, "ponyDiffusionV6XL.safetensors")
    cfg["vae"] = _pick_model_path(settings, "sdxl_vae.safetensors")
    xformers_value = cfg.get("xformers")
    if isinstance(xformers_value, str):
        xformers_key = xformers_value.strip().lower()
        if xformers_key == "sdpa":
            cfg["xformers"] = False
            cfg["sdpa"] = True
        elif xformers_key in {"xformers", "true", "yes", "on", "1"}:
            cfg["xformers"] = True
            cfg["sdpa"] = False
        elif xformers_key in {"none", "false", "no", "off", "0"}:
            cfg["xformers"] = False
            cfg["sdpa"] = bool(cfg.get("sdpa", False))
    elif isinstance(xformers_value, bool):
        cfg["xformers"] = xformers_value
        if xformers_value:
            cfg["sdpa"] = False

    # Training schedule
    epochs = _as_int(settings.get("trainer_epochs", cfg.get("max_train_epochs", 10)), 10)
    cfg["max_train_epochs"] = epochs
    cfg["epoch"] = epochs
    cfg["train_batch_size"] = _as_int(settings.get("trainer_batch_size", cfg.get("train_batch_size", 1)), 1)
    cfg["gradient_accumulation_steps"] = _as_int(settings.get("trainer_grad_accum", cfg.get("gradient_accumulation_steps", 1)), 1)
    cfg["network_dim"] = _as_int(settings.get("trainer_lora_rank", cfg.get("network_dim", 32)), 32)
    cfg["network_alpha"] = _as_int(settings.get("trainer_lora_alpha", cfg.get("network_alpha", 32)), 32)
    cfg["network_dropout"] = _as_float(settings.get("trainer_network_dropout", cfg.get("network_dropout", 0.0)), 0.0)
    cfg["network_module"] = "networks.lora"
    cfg["network_weights"] = None
    cfg["vae_batch_size"] = max(1, _as_int(settings.get("trainer_vae_batch_size", cfg.get("vae_batch_size", 1)), 1))
    cfg["learning_rate"] = _as_float(settings.get("trainer_learning_rate", cfg.get("learning_rate", 1e-4)), 1e-4)
    cfg["unet_lr"] = cfg["learning_rate"]
    cfg["text_encoder_lr"] = _as_float(settings.get("trainer_te_learning_rate", cfg.get("text_encoder_lr", cfg["learning_rate"])), cfg["learning_rate"])
    cfg["lr_scheduler"] = str(settings.get("trainer_lr_scheduler", cfg.get("lr_scheduler", "constant")) or "constant")
    cfg["lr_warmup_steps"] = _as_int(settings.get("trainer_lr_warmup_steps", cfg.get("lr_warmup_steps", 0)), 0)
    opt_name = str(settings.get("trainer_optimizer", cfg.get("optimizer", "AdamW")) or "AdamW")
    use_prodigy = _as_bool(settings.get("trainer_use_prodigy", False), False)
    cfg["optimizer"] = "Prodigy" if use_prodigy or opt_name.lower() == "prodigy" else "AdamW"
    cfg["max_grad_norm"] = _as_float(settings.get("trainer_max_grad_norm", cfg.get("max_grad_norm", 0)), 0.0)
    cfg["min_snr_gamma"] = _as_float(settings.get("trainer_min_snr_gamma", cfg.get("min_snr_gamma", 0)), 0.0)
    cfg["noise_offset"] = _as_float(settings.get("trainer_noise_offset", cfg.get("noise_offset", 0)), 0.0)
    cfg["weight_decay"] = _as_float(settings.get("trainer_weight_decay", cfg.get("weight_decay", 0)), 0.0)
    cfg["gradient_checkpointing"] = _as_bool(settings.get("trainer_gradient_checkpointing", cfg.get("gradient_checkpointing", True)), True)
    cfg["use_8bit_adam"] = _as_bool(settings.get("trainer_use_8bit_adam", cfg.get("use_8bit_adam", False)), False)
    cfg["train_on_input"] = True
    cfg["max_train_steps"] = _as_int(settings.get("trainer_max_train_steps", cfg.get("max_train_steps", 0)), 0)
    cfg["max_data_loader_n_workers"] = max(1, _as_int(settings.get("trainer_dataloader_workers", cfg.get("max_data_loader_n_workers", 1)), 1))

    # Resolution / buckets
    res = _as_int(settings.get("trainer_resolution", 1024), 1024)
    cfg["resolution"] = f"{res},{res}"
    cfg["max_resolution"] = f"{res},{res}"
    reso_steps = _as_int(settings.get("trainer_bucket_step", cfg.get("bucket_reso_steps", 64)), 64)
    cfg["bucket_reso_steps"] = reso_steps
    min_bucket_setting = _as_int(settings.get("trainer_bucket_min_reso", cfg.get("min_bucket_reso", res)), res)
    max_bucket_setting = _as_int(settings.get("trainer_bucket_max_reso", cfg.get("max_bucket_reso", res)), res)

    bounds_root = staged_dir if staged_dir else (DATASET_STAGING_ROOT / f"1_{name.split('_', 1)[-1]}")
    min_side, max_side = _image_bounds(bounds_root)
    if min_side is not None and max_side is not None:
        # Align bucket bounds to available image sizes to avoid tiny crop asserts
        max_bucket = max(reso_steps, min(max_bucket_setting, max_side - (max_side % reso_steps)))
        if max_bucket < reso_steps:
            max_bucket = reso_steps
        min_bucket = max(
            reso_steps,
            min(min_bucket_setting, max_bucket, min_side - (min_side % reso_steps) if min_side >= reso_steps else reso_steps),
        )
        cfg["min_bucket_reso"] = min_bucket
        cfg["max_bucket_reso"] = max_bucket
        cfg["resolution"] = f"{max_bucket},{max_bucket}"
        cfg["max_resolution"] = cfg["resolution"]
        cfg["bucket_no_upscale"] = False if min_side < min_bucket_setting else cfg.get("bucket_no_upscale", True)
    else:
        cfg["min_bucket_reso"] = min_bucket_setting
        cfg["max_bucket_reso"] = max_bucket_setting
        cfg["bucket_no_upscale"] = False if min_bucket_setting > reso_steps else cfg.get("bucket_no_upscale", True)

    # Captions
    cfg["clip_skip"] = _as_int(settings.get("trainer_clip_skip", cfg.get("clip_skip", 1)), 1)
    cfg["caption_dropout_rate"] = _as_float(settings.get("trainer_caption_dropout", cfg.get("caption_dropout_rate", 0.0)), 0.0)
    cfg["shuffle_caption"] = _as_bool(settings.get("trainer_shuffle_caption", cfg.get("shuffle_caption", False)), False)
    cfg["keep_tokens"] = _as_int(settings.get("trainer_keep_tokens", cfg.get("keep_tokens", 0)), 0)
    cfg["sample_every_n_epochs"] = cfg.get("sample_every_n_epochs", 1)
    user_prompts = _resolve_sample_prompts(settings)
    cfg["sample_prompts"] = _render_sample_prompts("{trigger}", user_prompts)

    # Logging
    cfg["log_with"] = "tensorboard"
    cfg["logging_dir"] = str(LOG_ROOT)
    cfg["log_tracker_config"] = None
    for key in ["save_every_n_steps", "sample_every_n_steps"]:
        if cfg.get(key, 0) in (0, None):
            cfg[key] = None
    # Retain epoch checkpoints instead of purging
    cfg["save_last_n_epochs"] = cfg.get("max_train_epochs", cfg.get("epoch", 10))
    # Strip Hugging Face upload settings entirely so no push is attempted
    for key in [
        "huggingface_repo_id",
        "huggingface_token",
        "huggingface_repo_type",
        "huggingface_repo_visibility",
        "huggingface_path_in_repo",
        "resume_from_huggingface",
        "save_state_to_huggingface",
        "save_state",
        "save_state_on_train_end",
    ]:
        cfg.pop(key, None)

    return cfg


def _venv_bin_dir() -> Optional[Path]:
    for cand in VENV_CANDIDATES:
        if cand.exists():
            return cand
    return None


def _count_images(folder: Path) -> int:
    valid_exts = {".png", ".jpg", ".jpeg", ".webp", ".bmp"}
    return sum(1 for p in folder.iterdir() if p.is_file() and p.suffix.lower() in valid_exts)


def build_kohya_plans(
    final_root: Path,
    settings: Dict[str, object],
    jobs_root: Optional[Path] = None,
    preset_override: Optional[Path] = None,
    allowed_run_ids: Optional[Iterable[str]] = None,
) -> List[KohyaPlan]:
    def _clean_staged_dir(staged_dir: Path) -> None:
        """
        Keep only valid image files (+ their captions), drop broken/other files to avoid training crashes.
        """
        valid_exts = {".png", ".jpg", ".jpeg", ".webp", ".bmp"}
        stems_with_images = set()
        min_side_allowed = 64
        for p in staged_dir.iterdir():
            if p.is_file() and p.suffix.lower() in valid_exts:
                stems_with_images.add(p.stem)

        for p in list(staged_dir.iterdir()):
            if p.is_dir():
                shutil.rmtree(p, ignore_errors=True)
                continue
            suffix = p.suffix.lower()
            if suffix in valid_exts:
                try:
                    from PIL import Image  # lazy import

                    with Image.open(p) as im:
                        w, h = im.size
                    if w <= 0 or h <= 0 or min(w, h) < min_side_allowed:
                        p.unlink(missing_ok=True)
                        cap = p.with_suffix(".txt")
                        cap.unlink(missing_ok=True)
                except Exception:
                    p.unlink(missing_ok=True)
                    cap = p.with_suffix(".txt")
                    cap.unlink(missing_ok=True)
            elif suffix == ".txt":
                # keep only if a paired image exists with same stem
                if p.stem not in stems_with_images:
                    p.unlink(missing_ok=True)
            else:
                p.unlink(missing_ok=True)

    def _is_run_folder(name: str) -> bool:
        """
        Accept only folders with a 6-digit runId prefix (e.g. 123456_name).
        Avoid staging stray folders like a bare 'furry'.
        """
        parts = name.split("_", 1)
        return len(parts) > 1 and parts[0].isdigit() and len(parts[0]) == 6

    def _matches_allowed(name: str) -> bool:
        if not allowed_run_ids:
            return True
        for run_id in allowed_run_ids:
            if not run_id:
                continue
            if name == run_id or name.startswith(f"{run_id}_"):
                return True
        return False

    if not final_root.exists():
        return []
    jobs_dir = jobs_root or (KOHYA_ROOT / "jobs")
    jobs_dir.mkdir(parents=True, exist_ok=True)
    preset_path = _detect_preset(preset_override)

    datasets = [
        p
        for p in sorted(final_root.iterdir())
        if p.is_dir()
        and p.name not in {"raw"}
        and not p.name.startswith("_")
        and (_is_run_folder(p.name) if allowed_run_ids else True)
        and _matches_allowed(p.name)
    ]
    plans: List[KohyaPlan] = []
    for ds in datasets:
        trigger = ds.name
        # Strip RunID prefix (e.g. 123456_name -> name)
        parts = trigger.split("_", 1)
        if parts and parts[0].isdigit() and len(parts[0]) == 6 and len(parts) > 1:
            trigger = parts[1]

        repeats = _as_int(settings.get("trainer_dataset_repeats", 1), 1)
        if repeats < 1:
            repeats = 1
        # Stage dataset into kohya layout: dataset/images/<repeats>_<trigger>
        staged_dir = DATASET_STAGING_ROOT / f"{repeats}_{trigger}"
        staged_dir.parent.mkdir(parents=True, exist_ok=True)
        if staged_dir.exists():
            shutil.rmtree(staged_dir)
        shutil.copytree(ds, staged_dir)
        _clean_staged_dir(staged_dir)
        image_count = _count_images(staged_dir)

        cfg = _build_config(ds, settings, preset_path, staged_dir=staged_dir)
        # Respect max_train_steps by trimming epochs so the planned total steps stay within the cap where possible.
        max_steps_setting = _as_int(settings.get("trainer_max_train_steps", cfg.get("max_train_steps", 0)), 0)
        if max_steps_setting and image_count > 0:
            steps_per_epoch = math.ceil(image_count / max(1, cfg["train_batch_size"]))
            steps_per_epoch = math.ceil(steps_per_epoch / max(1, cfg["gradient_accumulation_steps"]))
            if steps_per_epoch > 0:
                max_epochs_cap = max_steps_setting // steps_per_epoch
                if max_epochs_cap < 1:
                    max_epochs_cap = 1
                if cfg.get("max_train_epochs"):
                    cfg["max_train_epochs"] = min(cfg["max_train_epochs"], max_epochs_cap)
                    cfg["epoch"] = cfg["max_train_epochs"]
                cfg["max_train_steps"] = max_steps_setting
        # Write sample prompts to a file so Kohya does not treat prompt text as a path
        sample_file = jobs_dir / f"{ds.name}_sample_prompts.txt"
        user_prompts = _resolve_sample_prompts(settings)
        sample_file.write_text(_render_sample_prompts(trigger, user_prompts), encoding="utf-8")
        cfg["sample_prompts"] = str(sample_file)
        out_dir = Path(cfg["output_dir"])
        _ensure_dir_clean(out_dir)
        cfg_path = jobs_dir / f"{ds.name}.toml"
        cfg_path.write_text(toml.dumps(cfg), encoding="utf-8")
        plans.append(
            KohyaPlan(
                name=ds.name,
                dataset_dir=ds,
                staged_dir=staged_dir,
                config_path=cfg_path,
                output_dir=out_dir,
                epochs=cfg["max_train_epochs"],
            )
        )
    return plans


def _ensure_progress_table(db_path: Path) -> None:
    if broker_enabled():
        try:
            broker_exec("ensure_tables")
        except Exception:
            pass
        return
    try:
        conn = sqlite3.connect(db_path)
        cur = conn.cursor()
        cur.execute(PROGRESS_TABLE_SQL)
        conn.commit()
        conn.close()
    except Exception:
        pass


def _upsert_progress(
    db_path: Optional[Path],
    run_id: Optional[str],
    epoch: Optional[int],
    epoch_total: Optional[int],
    step: Optional[int],
    step_total: Optional[int],
    raw: str,
) -> None:
    if not db_path or not run_id:
        return
    if broker_enabled():
        try:
            broker_exec(
                "upsert_train_progress",
                {
                    "run_id": run_id,
                    "epoch": epoch,
                    "epoch_total": epoch_total,
                    "step": step,
                    "step_total": step_total,
                    "raw": raw,
                },
            )
        except Exception:
            pass
        return
    try:
        conn = sqlite3.connect(db_path)
        cur = conn.cursor()
        cur.execute(PROGRESS_TABLE_SQL)
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
            (run_id, epoch, epoch_total, step, step_total, raw),
        )
        cur.execute(
            "UPDATE Run SET lastStep=? WHERE runId=?;",
            (raw, run_id),
        )
        conn.commit()
        conn.close()
    except Exception:
        pass


def run_kohya_training(plan: KohyaPlan, hf_token: Optional[str] = None, run_id: Optional[str] = None) -> None:
    LOG_ROOT.mkdir(parents=True, exist_ok=True)
    log_path = LOG_ROOT / f"{plan.name}.log"
    db_path_env = os.environ.get("RUN_DB")
    db_path = Path(db_path_env) if db_path_env else None
    if db_path and run_id:
        _ensure_progress_table(db_path)

    env = os.environ.copy()
    bin_dir = _venv_bin_dir()
    if bin_dir:
        env["PATH"] = f"{bin_dir}:{env.get('PATH', '')}"
        env["VIRTUAL_ENV"] = str(bin_dir.parent)
    python_cmd = str((bin_dir / "python3") if bin_dir and (bin_dir / "python3").exists() else "python3")
    accel_bin = (bin_dir / "accelerate") if bin_dir and (bin_dir / "accelerate").exists() else None
    if hf_token:
        env["HF_TOKEN"] = hf_token
        env.setdefault("HUGGINGFACE_TOKEN", hf_token)
    script_path = KOHYA_ROOT / "sd-scripts" / "sdxl_train_network.py"
    if not script_path.exists():
        script_path = KOHYA_ROOT / "sd-scripts" / "train_network.py"
    if not script_path.exists():
        alt = KOHYA_ROOT / "train_network.py"
        if alt.exists():
            script_path = alt
        else:
            raise FileNotFoundError(f"kohya train_network.py not found under {KOHYA_ROOT}")

    # Replace trigger placeholder in sample prompts
    trigger = plan.name
    parts = trigger.split("_", 1)
    if parts and parts[0].isdigit() and len(parts[0]) == 6 and len(parts) > 1:
        trigger = parts[1]
    # Patch config file sample prompts with resolved trigger or ensure file exists
    try:
        cfg_data = toml.load(plan.config_path)
        sample_prompts = cfg_data.get("sample_prompts")
        if isinstance(sample_prompts, str):
            if "{trigger}" in sample_prompts:
                cfg_data["sample_prompts"] = sample_prompts.replace("{trigger}", trigger)
                plan.config_path.write_text(toml.dumps(cfg_data), encoding="utf-8")
            else:
                sample_path = Path(sample_prompts)
                if not sample_path.exists():
                    sample_path.write_text(_render_sample_prompts(trigger), encoding="utf-8")
    except Exception:
        pass

    cmd = (
        [str(accel_bin), "launch", str(script_path), "--config_file", str(plan.config_path)]
        if accel_bin
        else [python_cmd, str(script_path), "--config_file", str(plan.config_path)]
    )
    plan.output_dir.mkdir(parents=True, exist_ok=True)
    step_pattern = re.compile(r"steps:.*?(\d+)\s*/\s*(\d+)", re.IGNORECASE)
    epoch_pattern = re.compile(r"epoch\s+(\d+)\s*/\s*(\d+)", re.IGNORECASE)
    last_epoch = None
    last_epoch_total = None
    last_step = None
    last_step_total = None
    last_progress_at = 0.0
    with open(log_path, "w", encoding="utf-8") as log_file:
        proc = subprocess.Popen(
            cmd,
            cwd=KOHYA_ROOT,
            env=env,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            bufsize=1,
        )
        if proc.stdout:
            for line in proc.stdout:
                if not line:
                    break
                log_file.write(line)
                log_file.flush()
                m_step = step_pattern.search(line)
                if m_step:
                    try:
                        last_step = int(m_step.group(1))
                        last_step_total = int(m_step.group(2))
                    except Exception:
                        pass
                m_epoch = epoch_pattern.search(line)
                if m_epoch:
                    try:
                        last_epoch = int(m_epoch.group(1))
                        last_epoch_total = int(m_epoch.group(2))
                    except Exception:
                        pass
                now = time.time()
                if now - last_progress_at >= 5 and run_id:
                    raw = "train_progress"
                    parts = []
                    if last_epoch is not None and last_epoch_total is not None:
                        parts.append(f"epoch {last_epoch}/{last_epoch_total}")
                    if last_step is not None and last_step_total is not None:
                        parts.append(f"step {last_step}/{last_step_total}")
                    if parts:
                        raw = f"train_progress {' '.join(parts)}"
                    _upsert_progress(db_path, run_id, last_epoch, last_epoch_total, last_step, last_step_total, raw)
                    last_progress_at = now
        ret = proc.wait()

    if run_id and (last_step is not None or last_epoch is not None):
        raw = "train_progress"
        parts = []
        if last_epoch is not None and last_epoch_total is not None:
            parts.append(f"epoch {last_epoch}/{last_epoch_total}")
        if last_step is not None and last_step_total is not None:
            parts.append(f"step {last_step}/{last_step_total}")
        if parts:
            raw = f"train_progress {' '.join(parts)}"
        _upsert_progress(db_path, run_id, last_epoch, last_epoch_total, last_step, last_step_total, raw)

    if ret != 0:
        raise RuntimeError(f"kohya training failed for {plan.name} (exit {ret}); see {log_path}")
    # Mark completion for watcher
    plan.output_dir.mkdir(parents=True, exist_ok=True)
    (plan.output_dir / "TRAIN_DONE").write_text("done", encoding="utf-8")
