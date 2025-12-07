"""
Auto-trainer scaffold (no kohya_ss).

Currently: scans final_ready for character folders, computes a LoRA job plan
based on dataset size and a VRAM-friendly preset (<= 12GB), and writes a job
manifest to trainer/jobs. Training execution is not yet wired; workflow.py
logs a simulation message.
"""
from __future__ import annotations

import json
import math
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Iterable, List, Optional

# Defaults tailored for Pony SDXL LoRA with <=12GB VRAM
DEFAULT_BASE_MODEL = "darkstorm2150/pony-diffusion-xl-base-1.0"
DEFAULT_VAE = "madebyollin/sdxl-vae-fp16-fix"
DEFAULT_EPOCHS = 15
DEFAULT_RESOLUTION = 1024
MAX_VRAM_GB = 12
LOCAL_BASE_FILENAME = "ponyDiffusionV6XL.safetensors"
LOCAL_VAE_FILENAME = "sdxl_vae.safetensors"
MIN_TRAIN_STEPS = 6000
DEFAULT_NOISE_OFFSET = 0.03
DEFAULT_MIN_SNR_GAMMA = 0.0
DEFAULT_WEIGHT_DECAY = 0.05
DEFAULT_TE_LR_SCALE = 0.5  # te_lr = lr * scale

# Paths (bundle-local)
BUNDLE_ROOT = Path(__file__).resolve().parent.parent
FINAL_OUTPUT = BUNDLE_ROOT / "final_ready"
JOBS_DIR = BUNDLE_ROOT / "trainer" / "jobs"
CONFIG_PATH = BUNDLE_ROOT / "trainer" / "configs" / "trainer.config.json"
MODELS_DIR = BUNDLE_ROOT / "trainer" / "models"


@dataclass
class TrainingPlan:
    name: str
    dataset_path: str
    output_name: str
    base_model: str = DEFAULT_BASE_MODEL
    vae: str = DEFAULT_VAE
    base_model_path: Optional[str] = None
    vae_path: Optional[str] = None
    resolution: int = DEFAULT_RESOLUTION
    batch_size: int = 1
    grad_accum: int = 1
    epochs: int = DEFAULT_EPOCHS
    max_train_steps: int = 0
    learning_rate: float = 1e-4
    lr_scheduler: str = "cosine"
    lr_warmup_steps: int = 50
    fp16: bool = True
    use_8bit_adam: bool = True
    gradient_checkpointing: bool = True
    lora_rank: int = 16
    lora_alpha: int = 16
    te_lora_rank: int = 8
    te_lora_alpha: int = 8
    te_learning_rate: float = 5e-5
    clip_skip: int = 2
    network_dropout: float = 0.0
    caption_dropout: float = 0.0
    noise_offset: float = DEFAULT_NOISE_OFFSET
    min_snr_gamma: float = DEFAULT_MIN_SNR_GAMMA
    weight_decay: float = DEFAULT_WEIGHT_DECAY
    seed: int = 42
    notes: str = "auto-generated plan (simulation)"


def estimate_batch_size(num_images: int, resolution: int, max_vram_gb: int = MAX_VRAM_GB) -> int:
    # SDXL 1024px + LoRA typically fits batch 1 on ~12GB with checkpointing and fp16
    if max_vram_gb <= 12 or resolution >= 1024:
        return 1
    return 2


def estimate_grad_accum(num_images: int, batch_size: int) -> int:
    # Aim for effective batch ~4 on bigger sets (<=12-16GB), ~2 on tiny sets
    if num_images < 80:
        target = 2
    elif num_images < 200:
        target = 3
    else:
        target = 4
    return max(1, math.ceil(target / batch_size))


def compute_train_steps(num_images: int, epochs: int, batch_size: int, grad_accum: int) -> int:
    eff_batch = max(1, batch_size * grad_accum)
    return max(MIN_TRAIN_STEPS, math.ceil((num_images * epochs) / eff_batch))


def pick_resolution(num_images: int) -> int:
    # Keep 1024 for quality; drop to 896 for very large sets if VRAM constrained
    if num_images > 800 and MAX_VRAM_GB <= 12:
        return 896
    return DEFAULT_RESOLUTION


def pick_rank(num_images: int) -> int:
    if num_images > 800:
        return 64
    if num_images > 500:
        return 48
    if num_images > 250:
        return 32
    return 24


def pick_te_rank(unet_rank: int) -> int:
    return max(8, unet_rank // 2)


def pick_learning_rate(num_images: int, effective_batch: int) -> float:
    # Lower LR for stability on large sets; bump slightly with larger eff batch
    if num_images < 80:
        return 5.0e-5
    if effective_batch >= 4:
        return 1.0e-4
    return 7.5e-5


def iter_datasets(root: Path) -> Iterable[Path]:
    if not root.exists():
        return []
    for p in sorted(root.iterdir()):
        if p.is_dir() and not p.name.startswith("_") and p.name not in {"raw"}:
            yield p


def build_plan(dataset_dir: Path) -> TrainingPlan:
    images = list(dataset_dir.rglob("*.png")) + list(dataset_dir.rglob("*.jpg")) + list(dataset_dir.rglob("*.jpeg")) + list(dataset_dir.rglob("*.webp"))
    num_images = len(images)
    resolution = pick_resolution(num_images)
    batch_size = estimate_batch_size(num_images, resolution, MAX_VRAM_GB)
    grad_accum = estimate_grad_accum(num_images, batch_size)
    steps = compute_train_steps(num_images=num_images or 1, epochs=DEFAULT_EPOCHS, batch_size=batch_size, grad_accum=grad_accum)
    effective_batch = batch_size * grad_accum
    rank = pick_rank(num_images)
    te_rank = pick_te_rank(rank)
    lr = pick_learning_rate(num_images, effective_batch)
    warmup = max(50, int(steps * 0.03))
    base_model_path = (MODELS_DIR / LOCAL_BASE_FILENAME) if (MODELS_DIR / LOCAL_BASE_FILENAME).exists() else None
    vae_path = (MODELS_DIR / LOCAL_VAE_FILENAME) if (MODELS_DIR / LOCAL_VAE_FILENAME).exists() else None
    name = dataset_dir.name
    return TrainingPlan(
        name=name,
        dataset_path=str(dataset_dir),
        output_name=name,
        resolution=resolution,
        batch_size=batch_size,
        grad_accum=grad_accum,
        max_train_steps=steps,
        learning_rate=lr,
        te_learning_rate=lr * DEFAULT_TE_LR_SCALE,
        lr_warmup_steps=warmup,
        lora_rank=rank,
        lora_alpha=rank,
        te_lora_rank=te_rank,
        te_lora_alpha=te_rank,
        noise_offset=DEFAULT_NOISE_OFFSET,
        min_snr_gamma=DEFAULT_MIN_SNR_GAMMA,
        weight_decay=DEFAULT_WEIGHT_DECAY,
        base_model_path=str(base_model_path) if base_model_path else None,
        vae_path=str(vae_path) if vae_path else None,
        notes=f"auto plan: {num_images} imgs, res {resolution}, eff batch {effective_batch}, steps {steps}",
    )


def write_plan(plan: TrainingPlan) -> Path:
    JOBS_DIR.mkdir(parents=True, exist_ok=True)
    out_path = JOBS_DIR / f"{plan.name}.json"
    out_path.write_text(json.dumps(asdict(plan), indent=2), encoding="utf-8")
    return out_path


def run_planning(final_root: Path = FINAL_OUTPUT) -> List[Path]:
    written: List[Path] = []
    for ds in iter_datasets(final_root):
        plan = build_plan(ds)
        written.append(write_plan(plan))
    return written


if __name__ == "__main__":
    plans = run_planning()
    for p in plans:
        print(f"[plan] wrote {p}")
    if not plans:
        print("[plan] no datasets found in final_ready")
