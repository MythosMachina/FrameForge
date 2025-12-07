"""
LoRA training runner for SDXL (Pony base), minimal and self-contained (no kohya_ss).
Trains one dataset (folder under final_ready) per invocation, saving epoch checkpoints
as .safetensors in trainer/output/<name>.

Designed for <=12GB VRAM: batch=1, grad accumulation (from plan), fp16 autocast,
gradient checkpointing enabled.
"""
from __future__ import annotations

import json
import math
import os
from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional, Tuple

import torch
import huggingface_hub

# diffusers 0.27 expects cached_download; newer huggingface_hub removed it.
if not hasattr(huggingface_hub, "cached_download"):
    huggingface_hub.cached_download = huggingface_hub.hf_hub_download
from PIL import Image
from torch import nn
from torch.optim import AdamW
try:
    from prodigyopt import Prodigy
except Exception:
    Prodigy = None
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms

from diffusers import DDPMScheduler, StableDiffusionXLPipeline
from diffusers.models.attention_processor import AttnProcessor2_0
from safetensors.torch import save_file as save_safetensors

# Paths
BUNDLE_ROOT = Path(__file__).resolve().parent.parent
TRAIN_OUTPUT = BUNDLE_ROOT / "trainer" / "output"


@dataclass
class TrainingPlan:
    name: str
    dataset_path: str
    output_name: str
    base_model: str
    vae: str
    base_model_path: Optional[str]
    vae_path: Optional[str]
    resolution: int
    batch_size: int
    grad_accum: int
    epochs: int
    max_train_steps: int
    learning_rate: float
    lr_scheduler: str
    lr_warmup_steps: int
    fp16: bool
    use_8bit_adam: bool
    gradient_checkpointing: bool
    lora_rank: int
    lora_alpha: int
    te_lora_rank: int
    te_lora_alpha: int
    te_learning_rate: float
    clip_skip: int
    network_dropout: float
    caption_dropout: float
    noise_offset: float
    min_snr_gamma: float
    weight_decay: float
    seed: int
    notes: str


class ImageCaptionDataset(Dataset):
    def __init__(self, root: Path, resolution: int, caption_dropout: float = 0.0):
        self.items: List[Tuple[Path, str]] = []
        self.transform = transforms.Compose(
            [
                transforms.Resize((resolution, resolution), interpolation=transforms.InterpolationMode.BICUBIC),
                transforms.CenterCrop(resolution),
                transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),
                transforms.Normalize([0.5], [0.5]),
            ]
        )
        for img_path in root.rglob("*"):
            if img_path.suffix.lower() in {".png", ".jpg", ".jpeg", ".webp"}:
                cap_path = img_path.with_suffix(".txt")
                if cap_path.exists():
                    caption = cap_path.read_text(encoding="utf-8").strip()
                else:
                    caption = img_path.parent.name
                self.items.append((img_path, caption))
        self.caption_dropout = caption_dropout

    def __len__(self) -> int:
        return len(self.items)

    def __getitem__(self, idx: int):
        img_path, caption = self.items[idx]
        image = Image.open(img_path).convert("RGB")
        pixel_values = self.transform(image)
        if self.caption_dropout > 0.0 and torch.rand(1).item() < self.caption_dropout:
            caption = ""
        return {"pixel_values": pixel_values, "caption": caption}


def setup_lora_for_unet(unet: nn.Module, rank: int, alpha: int) -> List[nn.Parameter]:
    """
    Replace all attention processors with AttnProcessor2_0 wrapped by LoRA via LoraLoaderMixin.
    This avoids deprecated LoRAAttnProcessor paths on diffusers >=0.27.
    """
    # Start from a clean AttnProcessor2_0 base
    attn_procs = {name: AttnProcessor2_0() for name in unet.attn_processors.keys()}
    unet.set_attn_processor(attn_procs)

    # Manually inject LoRA layers onto UNet attention projections (q, k, v, out)
    trainable: List[nn.Parameter] = []

    def add_lora_to_linear(module: nn.Module, rank: int, alpha: int) -> None:
        if not isinstance(module, nn.Linear):
            return
        in_features = module.in_features
        out_features = module.out_features
        lora_down = nn.Linear(in_features, rank, bias=False)
        lora_up = nn.Linear(rank, out_features, bias=False)
        # scale factor
        scale = alpha / rank
        # Attach to module for forward hook usage
        module.lora_down = lora_down
        module.lora_up = lora_up
        module.lora_scale = scale
        trainable.extend(list(lora_down.parameters()))
        trainable.extend(list(lora_up.parameters()))

    # Collect target modules by name (only attention projections)
    modules = list(unet.named_modules())
    for name, module in modules:
        if not isinstance(module, nn.Linear):
            continue
        if not any(key in name for key in ["to_q", "to_k", "to_v", "to_out.0"]):
            continue
        add_lora_to_linear(module, rank, alpha)

    # Patch forward of Linear to include LoRA contribution
    orig_linear_forward = nn.Linear.forward
    def lora_forward(self, input):
        out = orig_linear_forward(self, input)
        if hasattr(self, "lora_down") and hasattr(self, "lora_up"):
            # ensure LoRA weights on same device as input
            if self.lora_down.weight.device != input.device:
                self.lora_down = self.lora_down.to(input.device)
                self.lora_up = self.lora_up.to(input.device)
            # ensure dtypes match input
            lora_input = input.to(self.lora_down.weight.dtype)
            lora_out = self.lora_up(self.lora_down(lora_input)) * self.lora_scale
            lora_out = lora_out.to(out.dtype)
            out = out + lora_out
        return out
    nn.Linear.forward = lora_forward

    return [p for p in trainable if p.requires_grad], add_lora_to_linear


def collect_lora_state_dict(unet: nn.Module) -> dict:
    state = {}
    for module_name, module in unet.named_modules():
        if hasattr(module, "lora_down") and hasattr(module, "lora_up"):
            state[f"{module_name}.lora_down.weight"] = module.lora_down.weight
            state[f"{module_name}.lora_up.weight"] = module.lora_up.weight
            state[f"{module_name}.lora_scale"] = torch.tensor(module.lora_scale, device=module.lora_down.weight.device)
    return state


def save_epoch_lora(unet: nn.Module, run_dir: Path, name: str, epoch: int) -> Path:
    run_dir.mkdir(parents=True, exist_ok=True)
    dst = run_dir / f"{name}_epoch{epoch}.safetensors"
    state = collect_lora_state_dict(unet)
    save_safetensors(state, dst)
    return dst


def write_done_marker(run_dir: Path, marker: str = "TRAIN_DONE") -> None:
    run_dir.mkdir(parents=True, exist_ok=True)
    (run_dir / marker).write_text("done", encoding="utf-8")


def train_one(plan: TrainingPlan, device: str = "cuda") -> None:
    torch.manual_seed(plan.seed)
    run_dir = TRAIN_OUTPUT / plan.output_name
    run_dir.mkdir(parents=True, exist_ok=True)
    print(f"[info] Trainer: {plan.name} | imgs {plan.max_train_steps * plan.batch_size} | steps {plan.max_train_steps} | rank {plan.lora_rank} | batch {plan.batch_size} | grad_accum {plan.grad_accum}")

    dtype = torch.float16 if plan.fp16 else torch.float32
    pipe: StableDiffusionXLPipeline
    if plan.base_model_path:
        try:
            pipe = StableDiffusionXLPipeline.from_single_file(
                plan.base_model_path,
                torch_dtype=dtype,
                use_safetensors=True,
            )
        except Exception as e:
            print(f"[warn] Failed loading base model from file {plan.base_model_path}: {e}")
            print(f"[warn] Falling back to pretrained ID {plan.base_model}")
            pipe = StableDiffusionXLPipeline.from_pretrained(
                plan.base_model,
                torch_dtype=dtype,
                use_safetensors=True,
                variant=None,
            )
    else:
        pipe = StableDiffusionXLPipeline.from_pretrained(
            plan.base_model,
            torch_dtype=dtype,
            use_safetensors=True,
            variant=None,
        )
    if plan.vae_path:
        try:
            pipe.vae = pipe.vae.from_single_file(plan.vae_path, torch_dtype=pipe.vae.dtype)
        except Exception as e:
            try:
                from diffusers import AutoencoderKL

                pipe.vae = AutoencoderKL.from_single_file(plan.vae_path, torch_dtype=pipe.vae.dtype)
            except Exception:
                print(f"[warn] Failed loading VAE from file {plan.vae_path}: {e}")
                print(f"[warn] Using pipeline VAE instead.")
    pipe.to(device)
    pipe.unet.enable_gradient_checkpointing()
    if hasattr(pipe, "text_encoder") and hasattr(pipe.text_encoder, "gradient_checkpointing_enable"):
        pipe.text_encoder.gradient_checkpointing_enable()
    if hasattr(pipe, "text_encoder_2") and hasattr(pipe.text_encoder_2, "gradient_checkpointing_enable"):
        pipe.text_encoder_2.gradient_checkpointing_enable()

    # Freeze base params
    pipe.vae.requires_grad_(False)
    pipe.text_encoder.requires_grad_(False)
    pipe.text_encoder_2.requires_grad_(False)
    pipe.unet.requires_grad_(False)

    # LoRA on UNet
    trainable_params, add_lora_to_linear = setup_lora_for_unet(pipe.unet, plan.lora_rank, plan.lora_alpha)

    # Add text encoder LoRA
    def add_lora_to_text_encoder(te, rank: int, alpha: int, params: List[nn.Parameter]) -> None:
        if te is None:
            return
        modules = list(te.named_modules())
        for name, module in modules:
            if not isinstance(module, nn.Linear):
                continue
            if not any(key in name for key in ["q_proj", "k_proj", "v_proj", "out_proj", "fc1", "fc2"]):
                continue
            add_lora_to_linear(module, rank, alpha)
            params.extend([p for p in getattr(module, "lora_down").parameters()])
            params.extend([p for p in getattr(module, "lora_up").parameters()])

    te_params: List[nn.Parameter] = []
    add_lora_to_text_encoder(getattr(pipe, "text_encoder", None), plan.te_lora_rank, plan.te_lora_alpha, te_params)
    add_lora_to_text_encoder(getattr(pipe, "text_encoder_2", None), plan.te_lora_rank, plan.te_lora_alpha, te_params)

    opt_params = [{"params": trainable_params, "lr": plan.learning_rate, "weight_decay": plan.weight_decay}]
    if te_params:
        opt_params.append({"params": te_params, "lr": plan.te_learning_rate, "weight_decay": plan.weight_decay})

    if Prodigy is not None:
        optimizer = Prodigy(opt_params, decouple=True, weight_decay=plan.weight_decay, betas=(0.9, 0.99), use_bias_correction=False)
    else:
        optimizer = AdamW(opt_params, weight_decay=plan.weight_decay)
    noise_scheduler = DDPMScheduler.from_config(pipe.scheduler.config)

    dataset = ImageCaptionDataset(Path(plan.dataset_path), plan.resolution, caption_dropout=plan.caption_dropout)
    dataloader = DataLoader(
        dataset,
        batch_size=plan.batch_size,
        shuffle=True,
        num_workers=min(4, os.cpu_count() or 1),
        pin_memory=True,
    )

    total_steps = plan.max_train_steps
    global_step = 0
    pipe.unet.train()

    # Precompute time_ids for SDXL conditioning
    def compute_time_ids(height: int, width: int, crop_top: int = 0, crop_left: int = 0):
        return [height, width, crop_top, crop_left, height, width]

    while global_step < total_steps:
        for batch in dataloader:
            pixel_values = batch["pixel_values"].to(device, dtype=pipe.unet.dtype, non_blocking=True)
            captions = batch["caption"]

            # Encode prompts
            encoded = pipe.encode_prompt(
                prompt=captions,
                device=device,
                num_images_per_prompt=1,
                do_classifier_free_guidance=False,
            )
            if isinstance(encoded, tuple):
                if len(encoded) == 2:
                    prompt_embeds, pooled_embeds = encoded
                elif len(encoded) >= 4:
                    prompt_embeds, _, pooled_embeds, _ = encoded[:4]
                else:
                    prompt_embeds = encoded[0]
                    pooled_embeds = encoded[-1]
            else:
                raise RuntimeError("Unexpected encode_prompt output")
            time_ids = torch.tensor(
                [compute_time_ids(plan.resolution, plan.resolution)],
                device=device,
                dtype=prompt_embeds.dtype,
            )
            time_ids = time_ids.repeat(pixel_values.shape[0], 1)

            # Encode images to latents
            latents = pipe.vae.encode(pixel_values).latent_dist.sample() * pipe.vae.config.scaling_factor

            # Sample noise and timesteps
            noise = torch.randn_like(latents)
            if plan.noise_offset > 0:
                noise = noise + plan.noise_offset * torch.randn_like(noise)
            timesteps = torch.randint(0, noise_scheduler.num_train_timesteps, (latents.shape[0],), device=device)
            timesteps = timesteps.long()
            noisy_latents = noise_scheduler.add_noise(latents, noise, timesteps)

            # UNet forward
            with torch.autocast(device_type="cuda", dtype=torch.float16, enabled=plan.fp16):
                model_pred = pipe.unet(
                    noisy_latents,
                    timesteps,
                    encoder_hidden_states=prompt_embeds,
                    added_cond_kwargs={"text_embeds": pooled_embeds, "time_ids": time_ids},
                ).sample
                target = noise
                # Min-SNR weighting (optional)
                if plan.min_snr_gamma > 0:
                    alphas_cumprod = noise_scheduler.alphas_cumprod.to(device=device, dtype=model_pred.dtype)
                    snr = (alphas_cumprod[timesteps] / (1 - alphas_cumprod[timesteps])).float()
                    weight = torch.minimum(plan.min_snr_gamma / snr, torch.ones_like(snr))
                    weight = weight.reshape(-1, *([1] * (model_pred.ndim - 1)))
                    loss = (weight * (model_pred - target) ** 2).mean()
                else:
                    loss = nn.functional.mse_loss(model_pred, target, reduction="mean")

            loss = loss / plan.grad_accum
            loss.backward()

            if (global_step + 1) % plan.grad_accum == 0:
                optimizer.step()
                optimizer.zero_grad()

            global_step += 1
            if global_step % max(plan.max_train_steps // plan.epochs, 1) == 0:
                current_epoch = math.ceil(global_step * plan.epochs / plan.max_train_steps)
                save_epoch_lora(pipe.unet, run_dir, plan.output_name, current_epoch)
                print(f"[info] Trainer: {plan.name} epoch {current_epoch}/{plan.epochs} saved")

            if global_step >= total_steps:
                break

    # Final save to ensure last epoch present
    save_epoch_lora(pipe.unet, run_dir, plan.output_name, plan.epochs)
    write_done_marker(run_dir)
    print(f"[info] Trainer: {plan.name} done, checkpoints in {run_dir}")
    torch.cuda.empty_cache()


def load_plan(path: Path) -> TrainingPlan:
    data = json.loads(path.read_text(encoding="utf-8"))
    # Backward compatibility for plans without TE/regularization fields
    data.setdefault("te_learning_rate", data.get("learning_rate", 5e-5))
    data.setdefault("te_lora_rank", data.get("lora_rank", 16))
    data.setdefault("te_lora_alpha", data.get("te_lora_rank", data.get("lora_rank", 16)))
    data.setdefault("noise_offset", data.get("noise_offset", 0.0))
    data.setdefault("min_snr_gamma", data.get("min_snr_gamma", 0.0))
    data.setdefault("weight_decay", data.get("weight_decay", 0.01))
    return TrainingPlan(**data)


if __name__ == "__main__":
    # For manual testing: python trainer/train_runner.py trainer/jobs/Character.json
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("plan", type=Path, help="Path to plan JSON")
    args = parser.parse_args()
    plan = load_plan(args.plan)
    train_one(plan)
