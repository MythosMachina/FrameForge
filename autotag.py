import json
import os
import re
import sqlite3
from dataclasses import dataclass, field
from functools import lru_cache
from pathlib import Path
from typing import Iterable, List, Optional, Tuple

import numpy as np
import timm
import torch
from huggingface_hub import hf_hub_download
from PIL import Image
from safetensors.torch import load_file as load_safetensors
from timm.data import resolve_data_config
from timm.data.transforms_factory import create_transform
from torchvision import transforms

from db_broker_client import broker_enabled, broker_query

DEFAULT_MODEL_ID = "SmilingWolf/wd-eva02-large-tagger-v3"
DEFAULT_GENERAL_THRESHOLD = 0.55
DEFAULT_CHARACTER_THRESHOLD = 0.4
COLOR_TERMS = (
    "blue", "red", "green", "purple", "pink", "orange", "yellow",
    "gold", "brown", "black", "white", "gray", "grey", "silver",
)
COLOR_PIXEL_FRACTION = 0.03  # minimum fraction of image that must show a color for tag verification
COLOR_CANONICAL = {
    "blue": "blue",
    "red": "red",
    "green": "green",
    "purple": "purple",
    "pink": "pink",
    "orange": "orange",
    "yellow": "yellow",
    "gold": "gold",
    "brown": "brown",
    "black": "black",
    "white": "white",
    "gray": "gray",
    "grey": "gray",
    "silver": "gray",
}
DEFAULT_MAX_TAGS = 30
CONFIG_FILE = Path(__file__).resolve().parent / "autotag.config.json"
MACROS_DIR = Path(__file__).resolve().parent / "preset" / "macros"
IMAGE_SUFFIXES = (".jpg", ".jpeg", ".png", ".webp", ".bmp")
TAG_FILE = "selected_tags.csv"
WEIGHTS_FILE = "model.safetensors"
CONFIG_JSON = "config.json"
CHARACTER_CATEGORY = 4
RATING_CATEGORY = 9

# Hue-based color buckets for optional verification (degrees)
COLOR_RULES = {
    "blue": {"ranges": [(190, 255)], "sat_min": 0.25, "val_min": 0.15},
    "red": {"ranges": [(0, 25), (335, 360)], "sat_min": 0.25, "val_min": 0.15},
    "green": {"ranges": [(90, 150)], "sat_min": 0.2, "val_min": 0.15},
    "purple": {"ranges": [(260, 300)], "sat_min": 0.2, "val_min": 0.15},
    "pink": {"ranges": [(300, 340)], "sat_min": 0.2, "val_min": 0.2},
    "orange": {"ranges": [(20, 45)], "sat_min": 0.2, "val_min": 0.2},
    "yellow": {"ranges": [(45, 75)], "sat_min": 0.2, "val_min": 0.2},
    "gold": {"ranges": [(35, 65)], "sat_min": 0.25, "val_min": 0.2},
    "brown": {"ranges": [(10, 50)], "sat_min": 0.2, "val_min": 0.1, "val_max": 0.7},
    "black": {"ranges": [(0, 360)], "sat_max": 0.35, "val_max": 0.25},
    "white": {"ranges": [(0, 360)], "sat_max": 0.15, "val_min": 0.85},
    "gray": {"ranges": [(0, 360)], "sat_max": 0.2, "val_min": 0.2, "val_max": 0.9},
}


RUN_PREFIX_RE = re.compile(r"^[0-9]{6}[_-]")


@dataclass
class TaggerSettings:
    model_id: str = DEFAULT_MODEL_ID
    general_threshold: float = DEFAULT_GENERAL_THRESHOLD
    character_threshold: float = DEFAULT_CHARACTER_THRESHOLD
    max_tags: int = DEFAULT_MAX_TAGS


def iter_images(root: Path) -> Iterable[Path]:
    for suffix in IMAGE_SUFFIXES:
        for path in root.rglob(f"*{suffix}"):
            # Ignore any images that live under a folder named "raw" (workspace scratch)
            if any(part == "raw" for part in path.parts):
                continue
            yield path


def load_config(path: Optional[Path] = None) -> TaggerSettings:
    """
    Load tagger settings from JSON file. Missing keys fall back to defaults.
    """
    config_path = path or CONFIG_FILE
    settings = TaggerSettings()
    config_presets: List[str] = []
    if config_path.exists():
        data = json.loads(config_path.read_text(encoding="utf-8"))
        settings.model_id = data.get("model_id", settings.model_id)
        config_presets = data.get("model_presets", []) or []
        settings.general_threshold = float(data.get("general_threshold", settings.general_threshold))
        settings.character_threshold = float(data.get("character_threshold", settings.character_threshold))
        settings.max_tags = int(data.get("max_tags", settings.max_tags))
        # Allow selecting model via first uncommented preset (use # or // to comment out)
        def pick_preset(presets: List[str]) -> Optional[str]:
            for candidate in presets:
                if not isinstance(candidate, str):
                    continue
                stripped = candidate.strip()
                if not stripped or stripped.startswith("#") or stripped.startswith("//"):
                    continue
                return stripped
            return None

        if (not settings.model_id) or str(settings.model_id).strip().startswith("#"):
            preset = pick_preset(config_presets)
            if preset:
                settings.model_id = preset
    else:
        # create a template for the user to edit next time
        config_path.write_text(
            json.dumps(
                {
                    "model_id": settings.model_id,
                    "model_presets": [
                        "SmilingWolf/wd-eva02-large-tagger-v3",
                        "# SmilingWolf/wd-swinv2-tagger-v3",
                        "# SmilingWolf/wd-convnext-tagger-v3",
                        "# SmilingWolf/wd-vit-tagger-v3",
                    ],
                    "general_threshold": settings.general_threshold,
                    "character_threshold": settings.character_threshold,
                    "max_tags": settings.max_tags,
                },
                indent=2,
            ),
            encoding="utf-8",
        )
    return settings


@lru_cache(maxsize=4)
def load_labels_and_model(
    model_id: str,
    device: str,
) -> Tuple[List[str], List[int], torch.nn.Module, transforms.Compose, torch.device]:
    """
    Download model + labels from HF, build timm model on CPU, and return inference transform.
    Cached per model_id to avoid reload.
    """
    device_obj = torch.device(device)
    tag_path = hf_hub_download(repo_id=model_id, filename=TAG_FILE)
    cfg_path = hf_hub_download(repo_id=model_id, filename=CONFIG_JSON)
    weights_path = hf_hub_download(repo_id=model_id, filename=WEIGHTS_FILE)

    with open(cfg_path, "r", encoding="utf-8") as f:
        cfg = json.load(f)

    # Labels / categories
    names: List[str] = []
    categories: List[int] = []
    with open(tag_path, "r", encoding="utf-8") as f:
        # skip header
        next(f)
        for line in f:
            parts = line.strip().split(",")
            if len(parts) < 3:
                continue
            _, name, category, *_ = parts
            names.append(name)
            categories.append(int(category))

    arch = cfg.get("architecture", "eva02_large_patch14_448")
    num_classes = len(names)
    model = timm.create_model(arch, num_classes=num_classes, pretrained=False)
    state = load_safetensors(weights_path)
    model.load_state_dict(state, strict=True)
    model.to(device_obj)
    model.eval()

    data_cfg = resolve_data_config(model=model, use_test_size=False)
    # override with provided mean/std if present; validate input_size against model
    pretrained_cfg = cfg.get("pretrained_cfg", {})
    mean = pretrained_cfg.get("mean", data_cfg["mean"])
    std = pretrained_cfg.get("std", data_cfg["std"])

    def _normalize_input_size(val: object) -> Optional[Tuple[int, int, int]]:
        if isinstance(val, (list, tuple)) and len(val) == 3:
            return tuple(int(x) for x in val)  # type: ignore[return-value]
        return None

    def _model_input_size() -> Tuple[int, int, int]:
        default_cfg = getattr(model, "default_cfg", None) or getattr(model, "pretrained_cfg", None) or {}
        cfg_input = _normalize_input_size(default_cfg.get("input_size"))
        if cfg_input:
            return cfg_input
        img_size = getattr(model, "img_size", None)
        if img_size is not None:
            if isinstance(img_size, (list, tuple)) and len(img_size) == 2:
                return (3, int(img_size[0]), int(img_size[1]))
            return (3, int(img_size), int(img_size))
        patch = getattr(model, "patch_embed", None)
        patch_size = getattr(patch, "img_size", None) if patch is not None else None
        if patch_size is not None:
            if isinstance(patch_size, (list, tuple)) and len(patch_size) == 2:
                return (3, int(patch_size[0]), int(patch_size[1]))
            return (3, int(patch_size), int(patch_size))
        return data_cfg["input_size"]

    expected_input = _model_input_size()
    cfg_input = _normalize_input_size(pretrained_cfg.get("input_size"))
    input_size = cfg_input if cfg_input == expected_input else expected_input
    transform = create_transform(
        input_size=tuple(input_size),
        mean=mean,
        std=std,
        interpolation=pretrained_cfg.get("interpolation", "bicubic"),
        crop_pct=pretrained_cfg.get("crop_pct", 1.0),
        crop_mode=pretrained_cfg.get("crop_mode", "center"),
    )

    return names, categories, model, transform, device_obj


def load_txt_list(path: Path) -> List[str]:
    if not path.exists():
        return []
    lines: List[str] = []
    for line in path.read_text(encoding="utf-8").splitlines():
        stripped = line.strip()
        if not stripped or stripped.startswith("#") or stripped.startswith("//"):
            continue
        lines.append(stripped)
    return lines


def load_autochar_lists(preset: Optional[str]) -> List[str]:
    """
    Load autochar block list from the RUN_DB AutoCharPreset table only.
    Multiple presets can be comma-separated; results are concatenated.
    No local file fallbacks to keep the DB as single source of truth.
    """
    block: List[str] = []
    db_path = os.environ.get("RUN_DB")
    names: List[str] = []
    if preset:
        names = [p.strip() for p in preset.split(",") if p.strip()]
    if broker_enabled():
        try:
            resp = broker_query("get_autochar_presets", {"names": names})
            rows = resp.get("data") or []
            if rows:
                for row in rows:
                    block += json.loads(row.get("block") or "[]")
                    print(f"[info] Autochar preset from DB: {row.get('name')}")
                return block
        except Exception as e:
            print(f"[warn] Autochar preset broker lookup failed ({e}); no patterns applied")
    # Only DB is authoritative
    if db_path and Path(db_path).exists():
        try:
            conn = sqlite3.connect(db_path)
            cur = conn.cursor()
            target_names = names or ["default"]
            rows = []
            for name in target_names:
                cur.execute("SELECT blockPatterns, allowPatterns FROM AutoCharPreset WHERE name=?", (name,))
                r = cur.fetchone()
                if r:
                    rows.append((name, r))
            if not rows and not preset:
                cur.execute("SELECT name, blockPatterns, allowPatterns FROM AutoCharPreset ORDER BY id ASC LIMIT 1;")
                first = cur.fetchone()
                if first:
                    rows.append((first[0], (first[1], first[2])))
            if rows:
                for name, (b_raw, _a_raw) in rows:
                    block += json.loads(b_raw or "[]")
                    print(f"[info] Autochar preset from DB: {name}")
                conn.close()
                return block
            conn.close()
        except Exception as e:
            print(f"[warn] Autochar preset DB lookup failed ({e}); no patterns applied")
    if preset:
        print(f"[warn] Autochar preset '{preset}' not found in DB; no patterns applied")
    return block


def strip_run_prefix(name: str) -> str:
    """
    Remove leading RunID prefixes (e.g., 123456_name) from folder names
    so trigger tags stay clean.
    """
    return RUN_PREFIX_RE.sub("", name, count=1)


def load_hsv(image_path: Path) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Load image and return H, S, V channels as float arrays:
    H in degrees [0, 360], S/V in [0, 1].
    """
    hsv = Image.open(image_path).convert("HSV")
    h, s, v = np.asarray(hsv, dtype=np.float32).transpose(2, 0, 1)
    h = h * (360.0 / 255.0)
    s = s / 255.0
    v = v / 255.0
    return h, s, v


def color_fraction(image_path: Path, color: str) -> float:
    """
    Estimate how much of the image matches the given color bucket.
    Returns fraction of pixels (0..1).
    """
    rule = COLOR_RULES.get(color)
    if not rule:
        return 0.0
    h, s, v = load_hsv(image_path)
    mask = np.ones_like(h, dtype=bool)
    sat_min = rule.get("sat_min", 0.0)
    sat_max = rule.get("sat_max", 1.0)
    val_min = rule.get("val_min", 0.0)
    val_max = rule.get("val_max", 1.0)
    mask &= s >= sat_min
    mask &= s <= sat_max
    mask &= v >= val_min
    mask &= v <= val_max
    hue_mask = np.zeros_like(h, dtype=bool)
    for low, high in rule.get("ranges", []):
        if low <= high:
            hue_mask |= (h >= low) & (h <= high)
        else:
            hue_mask |= (h >= low) | (h <= high)
    mask &= hue_mask
    if mask.size == 0:
        return 0.0
    return float(mask.sum() / mask.size)


def predict_tags(
    image_path: Path,
    labels: List[str],
    categories: List[int],
    model: torch.nn.Module,
    transform: transforms.Compose,
    general_threshold: float,
    character_threshold: float,
    max_tags: int,
    device: torch.device,
    trigger_tag: Optional[str] = None,
    verify_colors: bool = False,
    use_bgr: bool = False,
) -> str:
    image = Image.open(image_path).convert("RGB")
    tensor = transform(image).unsqueeze(0)
    if use_bgr:
        tensor = tensor[:, [2, 1, 0], :, :]
    tensor = tensor.to(device)
    with torch.inference_mode():
        if device.type == "cuda":
            with torch.cuda.amp.autocast():
                logits = model(tensor)
        else:
            logits = model(tensor)
    probs = torch.sigmoid(logits)[0].tolist()

    tags: List[Tuple[float, str]] = []
    for prob, name, category in zip(probs, labels, categories):
        if category == RATING_CATEGORY:
            continue  # skip ratings
        clean_name = name.replace("_", " ")
        lower_name = clean_name.lower()
        threshold = character_threshold if category == CHARACTER_CATEGORY else general_threshold
        if prob < threshold:
            continue
        tags.append((prob, clean_name))

    if verify_colors:
        filtered: List[Tuple[float, str]] = []
        for prob, name in tags:
            lower = name.lower()
            matched_term = next((term for term in COLOR_CANONICAL if term in lower), None)
            if matched_term:
                color_key = COLOR_CANONICAL[matched_term]
                frac = color_fraction(image_path, color_key)
                if frac < COLOR_PIXEL_FRACTION:
                    continue
            filtered.append((prob, name))
        tags = filtered

    tags.sort(key=lambda t: t[0], reverse=True)

    parts: List[str] = []
    if trigger_tag:
        parts.append(trigger_tag)

    remaining_slots = max(max_tags - len(parts), 0)
    for _, tag in tags:
        if remaining_slots <= 0:
            break
        parts.append(tag)
        remaining_slots -= 1

    return ", ".join(parts)


def _normalize_tag(text: str) -> str:
    return " ".join(text.strip().lower().replace("_", " ").split())

@lru_cache(maxsize=16)
def _load_macro_values(name: str) -> List[str]:
    txt_path = MACROS_DIR / f"{name}.txt"
    json_path = MACROS_DIR / f"{name}.json"
    if txt_path.exists():
        values: List[str] = []
        for line in txt_path.read_text(encoding="utf-8").splitlines():
            raw = line.strip()
            if not raw or raw.startswith("#") or raw.startswith("//"):
                continue
            for chunk in raw.split(","):
                item = chunk.strip()
                if item:
                    values.append(item)
        return values
    if json_path.exists():
        try:
            data = json.loads(json_path.read_text(encoding="utf-8"))
            values = data.get("values") if isinstance(data, dict) else None
            if isinstance(values, list):
                return [str(v).strip() for v in values if str(v).strip()]
        except Exception:
            return []
    return []


def _expand_macro_patterns(patterns: Iterable[str]) -> List[str]:
    expanded: List[str] = []
    for pat in patterns:
        raw = str(pat).strip()
        if raw.startswith("@"):
            macro_spec = raw[1:]
            macro_name, sep, suffix = macro_spec.partition(":")
            macro_name = macro_name.strip()
            if not macro_name:
                expanded.append(raw)
                continue
            values = _load_macro_values(macro_name)
            if not values:
                expanded.append(raw)
                continue
            if sep:
                suffix = suffix.strip()
                if suffix:
                    expanded.extend([f"{v} {suffix}" for v in values])
                    continue
            expanded.extend(values)
            continue
        expanded.append(raw)
    return expanded


def _match_wildcard(pattern: str, tag: str) -> bool:
    pat = _normalize_tag(pattern)
    tgt = _normalize_tag(tag)
    if "*" not in pat:
        return pat == tgt
    parts = pat.split("*")
    if parts and parts[0] and not tgt.startswith(parts[0]):
        return False
    if parts and parts[-1] and not tgt.endswith(parts[-1]):
        return False
    idx = 0
    for part in parts:
        if not part:
            continue
        pos = tgt.find(part, idx)
        if pos == -1:
            return False
        idx = pos + len(part)
    return True


def filter_tags(tag_line: str, patterns: Iterable[str]) -> str:
    """
    Remove tags matching any wildcard pattern (case-insensitive).
    Tag separator is ", " and the first tag is preserved.
    """
    tags = [t.strip() for t in tag_line.split(",") if t.strip()]
    if not tags:
        return tag_line
    expanded = _expand_macro_patterns(patterns)
    keep = [tags[0]]
    for tag in tags[1:]:
        if any(_match_wildcard(pat, tag) for pat in expanded):
            continue
        keep.append(tag)
    return ", ".join(keep)


def _use_bgr_for_model(model_id: str) -> bool:
    """
    WD taggers are often trained with BGR channel order; swap channels to match.
    """
    return "/wd-" in model_id.lower()


def tag_folder(
    root: Path,
    model_id: Optional[str] = None,
    general_threshold: Optional[float] = None,
    character_threshold: Optional[float] = None,
    max_tags: Optional[int] = None,
    config_path: Optional[Path] = None,
    device: Optional[str] = None,
    autochar_enabled: bool = False,
    autochar_patterns: Optional[List[str]] = None,
    autochar_allow: Optional[List[str]] = None,
    verify_colors: bool = False,
) -> None:
    if not root.exists():
        raise FileNotFoundError(f"Tag target not found: {root}")
    settings = load_config(config_path)
    if model_id is not None:
        settings.model_id = model_id
    if general_threshold is not None:
        settings.general_threshold = general_threshold
    if character_threshold is not None:
        settings.character_threshold = character_threshold
    if max_tags is not None:
        settings.max_tags = max_tags
    device_str = device or "cpu"
    device_obj = torch.device(device_str)
    if device_obj.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA requested for autotag but no GPU is available.")

    labels, categories, model, transform, device_obj = load_labels_and_model(settings.model_id, device_obj.type)
    use_bgr = _use_bgr_for_model(settings.model_id)
    dataset_dirs = [p for p in sorted(root.iterdir()) if p.is_dir()]
    if not dataset_dirs:
        dataset_dirs = [root] if root.is_dir() else []

    try:
        for ds in dataset_dirs:
            preset_name = None
            env_preset = os.environ.get("AUTOCHAR_PRESET", "").strip()
            preset_marker = ds / ".autochar_preset"
            # Prefer explicit env (webapp/CLI) over folder markers to keep DB presets authoritative
            if env_preset:
                preset_name = env_preset
            elif preset_marker.exists():
                preset_name = preset_marker.read_text(encoding="utf-8").strip() or None
            patterns = None
            if autochar_enabled and preset_name:
                patterns = load_autochar_lists(preset_name)
                print(f"[info] Autochar filtering enabled for {ds.name} ({len(patterns or [])} patterns, preset={preset_name})")

            trigger = strip_run_prefix(ds.name).strip("_- ")
            if not trigger:
                trigger = ds.name
            for image_path in iter_images(ds):
                tag_line = predict_tags(
                    image_path=image_path,
                    labels=labels,
                    categories=categories,
                    model=model,
                    transform=transform,
                    general_threshold=settings.general_threshold,
                    character_threshold=settings.character_threshold,
                    max_tags=settings.max_tags,
                    device=device_obj,
                    trigger_tag=trigger,
                    verify_colors=verify_colors,
                    use_bgr=use_bgr,
                )
                if patterns:
                    tag_line = filter_tags(tag_line, patterns)
                out_path = image_path.with_suffix(".txt")
                out_path.write_text(tag_line, encoding="utf-8")
    finally:
        # Release GPU memory after autotag to avoid lingering CUDA usage
        try:
            if model is not None:
                del model
            if categories is not None:
                del categories
            if labels is not None:
                del labels
            if transform is not None:
                del transform
            if device_obj and device_obj.type == "cuda":
                torch.cuda.empty_cache()
                torch.cuda.ipc_collect()
        except Exception:
            pass
