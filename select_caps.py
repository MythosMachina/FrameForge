from __future__ import annotations

"""
Selects up to 40 diverse images per character (e.g., MiniJupiter_1 + MiniJupiter_2)
and copies them into a consolidated output folder.
Runs stand-alone or is imported by workflow.py.
"""

import random
import re
import shutil
from pathlib import Path
from typing import Iterable, List, Tuple

import numpy as np
from PIL import Image

# Configuration (bundle-local paths by default)
BUNDLE_ROOT = Path(__file__).resolve().parent
SRC_ROOT = BUNDLE_ROOT / "30_work" / "raw"  # filled with capped frames
DEST_ROOT = BUNDLE_ROOT / "30_work"  # final picks land here (per character)
IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".webp", ".bmp"}
TARGET_PER_CHARACTER = 40
FACE_QUOTA = 10  # target number of close-ups; rest will be non-face
HAMMING_THRESHOLD = 6  # increase for more variety, decrease if too few images selected
HAMMING_RELAXED = 4  # fallback threshold if diversity is too strict
PRESET_DIRS = {"furry", "human", "dragon", "daemon"}


def iter_images(folder: Path) -> Iterable[Path]:
    for path in folder.rglob("*"):
        if path.is_file() and path.suffix.lower() in IMAGE_EXTS:
            yield path


def character_key(folder_name: str) -> str:
    """
    Collapse MiniJupiter_1, MiniJupiter_2 -> MiniJupiter
    If no trailing numeric part exists, use the full name.
    """
    if "_" in folder_name:
        base, last = folder_name.rsplit("_", 1)
        if last.isdigit():
            return base
    return folder_name


def phash(path: Path, size: int = 8) -> np.ndarray:
    img = Image.open(path).convert("L").resize((size * 4, size * 4), Image.LANCZOS)
    dct = np.fft.fft2(np.asarray(img, dtype=np.float32))
    dct_low = np.real(dct)[:size, :size]
    median = np.median(dct_low)
    bits = (dct_low > median).astype(np.uint8).flatten()
    return bits


def hamming(a: np.ndarray, b: np.ndarray) -> int:
    return int(np.count_nonzero(a != b))


def select_diverse(
    paths: List[Path],
    limit: int,
    seed_hashes: List[np.ndarray] | None = None,
) -> List[Tuple[Path, np.ndarray]]:
    """
    Select images that are at least HAMMING_THRESHOLD apart.
    If not enough are found, relax to HAMMING_RELAXED.
    """
    selected: List[Tuple[Path, np.ndarray]] = []
    used_paths: set[Path] = set()

    def run_with_threshold(threshold: int, seeds: List[np.ndarray]) -> None:
        hashes: List[np.ndarray] = list(seeds)
        nonlocal selected, used_paths
        for p in sorted(paths):
            if p in used_paths:
                continue
            try:
                h = phash(p)
            except Exception:
                continue
            if all(hamming(h, prev_hash) >= threshold for prev_hash in hashes):
                hashes.append(h)
                selected.append((p, h))
                used_paths.add(p)
            if len(selected) >= limit:
                break

    seeds = list(seed_hashes) if seed_hashes else []
    run_with_threshold(HAMMING_THRESHOLD, seeds)
    if len(selected) < limit:
        run_with_threshold(HAMMING_RELAXED, seeds)
    return selected


def run_selection(
    src_root: Path = SRC_ROOT,
    dest_root: Path = DEST_ROOT,
    face_quota: int = FACE_QUOTA,
    target_per_char: int = TARGET_PER_CHARACTER,
) -> None:
    if not src_root.exists():
        print(f"[warn] source root not found: {src_root}")
        return
    dest_root.mkdir(parents=True, exist_ok=True)

    def has_direct_images(folder: Path) -> bool:
        return any(child.is_file() and child.suffix.lower() in IMAGE_EXTS for child in folder.iterdir())

    def datasets_from_root(root: Path) -> List[tuple[Path, Optional[str]]]:
        datasets: List[tuple[Path, Optional[str]]] = []
        for folder in sorted(p for p in root.iterdir() if p.is_dir() and p.name != "00"):
            children = [c for c in sorted(folder.iterdir()) if c.is_dir()]
            if folder.name in PRESET_DIRS:
                for child in children:
                    datasets.append((child, folder.name))
                continue
            if children and not has_direct_images(folder):
                # Treat children as datasets; folder name is the preset/category
                for child in children:
                    datasets.append((child, folder.name))
            else:
                datasets.append((folder, None))
        return datasets

    for folder, preset in datasets_from_root(src_root):
        char = character_key(folder.name)
        all_images: List[Path] = list(iter_images(folder))

        # treat any path segment containing "face" as face images (including face crops)
        face_imgs = [p for p in all_images if any("face" in part.lower() for part in p.parts)]
        other_imgs = [p for p in all_images if p not in face_imgs]

        chosen_pairs: List[Tuple[Path, np.ndarray]] = []

        face_selected = select_diverse(face_imgs, limit=face_quota)
        chosen_pairs.extend(face_selected)

        face_hashes = [h for _, h in face_selected]
        remaining = target_per_char - len(chosen_pairs)
        # Split other images in three quadrants (start, middle, end) and sample each randomly
        if remaining > 0 and other_imgs:
            sorted_other = sorted(other_imgs)
            third = max(1, len(sorted_other) // 3)
            quadrants = [
                sorted_other[:third],
                sorted_other[third:2 * third],
                sorted_other[2 * third:],
            ]
            current_hashes = list(face_hashes)
            for quad in quadrants:
                if remaining <= 0:
                    break
                if not quad:
                    continue
                per_quad_limit = min(10, remaining)
                sample = random.sample(quad, min(len(quad), per_quad_limit))
                quad_selected = select_diverse(sample, limit=per_quad_limit, seed_hashes=current_hashes)
                chosen_pairs.extend(quad_selected)
                current_hashes.extend([h for _, h in quad_selected])
                remaining = target_per_char - len(chosen_pairs)

        # If still short, pull whatever is left (no diversity checks)
        if len(chosen_pairs) < target_per_char:
            missing = target_per_char - len(chosen_pairs)
            chosen_paths = {c[0] for c in chosen_pairs}
            for p in sorted(other_imgs):
                if p in chosen_paths:
                    continue
                chosen_pairs.append((p, np.array([])))
                if len(chosen_pairs) >= target_per_char:
                    break

        chosen = [p for p, _ in chosen_pairs]
        if not chosen:
            print(f"[skip] {char}: no images found")
            continue

        out_dir = dest_root / char
        out_dir.mkdir(parents=True, exist_ok=True)
        if preset:
            (out_dir / ".autochar_preset").write_text(preset, encoding="utf-8")
        for idx, src in enumerate(chosen, start=1):
            dest = out_dir / f"{idx}{src.suffix.lower()}"
            counter = 1
            while dest.exists():
                dest = out_dir / f"{idx}_{counter}{src.suffix.lower()}"
                counter += 1
            shutil.copy2(src, dest)
        print(f"[ok] {char}: copied {len(chosen)} images to {out_dir}")


def main() -> None:
    run_selection()


if __name__ == "__main__":
    main()
