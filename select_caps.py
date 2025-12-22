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
from typing import Iterable, List, Tuple, Optional

import numpy as np
from PIL import Image

# Configuration (bundle-local paths by default)
BUNDLE_ROOT = Path(__file__).resolve().parent
SYSTEM_ROOT = BUNDLE_ROOT / "_system"
SRC_ROOT = SYSTEM_ROOT / "workflow" / "raw"  # filled with capped frames
DEST_ROOT = SYSTEM_ROOT / "workflow" / "work"  # final picks land here (per character)
IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".webp", ".bmp"}
TARGET_PER_CHARACTER = 40
FACE_QUOTA = 10  # target number of close-ups; rest will be non-face
HAMMING_THRESHOLD = 6  # increase for more variety, decrease if too few images selected
HAMMING_RELAXED = 4  # fallback threshold if diversity is too strict


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
    threshold: int = HAMMING_THRESHOLD,
    relaxed: int = HAMMING_RELAXED,
) -> List[Tuple[Path, np.ndarray]]:
    """
    Select images that are at least HAMMING_THRESHOLD apart.
    If not enough are found, relax to HAMMING_RELAXED.
    """
    selected: List[Tuple[Path, np.ndarray]] = []
    used_paths: set[Path] = set()

    def run_with_threshold(thresh: int, seeds: List[np.ndarray]) -> None:
        hashes: List[np.ndarray] = list(seeds)
        nonlocal selected, used_paths
        for p in sorted(paths):
            if p in used_paths:
                continue
            try:
                h = phash(p)
            except Exception:
                continue
            if all(hamming(h, prev_hash) >= thresh for prev_hash in hashes):
                hashes.append(h)
                selected.append((p, h))
                used_paths.add(p)
            if len(selected) >= limit:
                break

    seeds = list(seed_hashes) if seed_hashes else []
    run_with_threshold(threshold, seeds)
    if len(selected) < limit:
        run_with_threshold(relaxed, seeds)
    return selected


def run_selection(
    src_root: Path = SRC_ROOT,
    dest_root: Path = DEST_ROOT,
    face_quota: int = FACE_QUOTA,
    target_per_char: int = TARGET_PER_CHARACTER,
    hamming_threshold: int = HAMMING_THRESHOLD,
    hamming_relaxed: int = HAMMING_RELAXED,
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
            if children and not has_direct_images(folder):
                # Treat the parent folder as a single dataset (merge children)
                datasets.append((folder, None))
            else:
                datasets.append((folder, None))
        return datasets

    for folder, preset in datasets_from_root(src_root):
        char = character_key(folder.name)
        all_images: List[Path] = list(iter_images(folder))

        # treat any path segment containing "face" as face images (including face crops)
        face_imgs = [p for p in all_images if any("face" in part.lower() for part in p.parts)]
        other_imgs = [p for p in all_images if p not in face_imgs]

        def source_key(path: Path) -> str:
            rel = path.relative_to(folder)
            return rel.parts[0] if len(rel.parts) > 1 else "_root"

        def group_by_source(paths: List[Path]) -> dict[str, List[Path]]:
            grouped: dict[str, List[Path]] = {}
            for p in paths:
                grouped.setdefault(source_key(p), []).append(p)
            return {k: sorted(v) for k, v in grouped.items()}

        chosen_pairs: List[Tuple[Path, np.ndarray]] = []

        # Face selection
        face_groups = group_by_source(face_imgs)
        face_selected: List[Tuple[Path, np.ndarray]] = []
        face_hashes: List[np.ndarray] = []
        for paths in face_groups.values():
            if len(face_selected) >= face_quota:
                break
            picked = select_diverse(paths, limit=1, seed_hashes=face_hashes, threshold=hamming_threshold, relaxed=hamming_relaxed)
            face_selected.extend(picked)
            face_hashes.extend([h for _, h in picked])
        remaining_face = max(face_quota - len(face_selected), 0)
        if remaining_face > 0:
            all_faces_sorted = sorted(face_imgs)
            extra_faces = select_diverse(all_faces_sorted, limit=remaining_face, seed_hashes=face_hashes, threshold=hamming_threshold, relaxed=hamming_relaxed)
            face_selected.extend(extra_faces)
            face_hashes.extend([h for _, h in extra_faces])
        if len(face_selected) > face_quota:
            face_selected = face_selected[:face_quota]
        chosen_pairs.extend(face_selected)

        # Non-face selection
        remaining = target_per_char - len(chosen_pairs)
        if remaining > 0 and other_imgs:
            other_groups = group_by_source(other_imgs)
            current_hashes = list(face_hashes)
            for paths in other_groups.values():
                if remaining <= 0:
                    break
                picked = select_diverse(paths, limit=1, seed_hashes=current_hashes, threshold=hamming_threshold, relaxed=hamming_relaxed)
                chosen_pairs.extend(picked)
                current_hashes.extend([h for _, h in picked])
                remaining = target_per_char - len(chosen_pairs)

            if remaining > 0:
                sorted_other = sorted(other_imgs)
                third = max(1, len(sorted_other) // 3)
                quadrants = [
                    sorted_other[:third],
                    sorted_other[third:2 * third],
                    sorted_other[2 * third:],
                ]
                for quad in quadrants:
                    if remaining <= 0:
                        break
                    if not quad:
                        continue
                    per_quad_limit = min(8, remaining)
                    sample = random.sample(quad, min(len(quad), per_quad_limit))
                    quad_selected = select_diverse(sample, limit=per_quad_limit, seed_hashes=current_hashes, threshold=hamming_threshold, relaxed=hamming_relaxed)
                    chosen_pairs.extend(quad_selected)
                    current_hashes.extend([h for _, h in quad_selected])
                    remaining = target_per_char - len(chosen_pairs)

        if len(chosen_pairs) < target_per_char:
            missing = target_per_char - len(chosen_pairs)
            chosen_paths = {c[0] for c in chosen_pairs}
            for p in sorted(other_imgs):
                if p in chosen_paths:
                    continue
                chosen_pairs.append((p, np.array([])))
                if len(chosen_pairs) >= target_per_char:
                    break

        if len(chosen_pairs) > target_per_char:
            chosen_pairs = chosen_pairs[:target_per_char]

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
