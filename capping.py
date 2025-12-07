import subprocess
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, List, Optional, Tuple
from urllib.request import urlretrieve

import numpy as np
from PIL import Image

# Configuration for frame extraction
VIDEO_EXTS = {".mp4", ".mov", ".mkv"}
FPS = 12  # export 12 frames per second
JPEG_QUALITY = 2  # lower is better quality for ffmpeg qscale (2 is near-lossless)
FACE_SAMPLE_STRIDE = 3  # analyze every Nth frame for face crops
MAX_FACE_CROPS = 200  # per video cap to avoid explosion
FACE_PAD_RATIO = 0.2  # 20% padding around detected box
FACE_MODEL_URL = "https://github.com/YapaLab/yolo-face/releases/download/v0.0.0/yolov12n-face.pt"
FACE_MODEL_PATH = Path(__file__).resolve().parent / "models_import" / "yolov12n-face.pt"


@dataclass
class Box:
    x1: float
    y1: float
    x2: float
    y2: float


def iter_videos(root: Path) -> Iterable[Path]:
    """Yield all video files under root that match VIDEO_EXTS."""
    for path in root.rglob("*"):
        if path.is_file() and path.suffix.lower() in VIDEO_EXTS:
            yield path


def cap_video(src: Path, out_dir: Path) -> None:
    """
    Export frames from a single video to out_dir using ffmpeg.
    Skips work if out_dir already has files.
    """
    out_dir.mkdir(parents=True, exist_ok=True)
    if any(out_dir.iterdir()):
        print(f"[skip] frames already exist: {out_dir}")
        return

    output_pattern = out_dir / "%06d.jpg"
    cmd = [
        "ffmpeg",
        "-loglevel",
        "warning",
        "-i",
        str(src),
        "-vf",
        f"fps={FPS}",
        "-qscale:v",
        str(JPEG_QUALITY),
        str(output_pattern),
    ]
    print(f"[cap ] {src} -> {out_dir}")
    subprocess.run(cmd, check=True)


def ensure_face_model() -> Optional[Path]:
    FACE_MODEL_PATH.parent.mkdir(parents=True, exist_ok=True)
    if FACE_MODEL_PATH.exists():
        return FACE_MODEL_PATH
    try:
        print(f"[face] downloading face model to {FACE_MODEL_PATH}")
        urlretrieve(FACE_MODEL_URL, FACE_MODEL_PATH)
        return FACE_MODEL_PATH
    except Exception as e:
        print(f"[face] download failed: {e}")
        return None


def load_face_detector():
    try:
        from ultralytics import YOLO
    except Exception:
        return None
    model_path = ensure_face_model()
    if model_path is None:
        return None
    try:
        return YOLO(model_path)
    except Exception as e:
        print(f"[face] failed to load detector: {e}")
        return None


def detect_faces(detector, image: Image.Image) -> list[Box]:
    if detector is None:
        return []
    # YOLO expects numpy array
    results = detector.predict(np.array(image), verbose=False)
    boxes: list[Box] = []
    for r in results:
        if not hasattr(r, "boxes"):
            continue
        for b in r.boxes:
            if b.conf is not None and float(b.conf) < 0.25:
                continue
            x1, y1, x2, y2 = b.xyxy[0].tolist()
            boxes.append(Box(x1, y1, x2, y2))
    return boxes


def crop_faces(detector, frames_dir: Path, stride: int = FACE_SAMPLE_STRIDE, max_crops: int = MAX_FACE_CROPS) -> int:
    """
    Scan frames in frames_dir for faces (sampling every Nth frame) and write crops to frames_dir/face.
    Returns number of crops written.
    """
    face_dir = frames_dir / "face"
    face_dir.mkdir(parents=True, exist_ok=True)
    written = 0
    frames = sorted(p for p in frames_dir.iterdir() if p.is_file() and p.suffix.lower() in {".jpg", ".jpeg", ".png", ".webp", ".bmp"})
    for idx, frame_path in enumerate(frames):
        if idx % stride != 0:
            continue
        if written >= max_crops:
            break
        try:
            img = Image.open(frame_path)
        except Exception:
            continue
        boxes = detect_faces(detector, img)
        for b in boxes:
            if written >= max_crops:
                break
            w, h = img.size
            pad_x = (b.x2 - b.x1) * FACE_PAD_RATIO
            pad_y = (b.y2 - b.y1) * FACE_PAD_RATIO
            x1 = max(0, int(b.x1 - pad_x))
            y1 = max(0, int(b.y1 - pad_y))
            x2 = min(w, int(b.x2 + pad_x))
            y2 = min(h, int(b.y2 + pad_y))
            crop = img.crop((x1, y1, x2, y2))
            out_name = face_dir / f"{frame_path.stem}_face{written+1}{frame_path.suffix.lower()}"
            try:
                crop.save(out_name)
                written += 1
            except Exception:
                continue
    return written


def cap_all(
    source_root: Path,
    capping_root: Path,
    facecap: bool = False,
) -> List[Path]:
    """
    Cap all videos under source_root into capping_root, mirroring the folder structure.
    Returns list of produced frame directories.
    """
    produced: List[Path] = []
    detector = load_face_detector() if facecap else None
    for video in sorted(iter_videos(source_root)):
        rel_parent = video.parent.relative_to(source_root)
        out_dir = capping_root / rel_parent / video.stem
        cap_video(video, out_dir)
        if facecap:
            crops = crop_faces(detector, out_dir)
            if crops:
                print(f"[face] {video} -> {crops} crops")
        produced.append(out_dir)
    return produced


if __name__ == "__main__":
    raise SystemExit("Use this module from workflow.py")
