import argparse
from pathlib import Path
from PIL import Image
import re


def save(img: Image.Image, path: Path) -> None:
    counter = 1
    stem, suffix = path.stem, path.suffix
    while path.exists():
        path = path.with_name(f"{stem}_{counter}{suffix}")
        counter += 1
    img.save(path)
    print(f"[write] {path}")


def process_image(src: Path, outdir: Path) -> None:
    base = src.stem
    ext = src.suffix.lower()

    with Image.open(src) as original:
        w, h = original.size

        outputs = []

        orig_path = outdir / f"{base}_original{ext}"
        save(original, orig_path)
        outputs.append(orig_path)

        crops = {
            "top_half":      (0, 0, w, h // 2),
            "bottom_half":   (0, h // 2, w, h),
            "top_left":      (0, 0, w // 2, h // 2),
            "top_right":     (w // 2, 0, w, h // 2),
            "bottom_left":   (0, h // 2, w // 2, h),
            "bottom_right":  (w // 2, h // 2, w, h),
        }

        for name, box in crops.items():
            cropped = original.crop(box)
            path = outdir / f"{base}_{name}{ext}"
            save(cropped, path)
            outputs.append(path)

    # flips aus allen erzeugten Bildern
    for path in list(outputs):
        with Image.open(path) as img:
            flipped = img.transpose(Image.FLIP_LEFT_RIGHT)
            flip_path = path.with_name(f"{path.stem}_flip{path.suffix}")
            save(flipped, flip_path)


def main(input_dir: Path, recursive: bool) -> None:
    pattern = re.compile(r"^[1-9][0-9]*\.(jpg|jpeg|png|bmp|webp)$", re.IGNORECASE)

    iterator = input_dir.rglob("*") if recursive else input_dir.iterdir()

    for file in iterator:
        if file.is_file() and pattern.match(file.name):
            print(f"Bearbeite {file} ...")
            process_image(file, file.parent)  # Output im gleichen Ordner wie Source
        else:
            # Nur im Top-Level-Mode etwas Ignoriere ausgeben, damit die Konsole nicht zuspammt
            if not recursive:
                print(f"Ignoriere {file.name}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Dataset-Erweiterung: Crops und Flips fuer numerische Bilder (optional rekursiv)."
    )
    parser.add_argument(
        "input_dir",
        type=Path,
        help="Pfad zum Eingabe-Ordner (Startpunkt)",
    )
    parser.add_argument(
        "-r", "--recursive",
        action="store_true",
        help="Rekursiv alle Unterordner verarbeiten (Bulk-Modus)"
    )
    args = parser.parse_args()

    main(args.input_dir, args.recursive)
