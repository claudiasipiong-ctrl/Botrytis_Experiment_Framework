"""Synthetic three-class software tests; never scientific evidence."""
from __future__ import annotations

# Stdlib bootstrap: records import/CLI/config failures before optional dependencies.
import sys as _boot_sys
from pathlib import Path as _BootPath
_boot_root = _BootPath(__file__).resolve().parents[1]
_boot_sys.path.insert(0, str(_boot_root / "src"))
from utils import bootstrap as _bootstrap
_bootstrap.install(__file__, _boot_root)

import argparse
from pathlib import Path
import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[1]/"src"))

import numpy as np
from PIL import Image, ImageDraw


def _make_image(rng: np.random.Generator, w: int, h: int) -> Image.Image:
    base = rng.integers(40, 120, size=(h // 8 + 1, w // 8 + 1, 3), dtype=np.uint8)
    img = Image.fromarray(base).resize((w, h), Image.Resampling.BICUBIC)
    arr = np.asarray(img).astype(np.int16) + rng.integers(-12, 12, size=(h, w, 3))
    return Image.fromarray(np.clip(arr, 0, 255).astype(np.uint8))


def generate(out: Path, n: int = 329, seed: int = 7, fruit_class: int = 0, sign_class: int = 2) -> dict[str, int]:
    """Escribe el dataset sintetico y devuelve conteos."""
    rng = np.random.default_rng(seed)
    (out / "images").mkdir(parents=True, exist_ok=True)
    (out / "labels").mkdir(parents=True, exist_ok=True)
    if fruit_class != 0 or sign_class != 2:
        raise ValueError("Synthetic classes require physical=0, suspect=1, auxiliary=2")
    (out / "classes.txt").write_text("arandano_a\narandano_sospechoso\nsigno_sospechoso_botrytis\n", encoding="utf-8")
    n_fruit = n_sign = n_exif = 0
    for i in range(n):
        landscape = rng.random() < 0.6
        w, h = (480, 360) if landscape else (360, 480)
        img = _make_image(rng, w, h)
        draw = ImageDraw.Draw(img)
        lines: list[str] = []
        boxes_px: list[tuple[int, float, float, float, float]] = []
        for _ in range(int(rng.integers(2, 13))):
            fw = rng.uniform(0.08, 0.2) * min(w, h)
            fh = fw * rng.uniform(0.85, 1.15)
            x1 = rng.uniform(0, w - fw)
            y1 = rng.uniform(0, h - fh)
            fruit = (x1, y1, x1 + fw, y1 + fh)
            from utils.pairing import box_iou
            if any(box_iou(fruit, b[1:]) > 0.30 for b in boxes_px if b[0] == 0):
                continue
            draw.ellipse(fruit, fill=(40, 60, int(rng.integers(120, 200))))
            boxes_px.append((fruit_class, *fruit))
            n_fruit += 1
            if rng.random() < 0.45:
                boxes_px.append((1, *fruit))
            if rng.random() < 0.30:  # auxiliary box independent of class
                cov_w = rng.uniform(0.1, 1.0)
                sx1, sx2 = x1 - 0.05 * fw, x1 + cov_w * fw
                sy1, sy2 = y1 - 0.02 * fh, y1 + fh * rng.uniform(0.6, 1.05)
                sign = (max(0.0, sx1), max(0.0, sy1), min(w, sx2), min(h, sy2))
                draw.rectangle(sign, outline=(240, 240, 240), width=1)
                boxes_px.append((sign_class, *sign))
                n_sign += 1
        for cid, bx1, by1, bx2, by2 in boxes_px:
            xc, yc = (bx1 + bx2) / 2 / w, (by1 + by2) / 2 / h
            lines.append(f"{cid} {xc:.6f} {yc:.6f} {(bx2 - bx1) / w:.6f} {(by2 - by1) / h:.6f}")
        name = f"IMG_{i + 1:04d}"
        exif_rot = rng.random() < 0.05
        if exif_rot:  # pixeles crudos rotados + tag 6: el marco ORIENTADO es el dibujado
            exif = Image.Exif()
            exif[0x0112] = 6
            img.rotate(90, expand=True).save(out / "images" / f"{name}.jpg", quality=95, exif=exif.tobytes())
            n_exif += 1
        else:
            img.save(out / "images" / f"{name}.jpg", quality=95)
        (out / "labels" / f"{name}.txt").write_text("\n".join(lines) + "\n", encoding="utf-8", newline="\n")
    return {"images": n, "fruits": n_fruit, "signs": n_sign, "exif_rotated": n_exif}


def main() -> None:
    """CLI."""
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--out", required=True, type=Path)
    ap.add_argument("--n", type=int, default=329)
    ap.add_argument("--seed", type=int, default=7)
    ap.add_argument("--fruit-class", type=int, default=0)
    ap.add_argument("--sign-class", type=int, default=2)
    a = ap.parse_args()
    print(generate(a.out, a.n, a.seed, a.fruit_class, a.sign_class))


if __name__ == "__main__":
    main()
