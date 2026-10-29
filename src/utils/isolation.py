"""Content integrity and image-level split inheritance, independent of ML backends."""
from __future__ import annotations
import hashlib
from pathlib import Path
import numpy as np
import pandas as pd
from utils.checks import MethodologyViolation
from utils.config_loader import cfg_path
from utils.hashing import sha256_file, verify_checksum_file, write_checksum_file
from utils.io import load_image_rgb, read_csv, read_json, write_csv, write_json
from utils.splits import SPLITS, load_split_manifest


def pixel_hash(path):
    im = load_image_rgb(path)
    return hashlib.sha256(str(im.size).encode() + im.tobytes()).hexdigest()


def assert_unique_content(table, keys):
    for key in keys:
        if key not in table:
            continue
        bad = table.groupby(key)["outer_split"].nunique()
        if (bad > 1).any():
            raise MethodologyViolation(f"Fuga de contenido entre splits ({key}): {bad[bad > 1].index.tolist()[:5]}")


def audit_source(images, manifest):
    if manifest.image_id.duplicated().any() or set(images.image_id) != set(manifest.image_id):
        raise MethodologyViolation("Imagenes/manifiesto no son una correspondencia unica")
    tab = images[["image_id", "image_path"]].merge(manifest[["image_id", "outer_split"]], validate="one_to_one")
    tab["file_sha256"] = [sha256_file(p) for p in tab.image_path]
    tab["pixel_sha256"] = [pixel_hash(p) for p in tab.image_path]
    assert_unique_content(tab, ["image_id", "file_sha256", "pixel_sha256"])
    return tab


def assert_crop_membership(crops, manifest):
    if crops.duplicated(["image_id", "fruit_id"]).any() or crops.crop_path.duplicated().any():
        raise MethodologyViolation("Crop duplicado en el manifiesto")
    split_of = manifest.set_index("image_id").outer_split
    if not crops.image_id.map(split_of).eq(crops.outer_split).all():
        raise MethodologyViolation("Crop no hereda el split de su imagen original")
    for r in crops.itertuples():
        p = Path(r.crop_path)
        if p.is_absolute() or ".." in p.parts or p.parts[0] != r.outer_split:
            raise MethodologyViolation(f"Ruta de crop fuera de su split: {p}")


def seal_files(root, paths):
    root = Path(root)
    records = {Path(p).relative_to(root).as_posix(): sha256_file(p) for p in paths}
    out = root / "content_integrity.json"
    write_json(out, {"files": records})
    write_checksum_file(out, out.with_suffix(".sha256"))


def verify_files(root):
    root = Path(root)
    p = root / "content_integrity.json"
    if not p.is_file() or not verify_checksum_file(p, p.with_suffix(".sha256")):
        raise MethodologyViolation(f"Falta/sello invalido de contenido: {root}. Regenerar 04/05")
    records = read_json(p)["files"]
    for rel, digest in records.items():
        f = root / rel
        if not f.is_file() or sha256_file(f) != digest:
            raise MethodologyViolation(f"Derivado alterado o ausente: {f}")
    # Reject stale image/label/crop files, including unlisted copies in another split.
    actual = {p.relative_to(root).as_posix() for p in root.rglob("*") if p.is_file()
              and p.suffix.lower() in {".png", ".jpg", ".jpeg", ".webp", ".bmp", ".tif", ".tiff", ".txt"}}
    if actual - set(records):
        raise MethodologyViolation(f"Archivos derivados no registrados: {sorted(actual - set(records))[:5]}")


def audit_crops(crops, root, manifest):
    assert_crop_membership(crops, manifest)
    tab = crops[["image_id", "fruit_id", "outer_split", "crop_path", "label"]].copy()
    tab["file_sha256"] = [sha256_file(Path(root) / p) for p in tab.crop_path]
    tab["pixel_sha256"] = [pixel_hash(Path(root) / p) for p in tab.crop_path]
    assert_unique_content(tab, ["file_sha256", "pixel_sha256"])
    return tab


def validate_derived_manifest(cfg, table, task):
    manifest = load_split_manifest(cfg_path(cfg, "paths.artifacts_dir"), cfg["outer_split"]["manifest_name"])
    if task == "classification":
        assert_crop_membership(table, manifest)
    else:
        if table.image_id.duplicated().any() or set(table.image_id) != set(manifest.image_id):
            raise MethodologyViolation("Tabla derivada del detector incompleta/duplicada")
        if not table.image_id.map(manifest.set_index("image_id").outer_split).eq(table.outer_split).all():
            raise MethodologyViolation("Tabla derivada del detector no hereda el split")
    verify_files(cfg_path(cfg, f"paths.{task}_dir"))


def near_duplicate_cross_split(cfg, manifest):
    """dHash is a review flag, not proof of shared source; never resample silently."""
    inv_path = cfg_path(cfg, "paths.artifacts_dir") / "dataset_inventory.csv"
    if not inv_path.is_file():
        return pd.DataFrame(columns=["image_a","image_b","split_a","split_b","hamming"])
    inv = read_csv(inv_path, dtype={"dhash": str})
    threshold = int(cfg["dataset"]["validation"]["near_duplicate_hamming"])
    split_of = manifest.set_index("image_id").outer_split.to_dict()
    rows = []
    if threshold >= 0 and "dhash" in inv:
        hashes = [(str(r.image_id), int(str(r.dhash), 16)) for r in inv.dropna(subset=["dhash"]).itertuples() if str(r.image_id) in split_of]
        for i, (a, ha) in enumerate(hashes):
            for b, hb in hashes[i+1:]:
                distance = (ha ^ hb).bit_count()
                if split_of[a] != split_of[b] and distance <= threshold:
                    rows.append({"image_a":a,"image_b":b,"split_a":split_of[a],"split_b":split_of[b],"hamming":distance})
    out = pd.DataFrame(rows, columns=["image_a","image_b","split_a","split_b","hamming"])
    write_csv(out, cfg_path(cfg, "paths.artifacts_dir") / "near_duplicate_cross_split.csv")
    return out


def require_near_duplicate_review(cfg, data):
    pairs = near_duplicate_cross_split(cfg, data.manifest)
    if not len(pairs):
        return
    path = data.artifacts_dir / "near_duplicate_review.json"
    review = read_json(path) if path.is_file() else {}
    if review.get("dataset_hash") != data.dataset_hash:
        raise MethodologyViolation(f"{len(pairs)} pares dHash cruzan splits: revisar near_duplicate_cross_split.csv y documentar near_duplicate_review.json (dataset_hash vigente)")
    decisions = {tuple(sorted((r["image_a"],r["image_b"]))):r for r in review.get("pairs",[])}
    for r in pairs.itertuples():
        d = decisions.get(tuple(sorted((r.image_a,r.image_b))), {})
        if d.get("decision") != "distinct_sources" or not d.get("evidence"):
            raise MethodologyViolation(f"Posible fuga entre {r.image_a}/{r.image_b}: se requiere evidencia de fuentes distintas. No se cambia automaticamente el split")
