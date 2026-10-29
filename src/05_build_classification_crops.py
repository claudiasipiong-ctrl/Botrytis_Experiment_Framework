"""05 - Class 0 crops, labelled by deterministic class 1 pairing."""
from __future__ import annotations

# Stdlib bootstrap: records import/CLI/config failures before optional dependencies.
import sys as _boot_sys
from pathlib import Path as _BootPath
_boot_root = _BootPath(__file__).resolve().parents[1]
_boot_sys.path.insert(0, str(_boot_root / "src"))
from utils import bootstrap as _bootstrap
_bootstrap.install(__file__, _boot_root)

import argparse
import shutil
import sys
from pathlib import Path
from typing import Any

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))

from utils.boxes import crop_window  # noqa: E402
from utils.checks import (  # noqa: E402
    MethodologyViolation, assert_crops_inherit_split, assert_pairing_labels_consistent, assert_same_image_same_fold,
)
from utils.config_loader import cfg_path  # noqa: E402
from utils.hashing import write_checksum_file  # noqa: E402
from utils.io import ensure_dir, load_image_rgb, save_image, to_posix, write_csv, write_json  # noqa: E402
from utils.logging_utils import RunContext, run_cli  # noqa: E402
from utils.metrics import classification_dataset_stats  # noqa: E402
from utils.prepared import load_prepared_data  # noqa: E402
from utils.splits import SPLITS  # noqa: E402
from utils.visualization import (  # noqa: E402
    plot_bbox_size_distribution, plot_class_balance_by_split, plot_crops_per_image, plot_dataset_distribution,
    plot_fruits_per_image, plot_gt_annotation_examples, plot_annotation_pairing_flow, plot_sign_coverage_distribution,
    select_gt_examples,
)

SCRIPT = "05_build_classification_crops"
MANIFEST_COLUMNS = [
    "image_id", "fruit_id", "x1", "y1", "x2", "y2", "max_sign_coverage", "label", "outer_split", "inner_fold",
    "n_sign_boxes", "best_sign_ordinal", "suspect_ordinal", "pairing_iou", "pairing_status", "crop_path", "crop_w", "crop_h", "image_w", "image_h",
]


def add_arguments(parser: argparse.ArgumentParser) -> None:
    """Argumentos propios del script."""
    g = parser.add_argument_group("05_build_classification_crops")
    g.add_argument("--clean", action="store_true", help="elimina los crops existentes (train/validation/test) antes de generar")
    g.add_argument("--manifest-only", action="store_true", help="solo manifiesto/estadisticas/figuras; no escribe crops")
    g.add_argument("--no-figures", action="store_true", help="no generar figuras")


def _crop_relpath(split: str, label: str, image_id: str, fruit_id: int, ext: str) -> str:
    return f"{split}/{label}/{image_id}__f{fruit_id:03d}.{ext}"


def main(ctx: RunContext, args: argparse.Namespace) -> None:
    """Build crops from paired annotations and check image-level inheritance."""
    cfg, log = ctx.cfg, ctx.logger
    from utils.guards import assert_not_frozen
    assert_not_frozen(cfg, "05_build_classification_crops")
    data = load_prepared_data(cfg, with_folds=True)
    assert data.manifest is not None and data.cls_folds is not None
    ctx.set_hashes(dataset_hash=data.dataset_hash, split_hash=data.split_hash)
    artifacts = cfg_path(cfg, "paths.artifacts_dir")
    out = cfg_path(cfg, "paths.classification_dir")
    ccfg = cfg["classification_dataset"]
    ext = "jpg" if str(ccfg["crop_format"]).lower() in ("jpg", "jpeg") else "png"
    margin = int(ccfg["margin_px"])
    pos, neg = cfg["classification_labels"]["positive_label"], cfg["classification_labels"]["negative_label"]

    # -- 1) GT por pairing ------------------------------------------------------------
    fr = data.fruits.copy()
    if fr.empty:
        raise MethodologyViolation("No hay frutos GT: no se puede construir el dataset de clasificacion")
    assert_pairing_labels_consistent(fr, cfg)
    split_of = data.manifest.set_index("image_id")["outer_split"]
    fold_of = data.cls_folds.set_index("image_id")["inner_fold_cls"]
    fr["outer_split"] = fr["image_id"].map(split_of)
    fr["inner_fold"] = fr["image_id"].map(fold_of).astype("Int64")
    fr.loc[fr["outer_split"] != "train", "inner_fold"] = pd.NA
    fr["crop_path"] = [_crop_relpath(r.outer_split, r.label, r.image_id, r.fruit_id, ext) for r in fr.itertuples(index=False)]
    log.info("GT = class 0 paired 1:1 to class 1; class 2 auxiliary")
    log.info("frutos=%d | %s=%d | %s=%d", len(fr), pos, int((fr["label"] == pos).sum()), neg, int((fr["label"] == neg).sum()))

    # -- 2) crops ------------------------------------------------------------------------
    if args.clean and not args.manifest_only:
        for s in SPLITS:
            shutil.rmtree(out / s, ignore_errors=True)
        log.info("--clean: eliminados crops previos en %s", out)
    crop_w: dict[tuple[str, int], int] = {}
    crop_h: dict[tuple[str, int], int] = {}
    if not args.manifest_only:
        n_done = 0
        for image_id, grp in fr.groupby("image_id", sort=True):
            src = data.images.loc[data.images["image_id"] == image_id, "image_path"].iloc[0]
            im = load_image_rgb(src)
            for r in grp.itertuples(index=False):
                left, top, right, bottom = crop_window((r.x1, r.y1, r.x2, r.y2), im.width, im.height, margin)
                crop = im.crop((left, top, right, bottom))
                save_image(crop, out / r.crop_path, fmt=ext, jpg_quality=int(ccfg["jpg_quality"]))
                crop_w[(image_id, r.fruit_id)], crop_h[(image_id, r.fruit_id)] = crop.width, crop.height
                n_done += 1
        log.info("%d crops escritos en %s", n_done, out)
        expected = set(fr["crop_path"])
        stale = [p for s in SPLITS if (out / s).exists() for p in (out / s).rglob("*")
                 if p.is_file() and to_posix(p, out) not in expected]
        if stale:
            raise MethodologyViolation(f"{len(stale)} crop(s) obsoleto(s) (p. ej. {stale[0].name}); ejecute con --clean.")
    fr["crop_w"] = [crop_w.get((r.image_id, r.fruit_id), pd.NA) for r in fr.itertuples(index=False)]
    fr["crop_h"] = [crop_h.get((r.image_id, r.fruit_id), pd.NA) for r in fr.itertuples(index=False)]
    fr["crop_w"], fr["crop_h"] = fr["crop_w"].astype("Int64"), fr["crop_h"].astype("Int64")

    # -- 3) manifiesto -------------------------------------------------------------------
    manifest = fr[MANIFEST_COLUMNS].sort_values(["image_id", "fruit_id"]).reset_index(drop=True)
    assert_crops_inherit_split(manifest, data.manifest)
    assert_same_image_same_fold(manifest, "inner_fold")
    if not args.manifest_only:
        missing = [p for p in manifest["crop_path"] if not (out / p).is_file()]
        if missing:
            raise MethodologyViolation(f"Faltan {len(missing)} crops en disco (p. ej. {missing[0]})")
    ensure_dir(out)
    man_path = out / ccfg["manifest_name"]
    from utils.pairing import SEMANTICS_VERSION
    from utils.semantics import semantic_fingerprint
    manifest["schema_version"]=4
    manifest["semantic_version"]=SEMANTICS_VERSION
    manifest["semantic_fingerprint"]=semantic_fingerprint(cfg)
    write_csv(manifest, man_path, float_format="%.10g")
    art_path = artifacts / "classification_manifest.csv"
    write_csv(manifest, art_path, float_format="%.10g")
    sha = write_checksum_file(art_path, art_path.with_suffix(".sha256"))
    log.info("manifiestos: %s | %s (sha256=%s)", man_path, art_path, sha)

    from utils.semantics import write_stamp
    from utils.isolation import audit_crops, seal_files
    if not args.manifest_only:
        content = audit_crops(manifest, out, data.manifest)
        write_csv(content, artifacts / "crop_content_audit.csv")
        seal_files(out, [man_path, *[out / p for p in manifest.crop_path]])
        write_stamp(out, cfg, data.dataset_hash, data.split_hash)
    else:
        (out / "semantic_stamp.json").unlink(missing_ok=True)
    # -- 4) estadisticas -----------------------------------------------------------------
    summary = data.summary.merge(data.manifest[["image_id", "outer_split"]], on="image_id")
    stats = classification_dataset_stats(manifest, summary, pos, neg)
    train_bal = stats["per_split"].get("train", {}).get("class_balance", {})
    if train_bal.get(pos) and train_bal.get(neg):
        n_tr = train_bal[pos] + train_bal[neg]
        stats["suggested_class_weights_train_INFORMATIVO"] = {
            pos: n_tr / (2 * train_bal[pos]), neg: n_tr / (2 * train_bal[neg]),
            "nota": "solo informativo; la estrategia ante desbalance es TODO BLOQUEANTE (sec. 38.6)",
        }
    stats.update({"dataset_hash": data.dataset_hash, "split_hash": data.split_hash, "manifest_sha256": sha,
                  "n_crops_written": 0 if args.manifest_only else int(len(manifest)), "crop_format": ext, "margin_px": margin})
    write_json(artifacts / "classification_stats.json", stats)
    write_csv(summary, artifacts / "classification_per_image.csv", float_format="%.6g")
    for p, d in ((man_path, "manifest.csv de crops"), (art_path, "classification_manifest.csv"),
                 (artifacts / "classification_stats.json", "estadisticas de clasificacion"),
                 (artifacts / "classification_per_image.csv", "resumen por imagen")):
        ctx.register_artifact(p, d)
    ctx.log_metrics(stats)
    cb = stats["class_balance"]
    log.info("TOTAL frutos=%d | %s=%d (%.1f %%) | %s=%d (%.1f %%) | razon de desbalance=%s",
             cb["total"], pos, cb[pos], cb[f"pct_{pos}"] or 0, neg, cb[neg], cb[f"pct_{neg}"] or 0,
             f"{cb['imbalance_ratio']:.2f}:1" if cb["imbalance_ratio"] else "n/a")
    if cb["imbalance_ratio"] is None:
        log.warning("Una de las clases tiene 0 ejemplos: el clasificador binario no es entrenable asi.")
    for s in SPLITS:
        ps = stats["per_split"].get(s)
        if ps:
            log.info("split %-10s imagenes=%d frutos=%d %s=%d", s, ps["n_images"], ps["n_fruits"], pos, ps["class_balance"][pos])

    # -- 5) figuras ----------------------------------------------------------------------
    if not args.no_figures:
        thr = float(cfg["pairing"]["min_iou"])
        ctx.save_figure(plot_dataset_distribution(summary, manifest, pos, neg), "dataset_distribucion_etiquetas")
        ctx.save_figure(plot_class_balance_by_split(manifest, pos, neg), "dataset_balance_clases_por_split")
        ctx.save_figure(plot_fruits_per_image(summary, "n_fruits"), "dataset_frutos_por_imagen")
        boxes = data.annotations[["kind", "x1", "y1", "x2", "y2", "image_w", "image_h"]]
        ctx.save_figure(plot_bbox_size_distribution(boxes), "dataset_tamanos_bbox")
        ctx.save_figure(plot_sign_coverage_distribution(manifest, None, pos, neg), "dataset_cobertura_signos")
        ctx.save_figure(plot_crops_per_image(summary, pos, neg), "dataset_crops_por_imagen")
        samples = _gt_samples(data, manifest, int(cfg["figures"].get("gt_examples", 6)), int(cfg["reproducibility"]["seed"]))
        ctx.save_figure(plot_gt_annotation_examples(samples, pos), "dataset_ejemplos_gt")
        ctx.save_figure(plot_annotation_pairing_flow(thr), "dataset_flujo_pairing")


def _gt_samples(data: Any, manifest: pd.DataFrame, n: int, seed: int) -> list[dict[str, Any]]:
    """Muestras para el grafico de ejemplos GT (frutos + signos de ``n`` imagenes variadas)."""
    manifest = manifest[manifest["outer_split"] == "train"].copy()
    ids = select_gt_examples(manifest, n, seed)
    samples: list[dict[str, Any]] = []
    for iid in ids:
        path = data.images.loc[data.images["image_id"] == iid, "image_path"].iloc[0]
        f = manifest[manifest["image_id"] == iid]
        s = data.annotations[(data.annotations["image_id"] == iid) & (data.annotations["kind"] == "sign")]
        n_pos = int((f["label"] == "con_signos").sum())
        samples.append({
            "image_path": path, "title": f"{iid}  ({n_pos}/{len(f)} frutos con_signos | {f['outer_split'].iloc[0]})",
            "fruits": f[["x1", "y1", "x2", "y2", "label", "max_sign_coverage"]].to_dict("records"),
            "signs": [tuple(v) for v in s[["x1", "y1", "x2", "y2"]].to_numpy(float)],
        })
    return samples


if __name__ == "__main__":
    raise SystemExit(run_cli(SCRIPT, main, description=__doc__ or SCRIPT, add_arguments=add_arguments))
