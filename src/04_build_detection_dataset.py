"""04 - Dataset de deteccion de una sola clase (Especificacion s2, sec. 7 y 31).

Por cada imagen: conserva SOLO las cajas GT de fruto, las remapea a la clase ``0`` (``fruit``)
y las escribe en formato YOLO. Las cajas de signo NO son targets del detector.

Estructura generada (``paths.detection_dir``)::

    images/{train,validation,test}/   labels/{train,validation,test}/
    manifests/split_manifest.csv  manifests/inner_folds_detection.csv
    data.yaml            (sin clave 'path': Ultralytics lo resuelve relativo al propio YAML)

El split y ``inner_fold_det`` se heredan de los artefactos sellados (02 y 03).

Ejemplo::

    python src/04_build_detection_dataset.py
    python src/04_build_detection_dataset.py --clean --link-mode hardlink
"""

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

import pandas as pd
import yaml

sys.path.insert(0, str(Path(__file__).resolve().parent))

from utils.boxes import xyxy_to_yolo  # noqa: E402
from utils.checks import MethodologyViolation, assert_detection_labels_fruit_only  # noqa: E402
from utils.config_loader import cfg_path  # noqa: E402
from utils.io import atomic_write_text, ensure_dir, link_or_copy, write_csv, write_json  # noqa: E402
from utils.logging_utils import RunContext, run_cli  # noqa: E402
from utils.prepared import load_prepared_data  # noqa: E402
from utils.splits import SPLITS, manifest_paths  # noqa: E402
from utils.visualization import plot_fruits_per_image  # noqa: E402

SCRIPT = "04_build_detection_dataset"
DET_CLASS_ID = 0  # FIJO (sec. 7.2): detector_classes {0: fruit}


def add_arguments(parser: argparse.ArgumentParser) -> None:
    """Argumentos propios del script."""
    g = parser.add_argument_group("04_build_detection_dataset")
    g.add_argument("--clean", action="store_true", help="elimina images/ labels/ manifests/ existentes antes de generar")
    g.add_argument("--link-mode", choices=("copy", "hardlink"), default=None, help="sobrescribe detection_dataset.link_mode")
    g.add_argument("--no-figures", action="store_true", help="no generar figuras")


def _format_label_line(box: tuple[float, float, float, float], w: int, h: int) -> str:
    xc, yc, bw, bh = xyxy_to_yolo(box, w, h)
    return f"{DET_CLASS_ID} {xc:.6f} {yc:.6f} {bw:.6f} {bh:.6f}"


def main(ctx: RunContext, args: argparse.Namespace) -> None:
    """Construye el dataset de deteccion y lo verifica."""
    cfg, log = ctx.cfg, ctx.logger
    from utils.guards import assert_not_frozen
    assert_not_frozen(cfg, "04_build_detection_dataset")
    data = load_prepared_data(cfg, with_folds=True)
    assert data.manifest is not None and data.det_folds is not None
    ctx.set_hashes(dataset_hash=data.dataset_hash, split_hash=data.split_hash)
    artifacts = cfg_path(cfg, "paths.artifacts_dir")
    out = cfg_path(cfg, "paths.detection_dir")
    dcfg = cfg["detection_dataset"]
    link_mode = args.link_mode or str(dcfg.get("link_mode", "copy"))
    dir_names = dcfg["split_dir_names"]
    if list(dcfg["class_names"].items()) != [(DET_CLASS_ID, "fruit")]:
        raise MethodologyViolation("detection_dataset.class_names debe ser exactamente {0: fruit} (sec. 7.2)")

    if args.clean:
        for sub in ("images", "labels", "manifests"):
            shutil.rmtree(out / sub, ignore_errors=True)
        log.info("--clean: eliminados images/ labels/ manifests/ en %s", out)

    split_of = data.manifest.set_index("image_id")["outer_split"]
    fold_of = data.det_folds.set_index("image_id")["inner_fold_det"]
    fruit_boxes = data.annotations[data.annotations["kind"] == "fruit"]
    by_image = {k: g for k, g in fruit_boxes.groupby("image_id")}

    expected_images: set[Path] = set()
    expected_labels: set[Path] = set()
    rows: list[dict[str, object]] = []
    used_mode = link_mode
    for img in data.images.itertuples(index=False):
        split = str(split_of[img.image_id])
        sdir = dir_names[split]
        src = Path(img.image_path)
        dst_img = out / "images" / sdir / src.name
        dst_lab = out / "labels" / sdir / f"{img.image_id}.txt"
        expected_images.add(dst_img)
        expected_labels.add(dst_lab)
        used_mode = link_or_copy(src, dst_img, link_mode)
        boxes = by_image.get(img.image_id)
        lines = []
        if boxes is not None:
            for b in boxes.sort_values("ordinal").itertuples(index=False):
                lines.append(_format_label_line((b.x1, b.y1, b.x2, b.y2), int(img.image_w), int(img.image_h)))
        ensure_dir(dst_lab.parent)
        atomic_write_text(dst_lab, "\n".join(lines) + ("\n" if lines else ""))
        rows.append({"image_id": img.image_id, "outer_split": split, "inner_fold_det": fold_of.get(img.image_id, pd.NA),
                     "n_fruits": len(lines), "image_file": src.name})

    # -- archivos obsoletos --------------------------------------------------------------
    stale = [p for p in (out / "images").rglob("*") if p.is_file() and p not in expected_images]
    stale += [p for p in (out / "labels").rglob("*") if p.is_file() and p not in expected_labels and p.suffix == ".txt"]
    if stale:
        raise MethodologyViolation(
            f"{len(stale)} fichero(s) obsoleto(s) en {out} (p. ej. {stale[0].name}); ejecute con --clean para regenerar desde cero."
        )

    # -- manifiestos y data.yaml ---------------------------------------------------------
    man_dir = ensure_dir(out / "manifests")
    csv_path, sha_path = manifest_paths(artifacts, str(cfg["outer_split"]["manifest_name"]))
    shutil.copy2(csv_path, man_dir / csv_path.name)
    shutil.copy2(sha_path, man_dir / sha_path.name)
    shutil.copy2(artifacts / "inner_folds_detection.csv", man_dir / "inner_folds_detection.csv")
    det_table = pd.DataFrame(rows).sort_values("image_id")
    det_table["inner_fold_det"] = det_table["inner_fold_det"].astype("Int64")
    write_csv(det_table, man_dir / "detection_images.csv")
    data_yaml = {
        "train": f"images/{dir_names['train']}", "val": f"images/{dir_names['validation']}",
        "test": f"images/{dir_names['test']}", "nc": 1, "names": {DET_CLASS_ID: "fruit"},
    }
    atomic_write_text(out / "data.yaml",
                      "# Dataset de deteccion (1 clase). Sin 'path': Ultralytics resuelve las rutas relativas a este fichero.\n"
                      "# 'test' NO debe usarse hasta freeze_manifest.json (sec. 26-27).\n"
                      + yaml.safe_dump(data_yaml, sort_keys=False, allow_unicode=True))

    from utils.semantics import write_stamp
    from utils.isolation import seal_files
    seal_files(out, [*expected_images, *expected_labels, man_dir / "detection_images.csv"])
    write_stamp(out, cfg, data.dataset_hash, data.split_hash)
    # -- verificaciones ------------------------------------------------------------------
    n_boxes = assert_detection_labels_fruit_only(out / "labels")
    counts: dict[str, dict[str, int]] = {}
    for s in SPLITS:
        n_img = len([p for p in (out / "images" / dir_names[s]).glob("*") if p.is_file()]) if (out / "images" / dir_names[s]).exists() else 0
        n_lab = len(list((out / "labels" / dir_names[s]).glob("*.txt"))) if (out / "labels" / dir_names[s]).exists() else 0
        n_exp = int((data.manifest["outer_split"] == s).sum())
        if not (n_img == n_lab == n_exp):
            raise MethodologyViolation(f"Split {s}: imagenes={n_img}, labels={n_lab}, esperado={n_exp}")
        counts[s] = {"images": n_img, "labels": n_lab, "fruit_boxes": int(det_table.loc[det_table["outer_split"] == s, "n_fruits"].sum())}
    expected_boxes = int(len(fruit_boxes))
    if n_boxes != expected_boxes:
        raise MethodologyViolation(f"Cajas escritas ({n_boxes}) != cajas de fruto GT ({expected_boxes})")
    in_sign_boxes = int((data.annotations["kind"] == "sign").sum())
    log.info("cajas de signo excluidas del detector: %d (no son targets, sec. 7.2)", in_sign_boxes)

    summary = {"link_mode": used_mode, "counts": counts, "n_fruit_boxes": n_boxes, "n_sign_boxes_excluded": in_sign_boxes,
               "data_yaml": str(out / "data.yaml"), "split_hash": data.split_hash, "dataset_hash": data.dataset_hash}
    write_json(out / "manifests" / "detection_dataset_summary.json", summary)
    for p, d in ((out / "data.yaml", "data.yaml del detector"), (man_dir / "detection_images.csv", "tabla de imagenes del detector")):
        ctx.register_artifact(p, d)
    ctx.log_metrics(summary)
    log.info("Dataset de deteccion listo en %s: %s", out, counts)

    if not args.no_figures:
        img_df = det_table.rename(columns={})[["image_id", "outer_split", "n_fruits"]]
        ctx.save_figure(plot_fruits_per_image(img_df, "n_fruits"), "deteccion_frutos_por_imagen")


if __name__ == "__main__":
    raise SystemExit(run_cli(SCRIPT, main, description=__doc__ or SCRIPT, add_arguments=add_arguments))
