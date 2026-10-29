"""01 - Validacion del dataset YOLO (Especificacion s2, sec. 3.3 y 31).

Comprueba por cada imagen:

* existencia del fichero de anotacion correspondiente (``<image_id>.txt``);
* formato YOLO valido (5 campos, ``class_id`` entero);
* ``class_id`` conocido: 0 fruto, 1 sospechoso adicional, 2 signo auxiliar;
* ``x_center, y_center, width, height`` en rango; ``width > 0``; ``height > 0``;
* area positiva tras convertir a pixeles y recortar a la imagen;
* ausencia de imagenes corruptas o truncadas;
* ausencia de imagenes duplicadas (contenido identico por SHA-256; casi-duplicados por dHash);
* unicidad de ``image_id`` (sin distinguir mayusculas: seguro en Windows).

Salidas (en ``artifacts/``): ``dataset_validation.json`` (incluye ``passed`` y ``dataset_hash``),
``dataset_inventory.csv`` y ``dataset_issues.csv``. Termina con exit 1 si hay ERRORES.

Incluye auditoria de todos los labels, pairing y stems huerfanos. IDs fijados en config.
``--label-only --dataset <ZIP/directorio> --output <directorio>`` no requiere imagenes.

Ejemplo::

    python src/01_validate_dataset.py --set classes.fruit_class_id=0 --set classes.sign_class_id=2
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
import sys
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import pandas as pd
from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parent))

from utils.boxes import clip_xyxy, yolo_to_xyxy  # noqa: E402
from utils.config_loader import cfg_path  # noqa: E402
from utils.dataset import (  # noqa: E402
    ERROR, VALIDATION_REPORT, WARNING, Issue, class_ids_from_cfg, dataset_dirs, discover_images,
    label_path_for, parse_yolo_label_file,
)
from utils.hashing import hamming_distance, image_dhash, sha256_file, sha256_text_file  # noqa: E402
from utils.io import get_image_size, image_orientation, write_csv, write_json  # noqa: E402
from utils.logging_utils import RunContext, run_cli  # noqa: E402
from utils.metrics import describe_series  # noqa: E402
from utils.visualization import (  # noqa: E402
    plot_annotation_class_counts, plot_bbox_size_distribution, plot_fruits_per_image,
)

SCRIPT = "01_validate_dataset"


class DatasetValidationError(RuntimeError):
    """La validacion encontro errores bloqueantes."""


def add_arguments(parser: argparse.ArgumentParser) -> None:
    """Argumentos propios del script."""
    g = parser.add_argument_group("01_validate_dataset")
    g.add_argument("--allow-count-mismatch", action="store_true",
                   help="no fallar si el numero de imagenes difiere de dataset.n_images_expected (solo pruebas)")
    g.add_argument("--no-figures", action="store_true", help="no generar figuras")


def _check_image(path: Path, min_side: int) -> tuple[dict[str, Any] | None, list[Issue]]:
    """Abre, verifica y decodifica una imagen. Devuelve (info, issues)."""
    issues: list[Issue] = []
    image_id = path.stem
    try:
        with Image.open(path) as im:
            im.verify()  # detecta cabeceras/CRC corruptos
        with Image.open(path) as im:
            im.load()  # detecta truncamiento (decodifica todos los pixeles)
            mode, fmt = im.mode, im.format
            dh = image_dhash(im)
    except Exception as exc:  # cualquier fallo de decodificacion = imagen corrupta
        issues.append(Issue(image_id, ERROR, "corrupt_image", f"{type(exc).__name__}: {exc}"))
        return None, issues
    w, h = get_image_size(path)
    orient = image_orientation(path)
    if min(w, h) < min_side:
        issues.append(Issue(image_id, WARNING, "small_image", f"Lado menor {min(w, h)} px < {min_side} px"))
    if mode != "RGB":
        issues.append(Issue(image_id, WARNING, "non_rgb_mode", f"Modo {mode}; se convertira a RGB al cargar"))
    return {"image_w": w, "image_h": h, "mode": mode, "format": fmt, "exif_orientation": orient, "dhash": f"{dh:016x}"}, issues


def _check_annotations(
    image_id: str, label_path: Path, w: int, h: int, fruit_ids: tuple[int, ...], sign_id: int, cfg_val: dict[str, Any],
) -> tuple[list[Issue], list[dict[str, Any]]]:
    """Valida anotaciones de una imagen. Devuelve (issues, cajas recortadas)."""
    tol = float(cfg_val["coord_tolerance"])
    min_side = float(cfg_val["min_box_side_px"])
    parsed = parse_yolo_label_file(label_path, image_id)
    issues = list(parsed.issues)
    boxes: list[dict[str, Any]] = []
    seen: set[tuple[int, float, float, float, float]] = set()
    for a in parsed.annotations:
        ln = a.line_no
        if a.class_id not in (*fruit_ids, 1, sign_id):
            issues.append(Issue(image_id, ERROR, "unknown_class", f"class_id={a.class_id} no es fruto({list(fruit_ids)}) ni signo({sign_id})", ln))
            continue
        if a.w <= 0 or a.h <= 0:
            issues.append(Issue(image_id, ERROR, "non_positive_size", f"width={a.w}, height={a.h} deben ser > 0", ln))
            continue
        bad = [n for n, v, lo, hi in (("x_center", a.xc, 0, 1), ("y_center", a.yc, 0, 1), ("width", a.w, 0, 1), ("height", a.h, 0, 1))
               if v < lo - tol or v > hi + tol]
        if bad:
            issues.append(Issue(image_id, ERROR, "out_of_range", f"valores fuera de [0,1]: {bad} ({a.xc}, {a.yc}, {a.w}, {a.h})", ln))
            continue
        raw = yolo_to_xyxy(a.xc, a.yc, a.w, a.h, w, h)
        clipped = clip_xyxy(raw, w, h)
        if max(-raw[0], -raw[1], raw[2] - w, raw[3] - h) > 0.5:
            issues.append(Issue(image_id, WARNING, "box_clipped", f"la caja excede la imagen mas de 0.5 px y se recorta: {tuple(round(v, 1) for v in raw)}", ln))
        area = max(0.0, clipped[2] - clipped[0]) * max(0.0, clipped[3] - clipped[1])
        if area <= 0:
            issues.append(Issue(image_id, ERROR, "zero_area_after_clip", f"area <= 0 tras convertir y recortar: {clipped}", ln))
            continue
        if min(clipped[2] - clipped[0], clipped[3] - clipped[1]) < min_side:
            issues.append(Issue(image_id, WARNING, "tiny_box", f"lado menor < {min_side} px: {tuple(round(v, 1) for v in clipped)}", ln))
        key = (a.class_id, round(a.xc, 6), round(a.yc, 6), round(a.w, 6), round(a.h, 6))
        if key in seen:
            issues.append(Issue(image_id, WARNING, "duplicate_box", "caja identica repetida en la misma imagen", ln))
        seen.add(key)
        boxes.append({
            "image_id": image_id, "kind": "fruit" if a.class_id in fruit_ids else "suspect" if a.class_id == 1 else "sign",
            "x1": clipped[0], "y1": clipped[1], "x2": clipped[2], "y2": clipped[3], "image_w": w, "image_h": h,
        })
    return issues, boxes


def main(ctx: RunContext, args: argparse.Namespace) -> None:
    """Valida todo el dataset y escribe inventario, incidencias e informe."""
    cfg, log = ctx.cfg, ctx.logger
    from utils.guards import assert_not_frozen
    assert_not_frozen(cfg, "01_validate_dataset")
    fruit_ids, sign_id = class_ids_from_cfg(cfg)  # aborta si siguen TODO
    images_dir, labels_dir = dataset_dirs(cfg)
    artifacts = cfg_path(cfg, "paths.artifacts_dir")
    dcfg = cfg["dataset"]
    vcfg = dcfg["validation"]
    log.info("images_dir=%s", images_dir)
    log.info("labels_dir=%s", labels_dir)
    log.info("class ids: fruto=%s signo=%s", list(fruit_ids), sign_id)

    issues: list[Issue] = []
    if not images_dir.is_dir():
        raise DatasetValidationError(f"No existe el directorio de imagenes: {images_dir}")
    if not labels_dir.is_dir():
        raise DatasetValidationError(f"No existe el directorio de anotaciones: {labels_dir}")

    files = discover_images(images_dir, dcfg["image_extensions"])
    log.info("%d imagenes encontradas", len(files))
    if not files:raise DatasetValidationError("PENDIENTE_USUARIO: images/ no contiene imagenes admitidas; no se valida ni inventa un dataset vacio")

    # -- unicidad de image_id (insensible a mayusculas) ----------------------------------
    by_id: dict[str, list[Path]] = defaultdict(list)
    for p in files:
        by_id[p.stem.lower()].append(p)
    for key, paths in by_id.items():
        if len(paths) > 1:
            issues.append(Issue(paths[0].stem, ERROR, "duplicate_image_id",
                                f"image_id repetido (insensible a mayusculas): {[p.name for p in paths]}"))

    n_expected = int(dcfg["n_images_expected"])
    if len(files) != n_expected:
        sev = WARNING if args.allow_count_mismatch or not dcfg.get("enforce_expected_count", True) else ERROR
        issues.append(Issue("*", sev, "unexpected_image_count", f"{len(files)} imagenes; se esperaban {n_expected} (sec. 3.1)"))

    # -- por imagen ---------------------------------------------------------------------
    inventory: list[dict[str, Any]] = []
    all_boxes: list[dict[str, Any]] = []
    label_ext = str(dcfg.get("label_extension", ".txt"))
    for i, path in enumerate(files, start=1):
        image_id = path.stem
        row: dict[str, Any] = {"image_id": image_id, "image_file": path.name, "image_sha256": sha256_file(path),
                               "label_file": None, "label_sha256": None, "n_fruit": 0, "n_sign": 0}
        info, img_issues = _check_image(path, int(vcfg["min_image_side_px"]))
        issues += img_issues
        lab = label_path_for(image_id, labels_dir, label_ext)
        if not lab.is_file():
            issues.append(Issue(image_id, ERROR, "missing_label", f"No existe la anotacion {lab.name}"))
        else:
            row["label_file"] = lab.name
            row["label_sha256"] = sha256_text_file(lab)
            if info is not None:
                a_issues, boxes = _check_annotations(image_id, lab, info["image_w"], info["image_h"], fruit_ids, sign_id, vcfg)
                issues += a_issues
                all_boxes += boxes
                row["n_fruit"] = sum(b["kind"] == "fruit" for b in boxes)
                row["n_sign"] = sum(b["kind"] == "sign" for b in boxes)
                if not boxes and not any(x.severity == ERROR and x.image_id == image_id for x in a_issues):
                    issues.append(Issue(image_id, WARNING, "no_annotations", "la imagen no tiene ninguna caja (fondo puro)"))
                elif row["n_fruit"] == 0 and not any(x.severity == ERROR and x.image_id == image_id for x in a_issues):
                    issues.append(Issue(image_id, WARNING, "no_fruit_boxes", "la imagen no tiene cajas de fruto"))
        row.update(info or {"image_w": None, "image_h": None, "mode": None, "format": None, "exif_orientation": None, "dhash": None})
        inventory.append(row)
        if i % 50 == 0 or i == len(files):
            log.info("procesadas %d/%d imagenes", i, len(files))

    # -- etiquetas huerfanas -------------------------------------------------------------
    image_stems = {p.stem for p in files}
    for lp in sorted(labels_dir.glob(f"*{label_ext}")):
        if lp.stem not in image_stems and lp.name.lower() != "classes.txt":
            issues.append(Issue(lp.stem, WARNING, "orphan_label", f"etiqueta sin imagen: {lp.name}"))

    # -- duplicados por contenido y casi-duplicados --------------------------------------
    inv = pd.DataFrame(inventory)
    dup = inv[inv["image_sha256"].duplicated(keep=False)].sort_values("image_sha256")
    for _, grp in dup.groupby("image_sha256"):
        ids = grp["image_id"].tolist()
        issues.append(Issue(ids[0], ERROR, "duplicate_image_content", f"imagenes con contenido identico (SHA-256): {ids}"))
    near = int(vcfg["near_duplicate_hamming"])
    n_near = 0
    if near >= 0:
        ok = inv.dropna(subset=["dhash"])
        ids, hashes = ok["image_id"].tolist(), [int(x, 16) for x in ok["dhash"]]
        exact_dups = set(dup["image_id"])
        for a in range(len(ids)):
            for b in range(a + 1, len(ids)):
                if ids[a] in exact_dups and ids[b] in exact_dups:
                    continue
                d = hamming_distance(hashes[a], hashes[b])
                if d <= near:
                    n_near += 1
                    issues.append(Issue(ids[a], WARNING, "near_duplicate", f"casi-duplicado de {ids[b]} (dHash Hamming={d}); riesgo de fuga si caen en splits distintos"))

    # -- hash del dataset ----------------------------------------------------------------
    from utils.dataset import compute_dataset_hash as current_hash
    from utils.semantics import semantic_fingerprint
    from utils.label_audit import audit_source, export_audit
    audit = audit_source(labels_dir.parent, cfg["pairing"]["min_iou"], labels_dir=labels_dir, images_dir=images_dir)
    export_audit(audit, artifacts / "label_audit")
    if audit[0]["anomalies_total"] and cfg["pairing"]["anomaly_policy"] == "error":
        issues.append(Issue("*", ERROR, "annotation_anomalies", f"{audit[0]['anomalies_total']} anomalies; inspect label_audit/"))
    dataset_hash, _ = current_hash(cfg)
    ctx.set_hashes(dataset_hash=dataset_hash)

    # -- resultados ----------------------------------------------------------------------
    n_err = sum(i.severity == ERROR for i in issues)
    n_warn = sum(i.severity == WARNING for i in issues)
    if vcfg.get("fail_on_warnings") and n_warn:
        n_err += n_warn
    by_code: dict[str, int] = defaultdict(int)
    for i in issues:
        by_code[f"{i.severity}:{i.code}"] += 1
    passed = n_err == 0

    inv_out = inv
    issues_df = pd.DataFrame([i.as_dict() for i in issues], columns=["image_id", "severity", "code", "line_no", "message"])
    write_csv(inv_out, artifacts / "dataset_inventory.csv")
    write_csv(issues_df, artifacts / "dataset_issues.csv")
    boxes_df = pd.DataFrame(all_boxes)
    n_fruit = int((boxes_df["kind"] == "fruit").sum()) if len(boxes_df) else 0
    n_sign = int((boxes_df["kind"] == "sign").sum()) if len(boxes_df) else 0
    report = {
        "n_labels_total": audit[0]["labels_total"], "n_labels_empty": audit[0]["labels_empty"],
        "all_label_boxes_by_class": audit[0]["boxes_by_class"],
        "images_without_label": audit[0]["images_without_label"],
        "labels_without_image": audit[0]["labels_without_image"],
        "pairing_valid_pairs_all_labels": audit[0]["valid_pairs"],
        "annotation_anomalies_all_labels": audit[0]["anomalies_total"],
        "n_suspicious_boxes_with_images": int((boxes_df["kind"] == "suspect").sum()) if len(boxes_df) else 0,
        "semantic_fingerprint": semantic_fingerprint(cfg),
        "passed": passed, "n_errors": n_err, "n_warnings": n_warn, "issues_by_code": dict(by_code),
        "dataset_hash": dataset_hash, "n_images": len(files), "n_images_expected": n_expected,
        "n_fruit_boxes": n_fruit, "n_sign_boxes": n_sign,
        "images_with_exif_rotation": int((inv["exif_orientation"].fillna(1) != 1).sum()),
        "n_near_duplicate_pairs": n_near,
        "class_ids": {"fruit": list(fruit_ids), "sign": sign_id},
        "fruits_per_image": describe_series(inv["n_fruit"]), "signs_per_image": describe_series(inv["n_sign"]),
        "images_dir": str(images_dir), "labels_dir": str(labels_dir),
        "generated_at": datetime.now(timezone.utc).isoformat(), "pc_id": ctx.pc_id,
    }
    write_json(artifacts / VALIDATION_REPORT, report)
    for f in ("dataset_inventory.csv", "dataset_issues.csv", VALIDATION_REPORT):
        ctx.register_artifact(artifacts / f, "validacion de dataset")
    ctx.log_metrics({k: v for k, v in report.items() if k not in ("images_dir", "labels_dir")})
    for issue in issues:
        if issue.severity == ERROR:
            log.error("%s [%s] %s%s", issue.image_id, issue.code, issue.message, f" (linea {issue.line_no})" if issue.line_no else "")
        else:
            log.warning("%s [%s] %s", issue.image_id, issue.code, issue.message)

    # -- figuras -------------------------------------------------------------------------
    if not args.no_figures:
        ctx.save_figure(plot_annotation_class_counts({"fruto": n_fruit, "signo": n_sign}, "Cajas anotadas por tipo"), "validacion_anotaciones_por_tipo")
        ctx.save_figure(plot_fruits_per_image(inv.rename(columns={"n_fruit": "n_fruits"}), "n_fruits"), "validacion_frutos_por_imagen")
        ctx.save_figure(plot_bbox_size_distribution(boxes_df), "validacion_tamanos_bbox")

    if not passed:
        raise DatasetValidationError(
            f"Validacion FALLIDA: {n_err} error(es), {n_warn} advertencia(s). Ver {artifacts / 'dataset_issues.csv'}. "
            f"Resumen: {dict(by_code)}"
        )
    log.info("Dataset VALIDO: %d imagenes, %d frutos, %d signos, %d advertencias. dataset_hash=%s", len(files), n_fruit, n_sign, n_warn, dataset_hash)


if __name__ == "__main__":
    if "--label-only" in sys.argv:
        from utils.label_audit import main as audit_main
        raise SystemExit(audit_main([a for a in sys.argv[1:] if a != "--label-only"]))
    raise SystemExit(run_cli(SCRIPT, main, description=__doc__ or SCRIPT, add_arguments=add_arguments))
