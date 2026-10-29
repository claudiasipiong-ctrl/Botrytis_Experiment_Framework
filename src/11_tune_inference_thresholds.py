"""11 - Ajuste de parametros de INFERENCIA con el validation externo (Especificacion s2, sec. 21).

* Detector: ``confidence threshold`` y ``NMS IoU`` (este solo si el modelo usa NMS: YOLOv8; YOLO26 y RT-DETR
  no lo necesitan). Se maximiza la metrica predeclarada (``inference_tuning.detector_threshold_metric``,
  F1) con matching uno-a-uno por IoU (``matching.threshold``) sobre el validation externo.
* Clasificador: ``probability_threshold`` de ``P(con_signos)`` que maximiza la metrica predeclarada
  (``inference_tuning.classifier_threshold_metric``: F1 o F2) sobre los GT crops del validation externo.

Estos umbrales son INDEPENDIENTES de el pairing GT de anotaciones y del HPO. Usa SOLO
validation externo; el test sigue BLOQUEADO. Salida: ``artifacts/inference_thresholds/<slug>.json`` (+ barridos CSV).

Ejemplo::

    python src/11_tune_inference_thresholds.py --architecture YOLOv8n
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
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parent))

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

from utils.architectures import ArchSpec  # noqa: E402
from utils.checks import MethodologyViolation  # noqa: E402
from utils.guards import GpuJobLock  # noqa: E402
from utils.hpo_results import final_checkpoint, predictions_path, stage_json_path  # noqa: E402
from utils.inference import load_detector, predict_paths  # noqa: E402
from utils.io import read_csv, read_json, write_csv, write_json  # noqa: E402
from utils.logging_utils import RunContext, run_cli  # noqa: E402
from utils.metrics import fbeta_score  # noqa: E402
from utils.metrics_classification import pick_threshold, threshold_sweep  # noqa: E402
from utils.metrics_detection import conf_sweep  # noqa: E402
from utils.pipeline_eval import gt_from_fruit_table  # noqa: E402
from utils.plots_experiment import plot_threshold_vs_metrics  # noqa: E402
from utils.run_common import add_arch_arguments, arch_from_args, guard_not_frozen, layout_for, prepare_for_inference  # noqa: E402

SCRIPT = "11_tune_inference_thresholds"


def add_arguments(parser: argparse.ArgumentParser) -> None:
    """Argumentos propios del script."""
    add_arch_arguments(parser)


def tune_detector(ctx: RunContext, spec: ArchSpec, data: Any) -> dict[str, Any]:
    """Barrido conf x NMS de un detector sobre el validation externo."""
    cfg, log = ctx.cfg, ctx.logger
    it = cfg["inference_tuning"]
    metric = str(it["detector_threshold_metric"])
    if metric not in ("f1", "f2"):
        raise MethodologyViolation("inference_tuning.detector_threshold_metric debe ser f1 o f2")
    ckpt = final_checkpoint(cfg, spec)
    model = load_detector(ckpt, spec)
    val_ids = sorted(data.manifest.loc[data.manifest["outer_split"] == "validation", "image_id"])
    paths = {r.image_id: Path(r.image_path) for r in data.images.itertuples(index=False) if r.image_id in set(val_ids)}
    gt = gt_from_fruit_table(data.fruits, val_ids)
    gts = [gt[i]["boxes"] for i in val_ids]
    confs = [float(c) for c in it["detector_conf_grid"]]
    nms_applies = spec.family == "yolov8"
    nms_grid = [float(v) for v in it["detector_nms_iou_grid"]] if nms_applies else [float(cfg["detector"]["val_nms_iou"])]
    iou_thr = float(cfg["matching"]["threshold"])
    all_sweeps: list[pd.DataFrame] = []
    for nms in nms_grid:
        preds = predict_paths(model, [paths[i] for i in val_ids], conf=min(confs), iou=nms, cfg=cfg)
        sw = conf_sweep(preds, gts, confs, iou_thr)
        if metric == "f2":
            sw["f2"] = [fbeta_score(p, r, 2.0) for p, r in zip(sw["precision"], sw["recall"])]
        all_sweeps.append(sw.assign(nms_iou=nms))
    sweep = pd.concat(all_sweeps, ignore_index=True)
    best_val = float(sweep[metric].max())
    tied = sweep[np.isclose(sweep[metric], best_val, atol=1e-12)].sort_values(["conf", "nms_iou"], ascending=[False, True])  # empate: mayor conf, menor NMS
    row = tied.iloc[0]
    out_sweep = stage_json_path(cfg, "inference_thresholds", spec).with_name(f"{spec.slug}_sweep.csv")
    write_csv(sweep, out_sweep, float_format="%.8g")
    log.info("UMBRALES %s: conf=%.2f nms_iou=%s -> %s=%.4f (P=%.3f R=%.3f; TP=%d FP=%d FN=%d)", spec.name, row["conf"],
             f"{row['nms_iou']:.2f}" if nms_applies else "n/a (sin NMS)", metric.upper(), best_val, row["precision"], row["recall"], int(row["tp"]), int(row["fp"]), int(row["fn"]))
    show = sweep[sweep["nms_iou"] == row["nms_iou"]].rename(columns={"conf": "threshold"})
    ctx.save_figure(plot_threshold_vs_metrics({spec.name: show.assign(f1=show["f1"])}, {spec.name: float(row["conf"])}, metric="f1"), f"det_conf_vs_f1_{spec.slug}",
                    data=show, sources=[out_sweep], copy_to_reports=False)
    return {"type": "detector", "conf": float(row["conf"]), "nms_iou": float(row["nms_iou"]) if nms_applies else None, "nms_applies": nms_applies,
            "metric": metric, "value": best_val, "precision": float(row["precision"]), "recall": float(row["recall"]), "tp": int(row["tp"]), "fp": int(row["fp"]),
            "fn": int(row["fn"]), "match_iou": iou_thr, "conf_grid": confs, "nms_grid": nms_grid if nms_applies else None, "sweep_csv": str(out_sweep),
            "checkpoint_sha256": read_json(stage_json_path(cfg, "outer_validation", spec))["checkpoint_sha256"]}


def tune_classifier(ctx: RunContext, spec: ArchSpec) -> dict[str, Any]:
    """Barrido del umbral de probabilidad de un clasificador sobre los GT crops del validation externo."""
    cfg, log = ctx.cfg, ctx.logger
    metric = str(cfg["inference_tuning"]["classifier_threshold_metric"])
    ov = read_json(stage_json_path(cfg, "outer_validation", spec))
    preds = read_csv(predictions_path(cfg, spec))
    sweep = threshold_sweep(preds["y_true"], preds["p_con_signos"], n_points=int(cfg["inference_tuning"]["classifier_threshold_grid_points"]))
    thr, val = pick_threshold(sweep, metric)
    row = sweep.iloc[(sweep["threshold"] - thr).abs().argmin()]
    out_sweep = stage_json_path(cfg, "inference_thresholds", spec).with_name(f"{spec.slug}_sweep.csv")
    write_csv(sweep, out_sweep, float_format="%.8g")
    log.info("UMBRAL %s: P(con_signos) >= %.3f -> %s=%.4f (P=%.3f R=%.3f)", spec.name, thr, metric.upper(), val, row["precision"], row["recall"])
    ctx.save_figure(plot_threshold_vs_metrics({spec.name: sweep}, {spec.name: thr}, metric="f1"), f"cls_umbral_vs_f1_{spec.slug}", data=sweep, sources=[out_sweep],
                    copy_to_reports=False)
    return {"type": "classifier", "threshold": float(thr), "metric": metric, "value": float(val), "precision": float(row["precision"]), "recall": float(row["recall"]),
            "f1": float(row["f1"]), "f2": float(row["f2"]), "sweep_csv": str(out_sweep), "n_points": int(len(sweep)), "distinct_from_annotation_pairing": True,
            "checkpoint_sha256": ov["checkpoint_sha256"]}


def main(ctx: RunContext, args: argparse.Namespace) -> None:
    """Ajusta los umbrales de inferencia de la arquitectura con el validation externo."""
    cfg = ctx.cfg
    spec = arch_from_args(args)
    data = prepare_for_inference(ctx, args)
    guard_not_frozen(ctx, "11_tune_inference_thresholds")
    if not stage_json_path(cfg, "outer_validation", spec).is_file():
        raise FileNotFoundError(f"Falta la evaluacion de validation de {spec.name}: ejecute 10_evaluate_outer_validation.py")
    ctx.note_split_access("validation", "threshold_tuning")
    ctx.add_metadata(architecture=spec.name, stage="threshold_tuning")
    if spec.is_detector:
        with GpuJobLock(cfg, ctx.pc_id, job=f"thresholds:{spec.slug}"):
            payload = tune_detector(ctx, spec, data)
    else:
        payload = tune_classifier(ctx, spec)
    payload.update(architecture=spec.name, slug=spec.slug, evaluated_on="outer_validation_20pct", split_hash=ctx.hashes.get("split_hash"),
                   dataset_hash=ctx.hashes.get("dataset_hash"), code_hash=ctx.hashes.get("code_hash"), test_used=False)
    out = stage_json_path(cfg, "inference_thresholds", spec)
    write_json(out, payload)
    ctx.register_artifact(out, "umbrales de inferencia")
    ctx.log_metrics(payload)


if __name__ == "__main__":
    raise SystemExit(run_cli(SCRIPT, main, description=__doc__ or SCRIPT, add_arguments=add_arguments, run_layout=layout_for("thresholds")))
