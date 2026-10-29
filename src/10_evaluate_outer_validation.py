"""10 - Evaluacion de UNA arquitectura entrenada con el 70 % sobre el validation externo 20 % (sec. 18, 20).

Detectores (``--architecture YOLOv8n``...): mAP@0.5:0.95, mAP@0.5, mAP@0.75, precision, recall, F1 (punto de
F1 maximo), AP por umbral de IoU (0.50:0.05:0.95), parametros y FLOPs. La evaluacion OFICIAL es la de
Ultralytics (la misma metrica del objetivo del HPO); se recalcula ademas de forma independiente (mAP con
interpolacion COCO de 101 puntos sobre las predicciones crudas guardadas en ``det_val_predictions.csv``).

Clasificadores: Average Precision de ``con_signos``, ROC-AUC, accuracy, precision, recall, F1, matriz de
confusion (umbral inicial 0.5; el umbral definitivo se ajusta en el paso 11), curvas PR/ROC y Grad-CAM.
Se evalua sobre GT crops del validation externo.

Escribe ``artifacts/outer_validation/<slug>.json`` (insumo de 11, 12 y 13) y deja su run
``runs/<arquitectura>/outer_validation/``. El test sigue BLOQUEADO.

Ejemplo::

    python src/10_evaluate_outer_validation.py --architecture Swin-Tiny
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
from utils.classifier_data import CropDataset, build_transforms, crops_for, load_crop_manifest, make_loader  # noqa: E402
from utils.classifier_models import load_classifier  # noqa: E402
from utils.classifier_train import predict_probs  # noqa: E402
from utils.config_loader import cfg_path  # noqa: E402
from utils.detector_data import write_outer_yaml  # noqa: E402
from utils.detector_train import arch_stats  # noqa: E402
from utils.guards import GpuJobLock  # noqa: E402
from utils.hashing import sha256_file  # noqa: E402
from utils.hpo_results import final_checkpoint, load_best_config, stage_json_path  # noqa: E402
from utils.inference import load_detector, predict_paths  # noqa: E402
from utils.io import read_json, write_csv, write_json  # noqa: E402
from utils.logging_utils import RunContext, run_cli  # noqa: E402
from utils.metrics import average_precision_np, binary_classification_report, map_over_iou, precision_recall_curve_np, roc_curve_np  # noqa: E402
from utils.metrics_detection import ap_over_iou_table  # noqa: E402
from utils.pipeline_eval import gt_from_fruit_table  # noqa: E402
from utils.plots_experiment import plot_ap_vs_iou, plot_confusion_matrices, plot_gradcam_grid, plot_pr_curves, plot_roc_curves  # noqa: E402
from utils.reporting import gradcam_samples  # noqa: E402
from utils.run_common import add_arch_arguments, arch_from_args, guard_not_frozen, layout_for, prepare_for_inference  # noqa: E402
from utils.training_common import resolve_device, ultralytics_device  # noqa: E402

SCRIPT = "10_evaluate_outer_validation"


def add_arguments(parser: argparse.ArgumentParser) -> None:
    """Argumentos propios del script."""
    add_arch_arguments(parser)
    parser.add_argument("--no-gradcam", action="store_true", help="omite Grad-CAM (clasificadores)")


def _common_payload(ctx: RunContext, spec: ArchSpec, cfg: dict[str, Any], best: dict[str, Any], ckpt: Path) -> dict[str, Any]:
    final_meta = read_json(ckpt.with_name("final_meta.json")) if ckpt.with_name("final_meta.json").is_file() else {}
    return {
        "architecture": spec.name, "slug": spec.slug, "task": spec.task, "config_id": best["config_id"], "hp": best["hp"], "final_epochs": best["final_epochs"],
        "robustness_mean": best["score_mean"], "robustness_sd": best["score_sd"], "baseline_score_mean": best.get("baseline_score_mean"),
        "checkpoint": str(ckpt), "checkpoint_sha256": sha256_file(ckpt), "vram_train_peak_mb": final_meta.get("vram_peak_mb"),
        "train_seconds": final_meta.get("train_seconds"), "dataset_hash": ctx.hashes.get("dataset_hash"), "split_hash": ctx.hashes.get("split_hash"),
        "code_hash": ctx.hashes.get("code_hash"), "environment_hash": ctx.env.get("environment_hash"), "pc_id": ctx.pc_id, "evaluated_on": "outer_validation_20pct",
    }


def evaluate_detector(ctx: RunContext, spec: ArchSpec, data: Any, best: dict[str, Any], ckpt: Path, *, split: str = "validation", export_evidence: bool = True) -> dict[str, Any]:
    """Evaluacion oficial (Ultralytics) + independiente (predicciones crudas) de un detector."""
    cfg, log = ctx.cfg, ctx.logger
    model = load_detector(ckpt, spec)
    yml = write_outer_yaml(cfg, ctx.run_dir, val_split=split, stage="final_test" if split == "test" else "outer_validation")
    d = cfg["detector"]
    half = str(d["precision"]).lower() == "fp16" and resolve_device(cfg) != "cpu"
    res = model.val(data=str(yml), split="val", imgsz=int(d["image_size"]), batch=int(d["batch_size"]), conf=float(d["val_conf"]), iou=float(d["val_nms_iou"]),
                    half=half, device=ultralytics_device(cfg), plots=False, verbose=False, project=str(ctx.run_dir), name=f"ultralytics_{split}", exist_ok=True)
    box = res.box
    ap_iou_official = [float(x) for x in box.all_ap[0]] if getattr(box.all_ap, "size", 0) else []
    official = {"map50_95": float(box.map), "map50": float(box.map50), "map75": float(box.map75), "precision_at_max_f1": float(box.mp), "recall_at_max_f1": float(box.mr),
                "f1_at_max_f1": float(box.f1[0]) if getattr(box.f1, "size", 0) else None, "ap_per_iou": ap_iou_official}

    val_ids = sorted(data.manifest.loc[data.manifest["outer_split"] == split, "image_id"])
    paths = {r.image_id: Path(r.image_path) for r in data.images.itertuples(index=False) if r.image_id in set(val_ids)}
    gt = gt_from_fruit_table(data.fruits, val_ids)
    preds = predict_paths(model, [paths[i] for i in val_ids], conf=float(d["val_conf"]), iou=float(d["val_nms_iou"]), cfg=cfg)
    gts = [gt[i]["boxes"] for i in val_ids]
    indep = map_over_iou(preds, gts)
    ap_tab = ap_over_iou_table(preds, gts)
    rows = [{"image_id": iid, "x1": b[0], "y1": b[1], "x2": b[2], "y2": b[3], "score": float(s)} for iid, (bx, sc) in zip(val_ids, preds) for b, s in zip(bx, sc)]
    pred_df = pd.DataFrame(rows, columns=["image_id", "x1", "y1", "x2", "y2", "score"])
    write_csv(pred_df, ctx.run_dir / f"det_{split}_predictions.csv", float_format="%.6f")
    shared_preds = stage_json_path(cfg, "outer_validation", spec).with_name(f"{spec.slug}_predictions.csv") if split == "validation" else ctx.run_dir / f"{spec.slug}_{split}_predictions.csv"
    write_csv(pred_df, shared_preds, float_format="%.6f")  # copia estable para 11/12/18 (el run puede moverse por ZIP)
    write_csv(ap_tab.assign(architecture=spec.name), ctx.run_dir / "ap_per_iou.csv", float_format="%.8g")
    log.info("%s %s | Ultralytics mAP50-95=%.4f mAP50=%.4f P=%.3f R=%.3f | independiente mAP50-95=%.4f mAP50=%.4f (dif=%.4f)", split.upper(), spec.name,
             official["map50_95"], official["map50"], official["precision_at_max_f1"], official["recall_at_max_f1"], indep["map50_95"], indep["map50"],
             abs(official["map50_95"] - indep["map50_95"]))
    stats = arch_stats(spec.slug, spec.family, int(d["image_size"]))
    ctx.save_figure(plot_ap_vs_iou(ap_tab.assign(architecture=spec.name)), f"det_ap_vs_iou_{spec.slug}", data=ap_tab, sources=[ctx.run_dir / "ap_per_iou.csv"], copy_to_reports=False)
    from utils.evidence import detection_evidence, finish_evidence
    from utils.evidence import split_dir
    evidence = detection_evidence(cfg, spec, split, data.manifest, paths, gt, preds) if export_evidence else {"root": str(split_dir(cfg, spec, split)), "n_images": len(val_ids)}
    components = {k: float(v) for k, v in getattr(res, "results_dict", {}).items() if "loss" in k}
    payload = {
        **_common_payload(ctx, spec, cfg, best, ckpt), "map50_95": official["map50_95"], "map50": official["map50"], "map75": official["map75"],
        "precision": official["precision_at_max_f1"], "recall": official["recall_at_max_f1"], "f1": official["f1_at_max_f1"], "ap_per_iou": ap_iou_official,
        "official_metric_source": "ultralytics_val", "independent": {"map50_95": indep["map50_95"], "map50": indep["map50"], "ap_per_iou": ap_tab["ap"].tolist(),
                                                                      "abs_diff_map50_95": abs(official["map50_95"] - indep["map50_95"]), "interpolation": "COCO-101"},
        "params_m": stats["params_m"], "gflops": stats["gflops"], "n_val_images": len(val_ids), "n_val_boxes": int(sum(len(g) for g in gts)),
        "predictions_file": str(shared_preds), "predictions_sha256": sha256_file(shared_preds),
    }
    payload.update(split=split, evaluated_on=split, evidence=evidence, loss=sum(components.values()) if components else None, loss_components=components)
    if split == "train":
        payload["n_train_images"] = payload.pop("n_val_images")
        payload["n_train_boxes"] = payload.pop("n_val_boxes")
        payload["evaluation_role"] = "post_training_resubstitution_no_selection"
    if split == "test":
        payload["n_test_images"] = payload.pop("n_val_images")
        payload["n_test_boxes"] = payload.pop("n_val_boxes")
    finish_evidence(payload)
    return payload


def evaluate_classifier(ctx: RunContext, spec: ArchSpec, data: Any, best: dict[str, Any], ckpt: Path, *, gradcam: bool, split: str = "validation", export_evidence: bool = True) -> dict[str, Any]:
    """Evaluacion de un clasificador sobre los GT crops del validation externo."""
    cfg, log = ctx.cfg, ctx.logger
    device = resolve_device(cfg)
    model, meta = load_classifier(ckpt, device)
    manifest = load_crop_manifest(cfg)
    va = crops_for(manifest, split=split, stage="final_test" if split == "test" else "outer_validation", cfg=cfg)
    root = cfg_path(cfg, "paths.classification_dir")
    norm = meta["norm"]
    ds = CropDataset(va, root, build_transforms(cfg, norm, train=False))
    loader = make_loader(ds, batch_size=64, shuffle=False, seed=0, workers=0)
    use_amp = device != "cpu" and str(cfg["classifier"]["precision"]).lower() == "fp16"
    prob, ytrue, idx, loss = predict_probs(model, loader, device, use_amp)
    preds = va.iloc[idx].assign(y_true=ytrue, p_con_signos=prob)[["crop_path", "image_id", "fruit_id", "y_true", "p_con_signos"]]
    write_csv(preds, ctx.run_dir / f"{split}_predictions.csv", float_format="%.8f")
    shared_preds = stage_json_path(cfg, "outer_validation", spec).with_name(f"{spec.slug}_predictions.csv") if split == "validation" else ctx.run_dir / f"{spec.slug}_{split}_predictions.csv"
    write_csv(preds, shared_preds, float_format="%.8f")  # copia estable para 11/13/18
    rep = binary_classification_report(ytrue, prob, threshold=0.5)
    write_json(ctx.run_dir / "classification_report.json", rep["classification_report"])
    error_preds = preds.assign(y_pred=(preds["p_con_signos"] >= .5).astype(int))
    error_preds = error_preds[error_preds["y_true"] != error_preds["y_pred"]].copy()
    error_preds["outcome"] = np.where(error_preds["y_true"] == 0, "FP", "FN")
    write_csv(error_preds, ctx.run_dir / "classification_fp_fn.csv")
    pcur, rcur, _ = precision_recall_curve_np(ytrue, prob)
    fpr, tpr, _ = roc_curve_np(ytrue, prob)
    write_csv(pd.DataFrame({"precision": pcur, "recall": rcur}), ctx.run_dir / "pr_curve.csv", float_format="%.8g")
    write_csv(pd.DataFrame({"fpr": fpr, "tpr": tpr}), ctx.run_dir / "roc_curve.csv", float_format="%.8g")
    ap = average_precision_np(ytrue, prob)
    log.info("%s %s | AP(con_signos)=%s ROC-AUC=%s acc@0.5=%.3f P=%.3f R=%.3f F1=%.3f (n=%d, positivos=%d)", split.upper(), spec.name, f"{ap:.4f}" if ap is not None else "n/a",
             f"{rep['roc_auc']:.4f}" if rep.get("roc_auc") is not None else "n/a", rep["accuracy"], rep["precision"], rep["recall"], rep["f1"], rep["n"], rep["n_positive"])
    prev = float(np.mean(ytrue)) if len(ytrue) else None
    src = [ctx.run_dir / f"{split}_predictions.csv"]
    ctx.save_figure(plot_pr_curves({spec.name: {"precision": pcur, "recall": rcur, "ap": ap or 0.0}}, prev), f"cls_curva_pr_{spec.slug}",
                    data=pd.DataFrame({"precision": pcur, "recall": rcur}), sources=src, copy_to_reports=False)
    ctx.save_figure(plot_roc_curves({spec.name: {"fpr": fpr, "tpr": tpr, "auc": rep.get("roc_auc") or 0.0}}), f"cls_curva_roc_{spec.slug}",
                    data=pd.DataFrame({"fpr": fpr, "tpr": tpr}), sources=src, copy_to_reports=False)
    ctx.save_figure(plot_confusion_matrices({spec.name: rep["confusion_matrix"]}), f"cls_matriz_confusion_{spec.slug}",
                    data=pd.DataFrame(rep["confusion_matrix"], columns=["pred_sin", "pred_con"]), sources=src, copy_to_reports=False)
    if gradcam and len(preds):
        samples, picked = gradcam_samples(cfg, spec, model, meta, preds, root, int(cfg["figures"]["gradcam_examples"]))
        ctx.save_figure(plot_gradcam_grid({spec.name: samples}, ncols=2 * max(1, len(samples))), f"cls_gradcam_{spec.slug}", data=picked, sources=src, copy_to_reports=False)
    from utils.evidence import classification_evidence, finish_evidence
    from utils.evidence import split_dir
    evidence = classification_evidence(cfg, spec, split, data.manifest, manifest, preds, root) if export_evidence else {"root": str(split_dir(cfg, spec, split)), "n_crops": len(preds)}
    payload = {
        **_common_payload(ctx, spec, cfg, best, ckpt), "average_precision_positive": ap, "roc_auc": rep.get("roc_auc"), "accuracy": rep["accuracy"], "precision": rep["precision"],
        "recall": rep["recall"], "f1": rep["f1"], "f2": rep["f2"], "confusion_matrix": rep["confusion_matrix"], "threshold_initial": 0.5, "val_loss": loss, "n_val": rep["n"],
        "n_val_positive": rep["n_positive"], "params_m": _cls_stats(spec, meta)["params_m"], "gflops": _cls_stats(spec, meta)["gflops"],
        "predictions_file": str(shared_preds), "predictions_sha256": sha256_file(shared_preds), "official_metric_source": "own_metrics_on_saved_predictions",
    }
    payload.update(split=split, evaluated_on=split, evidence=evidence, loss=loss)
    if split == "train":
        payload["train_eval_loss"] = payload.pop("val_loss")
        payload["n_train"] = payload.pop("n_val")
        payload["n_train_positive"] = payload.pop("n_val_positive")
        payload["evaluation_role"] = "post_training_resubstitution_no_augmentation_no_selection"
    if split == "test":
        payload["test_loss"] = payload.pop("val_loss")
        payload["n_test"] = payload.pop("n_val")
        payload["n_test_positive"] = payload.pop("n_val_positive")
    finish_evidence(payload)
    return payload


def _cls_stats(spec: ArchSpec, meta: dict[str, Any]) -> dict[str, float]:
    from utils.classifier_models import build_classifier
    from utils.model_stats import classifier_stats

    m, _ = build_classifier(spec, meta.get("hp"), pretrained=False)
    return classifier_stats(m, int(meta.get("image_size", 224)))


def main(ctx: RunContext, args: argparse.Namespace) -> None:
    """Evalua el checkpoint definitivo de la arquitectura sobre el validation externo."""
    cfg = ctx.cfg
    spec = arch_from_args(args)
    data = prepare_for_inference(ctx, args)
    if cfg['project']['profile'] == 'official':
        from utils.experiments import use_external
        use_external(cfg,'outer_validation',final_checkpoint(cfg,spec))
    guard_not_frozen(ctx, "10_evaluate_outer_validation")
    best = load_best_config(cfg, spec)
    ckpt = final_checkpoint(cfg, spec)
    if not ckpt.is_file():
        raise FileNotFoundError(f"No existe {ckpt}: ejecute 09_train_outer_candidate.py para {spec.name}")
    ctx.note_split_access("validation", "outer_validation")
    ctx.add_metadata(architecture=spec.name, stage="outer_validation", config_id=best["config_id"])
    with GpuJobLock(cfg, ctx.pc_id, job=f"outer_eval:{spec.slug}"):
        payload = evaluate_detector(ctx, spec, data, best, ckpt) if spec.is_detector else evaluate_classifier(ctx, spec, data, best, ckpt, gradcam=not args.no_gradcam)
    out = stage_json_path(cfg, "outer_validation", spec)
    write_json(out, payload)
    from utils.split_summary import export_split_summary
    export_split_summary(cfg, data)
    ctx.register_artifact(out, "metricas de validation externo")
    ctx.log_metrics(payload)
    ctx.set_checkpoint(ckpt, role="evaluated", config_id=best["config_id"])


if __name__ == "__main__":
    raise SystemExit(run_cli(SCRIPT, main, description=__doc__ or SCRIPT, add_arguments=add_arguments, run_layout=layout_for("outer_validation")))
