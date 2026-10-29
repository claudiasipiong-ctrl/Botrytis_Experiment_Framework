"""Tablas y figuras comparativas de la experimentacion, construidas SOLO con resultados reales de ``artifacts/``.

Usado por ``12_select_detector``/``13_select_classifier`` (comparacion en validation, antes del test) y por
``18_export_master_results`` (version final con latencia y VRAM de inferencia del benchmark).
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Mapping

import numpy as np
import pandas as pd

from utils.checks import MethodologyViolation
from utils.architectures import ArchSpec, archs
from utils.classifier_data import crops_for, load_crop_manifest
from utils.classifier_models import load_classifier
from utils.config_loader import cfg_path
from utils.gradcam import cam_for_image, overlay
from utils.hpo_results import final_checkpoint, predictions_path, stage_json_path
from utils.io import load_image_rgb, read_csv, read_json
from utils.logging_utils import RunContext
from utils.metrics import average_precision_np, binary_classification_report, precision_recall_curve_np, roc_auc_np, roc_curve_np
from utils.plots_experiment import (
    plot_accumulated_errors, plot_ap_vs_iou, plot_confusion_matrices, plot_detector_radar, plot_gradcam_grid, plot_metric_by_architecture, plot_metric_vs_cost,
    plot_pareto, plot_pr_curves, plot_recall_vs_latency, plot_roc_curves, plot_threshold_vs_metrics, plot_tp_fp_fn_counts,
)
from utils.training_common import resolve_device
from utils.visualization import plot_hybrid_pipeline_diagram, plot_latency_breakdown, plot_pipeline_error_matrix, plot_spatial_error_map, plot_tp_fp_fn_examples


def benchmark_dir(cfg: Mapping[str, Any]) -> Path:
    """``artifacts/benchmark``."""
    return cfg_path(cfg, "paths.artifacts_dir") / "benchmark"


def collect_validation_table(cfg: Mapping[str, Any], task: str) -> pd.DataFrame:
    """Una fila por arquitectura de ``task`` con sus metricas de validation externo y umbrales afinados.

    ``recall``/``precision``/``f1`` son los del PUNTO DE OPERACION afinado en el paso 11 (conf o umbral de
    probabilidad); las metricas independientes del umbral (mAP, AP) son las de la evaluacion de ``10``.
    """
    rows: list[dict[str, Any]] = []
    for spec in archs(task):
        p = stage_json_path(cfg, "outer_validation", spec)
        if not p.is_file():
            continue
        ov = read_json(p)
        from utils.semantics import semantic_fingerprint
        if ov.get("semantic_fingerprint") != semantic_fingerprint(cfg):
            raise MethodologyViolation("Validation obsoleta/no trazable: "+str(p))
        row: dict[str, Any] = {k: ov.get(k) for k in (
            "map50_95", "map50", "map75", "accuracy", "average_precision_positive", "roc_auc", "params_m", "gflops", "vram_train_peak_mb", "train_seconds",
            "robustness_mean", "robustness_sd", "baseline_score_mean", "config_id", "checkpoint_sha256", "f2")}
        row.update(architecture=spec.name, slug=spec.slug, task=task, precision_maxf1=ov.get("precision"), recall_maxf1=ov.get("recall"), f1_maxf1=ov.get("f1"),
                   dataset_hash=ov.get("dataset_hash"), split_hash=ov.get("split_hash"))
        tp = stage_json_path(cfg, "inference_thresholds", spec)
        if tp.is_file():
            th = read_json(tp)
            if th.get("semantic_fingerprint") != semantic_fingerprint(cfg):raise MethodologyViolation("Umbrales de otra semantica")
            row.update(precision=th.get("precision"), recall=th.get("recall"), f1=th.get("f1", th.get("value") if th.get("metric") == "f1" else None),
                       conf=th.get("conf"), nms_iou=th.get("nms_iou"), threshold=th.get("threshold"))
            if task == "detection" and th.get("precision") is not None and th.get("recall") is not None:
                p_, r_ = th["precision"], th["recall"]
                row["f1"] = (2 * p_ * r_ / (p_ + r_)) if (p_ + r_) else 0.0
        else:
            row.update(precision=ov.get("precision"), recall=ov.get("recall"), f1=ov.get("f1"))
        rows.append(row)
    return pd.DataFrame(rows)


def collect_ap_per_iou(cfg: Mapping[str, Any]) -> pd.DataFrame:
    """Tabla larga ``architecture, iou, ap`` (evaluacion oficial) de los detectores evaluados."""
    rows = []
    for spec in archs("detection"):
        p = stage_json_path(cfg, "outer_validation", spec)
        if p.is_file():
            aps = read_json(p).get("ap_per_iou") or []
            for i, ap in enumerate(aps):
                rows.append({"architecture": spec.name, "iou": round(0.5 + 0.05 * i, 2), "ap": ap})
    return pd.DataFrame(rows, columns=["architecture", "iou", "ap"])


def attach_benchmark(df: pd.DataFrame, cfg: Mapping[str, Any], task: str) -> pd.DataFrame:
    """Agrega ``latency_ms`` y ``vram_inference_mb`` del benchmark (si ``17_benchmark_latency.py`` ya corrio)."""
    p = benchmark_dir(cfg) / f"latency_{task}.csv"
    out = df.copy()
    if p.is_file() and not df.empty:
        b = read_csv(p)[["architecture", "latency_mean_ms", "latency_p95_ms", "vram_inference_mb"]].rename(columns={"latency_mean_ms": "latency_ms"})
        out = out.merge(b, on="architecture", how="left")
    return out


def save_detection_figures(ctx: RunContext, cfg: Mapping[str, Any], df: pd.DataFrame, *, sources: list[Path] | None = None) -> list[str]:
    """Figuras de comparacion de detectores (las que requieren latencia/VRAM solo si existe el benchmark)."""
    saved: list[str] = []
    src = sources or []
    d = df.copy()
    ap = collect_ap_per_iou(cfg)

    def put(fig: Any, name: str, data: pd.DataFrame) -> None:
        ctx.save_figure(fig, name, data=data, sources=src)
        saved.append(name)

    put(plot_metric_by_architecture(d, "map50_95", "mAP@0.5:0.95", "mAP@0.5:0.95 por arquitectura (validation externo)", ylim=(0, 1)), "det_map50_95_por_arquitectura", d[["architecture", "map50_95"]])
    put(plot_metric_by_architecture(d, "map50", "mAP@0.5", "mAP@0.5 por arquitectura (validation externo)", ylim=(0, 1)), "det_map50_por_arquitectura", d[["architecture", "map50"]])
    put(plot_detector_radar(d), "det_radar_detectores", d)
    put(plot_ap_vs_iou(ap), "det_ap_vs_iou", ap)
    put(plot_metric_vs_cost(d, "gflops", "FLOPs [G] (2 x MACs, batch 1)", title="mAP@0.5:0.95 vs FLOPs"), "det_map_vs_flops", d[["architecture", "gflops", "map50_95"]])
    put(plot_metric_vs_cost(d, "params_m", "Parametros [M]", title="mAP@0.5:0.95 vs parametros"), "det_map_vs_parametros", d[["architecture", "params_m", "map50_95"]])
    if "latency_ms" in d.columns and d["latency_ms"].notna().any():
        put(plot_recall_vs_latency(d), "det_recall_vs_latencia", d[["architecture", "latency_ms", "recall"]])
        put(plot_pareto(d, "latency_ms", "map50_95", "Latencia [ms / imagen]", "mAP@0.5:0.95", "Pareto mAP-latencia (BENCHMARK_PC)"), "det_pareto_map_latencia",
            d[["architecture", "latency_ms", "map50_95"]])
    if "vram_inference_mb" in d.columns and d["vram_inference_mb"].notna().any():
        put(plot_pareto(d, "vram_inference_mb", "map50_95", "VRAM pico de inferencia [MB]", "mAP@0.5:0.95", "Pareto mAP-VRAM (BENCHMARK_PC)"), "det_pareto_map_vram",
            d[["architecture", "vram_inference_mb", "map50_95"]])
    return saved


def _load_predictions(cfg: Mapping[str, Any], spec: ArchSpec) -> pd.DataFrame:
    return read_csv(predictions_path(cfg, spec))


def gradcam_samples(cfg: Mapping[str, Any], spec: ArchSpec, model: Any, meta: Mapping[str, Any], preds: pd.DataFrame, root: Path, n: int) -> tuple[list[dict[str, Any]], pd.DataFrame]:
    """Ejemplos Grad-CAM deterministas: TP mas confiables, TN mas claros, FN y FP mas graves (hasta ``n``)."""
    device = resolve_device(cfg)
    q = max(1, n // 4)
    pos = preds[preds["y_true"] == 1].sort_values("p_con_signos", ascending=False)
    neg = preds[preds["y_true"] == 0].sort_values("p_con_signos", ascending=True)
    picks = pd.concat([pos.head(q).assign(kind="TP"), neg.head(q).assign(kind="TN"), pos.tail(q).assign(kind="FN"),
                       neg.tail(q).assign(kind="FP")]).drop_duplicates("crop_path").head(n)
    threshold = float(read_json(stage_json_path(cfg, "inference_thresholds", spec)).get("threshold", .5)) if stage_json_path(cfg, "inference_thresholds", spec).is_file() else .5
    predicted = picks["p_con_signos"] >= threshold
    picks["kind"] = np.where(picks["y_true"] == 1, np.where(predicted, "TP", "FN"), np.where(predicted, "FP", "TN"))
    picks["classification_threshold"] = threshold
    out: list[dict[str, Any]] = []
    from utils.dataset import load_annotations
    from utils.boxes import crop_window
    from utils.spatial_audit import gradcam_sign_metrics
    from utils.classifier_data import load_crop_manifest
    _, annotations = load_annotations(cfg)
    manifest = load_crop_manifest(cfg).set_index("crop_path")
    spatial = []
    for r in picks.itertuples(index=False):
        img = load_image_rgb(root / r.crop_path)
        cam, p = cam_for_image(model, spec, meta["norm"], cfg, img, device)
        fruit = manifest.loc[r.crop_path]
        window = crop_window(tuple(fruit[k] for k in ("x1","y1","x2","y2")), int(fruit.image_w), int(fruit.image_h))
        signs = annotations[(annotations["image_id"]==fruit.image_id)&(annotations["class_id"]==2)]
        spatial.append({"crop_path":r.crop_path, **gradcam_sign_metrics(cam, signs[["x1","y1","x2","y2"]].to_numpy(float), window)})
        out.append({"crop": img, "overlay": overlay(img, cam), "title": f"{r.kind} | gt={'con' if r.y_true else 'sin'} | P={p:.2f}"})
    evidence = picks[["crop_path", "y_true", "p_con_signos", "kind", "classification_threshold"]].merge(pd.DataFrame(spatial),on="crop_path",how="left") if spatial else picks[["crop_path", "y_true", "p_con_signos", "kind", "classification_threshold"]]
    return out, evidence


def save_classification_figures(ctx: RunContext, cfg: Mapping[str, Any], df: pd.DataFrame, *, with_gradcam: bool = True, sources: list[Path] | None = None) -> list[str]:
    """Figuras de comparacion de clasificadores: PR, ROC, matriz de confusion, umbral vs F1, AP positiva y Grad-CAM."""
    saved: list[str] = []
    src = sources or []
    if df.empty:
        return saved
    pr: dict[str, Any] = {}
    roc: dict[str, Any] = {}
    cms: dict[str, Any] = {}
    sweeps: dict[str, pd.DataFrame] = {}
    chosen: dict[str, float] = {}
    prevalence = None
    samples: dict[str, list[dict[str, Any]]] = {}
    pick_rows: list[pd.DataFrame] = []
    root = cfg_path(cfg, "paths.classification_dir")
    for spec in archs("classification"):
        if spec.name not in set(df["architecture"]):
            continue
        preds = _load_predictions(cfg, spec)
        y, p = preds["y_true"].to_numpy(int), preds["p_con_signos"].to_numpy(float)
        pc, rc, _ = precision_recall_curve_np(y, p)
        fpr, tpr, _ = roc_curve_np(y, p)
        pr[spec.name] = {"precision": pc, "recall": rc, "ap": average_precision_np(y, p) or 0.0}
        roc[spec.name] = {"fpr": fpr, "tpr": tpr, "auc": roc_auc_np(y, p) or 0.0}
        thr = float(df.loc[df["architecture"] == spec.name, "threshold"].iloc[0]) if "threshold" in df.columns and pd.notna(df.loc[df["architecture"] == spec.name, "threshold"].iloc[0]) else 0.5
        cms[f"{spec.name} (umbral {thr:.2f})"] = binary_classification_report(y, p, threshold=thr)["confusion_matrix"]
        sw_path = stage_json_path(cfg, "inference_thresholds", spec).with_name(f"{spec.slug}_sweep.csv")
        if sw_path.is_file():
            sweeps[spec.name] = read_csv(sw_path)
            chosen[spec.name] = thr
        prevalence = float(np.mean(y))
        if with_gradcam:
            ck = final_checkpoint(cfg, spec)
            if ck.is_file():
                model, meta = load_classifier(ck, resolve_device(cfg))
                smp, picked = gradcam_samples(cfg, spec, model, meta, preds, root, int(cfg["figures"]["gradcam_examples"]))
                samples[spec.name] = smp
                pick_rows.append(picked.assign(architecture=spec.name))
    pr_data = pd.concat([pd.DataFrame({"architecture": k, "precision": v["precision"], "recall": v["recall"]}) for k, v in pr.items()]) if pr else pd.DataFrame()
    roc_data = pd.concat([pd.DataFrame({"architecture": k, "fpr": v["fpr"], "tpr": v["tpr"]}) for k, v in roc.items()]) if roc else pd.DataFrame()
    for fig, name, data in (
        (plot_pr_curves(pr, prevalence), "cls_curva_pr", pr_data), (plot_roc_curves(roc), "cls_curva_roc", roc_data),
        (plot_confusion_matrices(cms, title="Matriz de confusion en el umbral afinado (validation externo)"), "cls_matriz_confusion",
         pd.DataFrame([{"model": k, "tn": v[0][0], "fp": v[0][1], "fn": v[1][0], "tp": v[1][1]} for k, v in cms.items()])),
        (plot_threshold_vs_metrics(sweeps, chosen, metric="f1"), "cls_umbral_vs_f1", pd.concat([s.assign(architecture=k) for k, s in sweeps.items()]) if sweeps else pd.DataFrame()),
        (plot_metric_by_architecture(df, "average_precision_positive", "AP de con_signos (PR-AUC)", "AP de la clase positiva por arquitectura (validation externo)", ylim=(0, 1)),
         "cls_ap_positiva_por_arquitectura", df[["architecture", "average_precision_positive"]]),
    ):
        ctx.save_figure(fig, name, data=data, sources=src)
        saved.append(name)
    if with_gradcam and samples:
        ctx.save_figure(plot_gradcam_grid(samples, ncols=2 * max(len(v) for v in samples.values())), "cls_gradcam",
                        data=pd.concat(pick_rows) if pick_rows else pd.DataFrame(), sources=src)
        saved.append("cls_gradcam")
    return saved


def save_pipeline_figures(ctx: RunContext, res: Any, det_name: str, cls_name: str, *, suffix: str, sources: list[Path], n_examples: int) -> list[str]:
    """Figuras del pipeline desde un ``PipelineResult`` real (sufijo ``validacion`` o ``test``).

    Arquitectura completa, errores acumulados, TP/FP/FN (conteos y ejemplos), matriz de errores, mapa
    espacial de errores y latencia por etapa medida en esta evaluacion (no es el benchmark controlado).
    """
    s = res.summary
    names: list[str] = []
    rows_err = [{"label": suffix, "e_det": s["errors"]["e_det"], "e_clas": s["errors"]["e_clas"], "e_flujo": s["errors"]["e_flujo"], "n_gt": s["n_gt"], "n_ec": s["errors"]["n_ec"]}]
    counts = [{"label": suffix, "tp": s["tp"], "fp": s["fp"], "fn": s["fn"]}]
    ex = res.examples[:n_examples]
    for fig, name, data in (
        (plot_hybrid_pipeline_diagram(det_name, cls_name), f"pipe_arquitectura_completa_{suffix}", pd.DataFrame([{"detector": det_name, "classifier": cls_name}])),
        (plot_accumulated_errors(rows_err), f"pipe_errores_acumulados_{suffix}", pd.DataFrame(rows_err)),
        (plot_tp_fp_fn_counts(counts), f"pipe_tp_fp_fn_conteos_{suffix}", pd.DataFrame(counts)),
        (plot_tp_fp_fn_examples(ex), f"pipe_tp_fp_fn_{suffix}", pd.DataFrame([{"image": e["title"], "tp": len(e["tp"]), "fp": len(e["fp"]), "fn": len(e["fn"])} for e in ex])),
        (plot_pipeline_error_matrix(res.confusion), f"pipe_matriz_errores_{suffix}", pd.DataFrame(res.confusion)),
        (plot_spatial_error_map(res.errors), f"pipe_mapa_espacial_errores_{suffix}", res.errors),
        (plot_latency_breakdown(res.per_image), f"pipe_latencia_por_etapa_{suffix}", res.per_image),
    ):
        ctx.save_figure(fig, name, data=data, sources=sources)
        names.append(name)
    return names


def load_crops_table(cfg: Mapping[str, Any], split: str, stage: str) -> pd.DataFrame:
    """Crops GT de un split con guardia de acceso (atajo para reportes)."""
    return crops_for(load_crop_manifest(cfg), split=split, stage=stage, cfg=cfg)
