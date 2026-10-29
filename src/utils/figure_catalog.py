"""Catalogo de los graficos OBLIGATORIOS (deteccion, clasificacion, pipeline) y quien los genera."""

from __future__ import annotations

from typing import Any, Mapping

from utils.architectures import archs

DET = "deteccion"
CLS = "clasificacion"
PIPE = "pipeline"
HPO = "hpo"


def required_figures(cfg: Mapping[str, Any] | None = None) -> list[dict[str, str]]:
    """Lista ``{'name', 'group', 'description', 'produced_by'}`` de figuras obligatorias."""
    f: list[dict[str, str]] = [
        {"name": "det_map50_95_por_arquitectura", "group": DET, "description": "mAP@0.5:0.95 vs arquitectura", "produced_by": "12, 18"},
        {"name": "det_map50_por_arquitectura", "group": DET, "description": "mAP@0.5 vs arquitectura", "produced_by": "12, 18"},
        {"name": "det_recall_vs_latencia", "group": DET, "description": "Recall vs latencia", "produced_by": "18 (requiere 17)"},
        {"name": "det_radar_detectores", "group": DET, "description": "Radar chart de detectores", "produced_by": "12, 18"},
        {"name": "det_ap_vs_iou", "group": DET, "description": "AP vs umbral de IoU", "produced_by": "12, 18"},
        {"name": "det_map_vs_flops", "group": DET, "description": "mAP vs FLOPs", "produced_by": "12, 18"},
        {"name": "det_map_vs_parametros", "group": DET, "description": "mAP vs parametros", "produced_by": "12, 18"},
        {"name": "det_pareto_map_latencia", "group": DET, "description": "Pareto mAP-latencia", "produced_by": "18 (requiere 17)"},
        {"name": "det_pareto_map_vram", "group": DET, "description": "Pareto mAP-VRAM", "produced_by": "18 (requiere 17)"},
        {"name": "cls_curva_pr", "group": CLS, "description": "Curva PR", "produced_by": "13"},
        {"name": "cls_curva_roc", "group": CLS, "description": "Curva ROC", "produced_by": "13"},
        {"name": "cls_matriz_confusion", "group": CLS, "description": "Matriz de confusion", "produced_by": "13"},
        {"name": "cls_gradcam", "group": CLS, "description": "Grad-CAM", "produced_by": "13"},
        {"name": "cls_umbral_vs_f1", "group": CLS, "description": "Umbral vs F1", "produced_by": "13"},
        {"name": "cls_ap_positiva_por_arquitectura", "group": CLS, "description": "AP de la clase positiva por arquitectura", "produced_by": "13, 18"},
    ]
    for suffix in ("validacion", "test"):
        f += [
            {"name": f"pipe_arquitectura_completa_{suffix}", "group": PIPE, "description": f"Arquitectura completa del pipeline ({suffix})", "produced_by": "14, 16"},
            {"name": f"pipe_errores_acumulados_{suffix}", "group": PIPE, "description": f"Errores acumulados ({suffix})", "produced_by": "14, 16"},
            {"name": f"pipe_tp_fp_fn_{suffix}", "group": PIPE, "description": f"Ejemplos TP/FP/FN ({suffix})", "produced_by": "14, 16"},
            {"name": f"pipe_tp_fp_fn_conteos_{suffix}", "group": PIPE, "description": f"Conteos TP/FP/FN ({suffix})", "produced_by": "14, 16"},
        ]
    f.append({"name": "pipe_latencia_por_etapa", "group": PIPE, "description": "Latencia por etapa del pipeline congelado", "produced_by": "17"})
    for spec in archs():
        f.append({"name": f"hpo_curvas_{spec.slug}", "group": HPO, "description": f"Curvas HPO {spec.name}", "produced_by": "07, 18"})
        f.append({"name": f"hpo_distribuciones_{spec.slug}", "group": HPO, "description": f"Distribuciones Optuna {spec.name}", "produced_by": "07, 18"})
    return f


def complete_catalog():
    """All 60 scientific requests plus aliases, with pending/source contracts."""
    from utils.figures_all import catalog
    return catalog()
