"""Genera TODAS las figuras de ``utils/visualization.py`` con datos SINTETICOS de demostracion.

Sirve para (a) verificar que cada funcion de graficos funciona y (b) ver el formato de los
DataFrames que esperan los futuros modulos de experimentacion (06-18).
Las figuras se guardan con el prefijo ``DEMO_`` y NO representan resultados reales.

Ejemplo::

    python tools/demo_figures.py --out tmp_demo_figures
    python tools/demo_figures.py --out tmp_demo_figures --images-dir tmp_synth/images --labels-dir tmp_synth/labels
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

import numpy as np
import pandas as pd
from matplotlib.figure import Figure

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from utils import visualization as viz  # noqa: E402
from utils.boxes import iou_matrix  # noqa: E402
from utils.metrics import match_detections, pipeline_confusion_matrix  # noqa: E402

ARCHS_DET = ["YOLOv8n", "YOLOv8m", "YOLOv8l", "YOLO26n", "YOLO26s", "YOLO26l", "RT-DETR-R50"]
ARCHS_CLS = ["ResNet50", "ViT-B/16", "Swin-Tiny"]


def _history(rng: np.random.Generator, folds: int = 3, epochs: int = 40) -> pd.DataFrame:
    rows = []
    for f in range(folds):
        for e in range(1, epochs + 1):
            k = 1 - np.exp(-e / 10)
            rows.append({
                "epoch": e, "fold": f,
                "train_loss": 1.2 * np.exp(-e / 12) + 0.05 + rng.normal(0, 0.01),
                "val_loss": 1.2 * np.exp(-e / 14) + 0.18 + 0.002 * max(0, e - 28) + rng.normal(0, 0.02),
                "train_map50_95": 0.75 * k + rng.normal(0, 0.01), "val_map50_95": 0.62 * k + rng.normal(0, 0.015),
                "val_precision": 0.85 * k + rng.normal(0, 0.01), "val_recall": 0.80 * k + rng.normal(0, 0.012),
                "val_f1": 0.82 * k + rng.normal(0, 0.01), "val_map50": 0.88 * k + rng.normal(0, 0.01),
            })
    return pd.DataFrame(rows)


def _trials(rng: np.random.Generator, n: int = 20) -> pd.DataFrame:
    rows = []
    for i in range(n):
        lr = 10 ** rng.uniform(-5, -2)
        wd = 10 ** rng.uniform(-6, -3)
        opt = rng.choice(["SGD", "AdamW"])
        warm = float(rng.uniform(0, 5))
        score = 0.55 + 0.12 * np.exp(-((np.log10(lr) + 3.2) ** 2)) + 0.03 * (opt == "AdamW") + rng.normal(0, 0.01)
        state = "PRUNED" if (i > 4 and score < 0.6 and rng.random() < 0.6) else "COMPLETE"
        rows.append({"number": i, "value": score, "state": state, "lr0": lr, "weight_decay": wd, "optimizer": opt, "warmup_epochs": warm})
    return pd.DataFrame(rows)


def generate(out: Path, images_dir: Path | None = None, labels_dir: Path | None = None, seed: int = 0) -> list[Path]:
    """Genera las figuras demo y devuelve sus rutas."""
    rng = np.random.default_rng(seed)
    out.mkdir(parents=True, exist_ok=True)
    paths: list[Path] = []

    def save(fig: Figure, name: str) -> None:
        paths.extend(viz.save_figure(fig, out, f"DEMO_{name}", formats=("png",), dpi=110))

    # ----- Experimentacion -----
    hist = _history(rng)
    save(viz.plot_learning_curves(hist, "train_map50_95", "val_map50_95", "mAP@0.5:0.95", group_col="fold"), "curvas_aprendizaje")
    save(viz.plot_loss_curves(hist, group_col="fold"), "loss_train_val")
    save(viz.plot_metrics_per_epoch(hist, ["val_precision", "val_recall", "val_f1", "val_map50", "val_map50_95", "val_loss"], group_col="fold"), "metricas_por_epoca")
    trials = _trials(rng)
    save(viz.plot_hpo_trials_vs_score(trials, ylabel="mAP@0.5:0.95 (media de folds)"), "hpo_trial_vs_score")
    winners = pd.DataFrame({
        "architecture": np.repeat(ARCHS_DET[:4], 3), "lr0": 10 ** rng.uniform(-4, -2.5, 12),
        "weight_decay": 10 ** rng.uniform(-5.5, -3, 12), "optimizer": rng.choice(["SGD", "AdamW"], 12),
        "warmup_epochs": rng.uniform(0, 5, 12), "momentum": rng.uniform(0.7, 0.98, 12),
    })
    save(viz.plot_winning_hyperparameter_distribution(winners), "hiperparametros_ganadores")
    bvh = pd.DataFrame({"architecture": ARCHS_DET, "baseline": rng.uniform(0.45, 0.6, 7)})
    bvh["hpo_mean"] = bvh["baseline"] + rng.uniform(-0.01, 0.08, 7)
    bvh["hpo_sd"] = rng.uniform(0.004, 0.02, 7)
    save(viz.plot_baseline_vs_hpo(bvh, ylabel="mAP@0.5:0.95"), "baseline_vs_hpo")
    save(viz.plot_hyperparameter_heatmap(trials, "lr0", "weight_decay", bins=4, cbar_label="mAP@0.5:0.95"), "heatmap_hiperparametros")
    save(viz.plot_hyperparameter_heatmap(trials, "lr0", "optimizer", bins=5), "heatmap_hiperparametros_cat")
    save(viz.plot_parallel_coordinates(trials, ["lr0", "weight_decay", "optimizer", "warmup_epochs"]), "parallel_coordinates")

    # ----- Comparacion -----
    names = ARCHS_DET + ARCHS_CLS
    comp = pd.DataFrame({
        "architecture": names, "latency_ms": rng.uniform(2, 40, len(names)), "params_m": np.exp(rng.uniform(np.log(3), np.log(90), len(names))),
        "gflops": np.exp(rng.uniform(np.log(8), np.log(260), len(names))), "metric": rng.uniform(0.5, 0.9, len(names)),
        "train_minutes": rng.uniform(10, 240, len(names)), "vram_mb": rng.uniform(2000, 30000, len(names)),
        "task": ["deteccion"] * 7 + ["clasificacion"] * 3,
    })
    save(viz.plot_accuracy_latency_pareto(comp, group_col="task", metric_label="metrica principal"), "pareto_precision_latencia")
    save(viz.plot_performance_efficiency_comparison(comp), "rendimiento_eficiencia")
    save(viz.plot_params_flops_metric(comp), "parametros_flops_metrica")
    save(viz.plot_training_time_by_architecture(comp), "tiempo_entrenamiento")
    save(viz.plot_vram_usage_by_model(comp, total_vram_mb=49140), "uso_vram")

    # ----- Pipeline -----
    save(viz.plot_hybrid_pipeline_diagram("YOLOv8m", "ResNet50"), "diagrama_pipeline_hibrido")
    n_det = rng.integers(1, 14, 120)
    lat = pd.DataFrame({"image_id": [f"IMG_{i:04d}" for i in range(120)], "n_detections": n_det,
                        "t_detector_ms": rng.normal(12, 1, 120), "t_crop_preprocess_ms": 0.4 * n_det + rng.normal(0.5, 0.1, 120),
                        "t_classifier_ms": 1.1 * n_det + rng.normal(0.8, 0.2, 120)})
    lat["t_total_ms"] = lat[["t_detector_ms", "t_crop_preprocess_ms", "t_classifier_ms"]].sum(axis=1)
    save(viz.plot_latency_breakdown(lat), "latencia_descompuesta")
    cm = pipeline_confusion_matrix(["con_signos"] * 20 + ["sin_signos"] * 60, ["con_signos"] * 16 + ["sin_signos"] * 4 + ["sin_signos"] * 55 + ["con_signos"] * 5,
                                   {"con_signos": 3, "sin_signos": 7}, ["sin_signos"] * 4 + ["con_signos"])
    save(viz.plot_pipeline_error_matrix(cm), "matriz_errores_pipeline")
    errors = pd.DataFrame({
        "x_norm": np.r_[rng.beta(2, 5, 25), rng.beta(5, 2, 20), rng.uniform(0, 1, 15)],
        "y_norm": np.r_[rng.beta(2, 2, 25), rng.beta(5, 3, 20), rng.uniform(0, 1, 15)],
        "error_type": ["FP"] * 25 + ["FN"] * 20 + ["EC"] * 15,
    })
    save(viz.plot_spatial_error_map(errors), "mapa_espacial_errores")

    # ----- Ejemplos con imagenes (TP/FP/FN con el matching real) -----
    if images_dir and labels_dir and images_dir.is_dir():
        from utils.boxes import yolo_to_xyxy  # noqa: WPS433
        from utils.io import get_image_size  # noqa: WPS433

        samples = []
        for img in sorted(images_dir.glob("*.jpg"))[:3]:
            w, h = get_image_size(img)
            gt = [yolo_to_xyxy(*map(float, ln.split()[1:]), w, h) for ln in (labels_dir / f"{img.stem}.txt").read_text().splitlines() if ln.split() and ln.split()[0] == "0"]
            gt_arr = np.array(gt).reshape(-1, 4)
            pred = gt_arr + rng.normal(0, 4, gt_arr.shape)
            pred = pred[rng.random(len(pred)) > 0.25]                      # algunos FN
            pred = np.vstack([pred, [[10, 10, 60, 60]]]) if len(pred) else pred.reshape(-1, 4)  # un FP
            res = match_detections(pred, rng.uniform(0.3, 1, len(pred)), gt_arr, iou_threshold=0.5)
            samples.append({"image_path": img, "title": img.stem, "tp": [tuple(pred[p]) for p, _, _ in res["matches"]],
                            "fp": [tuple(pred[i]) for i in res["fp"]], "fn": [tuple(gt_arr[i]) for i in res["fn"]]})
            assert iou_matrix(gt_arr, gt_arr).shape[0] == len(gt_arr)
        save(viz.plot_tp_fp_fn_examples(samples), "ejemplos_tp_fp_fn")
    return paths


def main() -> None:
    """CLI."""
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--out", required=True, type=Path)
    ap.add_argument("--images-dir", type=Path, default=None)
    ap.add_argument("--labels-dir", type=Path, default=None)
    a = ap.parse_args()
    allowed=(Path(__file__).resolve().parents[2] / "estimaciones_exploratorias").resolve()
    target=a.out.resolve()
    if not target.is_relative_to(allowed):
        ap.error("Salidas ilustrativas solo en Codigo/estimaciones_exploratorias; nunca reports oficiales")
    target.mkdir(parents=True,exist_ok=True)
    (target/"ESTIMADO_ILUSTRATIVO.md").write_text("ESTIMADO/ILUSTRATIVO: datos sinteticos aleatorios, sin resultados propios ni valor experimental. Supuestos en tools/demo_figures.py.\n",encoding="utf-8")
    for p in generate(target, a.images_dir, a.labels_dir):
        print(p)


if __name__ == "__main__":
    main()
