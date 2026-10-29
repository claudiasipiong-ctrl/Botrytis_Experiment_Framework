"""Figuras de entrenamiento de un run (loss y metricas por epoca) desde el ``history.csv`` REAL del run."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pandas as pd

from utils.architectures import ArchSpec
from utils.logging_utils import RunContext
from utils.visualization import plot_loss_curves, plot_metrics_per_epoch

DET_METRICS = ["map50_95", "map50", "precision", "recall", "f1"]
CLS_METRICS = ["val_ap", "val_roc_auc", "val_f1", "val_accuracy", "val_precision", "val_recall"]


def save_training_figures(ctx: RunContext, spec: ArchSpec, history: pd.DataFrame, prefix: str, source_csv: Path | None = None, *, to_reports: bool = False) -> None:
    """Guarda ``<prefix>_loss`` y ``<prefix>_metricas_por_epoca`` (curvas por fold: media +- SD).

    Las figuras de cada trial/baseline viven solo en ``runs/<arq>/<etapa>/figures`` (``to_reports=False``).
    """
    if history is None or history.empty:
        ctx.logger.warning("sin historial de entrenamiento: no se generan figuras de %s", prefix)
        return
    from utils.evidence import training_outputs
    training_outputs(ctx.cfg, spec, history, ctx.run_dir, prefix)
    sources: list[Any] = [source_csv] if source_csv is not None and Path(source_csv).is_file() else []
    hist = history.copy()
    if "train_loss" in hist and hist["train_loss"].notna().any():
        train_only = hist[[c for c in ("fold", "epoch", "train_loss") if c in hist]].copy()
        train_only["val_loss"] = float("nan")
        ctx.save_figure(plot_loss_curves(train_only, "train_loss", "val_loss", group_col="fold" if "fold" in hist.columns else None),
                        f"{prefix}_train_loss", data=train_only, sources=sources, copy_to_reports=to_reports)
    if "val_loss" in hist and hist["val_loss"].notna().any():
        hist = hist.copy()
        hist["train_loss"] = float("nan")
        ctx.save_figure(plot_loss_curves(hist, "train_loss", "val_loss", group_col="fold" if "fold" in hist.columns else None),
                        f"{prefix}_inner_validation_loss", data=hist[[c for c in ("fold", "epoch", "train_loss", "val_loss") if c in hist.columns]], sources=sources, copy_to_reports=to_reports)
    cols = DET_METRICS if spec.is_detector else CLS_METRICS
    present = [c for c in cols if c in hist.columns and hist[c].notna().any()]
    if present:
        ctx.save_figure(plot_metrics_per_epoch(hist, present, group_col="fold" if "fold" in hist.columns else None), f"{prefix}_metricas_por_epoca",
                        data=hist[[c for c in ["fold", "epoch", *present] if c in hist.columns]], sources=sources, copy_to_reports=to_reports)
