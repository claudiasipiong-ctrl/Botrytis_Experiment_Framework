"""Nucleo comun de baseline (06), HPO (07) y robustez (08): evaluar UNA configuracion en los folds internos.

``evaluate_config`` entrena/valida la configuracion ``hp`` en cada fold interno (solo outer train) y
devuelve los scores por fold; escribe ``fold_metrics.csv`` y ``history.csv`` en el directorio del run. Un
``EpochReporter`` conecta cada epoca con Optuna (report + ``MedianPruner``) sin acoplar el entrenamiento.
"""

from __future__ import annotations

import shutil
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Mapping, Sequence

import numpy as np
import pandas as pd

from utils.checks import MethodologyViolation
from utils.architectures import ArchSpec
from utils.classifier_data import load_crop_manifest
from utils.classifier_train import load_fold_frames, train_classifier
from utils.detector_data import detection_images_table, write_fold_yaml
from utils.detector_train import train_detector
from utils.io import write_csv
from utils.logging_utils import RunContext
from utils.pretrained import weights_sha256
from utils.training_common import mean_sd


def objective_metric(spec: ArchSpec) -> str:
    """Nombre de la metrica objetivo: ``map50_95`` (detector) | ``average_precision_positive`` (clasificador)."""
    return "map50_95" if spec.is_detector else "average_precision_positive"


def intermediate_key(spec: ArchSpec) -> str:
    """Columna del historial por epoca que se reporta al pruner."""
    return "map50_95" if spec.is_detector else "val_ap"


@dataclass
class ConfigEvaluation:
    """Resultado de evaluar una configuracion en los folds."""

    fold_scores: list[float]
    mean: float
    sd: float | None
    folds: list[dict[str, Any]]
    train_seconds: float
    vram_peak_mb: float | None
    params_m: float
    gflops: float
    history: pd.DataFrame
    pruned: bool = False


def make_reporter(trial: Any | None, spec: ArchSpec, max_epochs: int, fold_idx: int) -> Callable[[int, Mapping[str, Any]], None] | None:
    """Callback por epoca: reporta a Optuna con paso global ``fold * max_epochs + epoca`` y poda si toca."""
    if trial is None:
        return None
    key = intermediate_key(spec)

    def _cb(step: int, row: Mapping[str, Any]) -> None:
        import optuna

        value = row.get(key)
        if value is None or (isinstance(value, float) and np.isnan(value)):
            return
        trial.report(float(value), step)
        if trial.should_prune():
            trial.set_user_attr("pruned_at_step", int(step))
            raise optuna.TrialPruned(f"podado en el paso {step} (fold {fold_idx}) con {key}={float(value):.4f}")

    return _cb


def evaluate_config(
    spec: ArchSpec, hp: Mapping[str, Any], cfg: Mapping[str, Any], ctx: RunContext, run_dir: Path, *, folds: Sequence[int], seed: int,
    max_epochs: int, patience: int, trial: Any | None = None, step_fold_width: int | None = None,
) -> ConfigEvaluation:
    """Entrena ``hp`` en cada fold de ``folds`` y agrega los resultados.

    Args:
        step_fold_width: Anchura de pasos por fold para el reporte a Optuna (por defecto ``max_epochs``).
    """
    log = ctx.logger.info
    width = int(step_fold_width or max_epochs)
    record_training_source(ctx, spec, cfg)
    manifest = load_crop_manifest(cfg) if not spec.is_detector else None
    det_table = detection_images_table(cfg) if spec.is_detector else None
    rows: list[dict[str, Any]] = []
    hists: list[pd.DataFrame] = []
    t0 = time.perf_counter()
    vram: float | None = None
    params_m = gflops = float("nan")
    for i, fold in enumerate(folds):
        from utils.experiments import fingerprint
        from utils.hashing import sha256_json,sha256_file
        from utils.durable import atomic_json
        import json,copy
        signature=sha256_json(dict(experiment=fingerprint(cfg),architecture=spec.slug,hp=dict(hp),seed=seed,fold=fold,epochs=max_epochs,patience=patience))
        cache=run_dir/f'fold{fold}'/'completed_fold.json'
        cache_history=cache.with_name('completed_history.csv')
        if cache.exists():
            prior=json.loads(cache.read_text(encoding='utf-8'))
            if prior['signature']!=signature or not cache_history.is_file() or sha256_file(cache_history)!=prior['history_sha256']:
                raise MethodologyViolation('Fold previo incompatible/alterado; no se sobrescribe: '+str(cache))
            rows.append(prior['row']);hists.append(pd.read_csv(cache_history))
            params_m,gflops=prior['params_m'],prior['gflops']
            vram=max([v for v in (vram,prior['vram_peak_mb']) if v is not None],default=None)
            log('fold %d ya completo y verificado; se conserva sin entrenar',fold)
            continue
        ctx.note_split_access("train", "hpo_trial")
        log("=== %s | fold %d (%d/%d) | seed=%d | hp=%s", spec.name, fold, i + 1, len(folds), seed, dict(hp))
        cb = make_reporter(trial, spec, max_epochs, i)
        if spec.is_detector:
            yml, counts = write_fold_yaml(cfg, fold, run_dir / f"fold{fold}", table=det_table)
            run = train_detector(spec, hp, cfg, data_yaml=yml, project_dir=run_dir, name=f"fold{fold}", seed=seed, epochs=max_epochs,
                                 patience=patience, epoch_callback=cb, step_offset=i * width, log=log)
            ctx.record_checkpoint(f"fold{fold}_best", sha256=run.weights_sha256, fold=fold, weights_discarded=True, pretrained_sha256=run.pretrained_sha256)
            shutil.rmtree(run_dir / f"fold{fold}" / "weights", ignore_errors=True)
            row = {"fold": fold, "score": run.score, "best_epoch": run.best_epoch, "epochs_run": run.epochs_run, "stopped_early": run.stopped_early,
                   "n_train": counts["n_train"], "n_val": counts["n_val"], "train_seconds": run.train_seconds, "vram_peak_mb": run.vram_peak_mb,
                   "micro_batch": run.micro_batch, "map50": run.final_metrics.get("map50"), "precision": run.final_metrics.get("precision"),
                   "recall": run.final_metrics.get("recall"), "f1": run.final_metrics.get("f1")}
        else:
            assert manifest is not None
            tr, va = load_fold_frames(manifest, cfg, fold)
            if set(va["y"].unique()) != {0,1}:
                raise MethodologyViolation("Fold interno sin ambas clases: AP no comparable; revisar estratificacion/anotaciones antes de HPO")
            runtime=copy.deepcopy(cfg)
            runtime['classifier']['resume_output']=str(run_dir/f'fold{fold}'/'resume.pt')
            runtime['classifier']['_resume_context']=dict(fold=fold,seed=seed,architecture=spec.slug)
            checkpoint=runtime['classifier'].get('resume_checkpoint')
            if checkpoint:
                side=Path(checkpoint).with_suffix(Path(checkpoint).suffix+'.resume.json')
                if json.loads(side.read_text(encoding='utf-8')).get('extra',{}).get('context',{}).get('fold')!=fold:
                    runtime['classifier'].pop('resume_checkpoint',None)
            run = train_classifier(spec, hp, runtime, tr, va, seed=seed, max_epochs=max_epochs, patience=patience, epoch_callback=cb,
                                   step_offset=i * width, log=log)
            ctx.record_checkpoint(f"fold{fold}_best", sha256=run.state_sha256, fold=fold, weights_discarded=True)
            m = run.metrics
            row = {"fold": fold, "score": run.score, "best_epoch": run.best_epoch, "epochs_run": run.epochs_run, "stopped_early": run.stopped_early,
                   "n_train": len(tr), "n_val": len(va), "train_seconds": run.train_seconds, "vram_peak_mb": run.vram_peak_mb, "micro_batch": run.micro_batch,
                   "accuracy": m.get("accuracy"), "precision": m.get("precision"), "recall": m.get("recall"), "f1": m.get("f1"), "roc_auc": m.get("roc_auc"),
                   "ap_undefined": run.ap_undefined}
            run.state = None
        if row.get("score") is None or row.get("ap_undefined"):
            raise MethodologyViolation("Objetivo HPO no definido; no registrar cero como observacion")
        params_m, gflops = run.params_m, run.gflops
        vram = max([v for v in (vram, run.vram_peak_mb) if v is not None], default=None)
        h = run.history.copy()
        h.insert(0, "fold", fold)
        from utils.search_spaces import config_id
        h["seed"] = int(seed)
        h["run_id"] = ctx.run_id+"_fold"+str(fold)
        h["architecture"] = spec.name
        h["config_id"] = config_id(hp)
        h["outer_split"] = "train"
        h["validation_role"] = "inner_fold_within_outer_train"
        hists.append(h)
        rows.append(row)
        write_csv(h,cache_history,float_format='%.12g')
        atomic_json(cache,dict(signature=signature,row=row,history_sha256=sha256_file(cache_history),params_m=params_m,gflops=gflops,vram_peak_mb=run.vram_peak_mb))
        log("fold %d: score=%.4f best_epoch=%d epochs=%d (%.1fs)", fold, row["score"], row["best_epoch"], row["epochs_run"], row["train_seconds"])
    hist = pd.concat(hists, ignore_index=True) if hists else pd.DataFrame()
    write_csv(pd.DataFrame(rows), run_dir / "fold_metrics.csv", float_format="%.8g")
    write_csv(hist, run_dir / "history.csv", float_format="%.8g")
    scores = [float(r["score"]) for r in rows]
    mean, sd = mean_sd(scores)
    return ConfigEvaluation(scores, float(mean if mean is not None else 0.0), sd, rows, time.perf_counter() - t0, vram, params_m, gflops, hist)


def record_training_source(ctx: RunContext, spec: ArchSpec, cfg: Mapping[str, Any]) -> None:
    """Deja en ``metadata.json`` la EVIDENCIA de que datos entrena el run (auditoria cientifica)."""
    from utils.config_loader import cfg_path
    from utils.hashing import sha256_file

    if spec.is_detector:
        ctx.add_metadata(detector_target="fruit_only", detection_dataset=str(cfg_path(cfg, "paths.detection_dir")))
    else:
        man = cfg_path(cfg, "paths.classification_dir") / str(cfg["classification_dataset"]["manifest_name"])
        ctx.add_metadata(train_source="ground_truth_crops", crop_manifest_sha256=sha256_file(man))


def weights_fingerprint(path: str) -> str:
    """SHA-256 de un fichero de pesos (cacheado)."""
    return weights_sha256(path)
