"""Entrenamiento de los 4 detectores con Ultralytics (un fold interno o el 70 % completo).

* YOLOv8n/l, YOLO26l y RT-DETR-R50, detector de UNA clase (``fruit``).
* Hiperparametros del espacio HPO -> argumentos de Ultralytics; ``imgsz=640``, batch 16 (micro-batch) con
  ``nbs=64`` (mismo effective batch size para todas), AMP fp16, ``pretrained`` siempre.
* Callback por epoca -> Optuna (reporte + ``MedianPruner``); el score del fold es el mejor mAP@0.5:0.95.
* ``fixed_epochs=True``: entrenamiento definitivo con el 70 % (sin early stopping; no se valida en
  ninguna epoca; el validation externo se evalua en el paso 10).
* RT-DETR-R50: ``backbone_lr_ratio`` (0.1) implementado separando los grupos de parametros del backbone.
"""

from __future__ import annotations

import gc
import re
import shutil
import time
from dataclasses import dataclass, field
from functools import lru_cache
from pathlib import Path
from typing import Any, Callable, Mapping

import pandas as pd

from utils.architectures import ArchSpec
from utils.model_stats import count_gflops, count_params_m
from utils.pretrained import configure_ultralytics, ensure_ultralytics_weights, weights_sha256
from utils.timing import free_cuda_memory, peak_vram_mb, reset_peak_vram
from utils.training_common import is_oom, resolve_device, ultralytics_device

EpochCallback = Callable[[int, Mapping[str, Any]], None]

_MAP = "metrics/mAP50-95(B)"
_MAP50 = "metrics/mAP50(B)"
_P = "metrics/precision(B)"
_R = "metrics/recall(B)"
_YAML = {"yolov8": "{slug}.yaml", "yolo26": "{slug}.yaml", "rtdetr": "rtdetr-resnet50.yaml"}


@dataclass
class DetectorRun:
    """Resultado de entrenar un detector."""

    score: float | None
    best_epoch: int
    epochs_run: int
    stopped_early: bool
    history: pd.DataFrame
    final_metrics: dict[str, Any]
    train_seconds: float
    vram_peak_mb: float | None
    micro_batch: int
    params_m: float
    gflops: float
    weights_path: str | None
    weights_sha256: str | None
    pretrained_sha256: str | None
    extra: dict[str, Any] = field(default_factory=dict)


@lru_cache(maxsize=None)
def arch_stats(slug: str, family: str, imgsz: int = 640) -> dict[str, float]:
    """Parametros (M) y GFLOPs de la arquitectura con ``nc=1`` (constantes por arquitectura)."""
    configure_ultralytics()
    from ultralytics.nn.tasks import DetectionModel, RTDETRDetectionModel

    yaml = _YAML[family].format(slug=slug)
    cls = RTDETRDetectionModel if family == "rtdetr" else DetectionModel
    model = cls(yaml, nc=1, verbose=False)
    return {"params_m": count_params_m(model), "gflops": count_gflops(model, (1, 3, imgsz, imgsz))}


def _new_model(spec: ArchSpec, cfg: Mapping[str, Any]) -> tuple[Any, str]:
    """``(modelo Ultralytics, sha256 de los pesos preentrenados)``."""
    configure_ultralytics()
    from ultralytics import RTDETR, YOLO

    from utils.weights import initialization_source
    resume=cfg.get("detector",{}).get("resume_checkpoint")
    if resume:
        from utils.resume import require_checkpoint
        source=require_checkpoint(resume,spec,cfg);sha=None
    else:
        source, sha = initialization_source(spec, cfg)
    return (RTDETR(str(source)) if spec.family == "rtdetr" else YOLO(str(source))), sha


def _rtdetr_backbone_lr_callback(ratio: float) -> Callable[[Any], None]:
    """Callback ``on_train_start``: divide los grupos de parametros en backbone (x ratio) y resto."""

    def _cb(trainer: Any) -> None:
        if not 0 < ratio <= 1: raise ValueError('backbone_lr_ratio fuera de (0,1]')
        backbone = {id(p) for n, p in trainer.model.named_parameters() if re.match(r"(module\.)?model\.[0-4]\.", n)}
        if not backbone: raise RuntimeError('Backend RT-DETR sin backbone identificable; no entrenar con LR incorrecto')
        opt = trainer.optimizer
        old = list(opt.param_groups)
        new_groups: list[dict[str, Any]] = []
        for g in old:
            rest = [p for p in g["params"] if id(p) not in backbone]
            back = [p for p in g["params"] if id(p) in backbone]
            for params, scale in ((rest, 1.0), (back, ratio)):
                if params:
                    ng = {k: v for k, v in g.items() if k != "params"}
                    ng["params"] = params
                    ng["lr"] = float(g["lr"]) * scale
                    ng["initial_lr"] = float(g.get("initial_lr", g["lr"])) * scale
                    new_groups.append(ng)
        opt.param_groups.clear()
        for ng in new_groups:
            opt.add_param_group(ng)
        sch = trainer.scheduler
        sch.lr_lambdas = [trainer.lf] * len(new_groups)
        sch.base_lrs = [g["initial_lr"] for g in new_groups]
        sch._last_lr = [g["lr"] for g in new_groups]
        trainer._botrytis_backbone_groups = [g["lr"] for g in new_groups]

    return _cb


def _train_args(spec: ArchSpec, hp: Mapping[str, Any], cfg: Mapping[str, Any], *, data_yaml: Path, epochs: int, patience: int, seed: int,
                micro_batch: int, project: Path, name: str, fixed_epochs: bool) -> dict[str, Any]:
    d = cfg["detector"]
    aug = dict(d["augmentation"])
    args: dict[str, Any] = dict(
        data=str(data_yaml), epochs=int(epochs), patience=int(patience), imgsz=int(d["image_size"]), batch=int(micro_batch),
        nbs=int(d["nominal_batch_size"]), device=ultralytics_device(cfg), workers=int(d.get("workers", 4)), seed=int(seed),
        deterministic=bool(cfg["reproducibility"]["deterministic"]), project=str(project), name=name, exist_ok=True,
        pretrained=bool(d["pretrained"]), amp=str(d["precision"]).lower() == "fp16", plots=False, verbose=False,
        val=not fixed_epochs, cos_lr=bool(d["baseline"]["cos_lr"]), single_cls=False, save=True, cache=False, **aug,
    )
    if d.get("resume_checkpoint"):
        args["resume"] = str(d["resume_checkpoint"])
    args["optimizer"] = str(hp["optimizer"])
    args["lr0"] = float(hp["lr0"])
    args["weight_decay"] = float(hp["weight_decay"])
    for k in ("lrf", "momentum", "warmup_epochs", "freeze", "mosaic", "mixup"):
        if k in hp:
            args[k] = hp[k]
    if spec.family == "rtdetr":
        args.setdefault("lrf", float(d["baseline"]["lrf"]))
    return args


def _collect(trainer: Any, history: list[dict[str, Any]], final: dict[str, Any], epoch_cb: EpochCallback | None, step_offset: int,
             log: Callable[[str], None], fixed_epochs: bool) -> None:
    """Callback ``on_fit_epoch_end``: acumula historial, reporta a Optuna, captura la evaluacion final."""
    metrics = dict(getattr(trainer, "metrics", {}) or {})
    is_final = trainer.epoch >= trainer.epochs  # ``final_eval`` (best.pt) se notifica con epoch == epochs
    if is_final:
        final.update({k: float(v) for k, v in metrics.items() if isinstance(v, (int, float))})
        try:
            ap = trainer.validator.metrics.box.all_ap
            final["ap_per_iou"] = [float(x) for x in ap[0]] if getattr(ap, "size", 0) else None
        except Exception:
            final["ap_per_iou"] = None
        return
    if getattr(trainer, "fitness", None) is None:
        return  # sin validacion en esta epoca (modo fixed_epochs): no hay metrica que registrar
    row: dict[str, Any] = {"epoch": int(trainer.epoch) + 1}
    try:
        row.update(trainer.label_loss_items(trainer.tloss))
    except Exception as exc:
        log("PENDIENTE: componentes loss no expuestos por backend: "+repr(exc))
    row.update({k: float(v) for k, v in metrics.items() if isinstance(v, (int, float))})
    for split in ("train", "val"):
        row[split+"_loss"] = None  # no existe loss agregada autentica expuesta por esta API
    p, r = row.get(_P), row.get(_R)
    row["f1"] = (2*p*r/(p+r) if p+r>0 else 0.0) if p is not None and r is not None else None
    row["map50_95"], row["map50"] = row.get(_MAP), row.get(_MAP50)
    history.append(row)
    log(f"epoca {row['epoch']}/{trainer.epochs} mAP50-95={row.get(_MAP)} mAP50={row.get(_MAP50)} P={p} R={r}")
    from utils.logging_utils import log_epoch
    log_epoch(row, seed=getattr(trainer.args,"seed",None), training_folder=str(getattr(trainer,"save_dir","")))
    if epoch_cb is not None and not fixed_epochs:
        epoch_cb(step_offset + int(trainer.epoch), row)  # puede lanzar optuna.TrialPruned


def train_detector(
    spec: ArchSpec, hp: Mapping[str, Any], cfg: Mapping[str, Any], *, data_yaml: Path, project_dir: Path, name: str, seed: int,
    epochs: int | None = None, patience: int | None = None, fixed_epochs: bool = False, epoch_callback: EpochCallback | None = None,
    step_offset: int = 0, log: Callable[[str], None] = print, keep_weights: bool = False,
) -> DetectorRun:
    """Entrena ``spec`` con ``hp``; ante CUDA OOM reintenta con micro-batch/2 (effective batch se conserva por ``nbs``)."""
    d = cfg["detector"]
    epochs = int(epochs if epochs is not None else d["max_epochs"])
    patience = 0 if fixed_epochs else int(patience if patience is not None else d["early_stopping_patience"])
    micro = int(d["batch_size"])
    attempt = 0
    while True:
        attempt += 1
        try:
            return _train_once(spec, hp, cfg, data_yaml=data_yaml, project_dir=project_dir, name=name if attempt == 1 else f"{name}_try{attempt}",
                               seed=seed, epochs=epochs, patience=patience, fixed_epochs=fixed_epochs, micro_batch=micro,
                               epoch_callback=epoch_callback, step_offset=step_offset, log=log, keep_weights=keep_weights)
        except Exception as exc:
            if not (is_oom(exc) and bool(d.get("oom_fallback", True)) and micro > 2):
                raise
            free_cuda_memory()
            micro //= 2
            log(f"CUDA OOM: reintentando con micro-batch={micro} (effective batch {d['nominal_batch_size']} por nbs)")


def fixed_epoch_trainer(spec):
    """Explicit trainer override supported by Model.train(trainer=...), pinned backend."""
    from ultralytics.models.yolo.detect.train import DetectionTrainer
    from ultralytics.models.rtdetr.train import RTDETRTrainer
    base = RTDETRTrainer if spec.family == 'rtdetr' else DetectionTrainer
    class FixedEpochTrainer(base):
        def validate(self):
            return {}, 0.0
        def final_eval(self):
            # Last checkpoint is saved by the base trainer; no external val is opened.
            return None
    return FixedEpochTrainer


def _train_once(spec: ArchSpec, hp: Mapping[str, Any], cfg: Mapping[str, Any], *, data_yaml: Path, project_dir: Path, name: str, seed: int,
                epochs: int, patience: int, fixed_epochs: bool, micro_batch: int, epoch_callback: EpochCallback | None, step_offset: int,
                log: Callable[[str], None], keep_weights: bool) -> DetectorRun:
    if resolve_device(cfg) == "cpu":
        log("ADVERTENCIA: entrenando en CPU (sin GPU CUDA); solo apto para pruebas.")
    model, pre_sha = _new_model(spec, cfg)
    def checkpoint_receipt(tr):
        from utils.resume import receipt
        path=Path(tr.last)
        if path.is_file():receipt(path,spec,cfg,{"seed":seed,"hp":dict(hp),"epoch":int(tr.epoch)+1})
    model.add_callback("on_model_save", checkpoint_receipt)
    history: list[dict[str, Any]] = []
    final: dict[str, Any] = {}
    model.add_callback("on_fit_epoch_end", lambda tr: _collect(tr, history, final, epoch_callback, step_offset, log, fixed_epochs))
    if spec.family == "rtdetr":
        model.add_callback("on_train_start", _rtdetr_backbone_lr_callback(float(hp.get("backbone_lr_ratio", 0.1))))
    args = _train_args(spec, hp, cfg, data_yaml=data_yaml, epochs=epochs, patience=patience, seed=seed, micro_batch=micro_batch,
                       project=project_dir, name=name, fixed_epochs=fixed_epochs)
    if fixed_epochs:
        args['trainer'] = fixed_epoch_trainer(spec)
    batch_contract = []
    def record_batch(tr):
        row = dict(epoch=int(tr.epoch), micro_batch=micro_batch,
                   nominal_batch=int(cfg["detector"]["nominal_batch_size"]), accumulate=getattr(tr,'accumulate',None),
                   learning_rates=[float(g['lr']) for g in tr.optimizer.param_groups])
        if not batch_contract or row != batch_contract[-1]: batch_contract.append(row)
    model.add_callback('on_train_batch_end',record_batch)
    reset_peak_vram()
    t0 = time.perf_counter()
    try:
        model.train(**args)
    finally:
        seconds = time.perf_counter() - t0
        vram = peak_vram_mb()
        try:
            del model
        except Exception:
            pass
        gc.collect()
        free_cuda_memory()
    save_dir = project_dir / name
    best_pt, last_pt = save_dir / "weights" / "best.pt", save_dir / "weights" / "last.pt"
    chosen = last_pt if fixed_epochs else (best_pt if best_pt.is_file() else last_pt)
    if not chosen.is_file(): raise RuntimeError('Backend no publico checkpoint final completo: '+str(chosen))
    w_sha = weights_sha256(str(chosen))
    hist = pd.DataFrame(history)
    if fixed_epochs:  # sin validacion: el historial (perdidas por epoca) sale de results.csv
        csv = save_dir / "results.csv"
        if csv.is_file():
            raw = pd.read_csv(csv)
            raw.columns = [c.strip() for c in raw.columns]
            tl = [c for c in raw.columns if c.startswith("train/")]
            hist = raw[["epoch", *tl]].copy()  # componentes originales; ninguna suma inventada
    if len(hist) and "map50_95" in hist and hist["map50_95"].notna().any():
        bi = int(hist["map50_95"].astype(float).idxmax())
        score, best_epoch = float(hist.loc[bi, "map50_95"]), int(hist.loc[bi, "epoch"])
    else:
        score, best_epoch = (float(final[_MAP]) if final.get(_MAP) is not None else None), epochs
    if fixed_epochs:  # checkpoint de ultima epoca; sin metrica externa durante entrenamiento
        score, best_epoch = (float(final[_MAP]) if final.get(_MAP) is not None else None), epochs
    stats = arch_stats(spec.slug, spec.family, int(cfg["detector"]["image_size"]))
    if w_sha is not None and not keep_weights:
        shutil.rmtree(save_dir / "weights", ignore_errors=True)
        chosen_path: str | None = None
    else:
        chosen_path = str(chosen) if chosen.is_file() else None
    p, r = final.get(_P), final.get(_R)
    final["f1"] = (2 * p * r / (p + r)) if p is not None and r is not None and (p + r) > 0 else None
    final["map50_95"], final["map50"], final["precision"], final["recall"] = final.get(_MAP), final.get(_MAP50), p, r
    return DetectorRun(
        score=score, best_epoch=best_epoch, epochs_run=len(hist) if len(hist) else epochs, stopped_early=bool(len(hist) and len(hist) < epochs and not fixed_epochs),
        history=hist, final_metrics=final, train_seconds=seconds, vram_peak_mb=vram, micro_batch=micro_batch, params_m=stats["params_m"],
        gflops=stats["gflops"], weights_path=chosen_path, extra={"batch_contract":batch_contract}, weights_sha256=w_sha, pretrained_sha256=pre_sha,
    )
