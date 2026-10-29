"""Entrenamiento de un clasificador binario sobre GT crops (un fold interno o el 70 % completo).

Caracteristicas (sec. 9-14):

* AdamW/SGD con calentamiento lineal + decaimiento coseno por iteracion (el planificador NO es HPO).
* Precision mixta fp16 (``torch.autocast`` + ``GradScaler``), batch efectivo 32 (acumulacion si hay OOM).
* Metrica de seleccion por epoca: Average Precision de ``con_signos`` en el fold de validacion interno.
* Early stopping (paciencia) y callback por epoca para que Optuna reporte/pode (``MedianPruner``).
* ``fixed_epochs=True`` (entrenamiento definitivo con el 70 %): numero de epocas fijado por los folds
  internos, SIN early stopping ni seleccion de epoca sobre el validation externo.
"""

from __future__ import annotations

import copy
import math
import time
from pathlib import Path
from dataclasses import dataclass, field
from typing import Any, Callable, Mapping

import numpy as np
import pandas as pd
import torch
from torch import nn

from utils.architectures import ArchSpec
from utils.classifier_data import (
    CropDataset, assert_deterministic_transform, build_transforms, class_weights, make_loader,
)
from utils.classifier_models import build_classifier
from utils.config_loader import cfg_path
from utils.metrics import binary_classification_report
from utils.model_stats import classifier_stats
from utils.reproducibility import set_global_seed
from utils.timing import free_cuda_memory, peak_vram_mb, reset_peak_vram
from utils.training_common import is_oom, resolve_device, state_dict_sha256

EpochCallback = Callable[[int, Mapping[str, Any]], None]


@dataclass
class ClassifierRun:
    """Resultado de entrenar un clasificador (un fold o el 70 % completo)."""

    score: float | None
    best_epoch: int
    epochs_run: int
    stopped_early: bool
    history: pd.DataFrame
    val_true: np.ndarray
    val_prob: np.ndarray
    val_index: np.ndarray
    metrics: dict[str, Any]
    train_seconds: float
    vram_peak_mb: float | None
    micro_batch: int
    accum_steps: int
    params_m: float
    gflops: float
    state_sha256: str
    state: dict[str, torch.Tensor] | None = field(default=None, repr=False)
    norm: dict[str, Any] = field(default_factory=dict)
    ap_undefined: bool = False


def build_optimizer(model: nn.Module, hp: Mapping[str, Any], cfg: Mapping[str, Any]) -> torch.optim.Optimizer:
    """AdamW o SGD; sin weight decay en sesgos y normalizaciones (parametros de 1 dimension)."""
    decay, no_decay = [], []
    for p in model.parameters():
        if p.requires_grad:
            (no_decay if p.ndim <= 1 else decay).append(p)
    wd = float(hp["weight_decay"])
    groups = [{"params": decay, "weight_decay": wd}, {"params": no_decay, "weight_decay": 0.0}]
    name = str(hp["optimizer"])
    if name == "AdamW":
        return torch.optim.AdamW(groups, lr=float(hp["lr"]))
    if name == "SGD":
        return torch.optim.SGD(groups, lr=float(hp["lr"]), momentum=float(cfg["classifier"].get("sgd_momentum", 0.9)))
    raise ValueError(f"Optimizador desconocido: {name!r}")


def lr_at(step: int, base_lr: float, warmup_steps: int, total_steps: int) -> float:
    """Calentamiento lineal y despues decaimiento coseno a 0."""
    if warmup_steps > 0 and step < warmup_steps:
        return base_lr * (step + 1) / warmup_steps
    progress = (step - warmup_steps) / max(1, total_steps - warmup_steps)
    return base_lr * 0.5 * (1.0 + math.cos(math.pi * min(1.0, progress)))


@torch.no_grad()
def predict_probs(model: nn.Module, loader: Any, device: str, use_amp: bool) -> tuple[np.ndarray, np.ndarray, np.ndarray, float]:
    """``(p_con_signos, y_true, indices, loss_ce)`` sobre un ``DataLoader`` deterministico."""
    model.eval()
    probs: list[np.ndarray] = []
    ys: list[np.ndarray] = []
    idxs: list[np.ndarray] = []
    total_loss, n = 0.0, 0
    for x, y, idx in loader:
        x = x.to(device, non_blocking=True)
        with torch.autocast(device_type="cuda", dtype=torch.float16, enabled=use_amp):
            logits = model(x)
        logits = logits.float()
        loss = nn.functional.cross_entropy(logits, y.to(device), reduction="sum")
        total_loss += float(loss.item())
        n += len(y)
        probs.append(torch.softmax(logits, dim=1)[:, 1].cpu().numpy())
        ys.append(y.numpy())
        idxs.append(idx.numpy())
    if not probs:
        return np.zeros(0), np.zeros(0, dtype=int), np.zeros(0, dtype=int), float("nan")
    return np.concatenate(probs), np.concatenate(ys), np.concatenate(idxs), total_loss / max(n, 1)


def _train_once(
    spec: ArchSpec, hp: Mapping[str, Any], cfg: Mapping[str, Any], train_df: pd.DataFrame, val_df: pd.DataFrame, *,
    seed: int, max_epochs: int, patience: int | None, fixed_epochs: bool, micro_batch: int, epoch_callback: EpochCallback | None,
    step_offset: int, log: Callable[[str], None], keep_state: bool,
) -> ClassifierRun:
    device = resolve_device(cfg)
    use_amp = device != "cpu" and str(cfg["classifier"]["precision"]).lower() == "fp16"
    cc = cfg["classifier"]
    batch = int(cc["batch_size"])
    accum = max(1, batch // micro_batch)
    set_global_seed(seed, deterministic=bool(cfg["reproducibility"]["deterministic"]),
                    cudnn_deterministic=bool(cfg["reproducibility"]["cudnn_deterministic"]),
                    cudnn_benchmark=bool(cfg["reproducibility"]["cudnn_benchmark"]))
    model, norm = build_classifier(spec, hp, pretrained=bool(cc["pretrained"]) and not cc.get("resume_checkpoint"), cfg=cfg)
    stats = classifier_stats(model, int(cc["image_size"]))
    model.to(device)

    root = cfg_path(cfg, "paths.classification_dir")
    train_tf = build_transforms(cfg, norm, train=True)
    eval_tf = build_transforms(cfg, norm, train=False)
    assert_deterministic_transform(eval_tf)
    workers = int(cc.get("workers", 0))
    train_ds = CropDataset(train_df, root, train_tf)
    has_val = len(val_df) > 0  # el entrenamiento definitivo con el 70 % no mira el validation externo
    val_ds = CropDataset(val_df, root, eval_tf) if has_val else None
    train_loader = make_loader(train_ds, batch_size=micro_batch, shuffle=True, seed=seed, workers=workers, drop_last=len(train_ds) > micro_batch)
    val_loader = make_loader(val_ds, batch_size=max(micro_batch, 32), shuffle=False, seed=seed, workers=workers) if val_ds is not None else None

    weights = class_weights(train_ds.y, str(cc["imbalance"]["strategy"]))
    criterion = nn.CrossEntropyLoss(weight=weights.to(device) if weights is not None else None, label_smoothing=float(hp.get("label_smoothing", 0.0)))
    opt = build_optimizer(model, hp, cfg)
    scaler = torch.amp.GradScaler("cuda", enabled=use_amp)
    steps_per_epoch = max(1, math.ceil(len(train_loader) / accum))
    total_steps = max_epochs * steps_per_epoch
    warm = int(hp.get("warmup_epochs", 0)) * steps_per_epoch
    base_lr = float(hp["lr"])

    history: list[dict[str, Any]] = []
    best = {"score": -1.0, "epoch": 0, "state": None, "prob": np.zeros(0), "true": np.zeros(0, dtype=int), "idx": np.zeros(0, dtype=int), "loss": float("inf")}
    bad_epochs, step = 0, 0
    stopped_early = False
    reset_peak_vram()
    t_train0 = time.perf_counter()
    last_prob = last_true = last_idx = None
    epoch = 0
    import random
    from utils.hashing import sha256_json
    from utils.semantics import semantic_fingerprint
    identity=sha256_json({"hp":dict(hp),"seed":seed,"train":train_df.to_dict("records"),"val":val_df.to_dict("records"),"semantics":semantic_fingerprint(cfg),"epochs":max_epochs,"micro_batch":micro_batch})
    start_epoch=1
    if cc.get("resume_checkpoint"):
        from utils.resume import require_checkpoint
        resume=require_checkpoint(cc["resume_checkpoint"],spec,cfg)
        saved=torch.load(resume,map_location=device,weights_only=False)  # only locally generated, hash verified
        if saved.get("identity")!=identity:raise RuntimeError("Resume de otro fold/seed/config/batch/datos")
        model.load_state_dict(saved["model"]);opt.load_state_dict(saved["optimizer"]);scaler.load_state_dict(saved["scaler"])
        history=saved["history"];best=saved["best"];bad_epochs=saved["bad_epochs"];step=saved["step"];start_epoch=saved["epoch"]+1
        random.setstate(saved["python_rng"]);np.random.set_state(saved["numpy_rng"]);torch.set_rng_state(saved["torch_rng"].cpu())
        if torch.cuda.is_available():torch.cuda.set_rng_state_all(saved["cuda_rng"])
        if train_loader.generator is not None:train_loader.generator.set_state(saved["loader_rng"].cpu())
        if start_epoch>max_epochs:raise RuntimeError("Checkpoint ya completo; no reanudar ni inventar nuevas epocas")
    for epoch in range(start_epoch, max_epochs + 1):
        t_ep = time.perf_counter()
        model.train()
        run_loss, n_seen = 0.0, 0
        opt.zero_grad(set_to_none=True)
        for bi, (x, y, _) in enumerate(train_loader):
            x, y = x.to(device, non_blocking=True), y.to(device, non_blocking=True)
            if bi % accum == 0:
                for g in opt.param_groups:
                    g["lr"] = lr_at(step, base_lr, warm, total_steps)
            with torch.autocast(device_type="cuda", dtype=torch.float16, enabled=use_amp):
                logits = model(x)
            loss = criterion(logits.float(), y)
            scaler.scale(loss / accum).backward()
            run_loss += float(loss.item()) * len(y)
            n_seen += len(y)
            if (bi + 1) % accum == 0 or (bi + 1) == len(train_loader):
                scaler.step(opt)
                scaler.update()
                opt.zero_grad(set_to_none=True)
                step += 1
        if val_loader is not None:
            prob, ytrue, vidx, vloss = predict_probs(model, val_loader, device, use_amp)
        else:
            prob, ytrue, vidx, vloss = np.zeros(0), np.zeros(0, dtype=int), np.zeros(0, dtype=int), float("nan")
        rep = binary_classification_report(ytrue, prob, threshold=0.5) if len(ytrue) else {}
        ap = rep.get("average_precision")
        ap_defined = ap is not None
        score = float(ap) if ap_defined else 0.0
        row = {
            "epoch": epoch, "train_loss": run_loss / max(n_seen, 1), "val_loss": vloss, "val_ap": float(ap) if ap_defined else None,
            "val_roc_auc": rep.get("roc_auc"), "val_f1": rep.get("f1"), "val_accuracy": rep.get("accuracy"),
            "val_precision": rep.get("precision"), "val_recall": rep.get("recall"),
            "lr": opt.param_groups[0]["lr"], "epoch_seconds": time.perf_counter() - t_ep,
        }
        history.append(row)
        from utils.logging_utils import log_epoch
        log_epoch(row,architecture=spec.slug,seed=seed)
        log(f"epoca {epoch}/{max_epochs} train_loss={row['train_loss']:.4f} val_loss={vloss:.4f} val_AP={score:.4f} f1@0.5={row['val_f1']}" if has_val
            else f"epoca {epoch}/{max_epochs} train_loss={row['train_loss']:.4f} (entrenamiento definitivo: sin validacion por epoca)")
        last_prob, last_true, last_idx = prob, ytrue, vidx
        improved = score > best["score"] + 1e-9 or (abs(score - best["score"]) <= 1e-9 and vloss < best["loss"] - 1e-9)
        if improved and not fixed_epochs:
            best.update(score=score, epoch=epoch, prob=prob, true=ytrue, idx=vidx, loss=vloss,
                        state={k: v.detach().cpu().clone() for k, v in model.state_dict().items()})  # siempre: se hashea el mejor checkpoint
            bad_epochs = 0
        elif not fixed_epochs:
            bad_epochs += 1
        resume_output=cc.get("resume_output")
        if resume_output:
            from utils.durable import atomic_torch_save
            from utils.resume import receipt
            path=Path(resume_output)
            atomic_torch_save({"identity":identity,"epoch":epoch,"model":model.state_dict(),"optimizer":opt.state_dict(),"scaler":scaler.state_dict(),"history":history,"best":best,"bad_epochs":bad_epochs,"step":step,"python_rng":random.getstate(),"numpy_rng":np.random.get_state(),"torch_rng":torch.get_rng_state(),"cuda_rng":torch.cuda.get_rng_state_all() if torch.cuda.is_available() else [],"loader_rng":train_loader.generator.get_state() if train_loader.generator is not None else None},path)
            receipt(path,spec,cfg,{"seed":seed,"epoch":epoch,"identity":identity})
        if epoch_callback is not None and not fixed_epochs:
            epoch_callback(step_offset + epoch - 1, row)  # puede lanzar optuna.TrialPruned
        if not fixed_epochs and patience is not None and bad_epochs >= patience:
            stopped_early = True
            log(f"early stopping en la epoca {epoch} (paciencia {patience})")
            break
    seconds = time.perf_counter() - t_train0
    if fixed_epochs:  # el modelo definitivo es el de la ULTIMA epoca (no se elige epoca con el validation externo)
        state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
        final_rep = binary_classification_report(last_true, last_prob, threshold=0.5) if last_true is not None and len(last_true) else {}
        best.update(score=float(final_rep.get("average_precision") or 0.0), epoch=epoch, prob=last_prob, true=last_true, idx=last_idx, state=state)
    hist = pd.DataFrame(history)
    bs = best["state"]
    rep_best = binary_classification_report(best["true"], best["prob"], threshold=0.5) if len(best["true"]) else {}
    result = ClassifierRun(
        score=(float(rep_best["average_precision"]) if rep_best.get("average_precision") is not None else None), best_epoch=int(best["epoch"]), epochs_run=len(history), stopped_early=stopped_early,
        history=hist, val_true=np.asarray(best["true"]), val_prob=np.asarray(best["prob"]), val_index=np.asarray(best["idx"]),
        metrics=rep_best, train_seconds=seconds, vram_peak_mb=peak_vram_mb(), micro_batch=micro_batch, accum_steps=accum,
        params_m=stats["params_m"], gflops=stats["gflops"], state_sha256=state_dict_sha256(bs) if bs is not None else "",
        state=bs if keep_state else None, norm={k: list(v) for k, v in norm.items()},
        ap_undefined=bool(len(best["true"]) and int(np.sum(best["true"])) == 0),
    )
    del model, opt, train_loader, val_loader, train_ds, val_ds
    free_cuda_memory()
    return result


def train_classifier(
    spec: ArchSpec, hp: Mapping[str, Any], cfg: Mapping[str, Any], train_df: pd.DataFrame, val_df: pd.DataFrame, *,
    seed: int, max_epochs: int | None = None, patience: int | None = None, fixed_epochs: bool = False,
    epoch_callback: EpochCallback | None = None, step_offset: int = 0, log: Callable[[str], None] = print, keep_state: bool = False,
) -> ClassifierRun:
    """Entrena con reintento automatico ante CUDA OOM (micro-batch/2, acumulacion x2; batch efectivo constante)."""
    import copy
    cfg=copy.deepcopy(dict(cfg))
    from utils.logging_utils import ACTIVE_RUN_DIR
    active=ACTIVE_RUN_DIR.get()
    if active:
        from utils.hashing import sha256_json
        name=sha256_json({"seed":seed,"hp":dict(hp),"train":train_df[[c for c in ("image_id","fruit_id") if c in train_df]].to_dict("records")})[:20]
        cfg["classifier"]["resume_output"]=str(Path(active)/("resume_"+name+".pt"))
    cc = cfg["classifier"]
    max_epochs = int(max_epochs if max_epochs is not None else cc["max_epochs"])
    patience = int(cc["early_stopping_patience"]) if patience is None and not fixed_epochs else patience
    micro = int(cc["batch_size"])
    while True:
        try:
            return _train_once(spec, hp, cfg, train_df, val_df, seed=seed, max_epochs=max_epochs, patience=patience, fixed_epochs=fixed_epochs,
                               micro_batch=micro, epoch_callback=epoch_callback, step_offset=step_offset, log=log, keep_state=keep_state)
        except Exception as exc:
            if not (is_oom(exc) and bool(cc.get("oom_fallback", True)) and micro > 2):
                raise
            free_cuda_memory()
            micro //= 2
            log(f"CUDA OOM: reintentando con micro-batch={micro} y acumulacion={int(cc['batch_size']) // micro} (batch efectivo {cc['batch_size']})")


def load_fold_frames(manifest: pd.DataFrame, cfg: Mapping[str, Any], fold: int) -> tuple[pd.DataFrame, pd.DataFrame]:
    """``(train, val)`` de un fold interno: SOLO outer train; validation = crops con ``inner_fold == fold``."""
    from utils.classifier_data import crops_for

    tr = crops_for(manifest, split="train", stage="hpo_trial", cfg=cfg, exclude_fold=fold)
    va = crops_for(manifest, split="train", stage="hpo_trial", cfg=cfg, only_fold=fold)
    if set(tr["image_id"]) & set(va["image_id"]):
        from utils.checks import MethodologyViolation

        raise MethodologyViolation("Fuga: una imagen aparece en train y validation del mismo fold interno")
    return tr, va


def copy_state(state: Mapping[str, torch.Tensor]) -> dict[str, torch.Tensor]:
    """Copia profunda de un ``state_dict``."""
    return copy.deepcopy(dict(state))
