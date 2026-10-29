"""Almacenamiento y construccion de estudios Optuna (sec. 10, 17).

* Un estudio por arquitectura: ``botrytis_<slug>`` (nunca un ``study_global``).
* ``load_if_exists=True``: un corte de luz/PC no pierde trials; se reanuda.
* TPE (multivariate, group, 5 trials de arranque) + MedianPruner (5 trials, 8 pasos de calentamiento).
* Backend: SQLite local (un fichero por estudio, sincronizable por ZIP/Drive; Estrategia_3PCs sec. 12) o
  PostgreSQL central si ``optuna.storage.backend: postgresql`` y existe ``OPTUNA_DATABASE_URL``.
"""

from __future__ import annotations

import os
import warnings
from pathlib import Path
from typing import Any, Mapping

import pandas as pd

from utils.architectures import ArchSpec
from utils.config_loader import BlockingTodoError, cfg_path, get_nested
from utils.io import ensure_dir


def _optuna() -> Any:
    import optuna

    warnings.filterwarnings("ignore", category=optuna.exceptions.ExperimentalWarning)
    optuna.logging.set_verbosity(optuna.logging.WARNING)
    return optuna


def study_db_path(cfg: Mapping[str, Any], spec: ArchSpec) -> Path:
    """Ruta del SQLite del estudio de ``spec``."""
    prefix = str(get_nested(cfg, "optuna.study_prefix", default="botrytis"))
    return ensure_dir(cfg_path(cfg, "paths.optuna_dir")) / f"{spec.study_name(prefix)}.db"


def storage_url(cfg: Mapping[str, Any], spec: ArchSpec) -> str:
    """URL de almacenamiento del estudio (SQLite local por estudio; no servidores)."""
    backend = str(get_nested(cfg, "optuna.storage.backend", default="sqlite")).lower()
    if str(backend).lower() != "sqlite":raise RuntimeError("Solo SQLite local permitido; no se conecta a servidores ni se buscan credenciales")
    return "sqlite:///" + study_db_path(cfg, spec).as_posix()


def make_sampler(cfg: Mapping[str, Any], seed: int | None) -> Any:
    """``TPESampler`` segun la sec. 10.2 (multivariate, group, arranque aleatorio de N trials)."""
    optuna = _optuna()
    s = cfg["optuna"]["sampler"]
    if str(s["name"]).upper() != "TPE":
        raise ValueError("La metodologia exige el sampler TPE (sec. 10.2)")
    return optuna.samplers.TPESampler(
        multivariate=bool(s["multivariate"]), group=bool(s["group"]), n_startup_trials=int(s["startup_trials"]), seed=seed,
    )


def make_pruner(cfg: Mapping[str, Any]) -> Any:
    """``MedianPruner`` segun la sec. 10.3 (5 trials de arranque, 8 pasos de calentamiento, intervalo 1)."""
    optuna = _optuna()
    p = cfg["optuna"]["pruner"]
    if str(p["name"]) != "MedianPruner":
        raise ValueError("La metodologia exige MedianPruner (sec. 10.3)")
    return optuna.pruners.MedianPruner(
        n_startup_trials=int(p["startup_trials"]), n_warmup_steps=int(p["warmup_epochs"]), interval_steps=int(p["interval_steps"]),
    )


def create_or_load_study(spec: ArchSpec, cfg: Mapping[str, Any], seed: int | None) -> Any:
    """Crea (o reanuda) el estudio INDEPENDIENTE de ``spec`` (direccion: maximizar)."""
    optuna = _optuna()
    prefix = str(get_nested(cfg, "optuna.study_prefix", default="botrytis"))
    study = optuna.create_study(
        study_name=spec.study_name(prefix), storage=storage_url(cfg, spec), sampler=make_sampler(cfg, seed),
        pruner=make_pruner(cfg), direction="maximize", load_if_exists=bool(get_nested(cfg, "optuna.load_if_exists", default=True)),
    )

    from utils.semantics import semantic_fingerprint
    identity=semantic_fingerprint(cfg)
    prior=study.user_attrs.get("semantic_fingerprint")
    if (prior and prior!=identity) or (study.trials and not prior):
        raise RuntimeError("Estudio Optuna obsoleto/incompatible; conservar y usar namespace v4 nuevo")
    study.set_user_attr("semantic_fingerprint",identity)
    study.set_user_attr("initialization",cfg.get("initialization"))
    return study


def load_study(spec: ArchSpec, cfg: Mapping[str, Any]) -> Any:
    """Carga un estudio existente (sin crearlo)."""
    optuna = _optuna()
    prefix = str(get_nested(cfg, "optuna.study_prefix", default="botrytis"))
    study=optuna.load_study(study_name=spec.study_name(prefix), storage=storage_url(cfg, spec))
    from utils.semantics import semantic_fingerprint
    if study.user_attrs.get("semantic_fingerprint")!=semantic_fingerprint(cfg):raise RuntimeError("Estudio de otra semantica; no reutilizar")
    return study


def budget_used(study: Any) -> int:
    """Trials que consumen presupuesto: COMPLETE y PRUNED (los FAIL se reintentan, con tope)."""
    optuna = _optuna()
    ok = (optuna.trial.TrialState.COMPLETE, optuna.trial.TrialState.PRUNED)
    return sum(1 for t in study.get_trials(deepcopy=False) if t.state in ok)


def failed_trials(study: Any) -> int:
    """Numero de trials FAIL del estudio."""
    optuna = _optuna()
    return sum(1 for t in study.get_trials(deepcopy=False) if t.state == optuna.trial.TrialState.FAIL)


def trials_table(study: Any) -> pd.DataFrame:
    """Una fila por trial: hiperparametros canonicos, fold scores, valor, estado, tiempo, VRAM, FLOPs, parametros.

    Las columnas ``hp_*`` salen de ``user_attrs['hp']`` (diccionario canonico), no de los nombres
    internos de Optuna.
    """
    rows: list[dict[str, Any]] = []
    for t in study.get_trials(deepcopy=False):
        ua = dict(t.user_attrs)
        row: dict[str, Any] = {
            "number": t.number, "state": t.state.name, "value": t.value,
            "datetime_start": t.datetime_start.isoformat() if t.datetime_start else None,
            "datetime_complete": t.datetime_complete.isoformat() if t.datetime_complete else None,
            "duration_s": (t.datetime_complete - t.datetime_start).total_seconds() if t.datetime_complete and t.datetime_start else None,
            "n_intermediate": len(t.intermediate_values),
        }
        for k in ("fold_scores", "config_id", "train_seconds", "vram_peak_mb", "gflops", "params_m", "run_dir", "pc_id", "n_folds_done", "pruned_at_step"):
            row[k] = ua.get(k)
        for k, v in (ua.get("hp") or {}).items():
            row[k] = v
        rows.append(row)
    df = pd.DataFrame(rows)
    return df
