"""06 - Baseline por arquitectura (Especificacion s2, sec. 11, 41).

Una ejecucion por arquitectura ANTES del HPO, con los hiperparametros baseline del anexo (sec. 41.2),
sobre los 3 folds internos (solo outer train) para que sea comparable con el objetivo del HPO. Sirve para:
verificar que el pipeline funciona, comprobar convergencia, detectar errores de etiquetas, medir VRAM y
tiempos y validar que las metricas no estan rotas. **NO se usa para descartar arquitecturas.**

Salidas (``runs/<arquitectura>/baseline/``): ``summary.json``, ``fold_metrics.csv``, ``history.csv``, figuras de
loss y metricas por epoca y los 6 ficheros obligatorios. Es idempotente: si ``summary.json`` existe, no repite.

Ejemplo::

    python src/06_run_baseline.py --architecture YOLOv8n --pc-id PC-01
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

sys.path.insert(0, str(Path(__file__).resolve().parent))

import numpy as np  # noqa: E402

from utils.architectures import section_for  # noqa: E402
from utils.guards import GpuJobLock  # noqa: E402
from utils.io import read_json, write_json  # noqa: E402
from utils.logging_utils import RunContext, run_cli  # noqa: E402
from utils.run_common import add_arch_arguments, arch_from_args, guard_not_frozen, layout_for, preflight_training  # noqa: E402
from utils.run_figures import save_training_figures  # noqa: E402
from utils.search_spaces import baseline_hyperparameters, config_id  # noqa: E402

SCRIPT = "06_run_baseline"


def add_arguments(parser: argparse.ArgumentParser) -> None:
    """Argumentos propios del script."""
    add_arch_arguments(parser)
    g = parser.add_argument_group("06_run_baseline")
    g.add_argument("--folds", nargs="*", type=int, default=None, help="folds internos a entrenar (por defecto todos)")
    g.add_argument("--force", action="store_true", help="repite aunque exista summary.json")


def main(ctx: RunContext, args: argparse.Namespace) -> None:
    """Entrena el baseline de una arquitectura y resume convergencia, VRAM y tiempos."""
    cfg, log = ctx.cfg, ctx.logger
    spec = arch_from_args(args)
    data = preflight_training(ctx, args)
    from utils.hpo_runner import evaluate_config, objective_metric
    guard_not_frozen(ctx, "06_run_baseline")
    sec = section_for(spec, cfg)
    base = dict(sec["baseline"])
    seed = int(base["seed"])
    ctx.set_seed(seed)
    k = int(cfg["inner_hpo"]["folds"])
    folds = list(args.folds) if args.folds else list(range(k))
    summary_path = ctx.run_dir / "summary.json"
    if summary_path.is_file() and not args.force:
        log.info("baseline ya ejecutado (%s): no se repite (--force para repetir)", summary_path)
        ctx.log_metrics(read_json(summary_path))
        return

    hp = baseline_hyperparameters(spec, cfg)
    max_epochs = int(base["epochs"])
    patience = int(base["patience"]) if spec.is_detector else int(sec["early_stopping_patience"])
    ctx.add_metadata(architecture=spec.name, stage="baseline", hp=hp, seed=seed, folds=folds)
    with GpuJobLock(cfg, ctx.pc_id, job=f"baseline:{spec.slug}"):
        ev = evaluate_config(spec, hp, cfg, ctx, ctx.run_dir, folds=folds, seed=seed, max_epochs=max_epochs, patience=patience)
    save_training_figures(ctx, spec, ev.history, "baseline", ctx.run_dir / "history.csv")

    first_last = []
    for _, g in ev.history.groupby("fold"):
        if "train_loss" in g and len(g) > 1:
            first_last.append(float(g["train_loss"].iloc[0]) > float(g["train_loss"].iloc[-1]))
    checks = {
        "score_above_zero": bool(ev.mean > 0.0),
        "train_loss_decreased_in_all_folds": bool(all(first_last)) if first_last else None,
        "metrics_finite": bool(np.isfinite(ev.fold_scores).all()),
        "all_folds_completed": len(ev.fold_scores) == len(folds),
    }
    if not all(v for v in checks.values() if v is not None):
        log.warning("BASELINE SOSPECHOSO (sec. 11: revisar etiquetas/convergencia antes del HPO): %s", checks)
    summary = {
        "architecture": spec.name, "slug": spec.slug, "task": spec.task, "stage": "baseline", "objective": objective_metric(spec), "hp": hp,
        "config_id": config_id(hp), "seed": seed, "folds": folds, "fold_scores": ev.fold_scores, "score_mean": ev.mean, "score_sd": ev.sd,
        "best_epochs": [f["best_epoch"] for f in ev.folds], "epochs_run": [f["epochs_run"] for f in ev.folds], "train_seconds": ev.train_seconds,
        "vram_peak_mb": ev.vram_peak_mb, "params_m": ev.params_m, "gflops": ev.gflops, "max_epochs": max_epochs, "patience": patience,
        "convergence_checks": checks, "used_to_discard_architectures": False, "dataset_hash": data.dataset_hash, "split_hash": data.split_hash, "pc_id": ctx.pc_id,
    }
    write_json(summary_path, summary)
    ctx.log_metrics(summary)
    ctx.register_artifact(summary_path, "resumen del baseline")
    log.info("BASELINE %s: %s medio = %.4f (folds %s) | %.1f min | VRAM pico %.0f MB", spec.name, objective_metric(spec), ev.mean,
             [round(s, 4) for s in ev.fold_scores], ev.train_seconds / 60, ev.vram_peak_mb or 0)


if __name__ == "__main__":
    raise SystemExit(run_cli(SCRIPT, main, description=__doc__ or SCRIPT, add_arguments=add_arguments, run_layout=layout_for("baseline")))
