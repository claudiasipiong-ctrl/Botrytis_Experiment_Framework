"""07 - Estudio HPO de UNA arquitectura (Especificacion s2, sec. 10, 12, 15, 18, 19).

* Un estudio Optuna INDEPENDIENTE por arquitectura (``botrytis_<slug>``), nunca ``study_global``.
* TPE (multivariate, group, 5 trials de arranque) + MedianPruner (5 trials, 8 pasos de calentamiento).
* Detector: 15 trials; clasificador: 20 trials. Trials SECUENCIALES (``n_jobs = 1``) dentro del estudio.
* Cada trial evalua los 3 folds internos (solo outer train); el objetivo es la media de los folds
  (detector: mAP@0.5:0.95; clasificador: AP de ``con_signos``). El validation externo y el test NO se tocan.
* Reanudable (``load_if_exists``): un corte no pierde trials; los trials RUNNING huerfanos se marcan FAIL.

Por trial deja ``runs/<arquitectura>/hpo/trial_NNN/`` con ``execution.log``, ``config.yaml``,
``environment.json``, ``metrics.json``, ``checkpoint_metadata.json``, ``fold_metrics.csv``, ``history.csv`` y
figuras; registra hiperparametros, scores por fold, objetivo, tiempo, VRAM pico, FLOPs y parametros.
A nivel de estudio: ``runs/<arquitectura>/hpo/{study_summary.json,trials.csv,intermediate_values.csv}`` y las
figuras ``hpo_curvas_<slug>`` / ``hpo_distribuciones_<slug>``.

Ejemplo::

    python src/07_run_hpo_study.py --architecture ResNet50 --pc-id PC-03
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

import pandas as pd  # noqa: E402

from orchestration.study_queue import StudyQueue  # noqa: E402
from utils.architectures import section_for, trials_for  # noqa: E402
from utils.checks import MethodologyViolation  # noqa: E402
from utils.guards import GpuJobLock, assert_equal_hpo_budget  # noqa: E402
from utils.io import write_csv, write_json  # noqa: E402
from utils.logging_utils import RunContext, run_cli  # noqa: E402
from utils.optuna_storage import budget_used, create_or_load_study, failed_trials, trials_table  # noqa: E402
from utils.plots_experiment import plot_hpo_curves, plot_optuna_distributions  # noqa: E402
from utils.run_common import add_arch_arguments, arch_from_args, guard_not_frozen, nested_run, preflight_training  # noqa: E402
from utils.run_figures import save_training_figures  # noqa: E402
from utils.search_spaces import (  # noqa: E402
    baseline_hyperparameters, config_id, enqueue_params_from_hp, load_search_spaces, search_space_hash, space_for, suggest_hyperparameters,
)

SCRIPT = "07_run_hpo_study"


def add_arguments(parser: argparse.ArgumentParser) -> None:
    """Argumentos propios del script."""
    add_arch_arguments(parser)
    g = parser.add_argument_group("07_run_hpo_study")
    g.add_argument("--trial-block-size", type=int, default=None, help="maximo de trials nuevos en este bloque; conserva presupuesto total 15/20")
    g.add_argument("--n-trials", type=int, default=None, help="presupuesto del estudio (por defecto el de la metodologia: 15/20)")
    g.add_argument("--allow-budget-override", action="store_true", help="permite un presupuesto distinto al oficial (queda DECLARADO en metadata.json)")


def fail_zombie_trials(study: Any, log: Any) -> int:
    """Recover only provably dead local trials through public Optuna APIs."""
    import optuna, socket, psutil
    n = 0
    for t in study.get_trials(deepcopy=False):
        if t.state != optuna.trial.TrialState.RUNNING: continue
        owner = t.user_attrs.get('process_owner',{})
        if owner.get('hostname') != socket.gethostname() or not owner.get('pid'):
            raise RuntimeError('Trial RUNNING sin propietario local verificable; recuperar explicitamente, nunca asumir zombie')
        try:
            proc = psutil.Process(owner['pid'])
            alive = abs(proc.create_time()-owner.get('created',0)) < 1
        except psutil.NoSuchProcess:
            alive = False
        if alive: raise RuntimeError('Trial RUNNING con proceso vivo; no se recupera')
        study.tell(t.number, state=optuna.trial.TrialState.FAIL)
        log.warning('trial %d: proceso local muerto verificado, FAIL via API publica',t.number)
        n += 1
    return n


def intermediate_table(study: Any) -> pd.DataFrame:
    """``trial, step, value`` de los valores intermedios reportados al pruner."""
    rows = [{"trial": t.number, "step": s, "value": v} for t in study.get_trials(deepcopy=False) for s, v in t.intermediate_values.items()]
    return pd.DataFrame(rows, columns=["trial", "step", "value"])


def write_study_outputs(ctx: RunContext, spec: Any, study: Any, hpo_dir: Path, *, make_figures: bool) -> pd.DataFrame:
    """Escribe ``trials.csv``, ``intermediate_values.csv``, ``study_summary.json`` (y las figuras del estudio)."""
    from utils.hpo_runner import objective_metric
    tt = trials_table(study)
    inter = intermediate_table(study)
    flat = tt.copy()
    if "fold_scores" in flat:
        flat["fold_scores"] = flat["fold_scores"].map(lambda v: "|".join(f"{x:.6g}" for x in v) if isinstance(v, (list, tuple)) else v)
    write_csv(flat, hpo_dir / "trials.csv", float_format="%.8g")
    write_csv(inter, hpo_dir / "intermediate_values.csv", float_format="%.8g")
    done = tt[tt["state"] == "COMPLETE"] if not tt.empty else tt
    best = None
    if len(done):
        b = done.sort_values(["value", "number"], ascending=[False, True]).iloc[0]
        best = {"number": int(b["number"]), "value": float(b["value"]), "config_id": b.get("config_id")}
    summary = {
        "architecture": spec.name, "study_name": study.study_name, "objective": objective_metric(spec), "direction": "maximize",
        "n_trials_total": int(len(tt)), "n_complete": int((tt["state"] == "COMPLETE").sum()) if len(tt) else 0,
        "n_pruned": int((tt["state"] == "PRUNED").sum()) if len(tt) else 0, "n_failed": int((tt["state"] == "FAIL").sum()) if len(tt) else 0,
        "best": best, "search_space_sha256": search_space_hash(),
    }
    write_json(hpo_dir / "study_summary.json", summary)
    if make_figures and len(tt):
        slug = spec.slug
        ctx.save_figure(plot_hpo_curves(tt, inter, title=f"HPO {spec.name}", ylabel=f"{objective_metric(spec)} (media de folds)"), f"hpo_curvas_{slug}",
                        data=tt[["number", "state", "value"]], sources=[hpo_dir / "trials.csv", hpo_dir / "intermediate_values.csv"])
        importances = None
        try:
            import optuna

            if len(done) >= 4:
                importances = optuna.importance.get_param_importances(study)
        except Exception as exc:  # pocas muestras o sklearn ausente: la figura sigue mostrando los slices
            ctx.logger.info("importancia de parametros no disponible: %s", exc)
        params = [c for c in space_for(spec, ctx.cfg) if c in tt.columns]
        ctx.save_figure(plot_optuna_distributions(tt, params, importances, title=f"Distribuciones Optuna {spec.name}"), f"hpo_distribuciones_{slug}",
                        data=tt[[c for c in ["number", "value", *params] if c in tt.columns]], sources=[hpo_dir / "trials.csv"])
    ctx.log_metrics({"study": summary})
    return tt


def main(ctx: RunContext, args: argparse.Namespace) -> None:
    """Ejecuta (o reanuda) el estudio HPO de la arquitectura."""
    import optuna

    cfg, log = ctx.cfg, ctx.logger
    spec = arch_from_args(args)
    data = preflight_training(ctx, args)
    from utils.hpo_runner import evaluate_config, objective_metric
    guard_not_frozen(ctx, "07_run_hpo_study")
    sec = section_for(spec, cfg)
    n_trials = trials_for(spec, cfg)
    if args.n_trials is not None and int(args.n_trials) != n_trials:
        if cfg['project']['profile']=='official':
            raise MethodologyViolation('Presupuesto oficial inmutable; trial-block-size limita el bloque sin reducir 15/20')
        assert_equal_hpo_budget(cfg, {spec.name: int(args.n_trials)}, declared_override=bool(args.allow_budget_override))
        n_trials = int(args.n_trials)
        ctx.add_metadata(budget_override_declared=True, n_trials=n_trials)
    seed = int(cfg["reproducibility"]["seed"])
    ctx.set_seed(seed)
    k = int(cfg["inner_hpo"]["folds"])
    max_epochs, patience = int(sec["max_epochs"]), int(sec["early_stopping_patience"])
    spaces = load_search_spaces()
    hpo_dir = ctx.run_dir.parent if ctx.run_dir.parent.name == "hpo" else ctx.run_dir
    queue = StudyQueue.from_cfg(cfg)
    prov = {"git_commit": ctx.env.get("git_commit"), "dataset_hash": ctx.hashes.get("dataset_hash"), "split_hash": ctx.hashes.get("split_hash")}
    ctx.add_metadata(architecture=spec.name, stage="hpo", n_trials_budget=n_trials, search_space_sha256=search_space_hash())

    with GpuJobLock(cfg, ctx.pc_id, job=f"hpo:{spec.slug}"):
        study = create_or_load_study(spec, cfg, seed)
        fail_zombie_trials(study, log)
        if spec.family == "yolo26" and bool(cfg["detector"].get("yolo26_enqueue_recipe_trial", True)) and len(study.trials) == 0:
            study.enqueue_trial(enqueue_params_from_hp(spec, baseline_hyperparameters(spec, cfg), cfg, spaces), skip_if_exists=True)
            log.info("YOLO26: trial 0 = recipe baseline (centro de busqueda, sec. 15.2)")
        max_failed = int(cfg["optuna"].get("max_failed_trials", 3))

        def objective(trial: Any) -> float:
            import os, socket, psutil
            trial.set_user_attr('process_owner',dict(hostname=socket.gethostname(),pid=os.getpid(),created=psutil.Process().create_time()))
            hp = suggest_hyperparameters(trial, spec, cfg, spaces)
            cid = config_id(hp)
            name = f"hpo/trial_{trial.number:03d}"
            trial.set_user_attr("hp", hp)
            trial.set_user_attr("config_id", cid)
            trial.set_user_attr("pc_id", ctx.pc_id)
            trial.set_user_attr("run_dir", str(ctx.runs_base / spec.slug / name))
            queue.record_trial(study.study_name, trial.number, "RUNNING", pc_id=ctx.pc_id, run_dir=trial.user_attrs["run_dir"], provenance=prov)
            run = nested_run(ctx, spec, name, extra_cfg={"architecture": spec.name, "study_name": study.study_name, "trial_number": trial.number,
                                                          "hp": hp, "config_id": cid, "seed": seed, "folds": list(range(k)), "max_epochs": max_epochs,
                                                          "patience": patience})
            with run:
                run.set_seed(seed)
                run.add_metadata(architecture=spec.name, stage="hpo", trial_number=trial.number, hp=hp, config_id=cid)
                try:
                    ev = evaluate_config(spec, hp, cfg, run, run.run_dir, folds=list(range(k)), seed=seed, max_epochs=max_epochs, patience=patience, trial=trial)
                except optuna.TrialPruned:
                    run.log_metrics({"architecture": spec.name, "trial_number": trial.number, "hp": hp, "status": "PRUNED",
                                     "pruned_at_step": trial.user_attrs.get("pruned_at_step")})
                    raise
                run.log_metrics({
                    "architecture": spec.name, "trial_number": trial.number, "config_id": cid, "hp": hp, "fold_scores": ev.fold_scores,
                    "objective": objective_metric(spec), "value": ev.mean, "value_sd": ev.sd, "train_seconds": ev.train_seconds,
                    "vram_peak_mb": ev.vram_peak_mb, "gflops": ev.gflops, "params_m": ev.params_m, "folds": ev.folds,
                })
                save_training_figures(run, spec, ev.history, "trial", run.run_dir / "history.csv")
            for key, val in (("fold_scores", ev.fold_scores), ("train_seconds", ev.train_seconds), ("vram_peak_mb", ev.vram_peak_mb),
                             ("gflops", ev.gflops), ("params_m", ev.params_m), ("n_folds_done", len(ev.fold_scores))):
                trial.set_user_attr(key, val)
            return ev.mean

        def after_trial(st: Any, ft: Any) -> None:
            mapped = {"COMPLETE": "COMPLETE", "PRUNED": "PRUNED", "FAIL": "FAILED", "RUNNING": "RUNNING"}.get(ft.state.name, "FAILED")
            queue.record_trial(st.study_name, ft.number, mapped, value=ft.value, pc_id=ctx.pc_id, run_dir=ft.user_attrs.get("run_dir"), provenance=prov, finished=True)
            write_study_outputs(ctx, spec, st, hpo_dir, make_figures=False)
            if failed_trials(st) > max_failed:
                log.error("demasiados trials FAILED (%d > %d): se detiene el estudio", failed_trials(st), max_failed)
                st.stop()

        remaining = n_trials - budget_used(study)
        log.info("estudio %s: %d/%d trials usados; restan %d (TPE + MedianPruner, n_jobs=1)", study.study_name, budget_used(study), n_trials, remaining)
        block = args.trial_block_size or cfg.get('experiment',{}).get('trial_block_size') or remaining
        if int(block) < 1: raise ValueError('trial-block-size debe ser positivo')
        block_target = min(n_trials, budget_used(study) + int(block))
        remaining = block_target - budget_used(study)
        guard_rounds = 0
        while remaining > 0 and failed_trials(study) <= max_failed and guard_rounds < n_trials + max_failed + 2:
            study.optimize(objective, n_trials=remaining, n_jobs=1, callbacks=[after_trial], catch=(Exception,))
            remaining = block_target - budget_used(study)
            guard_rounds += 1
        if failed_trials(study) > max_failed:
            write_study_outputs(ctx, spec, study, hpo_dir, make_figures=True)
            raise MethodologyViolation(f"El estudio {study.study_name} acumulo {failed_trials(study)} trials FAILED; revise los logs de runs/{spec.slug}/hpo/")

    tt = write_study_outputs(ctx, spec, study, hpo_dir, make_figures=True)
    done = tt[tt["state"] == "COMPLETE"]
    if budget_used(study) < n_trials:
        log.info("Bloque HPO guardado: %d/%d; job sigue PENDING",budget_used(study),n_trials)
        return
    if done.empty:
        raise MethodologyViolation(f"El estudio {study.study_name} no tiene ningun trial COMPLETE")
    log.info("HPO %s terminado: mejor trial %d valor %.4f | %d COMPLETE, %d PRUNED", spec.name, int(done.sort_values('value', ascending=False).iloc[0]['number']),
             float(done['value'].max()), len(done), int((tt['state'] == 'PRUNED').sum()))
    ctx.add_metadata(dataset_hash=data.dataset_hash, split_hash=data.split_hash)


if __name__ == "__main__":
    def _layout(a: argparse.Namespace) -> tuple[str, str, bool]:
        from utils.architectures import get_arch

        return get_arch(a.architecture).slug, "hpo/study_run", False

    raise SystemExit(run_cli(SCRIPT, main, description=__doc__ or SCRIPT, add_arguments=add_arguments, run_layout=_layout))
