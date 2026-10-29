"""08 - Robustez: top-3 configuraciones x 3 seeds (Especificacion s2, sec. 12.3, 16.5, 18).

Toma las top-3 configuraciones DISTINTAS del estudio de la arquitectura y repite cada una con las seeds
predeclaradas ``[17, 42, 73]`` sobre los 3 folds internos (mismo presupuesto que el HPO). Las tres seeds
son independientes: pueden repartirse entre las 3 PCs (``--config-id`` + ``--seed``). Despues
(``--aggregate-only``) se agrega media +- SD y se **elige la configuracion por la MEDIA entre seeds**, nunca
por un run individual; se congela en ``artifacts/best_configs/<slug>.json`` (hiperparametros + numero de
epocas definitivo = media de las mejores epocas de los folds/seeds de esa configuracion).

Modos::

    python src/08_repeat_top_configs.py --architecture YOLOv8n                   # todo en esta PC
    python src/08_repeat_top_configs.py --architecture YOLOv8n --config-id cfg_ab12cd34 --seed 17
    python src/08_repeat_top_configs.py --architecture YOLOv8n --aggregate-only
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

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

from utils.architectures import get_arch, section_for  # noqa: E402
from utils.checks import MethodologyViolation  # noqa: E402
from utils.guards import GpuJobLock, assert_one_selected_by_mean  # noqa: E402
from utils.hpo_results import best_config_path, top_configs  # noqa: E402
from utils.hpo_runner import evaluate_config, objective_metric  # noqa: E402
from utils.io import read_json, write_csv, write_json  # noqa: E402
from utils.logging_utils import RunContext, run_cli  # noqa: E402
from utils.optuna_storage import load_study  # noqa: E402
from utils.plots_experiment import plot_robustness  # noqa: E402
from utils.run_common import add_arch_arguments, arch_from_args, guard_not_frozen, nested_run, preflight_training  # noqa: E402
from utils.run_figures import save_training_figures  # noqa: E402
from utils.search_spaces import search_space_hash  # noqa: E402
from utils.training_common import mean_sd  # noqa: E402

SCRIPT = "08_repeat_top_configs"


def add_arguments(parser: argparse.ArgumentParser) -> None:
    """Argumentos propios del script."""
    add_arch_arguments(parser)
    g = parser.add_argument_group("08_repeat_top_configs")
    g.add_argument("--config-id", default=None, help="repetir solo esta configuracion (cfg_xxxxxxxx)")
    g.add_argument("--seed", type=int, default=None, help="repetir solo con esta seed (debe estar en robustness.seeds)")
    g.add_argument("--aggregate-only", action="store_true", help="no entrena: agrega los resultados existentes y congela la mejor configuracion")
    g.add_argument("--force", action="store_true", help="repite ejecuciones que ya tienen result.json")


def _layout(a: argparse.Namespace) -> tuple[str, str, bool]:
    slug = get_arch(a.architecture).slug
    if a.aggregate_only:
        return slug, "robustness/aggregate", True
    if a.config_id and a.seed is not None:
        return slug, f"robustness/{a.config_id}/seed_{a.seed}", True
    return slug, "robustness/run_all", False


def run_one(parent: RunContext, spec: Any, cfg: dict[str, Any], cand: dict[str, Any], seed: int, *, nested: bool, force: bool) -> dict[str, Any]:
    """Entrena ``cand`` con ``seed`` en los folds internos y escribe ``result.json`` en su run."""
    sec = section_for(spec, cfg)
    k = int(cfg["inner_hpo"]["folds"])
    cid = cand["config_id"]
    run = nested_run(parent, spec, f"robustness/{cid}/seed_{seed}", extra_cfg={"architecture": spec.name, "config_id": cid, "hp": cand["hp"], "seed": seed,
                                                                             "folds": list(range(k)), "hpo_trial_number": cand["trial_number"]}) if nested else parent
    result_path = run.run_dir / "result.json"
    if result_path.is_file() and not force:
        parent.logger.info("%s seed %d ya tiene result.json: se reutiliza", cid, seed)
        return dict(read_json(result_path))

    def _work(r: RunContext) -> dict[str, Any]:
        r.set_seed(seed)
        r.add_metadata(architecture=spec.name, stage="robustness", config_id=cid, seed=seed, hp=cand["hp"])
        ev = evaluate_config(spec, cand["hp"], cfg, r, r.run_dir, folds=list(range(k)), seed=seed, max_epochs=int(sec["max_epochs"]),
                             patience=int(sec["early_stopping_patience"]))
        save_training_figures(r, spec, ev.history, "robustez", r.run_dir / "history.csv")
        res = {"architecture": spec.name, "config_id": cid, "seed": seed, "hp": cand["hp"], "fold_scores": ev.fold_scores, "score": ev.mean,
               "best_epochs": [f["best_epoch"] for f in ev.folds], "epochs_run": [f["epochs_run"] for f in ev.folds], "train_seconds": ev.train_seconds,
               "vram_peak_mb": ev.vram_peak_mb, "gflops": ev.gflops, "params_m": ev.params_m, "hpo_trial_number": cand["trial_number"], "pc_id": r.pc_id}
        write_json(r.run_dir / "result.json", res)
        r.log_metrics(res)
        return res

    if nested:
        with run:
            return _work(run)
    return _work(run)


def aggregate(ctx: RunContext, spec: Any, cfg: dict[str, Any], cands: list[dict[str, Any]], rob_dir: Path) -> dict[str, Any]:
    """Agrega media +- SD por configuracion, elige por la MEDIA entre seeds y congela la mejor."""
    seeds = [int(s) for s in cfg["robustness"]["seeds"]]
    rows: list[dict[str, Any]] = []
    detail: dict[str, dict[int, dict[str, Any]]] = {}
    missing: list[str] = []
    for c in cands:
        for s in seeds:
            p = rob_dir / c["config_id"] / f"seed_{s}" / "result.json"
            if not p.is_file():
                missing.append(f"{c['config_id']}/seed_{s}")
                continue
            res = read_json(p)
            detail.setdefault(c["config_id"], {})[s] = res
            rows.append({"config_id": c["config_id"], "seed": s, "score": float(res["score"]), "fold_scores": res["fold_scores"], "best_epochs": res["best_epochs"]})
    if missing:
        raise MethodologyViolation(f"Faltan resultados de robustez (se elige por la media de las {len(seeds)} seeds): {missing}")
    per_seed = pd.DataFrame(rows)
    summ: list[dict[str, Any]] = []
    for c in cands:
        cid = c["config_id"]
        assert_one_selected_by_mean({s: [float(detail[cid][s]["score"])] for s in detail[cid]}, seeds)
        scores = [float(detail[cid][s]["score"]) for s in seeds]
        pooled = [float(x) for s in seeds for x in detail[cid][s]["fold_scores"]]
        m, sd = mean_sd(scores)
        pm, psd = mean_sd(pooled)
        epochs = [int(e) for s in seeds for e in detail[cid][s]["best_epochs"]]
        summ.append({"config_id": cid, "hpo_trial_number": c["trial_number"], "hpo_value": c["value"], "mean": m, "sd": sd, "pooled_mean": pm, "pooled_sd": psd,
                     "mean_best_epoch": float(np.mean(epochs)), "n_runs": len(pooled)})
    summary = pd.DataFrame(summ)
    order = summary.sort_values(["mean", "sd", "hpo_trial_number"], ascending=[False, True, True]).index[0]
    summary["selected"] = summary.index == order
    sel = summary.loc[order]
    write_csv(summary, rob_dir / "summary.csv", float_format="%.8g")
    write_csv(per_seed.assign(fold_scores=per_seed["fold_scores"].map(lambda v: "|".join(f"{x:.6g}" for x in v)),
                              best_epochs=per_seed["best_epochs"].map(lambda v: "|".join(str(x) for x in v))), rob_dir / "per_seed.csv", float_format="%.8g")
    ctx.save_figure(plot_robustness(per_seed, summary, title=f"Robustez {spec.name}: {objective_metric(spec)} (media ± SD entre seeds)"), f"robustez_{spec.slug}",
                    data=per_seed[["config_id", "seed", "score"]], sources=[rob_dir / "per_seed.csv", rob_dir / "summary.csv"])
    chosen = next(c for c in cands if c["config_id"] == sel["config_id"])
    base_path = ctx.runs_base / spec.slug / "baseline" / "summary.json"
    best = {
        "architecture": spec.name, "slug": spec.slug, "task": spec.task, "objective": objective_metric(spec), "config_id": chosen["config_id"], "hp": chosen["hp"],
        "score_mean": float(sel["mean"]), "score_sd": float(sel["sd"]) if pd.notna(sel["sd"]) else None, "pooled_mean": float(sel["pooled_mean"]),
        "pooled_sd": float(sel["pooled_sd"]) if pd.notna(sel["pooled_sd"]) else None, "per_seed_scores": {int(s): float(detail[chosen["config_id"]][s]["score"]) for s in seeds},
        "seeds": seeds, "final_epochs": max(1, int(round(float(sel["mean_best_epoch"])))), "hpo_trial_number": int(chosen["trial_number"]), "hpo_value": float(chosen["value"]),
        "baseline_score_mean": float(read_json(base_path)["score_mean"]) if base_path.is_file() else None, "selection_rule": "max media entre seeds (tie: menor SD, menor n.º de trial)",
        "selected_by_single_seed": False, "search_space_sha256": search_space_hash(), "dataset_hash": ctx.hashes.get("dataset_hash"), "split_hash": ctx.hashes.get("split_hash"),
        "code_hash": ctx.hashes.get("code_hash"), "pc_id": ctx.pc_id,
    }
    write_json(best_config_path(cfg, spec), best)
    write_json(rob_dir / "summary.json", {"architecture": spec.name, "candidates": summary.to_dict("records"), "selected": best})
    ctx.register_artifact(best_config_path(cfg, spec), "configuracion congelada de la arquitectura")
    ctx.log_metrics({"robustness": summary.to_dict("records"), "selected": best})
    ctx.logger.info("ROBUSTEZ %s: elegida %s con %.4f ± %s (media de %d seeds); epocas definitivas=%d", spec.name, best["config_id"], best["score_mean"],
                    f"{best['score_sd']:.4f}" if best["score_sd"] is not None else "n/a", len(seeds), best["final_epochs"])
    return best


def main(ctx: RunContext, args: argparse.Namespace) -> None:
    """Repite las top-k configuraciones con las seeds predeclaradas y/o agrega la robustez."""
    cfg, log = ctx.cfg, ctx.logger
    spec = arch_from_args(args)
    preflight_training(ctx, args)
    guard_not_frozen(ctx, "08_repeat_top_configs")
    seeds = [int(s) for s in cfg["robustness"]["seeds"]]
    if args.seed is not None and args.seed not in seeds:
        raise MethodologyViolation(f"La seed {args.seed} no esta predeclarada ({seeds}); las mismas 3 seeds se usan en todas las arquitecturas (sec. 12.3)")
    rob_dir = ctx.runs_base / spec.slug / "robustness"
    study = load_study(spec, cfg)
    cands = top_configs(study, int(cfg["robustness"]["top_k_configs"]))
    if not cands:
        raise MethodologyViolation(f"El estudio de {spec.name} no tiene trials COMPLETE: ejecute 07_run_hpo_study.py")
    log.info("top-%d configuraciones de %s: %s", len(cands), spec.name, [(c["config_id"], round(c["value"], 4)) for c in cands])
    ctx.add_metadata(architecture=spec.name, stage="robustness", candidates=[c["config_id"] for c in cands])

    if not args.aggregate_only:
        todo = [(c, s) for c in cands for s in seeds if (args.config_id is None or c["config_id"] == args.config_id) and (args.seed is None or s == args.seed)]
        if not todo:
            raise MethodologyViolation(f"--config-id {args.config_id} no esta entre las top configuraciones {[c['config_id'] for c in cands]}")
        single = args.config_id is not None and args.seed is not None
        with GpuJobLock(cfg, ctx.pc_id, job=f"repeat:{spec.slug}"):
            for c, s in todo:
                log.info("=== robustez %s %s seed %d ===", spec.name, c["config_id"], s)
                run_one(ctx, spec, cfg, c, s, nested=not single, force=args.force)
    if args.aggregate_only or (args.config_id is None and args.seed is None):
        aggregate(ctx, spec, cfg, cands, rob_dir)


if __name__ == "__main__":
    raise SystemExit(run_cli(SCRIPT, main, description=__doc__ or SCRIPT, add_arguments=add_arguments, run_layout=_layout))
