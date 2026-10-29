"""18 - Consolidacion final: master_results.csv, tablas, figuras y auditoria cientifica (sec. 31, 33).

* ``artifacts/master_results.csv``: UNA fila por run de ``runs/**`` con los campos de la sec. 33 (run_id,
  arquitectura, tarea, PC, seed, splits accedidos, folds, hiperparametros, hashes de git/dataset/split/entorno,
  inicio/fin, estado, metricas, latencia, hash del checkpoint).
* Tablas finales en ``reports/tables/``: todos los trials HPO, robustez (media +- SD), baseline vs mejor HPO,
  validation de detectores/clasificadores (con latencia y VRAM del benchmark) y resultados de test.
* Figuras finales de comparacion (mAP, radar, Pareto, FLOPs/parametros...; con latencia si ya corrio el 17),
  curvas HPO/Optuna por arquitectura (regeneradas desde los resultados si faltan) y eficiencia.
* Auditoria cientifica automatica (``reports/scientific_audit.json`` y ``.md``) y verificacion de los graficos
  OBLIGATORIOS con procedencia de datos reales (``reports/figure_audit.json``). Termina con exit 1 si alguna
  comprobacion FALLA (``--no-fail`` para informes parciales).
* ``artifacts/provenance.json``: hashes, versiones, PCs y SHA-256 de los artefactos clave.

Ejemplo::

    python src/18_export_master_results.py
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
import json
import sys
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parent))

import pandas as pd  # noqa: E402

from orchestration.pc_registry import PCRegistry  # noqa: E402
from orchestration.study_queue import StudyQueue  # noqa: E402
from utils.architectures import archs  # noqa: E402
from utils.config_loader import cfg_path  # noqa: E402
from utils.hashing import sha256_file  # noqa: E402
from utils.hpo_results import stage_json_path  # noqa: E402
from utils.io import read_csv, read_json, write_csv, write_json  # noqa: E402
from utils.logging_utils import RunContext, run_cli  # noqa: E402
from utils.optuna_storage import load_study, study_db_path  # noqa: E402
from utils.plots_experiment import plot_hpo_curves, plot_optuna_distributions  # noqa: E402
from utils.prepared import load_prepared_data  # noqa: E402
from utils.reporting import attach_benchmark, collect_validation_table, save_classification_figures, save_detection_figures  # noqa: E402
from utils.scientific_audit import audit_markdown, iter_runs, run_audit  # noqa: E402
from utils.search_spaces import space_for  # noqa: E402
from utils.visualization import (  # noqa: E402
    plot_baseline_vs_hpo, plot_params_flops_metric, plot_performance_efficiency_comparison, plot_training_time_by_architecture, plot_vram_usage_by_model,
    plot_winning_hyperparameter_distribution,
)

SCRIPT = "18_export_master_results"


def add_arguments(parser: argparse.ArgumentParser) -> None:
    """Argumentos propios del script."""
    parser.add_argument("--no-fail", action="store_true", help="no terminar con error si la auditoria tiene FAIL (informes parciales)")
    parser.add_argument("--allow-missing-figures", action="store_true", help="las figuras obligatorias ausentes son WARN (etapas aun no ejecutadas)")


def master_table(cfg: dict[str, Any], runs_dir: Path) -> pd.DataFrame:
    """Una fila por run con los campos de la sec. 33."""
    det_slugs = {a.slug for a in archs("detection")}
    rows: list[dict[str, Any]] = []
    for d in iter_runs(runs_dir):
        meta = read_json(d / "metadata.json")
        metrics = read_json(d / "metrics.json") if (d / "metrics.json").is_file() else {}
        ck = read_json(d / "checkpoint_metadata.json") if (d / "checkpoint_metadata.json").is_file() else {}
        ex = meta.get("extra", {}) or {}
        arch = meta.get("architecture")
        hashes = [c.get("sha256") for c in ck.get("checkpoints", []) if c.get("sha256")]
        rows.append({
            "run_id": meta.get("run_id"), "architecture": arch, "task": ("detection" if arch in det_slugs else "classification") if arch else "pipeline",
            "stage": ex.get("stage"), "script": meta.get("script"), "pc_id": meta.get("pc_id"), "seed": meta.get("seed"),
            "outer_split": ";".join(sorted({a["split"] for a in meta.get("splits_accessed", [])})), "inner_fold": ex.get("folds", "all" if ex.get("stage") in ("baseline", "hpo", "robustness") else ""),
            "hyperparameters": json.dumps(ex.get("hp"), sort_keys=True) if ex.get("hp") else "", "git_commit": meta.get("git_commit"), "dataset_hash": meta.get("dataset_hash"),
            "split_hash": meta.get("split_hash"), "environment_hash": meta.get("environment_hash"), "code_hash": meta.get("code_hash"), "start_time": meta.get("started_at"),
            "end_time": meta.get("finished_at"), "status": meta.get("status"), "duration_seconds": meta.get("duration_seconds"),
            "metrics": json.dumps(metrics, default=str)[:20000], "checkpoint_hash": ";".join(hashes), "run_dir": str(d),
        })
    return pd.DataFrame(rows)


def collect_hpo_tables(cfg: dict[str, Any], runs_dir: Path) -> tuple[pd.DataFrame, pd.DataFrame]:
    """``(todos los trials HPO, resumen de robustez)`` de las 7 arquitecturas."""
    trials, rob = [], []
    for spec in archs():
        p = runs_dir / spec.slug / "hpo" / "trials.csv"
        if p.is_file():
            trials.append(read_csv(p).assign(architecture=spec.name))
        r = runs_dir / spec.slug / "robustness" / "summary.csv"
        if r.is_file():
            rob.append(read_csv(r).assign(architecture=spec.name))
    return (pd.concat(trials, ignore_index=True) if trials else pd.DataFrame()), (pd.concat(rob, ignore_index=True) if rob else pd.DataFrame())


def regenerate_hpo_figures(ctx: RunContext, cfg: dict[str, Any], runs_dir: Path) -> int:
    """Curvas HPO y distribuciones Optuna de cada arquitectura desde ``trials.csv`` REAL (si faltan en reports/figures)."""
    rep = cfg_path(cfg, "paths.figures_dir")
    n = 0
    for spec in archs():
        hp_dir = runs_dir / spec.slug / "hpo"
        tp = hp_dir / "trials.csv"
        if not tp.is_file():
            continue
        tt = read_csv(tp)
        ip = hp_dir / "intermediate_values.csv"
        inter = read_csv(ip) if ip.is_file() else pd.DataFrame(columns=["trial", "step", "value"])
        if not (rep / f"hpo_curvas_{spec.slug}.png").is_file():
            ctx.save_figure(plot_hpo_curves(tt, inter, title=f"HPO {spec.name}"), f"hpo_curvas_{spec.slug}", data=tt[["number", "state", "value"]], sources=[tp, ip])
            n += 1
        if not (rep / f"hpo_distribuciones_{spec.slug}.png").is_file():
            importances = None
            try:
                import optuna

                if study_db_path(cfg, spec).is_file():
                    st = load_study(spec, cfg)
                    if sum(1 for t in st.trials if t.state.name == "COMPLETE") >= 4:
                        importances = optuna.importance.get_param_importances(st)
            except Exception:
                importances = None
            params = [c for c in space_for(spec, cfg) if c in tt.columns]
            ctx.save_figure(plot_optuna_distributions(tt, params, importances, title=f"Distribuciones Optuna {spec.name}"), f"hpo_distribuciones_{spec.slug}",
                            data=tt[[c for c in ["number", "value", *params] if c in tt.columns]], sources=[tp])
            n += 1
    return n


def efficiency_tables(cfg: dict[str, Any], runs_dir: Path) -> pd.DataFrame:
    """Baseline vs HPO y costes de entrenamiento por arquitectura."""
    rows = []
    best_dir = cfg_path(cfg, "paths.best_configs_dir")
    for spec in archs():
        bp, cp = runs_dir / spec.slug / "baseline" / "summary.json", best_dir / f"{spec.slug}.json"
        if not (bp.is_file() and cp.is_file()):
            continue
        b, c = read_json(bp), read_json(cp)
        fm = cfg_path(cfg, "paths.artifacts_dir") / "checkpoints" / spec.slug / "final_meta.json"
        f = read_json(fm) if fm.is_file() else {}
        rows.append({"architecture": spec.name, "task": spec.task, "baseline": b["score_mean"], "hpo_mean": c["score_mean"], "hpo_sd": c["score_sd"], "gflops": b.get("gflops"),
                     "params_m": b.get("params_m"), "vram_mb": f.get("vram_peak_mb") or b.get("vram_peak_mb"), "train_minutes": (f.get("train_seconds") or b.get("train_seconds")) / 60 if (f.get("train_seconds") or b.get("train_seconds")) is not None else None,
                     "baseline_minutes": b["train_seconds"] / 60 if b.get("train_seconds") is not None else None})
    return pd.DataFrame(rows)


def main(ctx: RunContext, args: argparse.Namespace) -> None:
    """Consolida todo, genera figuras/tablas finales y ejecuta la auditoria cientifica."""
    cfg, log = ctx.cfg, ctx.logger
    data = load_prepared_data(cfg, with_folds=True)
    ctx.set_hashes(dataset_hash=data.dataset_hash, split_hash=data.split_hash)
    art, runs_dir = cfg_path(cfg, "paths.artifacts_dir"), ctx.runs_base
    tables = cfg_path(cfg, "paths.tables_dir")
    tables.mkdir(parents=True, exist_ok=True)

    from utils.split_summary import export_split_summary
    export_split_summary(cfg, data)

    # -- 1) master_results.csv y tablas ----------------------------------------------------
    master = master_table(cfg, runs_dir)
    bench = art / "benchmark"
    lat = {t: read_csv(bench / f"latency_{t}.csv") for t in ("detection", "classification") if (bench / f"latency_{t}.csv").is_file()}
    if len(master) and lat:
        lm = pd.concat([v[["architecture", "latency_mean_ms"]].assign(slug=v["slug"]) for v in lat.values()])
        master["latency_ms"] = master["architecture"].map(dict(zip(lm["slug"], lm["latency_mean_ms"])))
    write_csv(master, art / "master_results.csv")
    ctx.register_artifact(art / "master_results.csv", "resultados maestros (una fila por run)")
    trials, rob = collect_hpo_tables(cfg, runs_dir)
    eff = efficiency_tables(cfg, runs_dir)
    dv = attach_benchmark(collect_validation_table(cfg, "detection"), cfg, "detection")
    cv = attach_benchmark(collect_validation_table(cfg, "classification"), cfg, "classification")
    for name, df in (("hpo_trials_all", trials), ("robustness_summary", rob), ("baseline_vs_hpo", eff), ("validation_detectors", dv), ("validation_classifiers", cv)):
        if len(df):
            write_csv(df, tables / f"{name}.csv", float_format="%.8g")
            ctx.register_artifact(tables / f"{name}.csv", name)
    test_p = art / "test_results.json"
    if test_p.is_file():
        t = read_json(test_p)
        flat = {"detector": t["detector"], "classifier": t["classifier"], **{f"det_{k}": v for k, v in t["detection_standard_metrics"].items() if k != "source"},
                **{f"pipeline_{k}": v for k, v in t["pipeline"]["detection"].items()}, **{k: v for k, v in t["pipeline"]["errors"].items()},
                **{f"cls_{k}": v for k, v in t["pipeline"]["classification_on_matched"].items() if not isinstance(v, (list, dict))}}
        write_csv(pd.DataFrame([flat]), tables / "test_results.csv", float_format="%.8g")
    log.info("master_results.csv: %d runs | trials HPO: %d | arquitecturas evaluadas: %d det + %d cls", len(master), len(trials), len(dv), len(cv))

    # -- 2) figuras finales ----------------------------------------------------------------
    n_hpo = regenerate_hpo_figures(ctx, cfg, runs_dir)
    saved: list[str] = []
    if len(dv):
        saved += save_detection_figures(ctx, cfg, dv, sources=[tables / "validation_detectors.csv"])
        d2 = dv.rename(columns={"map50_95": "metric"})
        ctx.save_figure(plot_performance_efficiency_comparison(d2, "metric", "architecture", "gflops", "latency_ms" if "latency_ms" in d2 else None, "mAP@0.5:0.95"),
                        "det_rendimiento_eficiencia", data=d2[["architecture", "metric", "gflops"]], sources=[tables / "validation_detectors.csv"])
    if len(cv):
        gradcam_missing = not (cfg_path(cfg, "paths.figures_dir") / "cls_gradcam.png").is_file()
        saved += save_classification_figures(ctx, cfg, cv, with_gradcam=gradcam_missing, sources=[tables / "validation_classifiers.csv"])
    if len(eff):
        ctx.save_figure(plot_baseline_vs_hpo(eff.rename(columns={"hpo_mean": "hpo_mean"}), "architecture", "baseline", "hpo_mean", "hpo_sd", "Objetivo (media de folds)"), "hpo_baseline_vs_mejor",
                        data=eff[["architecture", "baseline", "hpo_mean", "hpo_sd"]], sources=[tables / "baseline_vs_hpo.csv"])
        ctx.save_figure(plot_training_time_by_architecture(eff, "architecture", "train_minutes", None, "task"), "tiempo_entrenamiento_por_arquitectura",
                        data=eff[["architecture", "task", "train_minutes"]], sources=[tables / "baseline_vs_hpo.csv"])
        ctx.save_figure(plot_vram_usage_by_model(eff.dropna(subset=["vram_mb"]), "architecture", "vram_mb", ctx.env.get("gpu_vram_mb")), "vram_entrenamiento_por_modelo",
                        data=eff[["architecture", "vram_mb"]], sources=[tables / "baseline_vs_hpo.csv"])
        pf = eff.dropna(subset=["params_m", "gflops"]).rename(columns={"hpo_mean": "metric"})
        ctx.save_figure(plot_params_flops_metric(pf, "params_m", "gflops", "metric", "architecture", "objetivo HPO"), "parametros_flops_metrica",
                        data=pf[["architecture", "params_m", "gflops", "metric"]], sources=[tables / "baseline_vs_hpo.csv"])
        winners = []
        for spec in archs():
            cp = cfg_path(cfg, "paths.best_configs_dir") / f"{spec.slug}.json"
            if cp.is_file():
                winners.append({"architecture": spec.name, "task": spec.task, **read_json(cp)["hp"]})
        if winners:
            w = pd.DataFrame(winners)
            for task, cols in (("detection", ["optimizer", "lr0", "lrf", "momentum", "weight_decay", "warmup_epochs"]), ("classification", ["optimizer", "lr", "weight_decay", "warmup_epochs", "label_smoothing"])):
                sub = w[w["task"] == task].dropna(axis=1, how="all")
                if len(sub):
                    ctx.save_figure(plot_winning_hyperparameter_distribution(sub, [c for c in cols if c in sub.columns]), f"hiperparametros_ganadores_{task}", data=sub, sources=[cfg_path(cfg, "paths.best_configs_dir")])
    log.info("figuras regeneradas: %d curvas/distribuciones HPO + %d de comparacion", n_hpo, len(saved))

    # -- 3) auditoria cientifica -----------------------------------------------------------
    checks = run_audit(cfg, data, allow_missing_figures=args.allow_missing_figures, runs_dir=runs_dir)
    rep = cfg_path(cfg, "paths.reports_dir")
    rep.mkdir(parents=True, exist_ok=True)
    write_json(rep / "scientific_audit.json", {"checks": [c.as_dict() for c in checks], "dataset_hash": ctx.hashes.get("dataset_hash"), "split_hash": ctx.hashes.get("split_hash"),
                                               "code_hash": ctx.hashes.get("code_hash")})
    (rep / "SCIENTIFIC_AUDIT.md").write_text(audit_markdown(checks), encoding="utf-8", newline="\n")
    fig_check = next(c for c in checks if c.id == 7)
    write_json(rep / "figure_audit.json", fig_check.as_dict())
    for c in checks:
        (log.error if c.status == "FAIL" else log.warning if c.status == "WARN" else log.info)("AUDITORIA %d [%s] %s: %s", c.id, c.status, c.name, c.summary[:400])

    # -- 4) procedencia --------------------------------------------------------------------
    key = [art / n for n in ("split_manifest.csv", "split_manifest.sha256", "inner_folds_detection.csv", "inner_folds_classification.csv", "classification_manifest.csv",
                             "selection_policy.yaml", "selected_detector.json", "selected_classifier.json", "final_thresholds.json", "pipeline_validation.json",
                             "freeze_manifest.json", "test_results.json", "master_results.csv")]
    reg = PCRegistry(cfg)
    prov = {
        "dataset_hash": ctx.hashes.get("dataset_hash"), "split_hash": ctx.hashes.get("split_hash"), "code_hash": ctx.hashes.get("code_hash"), "environment_hash": ctx.env.get("environment_hash"),
        "git_commit": ctx.env.get("git_commit"), "versions": {k: ctx.env.get(k) for k in ("python_version", "torch", "torchvision", "timm", "ultralytics", "optuna", "numpy", "cuda_runtime", "cudnn")},
        "pcs_seen": {pc: (read_json(reg.environment_file(pc)).get("environment_hash") if reg.environment_file(pc).is_file() else None) for pc in reg.pcs},
        "environment_mismatches_between_pcs": reg.environment_mismatches(), "artifacts_sha256": {p.name: sha256_file(p) for p in key if p.is_file()},
        "queue_counts": StudyQueue.from_cfg(cfg).counts(), "n_runs": int(len(master)), "profile": cfg["project"].get("profile"),
        "outer_validation_files": {a.slug: str(stage_json_path(cfg, "outer_validation", a)) for a in archs() if stage_json_path(cfg, "outer_validation", a).is_file()},
        "audit_status": {c.id: c.status for c in checks},
    }
    write_json(art / "provenance.json", prov)
    ctx.register_artifact(art / "provenance.json", "procedencia completa")
    ctx.log_metrics({"n_runs": int(len(master)), "audit": {c.id: c.status for c in checks}, "figures_required_ok": fig_check.evidence.get("ok"), "figures_required": fig_check.evidence.get("required")})
    fails = [c for c in checks if c.status == "FAIL"]
    if fails and not args.no_fail:
        raise RuntimeError("Auditoria cientifica con FALLOS: " + "; ".join(f"[{c.id}] {c.name}: {c.summary[:200]}" for c in fails))


if __name__ == "__main__":
    raise SystemExit(run_cli(SCRIPT, main, description=__doc__ or SCRIPT, add_arguments=add_arguments))
