"""13 - Seleccion del clasificador (Especificacion s2, sec. 22.2, 22.3).

Aplica ``selection_policy.yaml`` a las 3 evaluaciones de validation externo: maximiza el Average
Precision de ``con_signos`` (sujeto a ``minimum_recall_positive`` si se declaro); empates por recall, F1 y
GFLOPs. Solo validation externo; el test sigue bloqueado. Genera las figuras de clasificacion (PR, ROC,
matriz de confusion, umbral vs F1, AP positiva por arquitectura y Grad-CAM) y ``artifacts/selected_classifier.json``.

Ejemplo::

    python src/13_select_classifier.py
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

from utils.architectures import archs  # noqa: E402
from utils.checks import MethodologyViolation  # noqa: E402
from utils.config_loader import cfg_path  # noqa: E402
from utils.hpo_results import final_checkpoint, load_best_config  # noqa: E402
from utils.io import read_json, write_csv, write_json  # noqa: E402
from utils.logging_utils import RunContext, run_cli  # noqa: E402
from utils.reporting import collect_validation_table, save_classification_figures  # noqa: E402
from utils.run_common import guard_not_frozen, prepare_for_inference  # noqa: E402
from utils.selection import materialize_policy, rank_candidates  # noqa: E402

SCRIPT = "13_select_classifier"


def add_arguments(parser: argparse.ArgumentParser) -> None:
    """Argumentos propios del script."""
    parser.add_argument("--allow-partial", action="store_true", help="permite seleccionar con menos de 3 clasificadores evaluados (solo pruebas)")
    parser.add_argument("--no-gradcam", action="store_true", help="omite la figura Grad-CAM")
    parser.add_argument("--skip-env-check", action="store_true", help="no comparar el entorno con la referencia")


def main(ctx: RunContext, args: argparse.Namespace) -> None:
    """Elige el clasificador aplicando la politica de seleccion predeclarada."""
    cfg, log = ctx.cfg, ctx.logger
    if cfg['project']['profile']=='official' and args.allow_partial:
        raise MethodologyViolation('--allow-partial no esta permitido en seleccion oficial')
    prepare_for_inference(ctx, args)
    guard_not_frozen(ctx, "13_select_classifier")
    policy_path, policy_sha = materialize_policy(cfg)
    pol = cfg["selection"]["classifier"]
    ctx.note_split_access("validation", "model_selection")
    df = collect_validation_table(cfg, "classification")
    expected = [a.name for a in archs("classification")]
    missing = [a for a in expected if a not in set(df.get("architecture", []))]
    if missing and not args.allow_partial:
        raise MethodologyViolation(f"Faltan clasificadores evaluados (10/11): {missing}. Se seleccionan entre los 3 en conjunto (sec. 20).")
    if "threshold" not in df or df["threshold"].isna().all():
        raise MethodologyViolation("Faltan los umbrales de probabilidad afinados (11_tune_inference_thresholds.py) de los clasificadores")
    ranked, winner = rank_candidates(df, primary=str(pol["primary_metric"]).replace("average_precision_positive", "average_precision_positive"),
                                     min_recall=pol.get("minimum_recall_positive"), recall_col="recall", tie_margin=float(pol.get("tie_margin", 0.0)),
                                     tie_breakers=pol.get("tie_breakers", []))
    if winner is None:
        raise MethodologyViolation("Ningun clasificador cumple el recall minimo declarado (minimum_recall_positive)")
    tables = cfg_path(cfg, "paths.tables_dir")
    tables.mkdir(parents=True, exist_ok=True)
    write_csv(ranked, tables / "ranking_classifiers.csv", float_format="%.8g")
    spec = next(a for a in archs("classification") if a.name == winner)
    row = ranked[ranked["architecture"] == winner].iloc[0]
    best = load_best_config(cfg, spec)
    th = read_json(cfg_path(cfg, "paths.artifacts_dir") / "inference_thresholds" / f"{spec.slug}.json")
    selected = {
        "architecture": spec.name, "slug": spec.slug, "task": "classification", "checkpoint": str(final_checkpoint(cfg, spec)), "checkpoint_sha256": row["checkpoint_sha256"],
        "config_id": best["config_id"], "hp": best["hp"], "image_size": int(cfg["classifier"]["image_size"]), "probability_threshold": th["threshold"],
        "threshold_metric": th["metric"], "positive_class": cfg["classifier"]["positive_class"], "primary_metric": pol["primary_metric"],
        "primary_value": float(row[pol["primary_metric"]]), "recall_at_operating_point": float(row["recall"]), "reason": row["reason"], "selection_policy_sha256": policy_sha,
        "selected_on": "outer_validation_20pct", "selected_by_single_run": False, "test_used": False,
        "ranking": ranked[["rank", "architecture", pol["primary_metric"], "recall", "f1", "gflops", "eligible", "reason"]].to_dict("records"),
        "split_hash": ctx.hashes.get("split_hash"), "dataset_hash": ctx.hashes.get("dataset_hash"), "code_hash": ctx.hashes.get("code_hash"),
    }
    out = cfg_path(cfg, "paths.artifacts_dir") / "selected_classifier.json"
    write_json(out, selected)
    ctx.register_artifact(out, "clasificador seleccionado")
    ctx.register_artifact(policy_path, "politica de seleccion")
    ctx.log_metrics({"selected": selected, "ranking": ranked.to_dict("records")})
    log.info("CLASIFICADOR SELECCIONADO: %s (%s=%.4f; %s)", spec.name, pol["primary_metric"], selected["primary_value"], row["reason"])
    save_classification_figures(ctx, cfg, ranked.sort_values("architecture", key=lambda s: s.map({a: i for i, a in enumerate(expected)})),
                                with_gradcam=not args.no_gradcam, sources=[tables / "ranking_classifiers.csv"])


if __name__ == "__main__":
    raise SystemExit(run_cli(SCRIPT, main, description=__doc__ or SCRIPT, add_arguments=add_arguments))
