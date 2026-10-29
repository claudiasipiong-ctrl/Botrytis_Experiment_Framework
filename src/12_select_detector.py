"""12 - Seleccion del detector (Especificacion s2, sec. 22.1, 22.3).

Aplica ``selection_policy.yaml`` (materializada en ``artifacts/selection_policy.yaml`` y ya inmutable) a las
4 evaluaciones de validation externo: maximiza mAP@0.5:0.95 (sujeto a ``minimum_recall`` si se declaro) y
resuelve empates con los ``tie_breakers`` declarados (recall, luego GFLOPs). Solo usa validation externo;
no mira el test. Genera las figuras de comparacion de detectores y ``artifacts/selected_detector.json``.

Ejemplo::

    python src/12_select_detector.py
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
from utils.reporting import collect_validation_table, save_detection_figures  # noqa: E402
from utils.run_common import guard_not_frozen, prepare_for_inference  # noqa: E402
from utils.selection import materialize_policy, rank_candidates  # noqa: E402

SCRIPT = "12_select_detector"


def add_arguments(parser: argparse.ArgumentParser) -> None:
    """Argumentos propios del script."""
    parser.add_argument("--allow-partial", action="store_true", help="permite seleccionar con menos de 4 detectores evaluados (solo pruebas)")
    parser.add_argument("--skip-env-check", action="store_true", help="no comparar el entorno con la referencia")


def main(ctx: RunContext, args: argparse.Namespace) -> None:
    """Elige el detector aplicando la politica de seleccion predeclarada."""
    cfg, log = ctx.cfg, ctx.logger
    if cfg['project']['profile']=='official' and args.allow_partial:
        raise MethodologyViolation('--allow-partial no esta permitido en seleccion oficial')
    prepare_for_inference(ctx, args)
    guard_not_frozen(ctx, "12_select_detector")
    policy_path, policy_sha = materialize_policy(cfg)
    pol = cfg["selection"]["detector"]
    ctx.note_split_access("validation", "model_selection")
    df = collect_validation_table(cfg, "detection")
    expected = [a.name for a in archs("detection")]
    missing = [a for a in expected if a not in set(df.get("architecture", []))]
    if missing and not args.allow_partial:
        raise MethodologyViolation(f"Faltan detectores evaluados (10/11): {missing}. Se seleccionan entre los 4 en conjunto (sec. 20).")
    if "recall" not in df or df["recall"].isna().all():
        raise MethodologyViolation("Faltan los umbrales afinados (11_tune_inference_thresholds.py) de los detectores")
    ranked, winner = rank_candidates(df, primary=str(pol["primary_metric"]), min_recall=pol.get("minimum_recall"), recall_col="recall",
                                     tie_margin=float(pol.get("tie_margin", 0.0)), tie_breakers=pol.get("tie_breakers", []))
    if winner is None:
        raise MethodologyViolation("Ningun detector cumple el recall minimo declarado (minimum_recall)")
    tables = cfg_path(cfg, "paths.tables_dir")
    tables.mkdir(parents=True, exist_ok=True)
    write_csv(ranked, tables / "ranking_detectors.csv", float_format="%.8g")
    spec = next(a for a in archs("detection") if a.name == winner)
    row = ranked[ranked["architecture"] == winner].iloc[0]
    best = load_best_config(cfg, spec)
    th = read_json(cfg_path(cfg, "paths.artifacts_dir") / "inference_thresholds" / f"{spec.slug}.json")
    selected = {
        "architecture": spec.name, "slug": spec.slug, "task": "detection", "checkpoint": str(final_checkpoint(cfg, spec)), "checkpoint_sha256": row["checkpoint_sha256"],
        "config_id": best["config_id"], "hp": best["hp"], "image_size": int(cfg["detector"]["image_size"]), "conf": th["conf"], "nms_iou": th["nms_iou"],
        "nms_applies": th["nms_applies"], "primary_metric": pol["primary_metric"], "primary_value": float(row[pol["primary_metric"]]), "recall_at_operating_point": float(row["recall"]),
        "reason": row["reason"], "selection_policy_sha256": policy_sha, "selected_on": "outer_validation_20pct", "selected_by_single_run": False, "test_used": False,
        "ranking": ranked[["rank", "architecture", pol["primary_metric"], "recall", "gflops", "eligible", "reason"]].to_dict("records"),
        "split_hash": ctx.hashes.get("split_hash"), "dataset_hash": ctx.hashes.get("dataset_hash"), "code_hash": ctx.hashes.get("code_hash"),
    }
    out = cfg_path(cfg, "paths.artifacts_dir") / "selected_detector.json"
    write_json(out, selected)
    ctx.register_artifact(out, "detector seleccionado")
    ctx.register_artifact(policy_path, "politica de seleccion")
    ctx.log_metrics({"selected": selected, "ranking": ranked.to_dict("records")})
    log.info("DETECTOR SELECCIONADO: %s (%s=%.4f; %s)", spec.name, pol["primary_metric"], selected["primary_value"], row["reason"])
    sources = [tables / "ranking_detectors.csv"]
    save_detection_figures(ctx, cfg, ranked.sort_values("architecture", key=lambda s: s.map({a: i for i, a in enumerate(expected)})), sources=sources)


if __name__ == "__main__":
    raise SystemExit(run_cli(SCRIPT, main, description=__doc__ or SCRIPT, add_arguments=add_arguments))
