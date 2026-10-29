"""03 - Folds internos para HPO (Especificacion s2, sec. 6 y 31).

* Opera UNICAMENTE sobre las imagenes de outer train (el validation externo y el test no
  participan jamas en los trials).
* Folds a nivel de imagen original (k = ``inner_hpo.folds``, RECOMENDADO 3); los crops heredaran
  el fold de su imagen (nunca folds sobre crops).
* Clasificacion: estratificado por ``has_positive_signs`` (+ bin opcional de % positivos).
* Deteccion: balanceado por bins de frutos por imagen. Todas las arquitecturas comparten folds.

Salidas (``artifacts/``): ``inner_folds_detection.csv``, ``inner_folds_classification.csv`` (+ ``.sha256``),
``inner_folds_metadata.json`` e ``image_manifest.csv`` (vista combinada outer_split + folds).
Requiere el split externo sellado (02). Usa ``inner_hpo.seed`` o, si es ``null``, ``outer_split.seed``.

Ejemplo::

    python src/03_create_inner_folds.py
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
from datetime import datetime, timezone
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from utils.checks import assert_folds_consistent_across_tasks, assert_folds_only_train  # noqa: E402
from utils.config_loader import cfg_path, require  # noqa: E402
from utils.folds import (  # noqa: E402
    CLS_FOLD_COL, DET_FOLD_COL, FoldIntegrityError, build_inner_folds, fold_balance_report, load_inner_folds,
    write_inner_folds,
)
from utils.io import write_csv, write_json  # noqa: E402
from utils.logging_utils import RunContext, run_cli  # noqa: E402
from utils.prepared import FOLDS_METADATA, IMAGE_MANIFEST, load_prepared_data  # noqa: E402
from utils.visualization import plot_inner_fold_balance  # noqa: E402

SCRIPT = "03_create_inner_folds"


def add_arguments(parser: argparse.ArgumentParser) -> None:
    """Argumentos propios del script."""
    g = parser.add_argument_group("03_create_inner_folds")
    g.add_argument("--force", action="store_true", help="sobrescribe folds existentes que difieran (DESACONSEJADO)")
    g.add_argument("--no-figures", action="store_true", help="no generar figuras")


def main(ctx: RunContext, args: argparse.Namespace) -> None:
    """Crea y sella los folds de deteccion y clasificacion."""
    cfg, log = ctx.cfg, ctx.logger
    from utils.guards import assert_not_frozen
    assert_not_frozen(cfg, "03_create_inner_folds")
    artifacts = cfg_path(cfg, "paths.artifacts_dir")
    data = load_prepared_data(cfg, with_split=True)
    assert data.manifest is not None
    ctx.set_hashes(dataset_hash=data.dataset_hash, split_hash=data.split_hash)

    seed = cfg["inner_hpo"].get("seed")
    if seed is None:
        seed = require(cfg, "outer_split.seed", spec_ref="sec. 5.3 (semilla de folds = seed del split si inner_hpo.seed es null)")
    if isinstance(seed, bool) or not isinstance(seed, int):
        raise ValueError(f"La semilla de folds debe ser un entero, recibido {seed!r}")
    ctx.set_seed(seed)
    k = int(cfg["inner_hpo"]["folds"])

    train_ids = set(data.manifest.loc[data.manifest["outer_split"] == "train", "image_id"])
    train_summary = data.summary[data.summary["image_id"].isin(train_ids)].reset_index(drop=True)
    log.info("outer train: %d imagenes -> %d folds (seed=%s)", len(train_summary), k, seed)
    if len(train_summary) < k:
        raise ValueError(f"Hay menos imagenes de train ({len(train_summary)}) que folds ({k})")

    det, cls = build_inner_folds(train_summary, cfg, seed)
    assert_folds_only_train(det, data.manifest, DET_FOLD_COL, k)
    assert_folds_only_train(cls, data.manifest, CLS_FOLD_COL, k)
    assert_folds_consistent_across_tasks(det, cls)

    # -- inmutabilidad -------------------------------------------------------------------
    try:
        old_det, old_cls = load_inner_folds(artifacts)
        exists = True
    except FoldIntegrityError as exc:
        exists = (artifacts / "inner_folds_detection.csv").exists()
        old_det = old_cls = None
        if exists and not args.force:
            raise
        if exists:
            log.warning("Folds existentes alterados/inconsistentes: %s", exc)
    if exists and old_det is not None and old_cls is not None:
        same = (old_det[["image_id", DET_FOLD_COL]].sort_values("image_id").reset_index(drop=True)
                .equals(det[["image_id", DET_FOLD_COL]].sort_values("image_id").reset_index(drop=True))
                and old_cls[["image_id", CLS_FOLD_COL]].sort_values("image_id").reset_index(drop=True)
                .equals(cls[["image_id", CLS_FOLD_COL]].sort_values("image_id").reset_index(drop=True)))
        if same:
            log.info("Los folds ya existen y son identicos a los recalculados: no se reescriben.")
        elif not args.force:
            raise FoldIntegrityError("Ya existen folds distintos a los que produce esta configuracion. Los folds son inmutables; use --force solo si es deliberado.")
        else:
            log.warning("--force: se sobrescriben folds existentes distintos.")

    info = write_inner_folds(det, cls, artifacts)
    from utils.semantics import semantic_fingerprint
    meta = {
        "semantic_fingerprint": semantic_fingerprint(cfg),
        "k": k, "seed": seed, "split_hash": data.split_hash, "dataset_hash": data.dataset_hash,
        "folds_hash": info["folds_hash"], "det_sha256": info["det_sha256"], "cls_sha256": info["cls_sha256"],
        "classification_stratification": cfg["inner_hpo"]["classification_stratification"],
        "classification_extra_positive_bin": cfg["inner_hpo"]["classification_extra_positive_bin"],
        "detection_balance": cfg["inner_hpo"]["detection_balance"],
        "detection_count_bins": cfg["inner_hpo"]["detection_count_bins"],
        "created_at": datetime.now(timezone.utc).isoformat(), "pc_id": ctx.pc_id,
    }
    write_json(artifacts / FOLDS_METADATA, meta)

    # -- vista combinada (el manifiesto del split permanece inmutable) -------------------
    combined = (data.manifest[["image_id", "outer_split", "split_hash"]]
                .merge(cls[["image_id", CLS_FOLD_COL]], on="image_id", how="left")
                .merge(det[["image_id", DET_FOLD_COL]], on="image_id", how="left"))
    combined[CLS_FOLD_COL] = combined[CLS_FOLD_COL].astype("Int64")
    combined[DET_FOLD_COL] = combined[DET_FOLD_COL].astype("Int64")
    combined = combined[["image_id", "outer_split", CLS_FOLD_COL, DET_FOLD_COL, "split_hash"]]
    if combined.loc[combined["outer_split"] != "train", [CLS_FOLD_COL, DET_FOLD_COL]].notna().any().any():
        raise FoldIntegrityError("Hay folds asignados a imagenes fuera de outer train")
    write_csv(combined, artifacts / IMAGE_MANIFEST)

    for p, d in ((artifacts / "inner_folds_detection.csv", "folds de deteccion"),
                 (artifacts / "inner_folds_classification.csv", "folds de clasificacion"),
                 (artifacts / FOLDS_METADATA, "metadatos de folds"), (artifacts / IMAGE_MANIFEST, "manifiesto combinado de imagenes")):
        ctx.register_artifact(p, d)

    # -- distribucion por fold (auditoria) -----------------------------------------------
    cls_rep = fold_balance_report(cls, CLS_FOLD_COL, "has_positive_signs")
    det_rep = fold_balance_report(det, DET_FOLD_COL)
    fruits_per_fold = det.groupby(DET_FOLD_COL)["n_fruits"].sum().astype(int).to_dict()
    pos_per_fold = cls.groupby(CLS_FOLD_COL)["n_positive"].sum().astype(int).to_dict()
    empty_pos = [f for f in range(k) if cls_rep["positive_images_per_fold"].get(f, 0) == 0]
    if empty_pos:
        log.warning("Folds de clasificacion SIN imagenes positivas: %s (AP indefinido en esos folds)", empty_pos)
    ctx.log_metrics({
        "k": k, "seed": seed, "split_hash": data.split_hash, "folds_hash": info["folds_hash"],
        "n_train_images": len(train_summary), "detection": {**det_rep, "fruits_per_fold": fruits_per_fold},
        "classification": {**cls_rep, "positive_fruits_per_fold": pos_per_fold},
    })
    log.info("deteccion: %s", det_rep)
    log.info("clasificacion: %s | frutos con_signos por fold: %s", cls_rep, pos_per_fold)

    if not args.no_figures:
        ctx.save_figure(plot_inner_fold_balance(det, cls), "folds_internos_balance")


if __name__ == "__main__":
    raise SystemExit(run_cli(SCRIPT, main, description=__doc__ or SCRIPT, add_arguments=add_arguments))
