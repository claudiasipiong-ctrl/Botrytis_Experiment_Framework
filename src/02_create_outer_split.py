"""02 - Split externo unico 70 / 20 / 10 por ``image_id`` (Especificacion s2, sec. 5 y 31).

* La unidad es la imagen original: frutos y crops HEREDARAN el split; nunca se dividen crops.
* Se ejecuta UNA sola vez. Si ya existe un manifiesto distinto, aborta (``--force`` lo archiva
  en ``artifacts/superseded/`` y obliga a regenerar folds y datasets derivados).
* Escribe ``artifacts/split_manifest.csv`` + ``artifacts/split_manifest.sha256`` (inmutables),
  ``artifacts/split_metadata.json`` y ``artifacts/test_lock.json`` (test BLOQUEADO).

Requiere ``outer_split.seed`` (TODO BLOQUEANTE, sec. 5.3: predeclararla ANTES de ver resultados)
y que ``01_validate_dataset.py`` haya pasado.

Ejemplo::

    python src/02_create_outer_split.py --set outer_split.seed=20260930
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
import shutil
import sys
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))

from utils.checks import assert_one_split_per_image  # noqa: E402
from utils.config_loader import cfg_path, get_nested, require  # noqa: E402
from utils.io import write_json  # noqa: E402
from utils.logging_utils import RunContext, run_cli  # noqa: E402
from utils.prepared import IMAGE_MANIFEST, load_prepared_data  # noqa: E402
from utils.splits import (  # noqa: E402
    SPLITS, SplitIntegrityError, allocate_counts, assign_outer_split, build_split_manifest, load_split_manifest,
    manifest_paths, write_split_manifest, write_test_lock,
)
from utils.visualization import plot_dataset_distribution  # noqa: E402

SCRIPT = "02_create_outer_split"
STRATIFY_OPTIONS = ("none", "has_positive_signs")


def add_arguments(parser: argparse.ArgumentParser) -> None:
    """Argumentos propios del script."""
    g = parser.add_argument_group("02_create_outer_split")
    g.add_argument("--force", action="store_true",
                   help="archiva el split existente en artifacts/superseded/ y crea uno nuevo (DESACONSEJADO)")
    g.add_argument("--no-figures", action="store_true", help="no generar figuras")


def _supersede(artifacts: Path, ctx: RunContext) -> Path:
    """Mueve el split previo y todo lo derivado de el a ``artifacts/superseded/<ts>/``."""
    dest = artifacts / "superseded" / datetime.now().strftime("%Y%m%d_%H%M%S")
    dest.mkdir(parents=True, exist_ok=True)
    names = ["split_manifest.csv", "split_manifest.sha256", "split_metadata.json", "test_lock.json", IMAGE_MANIFEST,
             "inner_folds_detection.csv", "inner_folds_detection.sha256", "inner_folds_classification.csv",
             "inner_folds_classification.sha256", "inner_folds_metadata.json", "classification_manifest.csv",
             "classification_manifest.sha256"]
    for n in names:
        src = artifacts / n
        if src.exists():
            shutil.move(str(src), str(dest / n))
    ctx.logger.warning("Split previo archivado en %s. REGENERE 03, 04 y 05: los artefactos derivados ya no son validos.", dest)
    return dest


def main(ctx: RunContext, args: argparse.Namespace) -> None:
    """Crea el split externo, lo sella y bloquea el test."""
    cfg, log = ctx.cfg, ctx.logger
    from utils.guards import assert_not_frozen
    assert_not_frozen(cfg, "02_create_outer_split")
    seed = require(cfg, "outer_split.seed", spec_ref="sec. 5.3 / 38.3")
    if isinstance(seed, bool) or not isinstance(seed, int):
        raise ValueError(f"outer_split.seed debe ser un entero, recibido {seed!r}")
    stratify = str(get_nested(cfg, "outer_split.stratify_by", default="none"))
    if stratify not in STRATIFY_OPTIONS:
        raise ValueError(f"outer_split.stratify_by invalido: {stratify!r}; opciones {STRATIFY_OPTIONS}")
    ctx.set_seed(seed)  # la seed registrada en metadata.json es la del split
    artifacts = cfg_path(cfg, "paths.artifacts_dir")
    manifest_name = str(cfg["outer_split"]["manifest_name"])

    data = load_prepared_data(cfg, with_split=False)
    ctx.set_hashes(dataset_hash=data.dataset_hash)
    ids = data.images["image_id"].tolist()
    ratios = {s: float(cfg["outer_split"][s]) for s in SPLITS}
    targets = allocate_counts(len(ids), ratios)
    log.info("n_images=%d  objetivo por split=%s  estratificacion=%s  seed=%s", len(ids), targets, stratify, seed)

    strata = None
    if stratify == "has_positive_signs":
        strata = {r.image_id: ("pos" if r.has_positive_signs else "neg") for r in data.summary.itertuples(index=False)}
        log.info("estratos: %s", pd.Series(strata).value_counts().to_dict())
    assignment = assign_outer_split(ids, ratios, seed, strata)
    manifest = build_split_manifest(assignment)
    assert_one_split_per_image(manifest)
    from utils.isolation import audit_source
    from utils.io import write_csv
    write_csv(audit_source(data.images, manifest), artifacts / "split_content_audit.csv")

    csv_path, sha_path = manifest_paths(artifacts, manifest_name)
    if csv_path.exists():
        try:
            existing = load_split_manifest(artifacts, manifest_name)
            same = existing[["image_id", "outer_split"]].equals(manifest[["image_id", "outer_split"]])
        except SplitIntegrityError as exc:
            if not args.force:
                raise
            log.warning("El split existente esta alterado/inconsistente: %s", exc)
            same = False
        from utils.io import read_json
        from utils.semantics import semantic_fingerprint
        old_meta = read_json(artifacts / "split_metadata.json") if (artifacts / "split_metadata.json").is_file() else {}
        compatible = old_meta.get("dataset_hash") == data.dataset_hash and old_meta.get("semantic_fingerprint") == semantic_fingerprint(cfg)
        if not compatible and not args.force:
            raise SplitIntegrityError("Existing split has incompatible dataset/semantics; explicitly regenerate with --force")
        same = same and compatible
        if same:
            log.info("El split ya existe y es IDENTICO al recalculado con la seed %s: no se reescribe (inmutable).", seed)
            ctx.set_hashes(split_hash=str(existing["split_hash"].iloc[0]))
            ctx.log_metrics({"created": False, "split_hash": str(existing["split_hash"].iloc[0])})
            return
        if not args.force:
            raise SplitIntegrityError(
                f"Ya existe {csv_path.name} y difiere del que produciria esta configuracion (seed/estratificacion/dataset). "
                "El split externo NO se vuelve a sortear (sec. 5.1). Si es un error deliberado use --force."
            )
        _supersede(artifacts, ctx)

    csv_path, sha_path, file_sha = write_split_manifest(manifest, artifacts, manifest_name)
    split_hash = str(manifest["split_hash"].iloc[0])
    lock_path = write_test_lock(artifacts, manifest, data.dataset_hash)
    counts = manifest["outer_split"].value_counts().reindex(list(SPLITS)).astype(int).to_dict()
    from utils.semantics import semantic_fingerprint
    meta = {
        "semantic_fingerprint": semantic_fingerprint(cfg),
        "seed": seed, "stratify_by": stratify, "ratios": ratios, "targets": targets, "counts": counts,
        "actual_fractions": {s: counts[s] / len(ids) for s in SPLITS},
        "n_images": len(ids), "split_hash": split_hash, "manifest_sha256": file_sha, "dataset_hash": data.dataset_hash,
        "created_at": datetime.now(timezone.utc).isoformat(), "pc_id": ctx.pc_id,
        "rounding": "mayor resto exacto; desempate train > validation > test",
    }
    write_json(artifacts / "split_metadata.json", meta)
    for p, d in ((csv_path, "manifiesto del split externo"), (sha_path, "sha256 del manifiesto"),
                 (lock_path, "bloqueo del test"), (artifacts / "split_metadata.json", "metadatos del split")):
        ctx.register_artifact(p, d)
    ctx.set_hashes(split_hash=split_hash)

    # -- verificacion por re-lectura -----------------------------------------------------
    reloaded = load_split_manifest(artifacts, manifest_name)
    assert_one_split_per_image(reloaded)
    if counts != {s: int(targets[s]) for s in SPLITS}:
        raise SplitIntegrityError(f"Conteos {counts} != objetivo {targets}")

    # -- estadisticas por split ----------------------------------------------------------
    per_split = {}
    fr = data.fruits.merge(manifest[["image_id", "outer_split"]], on="image_id")
    for s in SPLITS:
        sub = fr[fr["outer_split"] == s]
        per_split[s] = {"n_images": counts[s], "n_fruits": int(len(sub)), "n_con_signos": int((sub["label"] == cfg["classification_labels"]["positive_label"]).sum())}
    ctx.log_metrics({"created": True, "split_hash": split_hash, "manifest_sha256": file_sha, "counts": counts,
                     "fractions": meta["actual_fractions"], "per_split": per_split, "test_locked": True})
    log.info("split_hash=%s  manifest_sha256=%s", split_hash, file_sha)
    log.info("Conteos: %s | fracciones reales: %s", counts, {k: round(v, 4) for k, v in meta["actual_fractions"].items()})
    log.info("TEST BLOQUEADO: no usar en HPO, seleccion ni thresholds (desbloqueo solo con freeze_manifest.json)")

    if not args.no_figures:
        img_df = data.summary.merge(manifest[["image_id", "outer_split"]], on="image_id")
        ctx.save_figure(plot_dataset_distribution(img_df, fr, cfg["classification_labels"]["positive_label"], cfg["classification_labels"]["negative_label"]),
                        "split_distribucion_outer")


if __name__ == "__main__":
    raise SystemExit(run_cli(SCRIPT, main, description=__doc__ or SCRIPT, add_arguments=add_arguments))
