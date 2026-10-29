"""09 - Entrenamiento definitivo de una arquitectura con TODO el 70 % de train (Especificacion s2, sec. 18, 20).

Usa la configuracion congelada por ``08_repeat_top_configs.py`` (``artifacts/best_configs/<slug>.json``) y
entrena con todo el outer train. El numero de epocas es el fijado por los folds internos
(``final_epochs``): SIN early stopping y SIN elegir epoca con el validation externo (el modelo es el de la
ultima epoca); asi el 20 % externo queda limpio para comparar arquitecturas y ajustar inferencia.

Salida: ``artifacts/checkpoints/<slug>/final.pt`` (+ ``final_meta.json`` con SHA-256) y el run
``runs/<arquitectura>/final/`` con los 6 ficheros obligatorios y ``history.csv``.

Ejemplo::

    python src/09_train_outer_candidate.py --architecture ViT-B/16 --pc-id PC-03
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
from pathlib import Path
from typing import Any, Mapping

sys.path.insert(0, str(Path(__file__).resolve().parent))

import pandas as pd  # noqa: E402

from utils.architectures import ArchSpec  # noqa: E402
from utils.classifier_data import crops_for, load_crop_manifest  # noqa: E402
from utils.classifier_models import save_classifier  # noqa: E402
from utils.classifier_train import train_classifier  # noqa: E402
from utils.detector_data import write_outer_yaml  # noqa: E402
from utils.detector_train import train_detector  # noqa: E402
from utils.guards import GpuJobLock  # noqa: E402
from utils.hpo_results import final_checkpoint, load_best_config  # noqa: E402
from utils.hpo_runner import record_training_source  # noqa: E402
from utils.io import read_json, write_json  # noqa: E402
from utils.logging_utils import RunContext, run_cli  # noqa: E402
from utils.run_common import add_arch_arguments, arch_from_args, guard_not_frozen, layout_for, preflight_training  # noqa: E402
from utils.run_figures import save_training_figures  # noqa: E402
from utils.hashing import sha256_file  # noqa: E402

SCRIPT = "09_train_outer_candidate"


def add_arguments(parser: argparse.ArgumentParser) -> None:
    """Argumentos propios del script."""
    add_arch_arguments(parser)
    parser.add_argument("--force", action="store_true", help="reentrena aunque ya exista final.pt de la misma configuracion")


def main(ctx: RunContext, args: argparse.Namespace) -> None:
    """Entrena la configuracion congelada con el 70 % completo y guarda el checkpoint definitivo."""
    cfg, log = ctx.cfg, ctx.logger
    spec = arch_from_args(args)
    data = preflight_training(ctx, args)
    guard_not_frozen(ctx, "09_train_outer_candidate")
    best = load_best_config(cfg, spec)
    hp, epochs = best["hp"], int(best["final_epochs"])
    seed = int(cfg["reproducibility"]["seed"])
    ctx.set_seed(seed)
    ckpt = final_checkpoint(cfg, spec)
    meta_path = ckpt.with_name("final_meta.json")
    if ckpt.exists() and args.force:
        raise RuntimeError('--force no sobrescribe candidatos historicos; cree nueva ronda')
    if ckpt.is_file() and meta_path.is_file():
        meta = read_json(meta_path)
        if meta.get("config_id") == best["config_id"] and meta.get("sha256") == sha256_file(ckpt):
            if meta.get("dataset_hash") != data.dataset_hash or meta.get("split_hash") != data.split_hash:
                from utils.checks import MethodologyViolation
                raise MethodologyViolation("Checkpoint reutilizado pertenece a otros datos/split; reentrenar antes de evaluar")
            log.info("checkpoint definitivo ya existe para %s (%s): no se reentrena (--force para repetir)", spec.name, best["config_id"])
            ctx.set_checkpoint(ckpt, role="final", config_id=best["config_id"])
            ctx.log_metrics(meta)
            return
    ctx.add_metadata(architecture=spec.name, stage="outer_train", hp=hp, final_epochs=epochs, config_id=best["config_id"], seed=seed)
    ctx.note_split_access("train", "outer_train")
    record_training_source(ctx, spec, cfg)
    log.info("entrenamiento definitivo %s: config %s, %d epocas fijas, 70 %% completo (seed %d)", spec.name, best["config_id"], epochs, seed)
    with GpuJobLock(cfg, ctx.pc_id, job=f"outer_train:{spec.slug}"):
        if spec.is_detector:
            yml = write_outer_yaml(cfg, ctx.run_dir, val_split="train", stage="outer_train")
            run = train_detector(spec, hp, cfg, data_yaml=yml, project_dir=ctx.run_dir, name="train", seed=seed, epochs=epochs, fixed_epochs=True,
                                 log=log.info, keep_weights=True)
            src = Path(run.weights_path) if run.weights_path else ctx.run_dir / "train" / "weights" / "last.pt"
            ckpt.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(src, ckpt)
            info = {"vram_peak_mb": run.vram_peak_mb, "train_seconds": run.train_seconds, "params_m": run.params_m, "gflops": run.gflops,
                    "pretrained_sha256": run.pretrained_sha256, "history": run.history, "micro_batch": run.micro_batch, "batch_contract":run.extra.get("batch_contract")}
        else:
            manifest = load_crop_manifest(cfg)
            tr = crops_for(manifest, split="train", stage="outer_train", cfg=cfg)
            run = train_classifier(spec, hp, cfg, tr, pd.DataFrame(columns=list(tr.columns)), seed=seed, max_epochs=epochs, fixed_epochs=True, log=log.info, keep_state=True)
            assert run.state is not None
            save_classifier(ckpt, _rebuild_model(spec, hp, run.state), spec, hp, run.norm, epochs=epochs, config_id=best["config_id"], seed=seed)
            info = {"vram_peak_mb": run.vram_peak_mb, "train_seconds": run.train_seconds, "params_m": run.params_m, "gflops": run.gflops,
                    "pretrained_sha256": None, "history": run.history, "micro_batch": run.micro_batch, "norm": run.norm}
    sha = sha256_file(ckpt)
    hist = info.pop("history")
    meta = {"architecture": spec.name, "slug": spec.slug, "config_id": best["config_id"], "hp": hp, "epochs": epochs, "seed": seed, "sha256": sha,
            "size_bytes": ckpt.stat().st_size, "checkpoint": str(ckpt), "trained_on": "outer_train_100pct", "early_stopping": False, "epoch_selected_on_validation": False,
            "dataset_hash": data.dataset_hash, "split_hash": data.split_hash, "code_hash": ctx.hashes.get("code_hash"), "pc_id": ctx.pc_id,
            **{k: v for k, v in info.items() if k != "norm"}, **({"norm": info["norm"]} if "norm" in info else {})}
    write_json(meta_path, meta)
    ctx.set_checkpoint(ckpt, role="final", config_id=best["config_id"], epochs=epochs)
    ctx.register_artifact(meta_path, "metadatos del checkpoint definitivo")
    ctx.log_metrics(meta)
    hist.to_csv(ctx.run_dir / "history.csv", index=False, lineterminator="\n")
    save_training_figures(ctx, spec, hist, "final", ctx.run_dir / "history.csv", to_reports=False)
    import importlib
    evaluator = importlib.import_module("10_evaluate_outer_validation")
    with GpuJobLock(cfg, ctx.pc_id, job=f"train_metrics:{spec.slug}"):
        fn = evaluator.evaluate_detector if spec.is_detector else evaluator.evaluate_classifier
        kwargs = {} if spec.is_detector else {"gradcam": False}
        train_metrics = fn(ctx, spec, data, best, ckpt, split="train", export_evidence=False, **kwargs)
    from utils.io import write_json as save_json
    save_json(ctx.run_dir / "train_evaluation.json", train_metrics)
    log.info("CHECKPOINT DEFINITIVO %s: %s (sha256=%s) | %.1f min", spec.name, ckpt, sha[:16], float(info["train_seconds"]) / 60)


def _rebuild_model(spec: ArchSpec, hp: Mapping[str, Any], state: Mapping[str, Any]) -> Any:
    """Reconstruye el modelo (sin descargar pesos) con los del entrenamiento para serializarlo."""
    from utils.classifier_models import build_classifier

    model, _ = build_classifier(spec, hp, pretrained=False)
    model.load_state_dict(state)
    return model


if __name__ == "__main__":
    raise SystemExit(run_cli(SCRIPT, main, description=__doc__ or SCRIPT, add_arguments=add_arguments, run_layout=layout_for("final")))
