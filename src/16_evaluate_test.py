"""16 - Evaluacion FINAL sobre el 10 % test (Especificacion s2, sec. 27-28).

* Verifica que el experimento siga congelado (manifiesto sellado, hashes de codigo/datos/split, checkpoints,
  umbrales y versiones) y recien entonces abre el test (``assert_split_allowed('test', 'final_test')``).
* UNA sola evaluacion: detector (P, R, F1, mAP@0.5, mAP@0.5:0.95), clasificacion sobre frutos correctamente
  asociados (accuracy, P, R, F1, ROC-AUC, AP, matriz de confusion) y errores ``E_det``, ``E_clas``, ``E_flujo``
  (con FP/duplicados como error de localizacion).
* ``artifacts/test_results.json`` es de escritura unica: una segunda ejecucion se rechaza (no se reajusta
  nada y se vuelve a mirar). ``--reproduce`` recalcula en un run aparte y COMPARA con los resultados oficiales
  (sin sobrescribirlos) para demostrar reproducibilidad.

Ejemplo::

    python src/16_evaluate_test.py
    python src/16_evaluate_test.py --reproduce
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
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parent))

from utils.architectures import get_arch  # noqa: E402
from utils.checks import MethodologyViolation  # noqa: E402
from utils.classifier_models import load_classifier  # noqa: E402
from utils.config_loader import cfg_path  # noqa: E402
from utils.detector_data import write_outer_yaml  # noqa: E402
from utils.freeze import load_freeze, verify_freeze  # noqa: E402
from utils.guards import GpuJobLock  # noqa: E402
from utils.hashing import write_checksum_file  # noqa: E402
from utils.inference import load_detector  # noqa: E402
from utils.io import read_json, write_csv, write_json  # noqa: E402
from utils.logging_utils import RunContext, run_cli  # noqa: E402
from utils.pipeline_eval import evaluate_pipeline, gt_from_fruit_table  # noqa: E402
from utils.reporting import save_pipeline_figures  # noqa: E402
from utils.run_common import prepare_for_inference  # noqa: E402
from utils.splits import assert_split_allowed  # noqa: E402
from utils.training_common import resolve_device, ultralytics_device  # noqa: E402

SCRIPT = "16_evaluate_test"


def add_arguments(parser: argparse.ArgumentParser) -> None:
    """Argumentos propios del script."""
    parser.add_argument("--reproduce", action="store_true", help="recalcula y COMPARA con test_results.json sin sobrescribirlo")
    parser.add_argument("--retry-after-crash", action="store_true", help="permite reintentar si una ejecucion anterior se interrumpio SIN escribir resultados")
    parser.add_argument("--skip-env-check", action="store_true", help="no comparar el entorno con la referencia")


def _flat(res_summary: dict[str, Any]) -> dict[str, float | None]:
    c = res_summary["classification_on_matched"]
    return {"e_det": res_summary["errors"]["e_det"], "e_clas": res_summary["errors"]["e_clas"], "e_flujo": res_summary["errors"]["e_flujo"], "tp": res_summary["tp"],
            "fp": res_summary["fp"], "fn": res_summary["fn"], "cls_accuracy": c.get("accuracy"), "cls_f1": c.get("f1"), "cls_ap": c.get("average_precision")}


def main(ctx: RunContext, args: argparse.Namespace) -> None:
    """Abre el test UNA vez con el pipeline congelado y reporta las metricas finales."""
    cfg, log = ctx.cfg, ctx.logger
    data = prepare_for_inference(ctx, args)
    art = cfg_path(cfg, "paths.artifacts_dir")
    freeze = load_freeze(cfg)
    verify_freeze(cfg, freeze, ctx.hashes, ctx.env)
    assert_split_allowed("test", "final_test", cfg)
    results_path = art / "test_results.json"
    started_path = art / "test_evaluation_started.json"
    if results_path.is_file() and not args.reproduce:
        raise MethodologyViolation("El test ya fue evaluado (artifacts/test_results.json). NO se reevalua ni se reajusta nada (sec. 27); use --reproduce para verificar reproducibilidad.")
    if started_path.is_file() and not results_path.is_file() and not (args.retry_after_crash or args.reproduce):
        raise MethodologyViolation("Una evaluacion de test anterior se interrumpio sin resultados; use --retry-after-crash (queda registrado)")
    if args.reproduce and not results_path.is_file():
        raise MethodologyViolation("--reproduce requiere resultados oficiales previos")
    if not args.reproduce:
        write_json(started_path, {"started_at": datetime.now(timezone.utc).isoformat(), "pc_id": ctx.pc_id, "freeze_sha256": cfg_path(cfg, "test.freeze_manifest").with_suffix(".sha256").read_text(encoding="utf-8").split()[0]})

    det_spec, cls_spec = get_arch(freeze["detector"]["architecture"]), get_arch(freeze["classifier"]["architecture"])
    m = freeze["pipeline"]["matching_rule"]
    match_iou = float(m["threshold"])
    ctx.note_split_access("test", "final_test")
    ctx.add_metadata(detector=det_spec.name, classifier=cls_spec.name, stage="final_test", reproduce=bool(args.reproduce), freeze_sha256=cfg_path(cfg, "test.freeze_manifest").with_suffix(".sha256").read_text(encoding="utf-8").split()[0])
    test_ids = sorted(data.manifest.loc[data.manifest["outer_split"] == "test", "image_id"])
    paths = {r.image_id: Path(r.image_path) for r in data.images.itertuples(index=False) if r.image_id in set(test_ids)}
    gt = gt_from_fruit_table(data.fruits, test_ids)
    log.info("TEST ABIERTO: %d imagenes, %d frutos GT | detector=%s clasificador=%s (congelados)", len(test_ids), sum(len(g["boxes"]) for g in gt.values()), det_spec.name, cls_spec.name)

    import importlib
    from utils.run_common import nested_run
    from utils.config_loader import resolve_path
    evaluator = importlib.import_module("10_evaluate_outer_validation")
    comparisons = {}
    # Reproduction writes evidence below its own run and never overwrites official results.
    if args.reproduce:
        cfg = dict(cfg)
        cfg["paths"] = dict(cfg["paths"], results_dir=str(ctx.run_dir / "reproduction_results"))
    with GpuJobLock(cfg, ctx.pc_id, job="final_test"):
        for candidate in freeze["comparison"]["candidates"]:
            spec = get_arch(candidate["architecture"])
            label = f"test_reproduction/{ctx.run_dir.name}" if args.reproduce else "final_test"
            with nested_run(ctx, spec, label, reuse_existing=False) as child:
                child.cfg = cfg
                child.set_hashes(dataset_hash=data.dataset_hash, split_hash=data.split_hash)
                child.note_split_access("test", "final_test")
                child.add_metadata(architecture=spec.name, stage="final_test", reproduce=bool(args.reproduce))
                fn = evaluator.evaluate_detector if spec.is_detector else evaluator.evaluate_classifier
                kwargs = {} if spec.is_detector else {"gradcam": False}
                payload = fn(child, spec, data, candidate["best"], Path(candidate["checkpoint"]), split="test", **kwargs)
                child.log_metrics(payload)
                comparisons[spec.slug] = payload
        detector = load_detector(freeze["detector"]["checkpoint"], det_spec)
        classifier, meta = load_classifier(freeze["classifier"]["checkpoint"], resolve_device(cfg))
        d = cfg["detector"]
        conf, nms = float(freeze["detector"]["confidence_threshold"]), freeze["detector"]["nms_iou"]
        nms_v = float(nms) if nms is not None else float(d["val_nms_iou"])
        yml = write_outer_yaml(cfg, ctx.run_dir, val_split="test", stage="final_test")
        half = str(d["precision"]).lower() == "fp16" and resolve_device(cfg) != "cpu"
        ul = detector.val(data=str(yml), split="val", imgsz=int(d["image_size"]), batch=int(d["batch_size"]), conf=float(d["val_conf"]), iou=float(d["val_nms_iou"]),
                          half=half, device=ultralytics_device(cfg), plots=False, verbose=False, project=str(ctx.run_dir), name="ultralytics_test", exist_ok=True)
        res = evaluate_pipeline(cfg, detector=detector, classifier=classifier, norm=meta["norm"], image_paths=paths, gt=gt, conf=conf, nms_iou=nms_v,
                                cls_threshold=float(freeze["classifier"]["probability_threshold"]), match_iou=match_iou, max_examples=int(cfg["figures"]["tp_fp_fn_examples"]))

    box = ul.box
    detection_std = {"map50_95": float(box.map), "map50": float(box.map50), "precision_at_max_f1": float(box.mp), "recall_at_max_f1": float(box.mr),
                     "f1_at_max_f1": float(box.f1[0]) if getattr(box.f1, "size", 0) else None, "source": "ultralytics_val"}
    results = {
        "comparison_models": comparisons,
        "comparison_use": "descriptive_only; no selection or tuning from test",
        "evaluated_at": datetime.now(timezone.utc).isoformat(), "pc_id": ctx.pc_id, "n_test_images": len(test_ids), "detector": det_spec.name, "classifier": cls_spec.name,
        "detection_standard_metrics": detection_std, "detection_at_frozen_operating_point": res.summary["detection"], "pipeline": res.summary,
        "freeze_sha256": cfg_path(cfg, "test.freeze_manifest").with_suffix(".sha256").read_text(encoding="utf-8").split()[0], "dataset_hash": ctx.hashes.get("dataset_hash"),
        "split_hash": ctx.hashes.get("split_hash"), "code_hash": ctx.hashes.get("code_hash"), "thresholds_changed_after_test": False,
    }
    write_csv(res.detections, ctx.run_dir / "pipeline_detections_test.csv", float_format="%.6f")
    write_csv(res.misses, ctx.run_dir / "pipeline_misses_test.csv", float_format="%.6f")
    write_csv(res.per_image, ctx.run_dir / "per_image_timing_test.csv", float_format="%.4f")
    e = res.summary["errors"]
    log.info("TEST | detector mAP50-95=%.4f mAP50=%.4f | P=%.3f R=%.3f F1=%.3f | E_det=%.3f E_clas=%s E_flujo=%.3f", detection_std["map50_95"], detection_std["map50"],
             res.summary["detection"]["precision"], res.summary["detection"]["recall"], res.summary["detection"]["f1"], e["e_det"] or 0,
             f"{e['e_clas']:.3f}" if e["e_clas"] is not None else "n/a", e["e_flujo"] or 0)
    if args.reproduce:
        official = read_json(results_path) if results_path.is_file() else None
        check = {"reproduced": True, "official_exists": official is not None}
        if official is not None:
            a, b = _flat(official["pipeline"]), _flat(results["pipeline"])
            check.update(identical=a == b, official=a, reproduction=b, map50_95_diff=abs(official["detection_standard_metrics"]["map50_95"] - detection_std["map50_95"]))
            log.info("REPRODUCCION: %s", "IDENTICA" if a == b else f"DIFERENCIAS {a} vs {b}")
        if official is not None:
            check["comparisons_identical"] = all(
                {k: v for k, v in payload.items() if k in ("map50_95", "map50", "accuracy", "f1", "roc_auc", "loss")} ==
                {k: v for k, v in official["comparison_models"][slug].items() if k in ("map50_95", "map50", "accuracy", "f1", "roc_auc", "loss")}
                for slug, payload in comparisons.items())
        write_json(ctx.run_dir / "reproduction_check.json", check)
        ctx.log_metrics({"reproduction_check": check})
        return
    from utils.split_summary import export_split_summary
    export_split_summary(cfg, data)
    write_json(results_path, results)
    write_checksum_file(results_path, results_path.with_suffix(".sha256"))
    ctx.register_artifact(results_path, "resultados FINALES de test")
    ctx.log_metrics(results)
    save_pipeline_figures(ctx, res, det_spec.name, cls_spec.name, suffix="test", sources=[ctx.run_dir / "pipeline_detections_test.csv", ctx.run_dir / "pipeline_misses_test.csv",
                                                                                          ctx.run_dir / "per_image_timing_test.csv"], n_examples=int(cfg["figures"]["tp_fp_fn_examples"]))
    log.info("Resultados de test guardados en %s. NO cambiar nada a partir de ahora (sec. 27).", results_path)


if __name__ == "__main__":
    raise SystemExit(run_cli(SCRIPT, main, description=__doc__ or SCRIPT, add_arguments=add_arguments))
