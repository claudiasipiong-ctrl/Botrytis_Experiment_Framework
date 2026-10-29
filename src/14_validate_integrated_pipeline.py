"""14 - Validacion del pipeline integrado con PREDICTED crops en el validation externo (sec. 23-25, 28).

Detector seleccionado + clasificador seleccionado: imagen completa -> cajas predichas -> crops de la imagen
original -> ``P(con_signos)`` -> umbral. El clasificador se entreno con GT crops, pero el pipeline real
produce crops desplazados/parciales: esta etapa es OBLIGATORIA antes del test (sec. 24).

* Matching detector-GT: IoU >= ``matching.threshold``, uno-a-uno por confianza (independiente del pairing GT).
* Errores: ``E_det = FN/N_GT``, ``E_clas = EC/N_matched``, ``E_flujo = (FN+EC)/N_GT``; FP/duplicados = error de
  localizacion.
* Con ``--retune-classifier-threshold`` el umbral del clasificador se reajusta sobre los crops PREDICHOS del
  validation (sigue siendo solo validation externo; el detector no se reentrena: "resolver errores de
  implementacion, no retunear pesos", sec. 36.35).

Salidas: ``artifacts/pipeline_validation.json``, ``artifacts/final_thresholds.json`` y las figuras del pipeline
(arquitectura completa, TP/FP/FN, errores acumulados, matriz de errores, mapa espacial, latencia por etapa de la validacion).

Ejemplo::

    python src/14_validate_integrated_pipeline.py --retune-classifier-threshold
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

from utils.architectures import get_arch  # noqa: E402
from utils.checks import MethodologyViolation  # noqa: E402
from utils.classifier_models import load_classifier  # noqa: E402
from utils.config_loader import cfg_path  # noqa: E402
from utils.guards import GpuJobLock  # noqa: E402
from utils.inference import load_detector  # noqa: E402
from utils.io import read_json, write_csv, write_json  # noqa: E402
from utils.logging_utils import RunContext, run_cli  # noqa: E402
from utils.metrics_classification import pick_threshold, threshold_sweep  # noqa: E402
from utils.pipeline_eval import evaluate_pipeline, gt_from_fruit_table, summarize_from_table  # noqa: E402
from utils.reporting import save_pipeline_figures  # noqa: E402
from utils.run_common import guard_not_frozen, prepare_for_inference  # noqa: E402
from utils.training_common import resolve_device  # noqa: E402

SCRIPT = "14_validate_integrated_pipeline"


def add_arguments(parser: argparse.ArgumentParser) -> None:
    """Argumentos propios del script."""
    parser.add_argument("--retune-classifier-threshold", action="store_true", help="reajusta el umbral del clasificador sobre los crops PREDICHOS del validation")
    parser.add_argument("--skip-env-check", action="store_true", help="no comparar el entorno con la referencia")


def main(ctx: RunContext, args: argparse.Namespace) -> None:
    """Ejecuta el pipeline integrado sobre el validation externo y fija los umbrales definitivos."""
    cfg, log = ctx.cfg, ctx.logger
    data = prepare_for_inference(ctx, args)
    guard_not_frozen(ctx, "14_validate_integrated_pipeline")
    art = cfg_path(cfg, "paths.artifacts_dir")
    for f in ("selected_detector.json", "selected_classifier.json"):
        if not (art / f).is_file():
            raise MethodologyViolation(f"Falta artifacts/{f}: ejecute 12_select_detector.py y 13_select_classifier.py (sec. 22)")
    sd, sc = read_json(art / "selected_detector.json"), read_json(art / "selected_classifier.json")
    det_spec, cls_spec = get_arch(sd["architecture"]), get_arch(sc["architecture"])
    m = cfg["matching"]
    if str(m["metric"]).upper().startswith("TODO") or str(m["threshold"]).upper().startswith("TODO"):
        raise MethodologyViolation("El matching detector-GT (sec. 25) debe estar definido")
    match_iou = float(m["threshold"])
    ctx.note_split_access("validation", "pipeline_validation")
    ctx.add_metadata(detector=det_spec.name, classifier=cls_spec.name, stage="pipeline_validation")

    val_ids = sorted(data.manifest.loc[data.manifest["outer_split"] == "validation", "image_id"])
    paths = {r.image_id: Path(r.image_path) for r in data.images.itertuples(index=False) if r.image_id in set(val_ids)}
    gt = gt_from_fruit_table(data.fruits, val_ids)
    with GpuJobLock(cfg, ctx.pc_id, job="pipeline_validation"):
        detector = load_detector(sd["checkpoint"], det_spec)
        classifier, meta = load_classifier(sc["checkpoint"], resolve_device(cfg))
        nms = float(sd["nms_iou"]) if sd["nms_iou"] is not None else float(cfg["detector"]["val_nms_iou"])
        res = evaluate_pipeline(cfg, detector=detector, classifier=classifier, norm=meta["norm"], image_paths=paths, gt=gt, conf=float(sd["conf"]), nms_iou=nms,
                                cls_threshold=float(sc["probability_threshold"]), match_iou=match_iou, max_examples=int(cfg["figures"]["tp_fp_fn_examples"]))
    thr_source = "isolated_gt_crops_validation"
    final_thr = float(sc["probability_threshold"])
    summary = res.summary
    if args.retune_classifier_threshold:
        matched = res.detections[res.detections["matched_fruit_id"].notna()]
        if matched["gt_label"].nunique() == 2:
            y = (matched["gt_label"] == cfg["classification_labels"]["positive_label"]).astype(int)
            sw = threshold_sweep(y, matched["p_con_signos"], n_points=int(cfg["inference_tuning"]["classifier_threshold_grid_points"]))
            new_thr, val = pick_threshold(sw, str(cfg["inference_tuning"]["classifier_threshold_metric"]))
            log.info("umbral del clasificador sobre crops PREDICHOS: %.3f (aislado en GT crops: %.3f) -> %s=%.4f", new_thr, final_thr, cfg["inference_tuning"]["classifier_threshold_metric"], val)
            op = {**summary["operating_point"], "cls_threshold": new_thr}
            summary, det_tab, cm = summarize_from_table(res.detections, res.misses, n_images=summary["n_images"], pos=cfg["classification_labels"]["positive_label"],
                                                        neg=cfg["classification_labels"]["negative_label"], op=op)
            res.detections, res.confusion = det_tab, cm
            res.summary = summary
            final_thr, thr_source = new_thr, "retuned_on_predicted_crops_validation"
        else:
            log.warning("no hay ambas clases entre los frutos asociados: no se reajusta el umbral")
    summary["timing_note"] = "tiempos medidos en esta PC durante la validacion; NO son el benchmark controlado (17_benchmark_latency.py)"

    write_csv(res.detections, ctx.run_dir / "pipeline_detections.csv", float_format="%.6f")
    write_csv(res.misses, ctx.run_dir / "pipeline_misses.csv", float_format="%.6f")
    write_csv(res.per_image, ctx.run_dir / "per_image_timing.csv", float_format="%.4f")
    e = summary["errors"]
    log.info("PIPELINE %s + %s (validation): det P=%.3f R=%.3f | E_det=%.3f E_clas=%s E_flujo=%.3f | TP=%d FP=%d FN=%d EC=%d",
             det_spec.name, cls_spec.name, summary["detection"]["precision"], summary["detection"]["recall"], e["e_det"] or 0, f"{e['e_clas']:.3f}" if e["e_clas"] is not None else "n/a",
             e["e_flujo"] or 0, summary["tp"], summary["fp"], summary["fn"], e["n_ec"])
    srcs = [ctx.run_dir / "pipeline_detections.csv", ctx.run_dir / "pipeline_misses.csv", ctx.run_dir / "per_image_timing.csv"]
    save_pipeline_figures(ctx, res, det_spec.name, cls_spec.name, suffix="validacion", sources=srcs, n_examples=int(cfg["figures"]["tp_fp_fn_examples"]))

    final = {
        "detector": {"architecture": det_spec.name, "checkpoint_sha256": sd["checkpoint_sha256"], "conf": float(sd["conf"]), "nms_iou": sd["nms_iou"], "nms_applies": sd["nms_applies"]},
        "classifier": {"architecture": cls_spec.name, "checkpoint_sha256": sc["checkpoint_sha256"], "probability_threshold": final_thr, "threshold_source": thr_source,
                       "positive_class": cfg["classifier"]["positive_class"]},
        "matching": {"metric": m["metric"], "threshold": match_iou, "one_to_one_rule": m["one_to_one_rule"], "duplicate_policy": m["duplicate_policy"]},
        "crop_strategy": "ventana entera de la caja predicha sobre la imagen original, sin margen (margin_px=0)", "tuned_on": "outer_validation_20pct", "test_used": False,
    }
    write_json(art / "final_thresholds.json", final)
    write_json(art / "pipeline_validation.json", {"status": "PASS", "summary": summary, "final_thresholds": final, "dataset_hash": ctx.hashes.get("dataset_hash"),
                                                  "split_hash": ctx.hashes.get("split_hash"), "code_hash": ctx.hashes.get("code_hash"), "predicted_crops_used": True,
                                                  "evaluated_on": "outer_validation_20pct", "detections_file": str(ctx.run_dir / "pipeline_detections.csv")})
    write_csv(res.detections, art / "pipeline_validation_detections.csv", float_format="%.6f")
    write_csv(res.misses, art / "pipeline_validation_misses.csv", float_format="%.6f")
    ctx.register_artifact(art / "pipeline_validation.json", "validacion del pipeline integrado")
    ctx.register_artifact(art / "final_thresholds.json", "umbrales definitivos")
    ctx.log_metrics({"summary": summary, "final_thresholds": final})


if __name__ == "__main__":
    raise SystemExit(run_cli(SCRIPT, main, description=__doc__ or SCRIPT, add_arguments=add_arguments))
