"""Auditable final counts, metrics and complete artifact inventory by task/model/split."""
from pathlib import Path
import pandas as pd
from utils.architectures import archs
from utils.checks import MethodologyViolation
from utils.config_loader import cfg_path
from utils.evidence import result_root, split_dir
from utils.hashing import sha256_file
from utils.io import read_csv, read_json, write_csv, write_json
from utils.isolation import audit_crops, audit_source, assert_unique_content, validate_derived_manifest
from utils.splits import SPLITS


def export_split_summary(cfg, data):
    root = result_root(cfg); root.mkdir(parents=True, exist_ok=True)
    crops = read_csv(cfg_path(cfg, "paths.classification_dir") / cfg["classification_dataset"]["manifest_name"])
    validate_derived_manifest(cfg, crops, "classification")
    source = audit_source(data.images, data.manifest)
    crop_table = audit_crops(crops, cfg_path(cfg, "paths.classification_dir"), data.manifest)
    write_csv(source, root / "source_content_audit.csv")
    write_csv(crop_table, root / "crop_content_audit.csv")
    split_of = data.manifest.set_index("image_id").outer_split.to_dict()
    models, prediction_contents, artifact_paths = [], [], []
    for spec in archs():
        for split in SPLITS:
            out = split_dir(cfg, spec, split)
            metric_path = out / "metrics" / "metrics.json"
            item = {"model": spec.name, "task": spec.task, "split": split, "root": str(out),
                    "status": "not_evaluated", "metrics": None}
            if metric_path.is_file():
                stored = read_json(metric_path)
                if any(stored.get(k) != expected for k, expected in (("dataset_hash", data.dataset_hash), ("split_hash", data.split_hash))):
                    raise MethodologyViolation(f"Metricas obsoletas: {spec.name}/{split}")
            if split == "train":
                histories = sorted(out.rglob("history.csv")) if out.exists() else []
                item.update(status="training_recorded" if histories else "not_trained", histories=[str(p) for p in histories],
                            metrics=read_json(metric_path) if metric_path.is_file() else None)
            elif metric_path.is_file():
                metrics = read_json(metric_path)
                if metrics.get("dataset_hash") != data.dataset_hash or metrics.get("split_hash") != data.split_hash:
                    raise MethodologyViolation(f"Resultados obsoletos {spec.name}/{split}")
                if metrics.get("architecture") != spec.name or metrics.get("split") != split:
                    raise MethodologyViolation("Identidad de modelo/split incorrecta")
                index_path = out / "artifact_index.json"
                if not index_path.is_file():
                    raise MethodologyViolation("Evidencia incompleta sin indice de artefactos")
                registered = read_json(index_path)["artifacts"]
                for entry in registered:
                    p = out / entry["path"]
                    if not p.is_file() or sha256_file(p) != entry["sha256"]:
                        raise MethodologyViolation(f"Artefacto alterado: {p}")
                actual = {p.relative_to(out).as_posix() for p in out.rglob("*") if p.is_file() and p != index_path}
                if actual != {r["path"] for r in registered}:
                    raise MethodologyViolation("Archivos no registrados en evidencia")
                preds = read_csv(out / "metrics" / "predictions.csv")
                for r in preds.itertuples():
                    if split_of.get(r.image_id) != split or r.split != split or r.model != spec.name:
                        raise MethodologyViolation("Fuga/identidad incorrecta en predicciones exportadas")
                    prediction_contents.append({"image_id": r.image_id, "outer_split": split, "file_sha256": r.crop_sha256})
                if spec.is_detector:
                    images = read_csv(out / "metrics" / "images.csv")
                    expected = set(data.manifest.loc[data.manifest.outer_split == split, "image_id"])
                    if set(images.image_id) != expected or images.image_id.duplicated().any():
                        raise MethodologyViolation("Detector no exporto todas las imagenes del split")
                else:
                    expected = crops[crops.outer_split == split]
                    if set(zip(preds.image_id, preds.fruit_id)) != set(zip(expected.image_id, expected.fruit_id)):
                        raise MethodologyViolation("Clasificador no exporto todos los crops del split")
                item.update(status="evaluated", metrics=metrics, n_predictions=len(preds), artifacts=[str(out / r["path"]) for r in registered])
                artifact_paths.extend(item["artifacts"])
            models.append(item)
    if prediction_contents:
        assert_unique_content(pd.DataFrame(prediction_contents), ["image_id", "file_sha256"])
    counts = {s: {"images": int((data.manifest.outer_split == s).sum()), "crops": int((crops.outer_split == s).sum()),
                   "class_counts": {label: int(((crops.outer_split == s) & (crops.label == label)).sum()) for label in ("sin_signos", "con_signos")}} for s in SPLITS}
    # dHash flags are for manual review, never automatic deletion or split resampling.
    issues = cfg_path(cfg, "paths.artifacts_dir") / "dataset_issues.csv"
    near = read_csv(issues) if issues.is_file() else pd.DataFrame()
    near = near[near.code == "near_duplicate"] if "code" in near else pd.DataFrame()
    from utils.isolation import near_duplicate_cross_split
    cross_pairs = near_duplicate_cross_split(cfg, data.manifest)
    summary = {"near_duplicate_cross_split_pairs": cross_pairs.to_dict("records"), "counts": counts, "dataset_hash": data.dataset_hash, "split_hash": data.split_hash, "models": models,
               "exact_content_overlap": False, "near_duplicate_flags_for_review": len(near),
               "scope": "exact files/pixels and image/crop identity; same scene/plant/fruit groups cannot be ruled out without collection metadata",
               "test_policy": "only after freeze; descriptive comparison; never reselect", "artifacts": artifact_paths}
    write_json(root / "summary.json", summary)
    lines = ["# Resumen por tarea, modelo y split", "", f"Dataset: `{data.dataset_hash}`", f"Split: `{data.split_hash}`", "", "| Split | Imagenes | Crops | Sin signos | Con signos |", "|---|---:|---:|---:|---:|"]
    for s, c in counts.items():
        lines.append(f"| {s} | {c['images']} | {c['crops']} | {c['class_counts']['sin_signos']} | {c['class_counts']['con_signos']} |")
    lines += ["", "Cruces exactos por identidad, SHA-256 y pixeles: ninguno.", f"Alertas dHash pendientes de revision: {len(near)}. La independencia entre tomas de una misma escena/fruto requiere metadatos de recoleccion.", "", "| Tarea | Modelo | Split | Estado | mAP50-95 | Accuracy | F1 | Ruta |", "|---|---|---|---|---:|---:|---:|---|"]
    for item in models:
        m = item['metrics'] or {}
        lines.append(f"| {item['task']} | {item['model']} | {item['split']} | {item['status']} | {m.get('map50_95', '—')} | {m.get('accuracy', '—')} | {m.get('f1', '—')} | `{item['root']}` |")
    lines += ["", "Metricas completas y rutas de TODOS los artefactos en `summary.json`; cada evaluacion tiene `artifact_index.json` con hashes.", "Las perdidas no expuestas por el backend se declaran ausentes. Test y validation son evaluaciones de un checkpoint; no tienen curvas por epoca."]
    (root / "SUMMARY.md").write_text("\n".join(lines)+"\n", encoding="utf-8")
    for task in ("detection", "classification"):
        for split in ("validation", "test"):
            rows = [{"model":i["model"], "split":split, **{k:v for k,v in (i["metrics"] or {}).items() if k in ("map50", "map50_95", "precision", "recall", "f1", "accuracy", "roc_auc", "loss")}}
                    for i in models if i["task"] == task and i["split"] == split and i["status"] == "evaluated"]
            if rows:
                out = root / "comparisons" / task / split; (out / "plots").mkdir(parents=True,exist_ok=True)
                table = pd.DataFrame(rows); write_csv(table, out / "metrics.csv")
                import matplotlib.pyplot as plt
                for metric in ("map50_95", "accuracy", "f1", "roc_auc"):
                    if metric not in table or not table[metric].notna().any():
                        continue
                    fig,ax=plt.subplots(figsize=(7,4)); ax.bar(table.model, table[metric]); ax.tick_params(axis="x", labelrotation=35)
                    ax.set(ylabel=metric,title=f"{task} | {split}"); fig.tight_layout(); fig.savefig(out / "plots" / f"{metric}.png",dpi=160); plt.close(fig)
    summary["artifacts"] = [str(p) for p in sorted(root.rglob("*")) if p.is_file()]
    write_json(root / "summary.json", summary)
    return summary
