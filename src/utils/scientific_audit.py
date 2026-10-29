"""Auditoria cientifica automatica del experimento (se ejecuta en ``18_export_master_results.py``).

Verifica sobre los ARTEFACTOS REALES (no sobre supuestos):

1. el clasificador se entrena con GT crops (nunca con crops de un detector);
2. el pairing clase 0/clase 1 crea etiquetas; clase 2 auxiliar;
3. el detector solo detecta frutos (labels de una clase; las cajas de signo no son targets);
4. el test permanece bloqueado hasta el freeze (ningun run lo toco antes; solo ``16_evaluate_test``);
5. no hay fuga de datos (split por imagen, crops/folds heredados, sin duplicados entre splits);
6. las metricas son reproducibles (se recalculan desde las predicciones guardadas);
7. los graficos tienen datos reales (procedencia, n_points > 0, sin figuras sinteticas);
8. cada ejecucion deja sus logs (los 6 ficheros obligatorios) y los 4 hashes;
9. el presupuesto/protocolo HPO se cumplio (estudios independientes, 15/20 trials, top-3 x 3 seeds).
"""

from __future__ import annotations

import ast
import re
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Iterator, Mapping

import numpy as np

from utils.architectures import archs
from utils.checks import MethodologyViolation, assert_crops_inherit_split, assert_detection_labels_fruit_only, assert_same_image_same_fold
from utils.config_loader import PROJECT_ROOT, cfg_path
from utils.figure_catalog import required_figures
from utils.hashing import hamming_distance, sha256_file
from utils.hpo_results import predictions_path
from utils.io import read_csv, read_json
from utils.logging_utils import RUN_FILES
from utils.metrics import average_precision_np, binary_classification_report, map_over_iou, roc_auc_np
from utils.pipeline_eval import summarize_from_table
from utils.prepared import PreparedData

PASS, FAIL, WARN, NA = "PASS", "FAIL", "WARN", "N/A"
SRC = PROJECT_ROOT / "src"


@dataclass
class AuditCheck:
    """Resultado de una comprobacion de la auditoria."""

    id: int
    name: str
    status: str
    summary: str
    evidence: dict[str, Any] = field(default_factory=dict)

    def as_dict(self) -> dict[str, Any]:
        """Representacion serializable."""
        return asdict(self)


def iter_runs(runs_dir: Path) -> Iterator[Path]:
    """Directorios de run (los que contienen ``metadata.json``), ordenados."""
    if not runs_dir.is_dir():
        return iter(())
    return iter(sorted(p.parent for p in runs_dir.rglob("metadata.json") if "_bootstrap" not in p.parts and "_powershell" not in p.parts))


def _before(a: str | None, b: str) -> bool:
    """``a < b`` comparando instantes (las fechas pueden venir con husos horarios distintos: local vs UTC)."""
    from datetime import datetime, timezone

    def parse(s: str) -> datetime:
        d = datetime.fromisoformat(s)
        return d if d.tzinfo else d.replace(tzinfo=timezone.utc)

    return a is None or parse(a) < parse(b)


def _load_runs(runs_dir: Path) -> list[dict[str, Any]]:
    out = []
    for d in iter_runs(runs_dir):
        try:
            meta = read_json(d / "metadata.json")
        except Exception:
            meta = {"status": "UNREADABLE"}
        out.append({"dir": d, "meta": meta})
    return out


def _referenced_names(path: Path, names: set[str]) -> set[str]:
    """Nombres de ``names`` que aparecen como llamadas/atributos/imports en el fichero (AST)."""
    try:
        tree = ast.parse(path.read_text(encoding="utf-8"))
    except SyntaxError:
        return set()
    found: set[str] = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Name) and node.id in names:
            found.add(node.id)
        elif isinstance(node, ast.Attribute) and node.attr in names:
            found.add(node.attr)
        elif isinstance(node, (ast.ImportFrom, ast.Import)):
            for a in node.names:
                if a.name in names:
                    found.add(a.name)
    return found


def _imports(path: Path) -> set[str]:
    tree = ast.parse(path.read_text(encoding="utf-8"))
    mods: set[str] = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.ImportFrom) and node.module:
            mods.add(node.module)
        elif isinstance(node, ast.Import):
            mods.update(a.name for a in node.names)
    return mods


# --------------------------------------------------------------------------- #
def check_gt_crops(cfg: Mapping[str, Any], data: PreparedData, runs: list[dict[str, Any]]) -> AuditCheck:
    """(1) El clasificador usa GT crops durante el entrenamiento."""
    root = cfg_path(cfg, "paths.classification_dir")
    man_path = root / str(cfg["classification_dataset"]["manifest_name"])
    ev: dict[str, Any] = {}
    problems: list[str] = []
    if not man_path.is_file():
        return AuditCheck(1, "clasificador con GT crops", FAIL, "no existe el manifiesto de crops", {"manifest": str(man_path)})
    man = read_csv(man_path)
    fr = data.fruits
    m = man.merge(fr, on=["image_id", "fruit_id"], suffixes=("", "_gt"))
    ev["n_manifest"], ev["n_gt_fruits"], ev["n_joined"] = len(man), len(fr), len(m)
    if not (len(man) == len(fr) == len(m)):
        problems.append("el manifiesto de crops no coincide 1:1 con los frutos GT")
    else:
        d = max(float(np.abs(m[c] - m[f"{c}_gt"]).max()) for c in ("x1", "y1", "x2", "y2"))
        ev["max_box_abs_diff_px"] = d
        if d > 1e-4:
            problems.append(f"las cajas de los crops difieren de las cajas GT (max {d:.3g} px)")
    missing = [p for p in man["crop_path"] if not (root / p).is_file()]
    ev["missing_crop_files"] = len(missing)
    if missing:
        problems.append(f"{len(missing)} crops del manifiesto no existen en disco")
    man_sha = sha256_file(man_path)
    bad_runs: list[str] = []
    n_cls_runs = 0
    cls_slugs = {a.slug for a in archs("classification")}
    for r in runs:
        meta = r["meta"]
        if meta.get("architecture") in cls_slugs and meta.get("extra", {}).get("stage") in ("baseline", "hpo", "robustness", "outer_train") and (r["dir"] / "history.csv").is_file():
            n_cls_runs += 1
            ex = meta.get("extra", {})
            if ex.get("train_source") != "ground_truth_crops" or ex.get("crop_manifest_sha256") != man_sha:
                bad_runs.append(meta.get("run_id", str(r["dir"])))
    ev["classifier_training_runs"], ev["runs_without_gt_crop_evidence"] = n_cls_runs, bad_runs[:10]
    if bad_runs:
        problems.append(f"{len(bad_runs)} runs de clasificador sin evidencia de GT crops")
    banned = {"predict_detector", "predict_paths", "load_detector", "evaluate_pipeline", "ultralytics"}
    for f in ("classifier_train.py", "classifier_data.py", "classifier_models.py"):
        imp = _imports(SRC / "utils" / f)
        hit = {i for i in imp if any(b in i for b in ("inference", "pipeline_eval", "ultralytics", "detector_"))} | _referenced_names(SRC / "utils" / f, banned)
        if hit:
            problems.append(f"{f} referencia codigo de detector/inferencia: {sorted(hit)}")
    ev["code_checked"] = ["classifier_train.py", "classifier_data.py", "classifier_models.py"]
    return AuditCheck(1, "clasificador con GT crops", FAIL if problems else PASS, "; ".join(problems) or "crops = cajas GT del fruto; entrenamiento sin predicciones del detector", ev)


def check_annotation_pairing(cfg: Mapping[str, Any], data: PreparedData) -> AuditCheck:
    """Classification follows annotation pairs, independently of class 2."""
    ev = {"source": "class_0_class_1_bipartite_pairing", "min_iou": cfg["pairing"]["min_iou"]}
    problems = []
    man_path = cfg_path(cfg, "paths.classification_dir") / str(cfg["classification_dataset"]["manifest_name"])
    if man_path.is_file():
        man = read_csv(man_path)
        m = man.merge(data.fruits[["image_id", "fruit_id", "label"]], on=["image_id", "fruit_id"], suffixes=("", "_gt"), validate="one_to_one")
        ev["labels_recomputed_from_raw_annotations"] = len(m)
        ev["label_mismatches"] = int((m["label"] != m["label_gt"]).sum())
        if len(m) != len(man) or ev["label_mismatches"]:
            problems.append("Manifest differs from paired raw annotation labels")
    return AuditCheck(2, "GT from annotation pairing", FAIL if problems else PASS, "; ".join(problems) or "class 2 is auxiliary only", ev)


def check_detector_fruit_only(cfg: Mapping[str, Any], data: PreparedData) -> AuditCheck:
    """(3) El detector solo detecta frutos."""
    det = cfg_path(cfg, "paths.detection_dir")
    ev: dict[str, Any] = {}
    problems: list[str] = []
    try:
        n = assert_detection_labels_fruit_only(det / "labels")
        ev["boxes_checked"] = n
        n_fruit = int((data.annotations["kind"] == "fruit").sum())
        ev["fruit_boxes_in_raw_annotations"], ev["sign_boxes_excluded"] = n_fruit, int((data.annotations["kind"] == "sign").sum())
        if n != n_fruit:
            problems.append(f"cajas en labels ({n}) != cajas de fruto GT ({n_fruit})")
    except MethodologyViolation as exc:
        problems.append(str(exc))
    except FileNotFoundError:
        problems.append("no existe data/detection/labels")
    yml = det / "data.yaml"
    if yml.is_file():
        import yaml

        y = yaml.safe_load(yml.read_text(encoding="utf-8"))
        ev["data_yaml"] = {"nc": y.get("nc"), "names": y.get("names")}
        if y.get("nc") != 1 or list((y.get("names") or {}).values()) != ["fruit"]:
            problems.append("data.yaml no es de una sola clase 'fruit'")
    return AuditCheck(3, "el detector solo detecta frutos", FAIL if problems else PASS, "; ".join(problems) or "todas las cajas de entrenamiento son clase 0 (fruit); nc=1", ev)


def check_test_locked(cfg: Mapping[str, Any], runs: list[dict[str, Any]]) -> AuditCheck:
    """(4) El test permanece bloqueado hasta el freeze."""
    art = cfg_path(cfg, "paths.artifacts_dir")
    freeze_p = cfg_path(cfg, "test.freeze_manifest")
    ev: dict[str, Any] = {"test_lock_exists": (art / "test_lock.json").is_file(), "freeze_exists": freeze_p.is_file()}
    problems: list[str] = []
    frozen_at = read_json(freeze_p).get("frozen_at") if freeze_p.is_file() else None
    ev["frozen_at"] = frozen_at
    touched: list[dict[str, Any]] = []
    val_in_trials: list[str] = []
    for r in runs:
        meta = r["meta"]
        for acc in meta.get("splits_accessed") or []:
            if acc["split"] == "test":
                touched.append({"run": meta.get("run_id"), "script": meta.get("script"), "stage": acc["stage"], "started_at": meta.get("started_at")})
            if acc["split"] == "validation" and acc["stage"] == "hpo_trial":
                val_in_trials.append(str(meta.get("run_id")))
    ev["runs_that_accessed_test"] = touched
    for t in touched:
        if t["script"] != "16_evaluate_test" or t["stage"] != "final_test":
            problems.append(f"run {t['run']} ({t['script']}) accedio al test en la etapa {t['stage']}")
        elif frozen_at is None or _before(t["started_at"], frozen_at):
            problems.append(f"run {t['run']} abrio el test antes del freeze")
    if val_in_trials:
        problems.append(f"validation externo usado dentro de trials de HPO: {val_in_trials[:5]}")
    if not (art / "test_lock.json").is_file():
        problems.append("falta test_lock.json (el split externo no bloqueo el test)")
    if touched and not freeze_p.is_file():
        problems.append("hay accesos a test sin freeze_manifest.json")
    ev["validation_used_in_hpo_trials"] = val_in_trials[:5]
    status = FAIL if problems else PASS
    return AuditCheck(4, "test bloqueado hasta el freeze", status, "; ".join(problems) or ("test sin abrir (aun no hay freeze)" if not freeze_p.is_file() else "solo 16_evaluate_test abrio el test, despues del freeze"), ev)


def check_no_leakage(cfg: Mapping[str, Any], data: PreparedData) -> AuditCheck:
    """(5) No existe fuga de datos."""
    ev: dict[str, Any] = {}
    problems: list[str] = []
    warns: list[str] = []
    man = data.manifest
    assert man is not None
    if man["image_id"].duplicated().any():
        problems.append("imagen en mas de un split")
    ev["images_per_split"] = man["outer_split"].value_counts().to_dict()
    cm_path = cfg_path(cfg, "paths.classification_dir") / str(cfg["classification_dataset"]["manifest_name"])
    if cm_path.is_file():
        cm = read_csv(cm_path)
        try:
            assert_crops_inherit_split(cm, man)
            assert_same_image_same_fold(cm, "inner_fold")
            ev["crops_inherit_split_and_fold"] = True
        except MethodologyViolation as exc:
            problems.append(str(exc))
    if data.cls_folds is not None and data.det_folds is not None:
        tr = set(man.loc[man["outer_split"] == "train", "image_id"])
        ev["folds_cover_outer_train_only"] = set(data.cls_folds["image_id"]) == tr == set(data.det_folds["image_id"])
        if not ev["folds_cover_outer_train_only"]:
            problems.append("los folds internos no cubren exactamente outer train")
    inv_path = cfg_path(cfg, "paths.artifacts_dir") / "dataset_inventory.csv"
    if inv_path.is_file():
        inv = read_csv(inv_path).merge(man[["image_id", "outer_split"]], on="image_id")
        dup = inv[inv.duplicated("image_sha256", keep=False)]
        cross = dup.groupby("image_sha256")["outer_split"].nunique()
        ev["identical_images_across_splits"] = int((cross > 1).sum())
        if (cross > 1).any():
            problems.append(f"{int((cross > 1).sum())} imagenes identicas (SHA-256) en splits distintos")
        ok = inv.dropna(subset=["dhash"]).reset_index(drop=True)
        thr = int(cfg["dataset"]["validation"]["near_duplicate_hamming"])
        near: list[tuple[str, str]] = []
        if thr >= 0 and len(ok) < 2000:
            h = [int(x, 16) for x in ok["dhash"]]
            for i in range(len(ok)):
                for j in range(i + 1, len(ok)):
                    if ok.loc[i, "outer_split"] != ok.loc[j, "outer_split"] and hamming_distance(h[i], h[j]) <= thr:
                        near.append((ok.loc[i, "image_id"], ok.loc[j, "image_id"]))
        ev["near_duplicate_pairs_across_splits"] = near[:10]
        ev["n_near_duplicate_pairs_across_splits"] = len(near)
        if near:
            warns.append(f"{len(near)} pares de casi-duplicados (dHash <= {thr}) entre splits distintos: riesgo de fuga")
    status = FAIL if problems else (WARN if warns else PASS)
    return AuditCheck(5, "sin fuga de datos", status, "; ".join(problems + warns) or "split por imagen; crops y folds heredados; sin duplicados entre splits", ev)


def check_reproducible(cfg: Mapping[str, Any], data: PreparedData, runs: list[dict[str, Any]]) -> AuditCheck:
    """(6) Las metricas se reproducen desde las predicciones guardadas."""
    art = cfg_path(cfg, "paths.artifacts_dir")
    ev: dict[str, Any] = {"recomputed": []}
    problems: list[str] = []
    for spec in archs("classification"):
        p = art / "outer_validation" / f"{spec.slug}.json"
        if not p.is_file():
            continue
        ov = read_json(p)
        preds = read_csv(predictions_path(cfg, spec))
        rep = binary_classification_report(preds["y_true"], preds["p_con_signos"], threshold=0.5)
        ap, auc = average_precision_np(preds["y_true"], preds["p_con_signos"]), roc_auc_np(preds["y_true"], preds["p_con_signos"])
        for key, val in (("average_precision_positive", ap), ("roc_auc", auc), ("f1", rep["f1"]), ("accuracy", rep["accuracy"])):
            stored = ov.get(key)
            if stored is not None and val is not None and abs(float(stored) - float(val)) > 1e-9:
                problems.append(f"{spec.name}: {key} guardado {stored} != recalculado {val}")
        ev["recomputed"].append({"architecture": spec.name, "file": str(predictions_path(cfg, spec))})
    gt_cache: dict[str, np.ndarray] = {}
    for spec in archs("detection"):
        p = art / "outer_validation" / f"{spec.slug}.json"
        if not p.is_file():
            continue
        ov = read_json(p)
        preds = read_csv(predictions_path(cfg, spec))
        val_ids = sorted(data.manifest.loc[data.manifest["outer_split"] == "validation", "image_id"])  # type: ignore[union-attr]
        fr = data.fruits[data.fruits["image_id"].isin(val_ids)]
        for iid in val_ids:
            gt_cache[iid] = fr.loc[fr["image_id"] == iid, ["x1", "y1", "x2", "y2"]].to_numpy(float)
        grouped = {k: g for k, g in preds.groupby("image_id")}
        pl = [(grouped[i][["x1", "y1", "x2", "y2"]].to_numpy(float), grouped[i]["score"].to_numpy(float)) if i in grouped else (np.zeros((0, 4)), np.zeros(0)) for i in val_ids]
        m = map_over_iou(pl, [gt_cache[i] for i in val_ids])
        stored = (ov.get("independent") or {}).get("map50_95")
        if stored is not None and abs(float(stored) - m["map50_95"]) > 1e-9:
            problems.append(f"{spec.name}: mAP independiente guardado {stored} != recalculado {m['map50_95']}")
        ev["recomputed"].append({"architecture": spec.name, "file": str(predictions_path(cfg, spec)), "map50_95_recomputed": m["map50_95"]})
    for name, tag in (("pipeline_validation.json", "validacion"),):
        pv = art / name
        det_csv = art / "pipeline_validation_detections.csv"
        if pv.is_file() and det_csv.is_file():
            s = read_json(pv)["summary"]
            d, ms = read_csv(det_csv), read_csv(art / "pipeline_validation_misses.csv")
            new, _, _ = summarize_from_table(d, ms, n_images=s["n_images"], pos=cfg["classification_labels"]["positive_label"], neg=cfg["classification_labels"]["negative_label"], op=s["operating_point"])
            for k in ("e_det", "e_clas", "e_flujo"):
                a, b = s["errors"][k], new["errors"][k]
                if (a is None) != (b is None) or (a is not None and abs(a - b) > 1e-12):
                    problems.append(f"pipeline {tag}: {k} guardado {a} != recalculado {b}")
            ev["recomputed"].append({"pipeline": tag, "e_flujo": new["errors"]["e_flujo"]})
    seedless = [str(r["meta"].get("run_id")) for r in runs if r["meta"].get("extra", {}).get("stage") in ("baseline", "hpo", "robustness", "outer_train") and r["meta"].get("seed") is None]
    ev["training_runs_without_seed"] = seedless[:10]
    if seedless:
        problems.append(f"{len(seedless)} runs de entrenamiento sin seed registrada")
    if not ev["recomputed"]:
        return AuditCheck(6, "metricas reproducibles", NA, "aun no hay evaluaciones que recalcular", ev)
    return AuditCheck(6, "metricas reproducibles", FAIL if problems else PASS, "; ".join(problems) or f"{len(ev['recomputed'])} conjuntos de metricas recalculados desde predicciones guardadas: coinciden", ev)


def check_figures(cfg: Mapping[str, Any], *, allow_missing: bool = False) -> AuditCheck:
    """(7) Los graficos obligatorios existen y tienen datos reales."""
    rep = cfg_path(cfg, "paths.figures_dir")
    fmt = str(cfg["figures"]["formats"][0])
    rows: list[dict[str, Any]] = []
    problems: list[str] = []
    for f in required_figures(cfg):
        img = rep / f"{f['name']}.{fmt}"
        prov = rep / f"{f['name']}.provenance.json"
        row = {**f, "exists": img.is_file(), "n_points": None, "has_data": None, "synthetic": None}
        if img.is_file() and prov.is_file():
            pv = read_json(prov)
            row.update(n_points=pv.get("n_points"), has_data=pv.get("has_data"), synthetic=pv.get("synthetic"))
            if not pv.get("has_data") or pv.get("synthetic"):
                problems.append(f"{f['name']}: sin datos reales (n_points={pv.get('n_points')}, synthetic={pv.get('synthetic')})")
        elif img.is_file():
            problems.append(f"{f['name']}: sin provenance.json")
        else:
            problems.append(f"{f['name']}: FALTA")
        rows.append(row)
    demo = [p.name for p in rep.glob("DEMO_*")] if rep.is_dir() else []
    if demo:
        problems.append(f"figuras DEMO (sinteticas) en reports/figures: {demo[:3]}")
    n_ok = sum(1 for r in rows if r["exists"] and r["has_data"] and not r["synthetic"])
    ev = {"required": len(rows), "ok": n_ok, "missing": [r["name"] for r in rows if not r["exists"]], "table": rows}
    status = PASS if not problems else (WARN if allow_missing else FAIL)
    return AuditCheck(7, "graficos con datos reales", status, f"{n_ok}/{len(rows)} figuras obligatorias OK" + (("; " + "; ".join(problems[:8]) + (" ..." if len(problems) > 8 else "")) if problems else ""), ev)


def check_logs(cfg: Mapping[str, Any], runs: list[dict[str, Any]]) -> AuditCheck:
    """(8) Cada ejecucion deja sus logs y los 4 hashes."""
    bad: list[str] = []
    no_hash: list[str] = []
    for r in runs:
        d, meta = r["dir"], r["meta"]
        miss = [f for f in RUN_FILES if not (d / f).is_file()]
        if not (d / "figures").is_dir():
            miss.append("figures/")
        if (d / "execution.log").is_file() and not (d / "execution.log").read_text(encoding="utf-8", errors="replace").strip():
            miss.append("execution.log vacio")
        if miss:
            bad.append(f"{meta.get('run_id', d.name)}: falta {miss}")
        if meta.get("script", "").split("_")[0] in {f"{i:02d}" for i in range(6, 19)} or meta.get("script") == "worker":
            for h in ("dataset_hash", "code_hash", "environment_hash", "split_hash"):
                if not meta.get(h):
                    no_hash.append(f"{meta.get('run_id')}:{h}")
    ev = {"runs": len(runs), "incomplete_runs": bad[:10], "missing_hashes": no_hash[:10]}
    problems = []
    if bad:
        problems.append(f"{len(bad)} runs incompletos")
    if no_hash:
        problems.append(f"{len(no_hash)} hashes ausentes")
    return AuditCheck(8, "cada ejecucion deja logs y hashes", FAIL if problems else (PASS if runs else NA), "; ".join(problems) or f"{len(runs)} runs con execution.log, config.yaml, environment.json, metrics.json, checkpoint_metadata.json y hashes", ev)


def check_protocol(cfg: Mapping[str, Any], runs_dir: Path | None = None) -> AuditCheck:
    """(9) Presupuesto y protocolo HPO: estudios independientes, 15/20 trials, top-3 x 3 seeds."""
    ev: dict[str, Any] = {"studies": {}}
    problems: list[str] = []
    warns: list[str] = []
    runs_root = runs_dir or cfg_path(cfg, "paths.runs_dir")
    try:
        from utils.optuna_storage import budget_used, load_study, study_db_path
    except Exception as exc:
        return AuditCheck(9, "protocolo HPO", NA, f"optuna no disponible: {exc}", {})
    found = 0
    for spec in archs():
        if not study_db_path(cfg, spec).is_file():
            continue
        found += 1
        st = load_study(spec, cfg)
        used = budget_used(st)
        want = int(cfg["detector" if spec.is_detector else "classifier"]["trials_per_architecture"])
        ev["studies"][spec.name] = {"study_name": st.study_name, "budget_used": used, "budget": want}
        if used != want:
            problems.append(f"{spec.name}: {used} trials usados (presupuesto {want})")
        if spec.study_name(str(cfg["optuna"].get("study_prefix", "botrytis"))) != st.study_name or "global" in st.study_name:
            problems.append(f"{spec.name}: nombre de estudio invalido {st.study_name}")
        rob = runs_root / spec.slug / "robustness" / "summary.json"
        if rob.is_file():
            s = read_json(rob)
            ev["studies"][spec.name]["robustness_candidates"] = len(s["candidates"])
            n_complete = int(((runs_root / spec.slug / "hpo" / "trials.csv").is_file() and (read_csv(runs_root / spec.slug / "hpo" / "trials.csv")["state"] == "COMPLETE").sum()) or 0)
            if len(s["candidates"]) != int(cfg["robustness"]["top_k_configs"]):
                if n_complete < int(cfg["robustness"]["top_k_configs"]) and len(s["candidates"]) == n_complete:
                    warns.append(f"{spec.name}: solo {n_complete} trials COMPLETE (el resto fue podado) < top-{cfg['robustness']['top_k_configs']}")
                else:
                    problems.append(f"{spec.name}: {len(s['candidates'])} configuraciones en robustez (esperadas {cfg['robustness']['top_k_configs']})")
            if s["selected"].get("seeds") != list(cfg["robustness"]["seeds"]):
                problems.append(f"{spec.name}: seeds de robustez {s['selected'].get('seeds')} != predeclaradas")
    if not found:
        return AuditCheck(9, "protocolo HPO", NA, "aun no hay estudios Optuna", ev)
    status = FAIL if problems else (WARN if warns else PASS)
    return AuditCheck(9, "protocolo HPO", status, "; ".join(problems + warns) or f"{found} estudios independientes con el presupuesto oficial y robustez top-k x seeds predeclaradas", ev)


def run_audit(cfg: Mapping[str, Any], data: PreparedData, *, allow_missing_figures: bool = False, runs_dir: Path | None = None) -> list[AuditCheck]:
    """Ejecuta las 9 comprobaciones sobre ``runs_dir`` (por defecto ``paths.runs_dir``; usar el ``--runs-dir`` real de la ejecucion)."""
    root = runs_dir or cfg_path(cfg, "paths.runs_dir")
    runs = _load_runs(root)
    return [
        check_gt_crops(cfg, data, runs), check_annotation_pairing(cfg, data), check_detector_fruit_only(cfg, data), check_test_locked(cfg, runs),
        check_no_leakage(cfg, data), check_reproducible(cfg, data, runs), check_figures(cfg, allow_missing=allow_missing_figures), check_logs(cfg, runs), check_protocol(cfg, root),
    ]


def audit_markdown(checks: list[AuditCheck]) -> str:
    """Informe en Markdown de la auditoria."""
    lines = ["# Auditoria cientifica automatica", "", "| # | Comprobacion | Estado | Resumen |", "|---|---|---|---|"]
    for c in checks:
        lines.append(f"| {c.id} | {c.name} | **{c.status}** | {re.sub(r'[|]', '/', c.summary)[:300]} |")
    return "\n".join(lines) + "\n"
