"""Integration 00-05 with synthetic fixtures, independent annotation identity verification."""
from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import pandas as pd
import pytest
from PIL import Image, ImageOps

from conftest import SRC, Workspace
from utils.io import read_csv, read_json, read_yaml
from utils.logging_utils import RUN_FILES

SCRIPTS = ["00_check_environment", "01_validate_dataset", "02_create_outer_split", "03_create_inner_folds",
           "04_build_detection_dataset", "05_build_classification_crops"]


# ----------------------------- verificacion independiente ------------------------------- #
def _oriented_size(path: Path) -> tuple[int, int]:
    with Image.open(path) as im:
        return ImageOps.exif_transpose(im).size


def _independent_truth(ws: Workspace) -> pd.DataFrame:
    """Synthetic positives have exact copies of class 0 boxes as class 1."""
    rows=[]
    for lab in sorted((ws.raw / "labels").glob("*.txt")):
        parsed=[line.split() for line in lab.read_text().splitlines() if line.strip()]
        suspects={tuple(map(float,r[1:])) for r in parsed if int(r[0])==1}
        fruits=[tuple(map(float,r[1:])) for r in parsed if int(r[0])==0]
        for fid,f in enumerate(fruits):
            rows.append({"image_id":lab.stem,"fruit_id":fid,"label":"con_signos" if f in suspects else "sin_signos"})
    return pd.DataFrame(rows)

# ------------------------------------ pruebas ------------------------------------------- #
def test_every_run_is_fully_traceable(pipeline_ws: Workspace) -> None:
    runs = [p for p in (pipeline_ws.runs / "_pipeline").iterdir() if p.is_dir()]
    assert len(runs) == len(SCRIPTS)
    by_script = {}
    for run in runs:
        for name in RUN_FILES:
            assert (run / name).is_file(), f"{run.name}: falta {name}"
        meta = read_json(run / "metadata.json")
        assert meta["status"] == "COMPLETE" and meta["pc_id"] == "PC-01" and meta["end_time"] and meta["start_time"]
        by_script[meta["script"]] = meta
    dataset_hash = by_script["01_validate_dataset"]["dataset_hash"]
    split_hash = by_script["02_create_outer_split"]["split_hash"]
    assert dataset_hash and split_hash
    for s in SCRIPTS[2:]:
        assert by_script[s]["dataset_hash"] == dataset_hash, s
    for s in SCRIPTS[3:]:
        assert by_script[s]["split_hash"] == split_hash, s
    assert by_script["02_create_outer_split"]["seed"] == pipeline_ws.seed
    assert (pipeline_ws.last_run("05_build_classification_crops") / "figures").glob("*.png")


def test_outer_split_70_20_10_by_image(pipeline_ws: Workspace) -> None:
    man = read_csv(pipeline_ws.artifacts / "split_manifest.csv")
    assert list(man.columns) == ["image_id", "outer_split", "split_hash"]
    assert man["outer_split"].value_counts().to_dict() == {"train": 230, "validation": 66, "test": 33}
    assert man["image_id"].is_unique and len(man) == 329
    sha = (pipeline_ws.artifacts / "split_manifest.sha256").read_text().split()
    import hashlib

    assert sha[0] == hashlib.sha256((pipeline_ws.artifacts / "split_manifest.csv").read_bytes()).hexdigest()
    assert sha[1] == "split_manifest.csv"
    assert json.loads((pipeline_ws.artifacts / "test_lock.json").read_text())["locked"] is True


def test_classification_manifest_matches_independent_pairing(pipeline_ws: Workspace) -> None:
    truth = _independent_truth(pipeline_ws)
    man = read_csv(pipeline_ws.artifacts / "classification_manifest.csv")
    for col in ("image_id", "fruit_id", "x1", "y1", "x2", "y2", "max_sign_coverage", "label", "outer_split", "inner_fold"):
        assert col in man.columns
    assert len(man) == len(truth) == int(read_json(pipeline_ws.last_run("01_validate_dataset") / "metrics.json")["n_fruit_boxes"])
    merged = man.merge(truth, on=["image_id", "fruit_id"], suffixes=("", "_truth"))
    assert len(merged) == len(man)
    assert (merged["label"] == merged["label_truth"]).all()
    assert set(man["label"]) == {"con_signos", "sin_signos"}
    # manifiesto duplicado en data/classification identico byte a byte
    assert (pipeline_ws.data / "derived_v4" / "classification" / "manifest.csv").read_bytes() == (pipeline_ws.artifacts / "classification_manifest.csv").read_bytes()


def test_no_leakage_between_splits_and_folds(pipeline_ws: Workspace) -> None:
    man = read_csv(pipeline_ws.artifacts / "split_manifest.csv").set_index("image_id")["outer_split"]
    crops = read_csv(pipeline_ws.artifacts / "classification_manifest.csv")
    assert (crops["image_id"].map(man) == crops["outer_split"]).all()
    # cada crop existe en disco dentro de su split/clase
    for p, split, label in zip(crops["crop_path"], crops["outer_split"], crops["label"]):
        assert p.startswith(f"{split}/{label}/")
        assert (pipeline_ws.data / "derived_v4" / "classification" / p).is_file()
    # ninguna imagen fuente aparece en dos splits (en disco)
    seen: dict[str, str] = {}
    for f in (pipeline_ws.data / "derived_v4" / "classification").rglob("*.png"):
        image_id = f.name.split("__f")[0]
        split = f.relative_to(pipeline_ws.data / "derived_v4" / "classification").parts[0]
        assert seen.setdefault(image_id, split) == split
    # folds: solo train, mismo fold por imagen, y en val/test vacio
    assert crops.loc[crops["outer_split"] != "train", "inner_fold"].isna().all()
    train = crops[crops["outer_split"] == "train"]
    assert train["inner_fold"].notna().all() and (train.groupby("image_id")["inner_fold"].nunique() == 1).all()
    cls = read_csv(pipeline_ws.artifacts / "inner_folds_classification.csv")
    det = read_csv(pipeline_ws.artifacts / "inner_folds_detection.csv")
    train_ids = set(man[man == "train"].index)
    assert set(cls["image_id"]) == set(det["image_id"]) == train_ids
    assert set(cls["inner_fold_cls"]) == set(det["inner_fold_det"]) == {0, 1, 2}
    assert (train.groupby("image_id")["inner_fold"].first() == cls.set_index("image_id")["inner_fold_cls"].reindex(train["image_id"].unique())).all()
    combined = read_csv(pipeline_ws.artifacts / "image_manifest.csv")
    assert list(combined.columns) == ["image_id", "outer_split", "inner_fold_cls", "inner_fold_det", "split_hash"]
    assert combined.loc[combined["outer_split"] != "train", ["inner_fold_cls", "inner_fold_det"]].isna().all().all()


def test_detection_dataset_is_single_class_fruit_only(pipeline_ws: Workspace) -> None:
    det = pipeline_ws.data / "derived_v4" / "detection"
    n_boxes = 0
    for lab in (det / "labels").rglob("*.txt"):
        for line in lab.read_text().splitlines():
            parts = line.split()
            assert parts[0] == "0" and len(parts) == 5 and all(0.0 <= float(v) <= 1.0 for v in parts[1:])
            n_boxes += 1
    truth = _independent_truth(pipeline_ws)
    assert n_boxes == len(truth)  # las cajas de signo NO estan como target
    man = read_csv(pipeline_ws.artifacts / "split_manifest.csv")
    for split, n in man["outer_split"].value_counts().items():
        assert len(list((det / "images" / split).glob("*.jpg"))) == n == len(list((det / "labels" / split).glob("*.txt")))
    y = read_yaml(det / "data.yaml")
    assert y["nc"] == 1 and y["names"] == {0: "fruit"} and "path" not in y
    assert (det / "manifests" / "split_manifest.csv").read_bytes() == (pipeline_ws.artifacts / "split_manifest.csv").read_bytes()
    # mismas imagenes de validation/test en ambas ramas
    crops = read_csv(pipeline_ws.artifacts / "classification_manifest.csv")
    for split in ("validation", "test"):
        ids_det = {p.stem for p in (det / "labels" / split).glob("*.txt")}
        assert set(crops.loc[crops["outer_split"] == split, "image_id"]) <= ids_det


def test_dataset_figures_generated(pipeline_ws: Workspace) -> None:
    expected = ["dataset_distribucion_etiquetas", "dataset_balance_clases_por_split", "dataset_frutos_por_imagen", "dataset_tamanos_bbox",
                "dataset_cobertura_signos", "dataset_crops_por_imagen", "dataset_ejemplos_gt", "dataset_flujo_pairing",
                "validacion_anotaciones_por_tipo", "validacion_frutos_por_imagen", "validacion_tamanos_bbox", "split_distribucion_outer",
                "folds_internos_balance", "deteccion_frutos_por_imagen"]
    for name in expected:
        assert (pipeline_ws.figs / f"{name}.png").stat().st_size > 5_000, name
    run_figs = pipeline_ws.last_run("05_build_classification_crops") / "figures"
    assert len(list(run_figs.glob("*.png"))) == 8


def test_classification_stats_reported(pipeline_ws: Workspace) -> None:
    stats = read_json(pipeline_ws.artifacts / "classification_stats.json")
    cb = stats["class_balance"]
    assert cb["total"] == cb["con_signos"] + cb["sin_signos"] and cb["imbalance_ratio"] >= 1
    assert {"train", "validation", "test"} == set(stats["per_split"])
    for key in ("fruits_per_image", "positive_crops_per_image", "pct_positive_crops_per_image"):
        assert stats[key]["count"] == 329


def test_rerunning_02_is_idempotent_and_seed_change_is_refused(pipeline_ws: Workspace) -> None:
    csv = pipeline_ws.artifacts / "split_manifest.csv"
    before = csv.read_bytes()
    res = pipeline_ws.run("02_create_outer_split")
    assert res.returncode == 0 and csv.read_bytes() == before
    assert read_json(pipeline_ws.last_run("02_create_outer_split") / "metrics.json")["created"] is False
    res = pipeline_ws.run("02_create_outer_split", "--set", "outer_split.seed=999")
    assert res.returncode == 1 and csv.read_bytes() == before
    meta = read_json(pipeline_ws.last_run("02_create_outer_split") / "metadata.json")
    assert meta["status"] == "FAILED" and meta["error"]["type"] == "FixedDecisionViolation"


def test_tampered_manifest_blocks_downstream(pipeline_ws: Workspace, tmp_path: Path) -> None:
    import shutil

    ws2 = Workspace(tmp_path)
    shutil.copytree(pipeline_ws.raw, ws2.raw)
    shutil.copytree(pipeline_ws.artifacts, ws2.artifacts)
    csv = ws2.artifacts / "split_manifest.csv"
    csv.write_bytes(csv.read_bytes().replace(b",train,", b",test,", 1))
    for script in ("03_create_inner_folds", "04_build_detection_dataset", "05_build_classification_crops"):
        res = ws2.run(script)
        assert res.returncode == 1, script
        assert read_json(ws2.last_run(script) / "metadata.json")["error"]["type"] == "SplitIntegrityError"


def test_dataset_change_after_validation_is_detected(pipeline_ws: Workspace, tmp_path: Path) -> None:
    import shutil

    ws2 = Workspace(tmp_path)
    shutil.copytree(pipeline_ws.raw, ws2.raw)
    shutil.copytree(pipeline_ws.artifacts, ws2.artifacts)
    lab = next((ws2.raw / "labels").glob("*.txt"))
    lab.write_text(lab.read_text() + "0 0.5 0.5 0.1 0.1\n")
    res = ws2.run("03_create_inner_folds")
    assert res.returncode == 1
    assert read_json(ws2.last_run("03_create_inner_folds") / "metadata.json")["error"]["type"] == "DatasetNotValidatedError"


def test_environment_mismatch_blocks_run(pipeline_ws: Workspace, tmp_path: Path) -> None:
    import shutil

    ws2 = Workspace(tmp_path)
    shutil.copytree(pipeline_ws.raw, ws2.raw)
    shutil.copytree(pipeline_ws.artifacts, ws2.artifacts)
    ref = ws2.artifacts / "environments" / "environment_reference.json"
    data = json.loads(ref.read_text())
    data["gpu_model"] = "OTRA GPU DISTINTA"
    data["torch"] = "9.9.9"
    ref.write_text(json.dumps(data))
    (ws2.artifacts / "environments" / "environment_pc01.json").unlink()
    res = ws2.run("00_check_environment")
    assert res.returncode == 1
    meta = read_json(ws2.last_run("00_check_environment") / "metadata.json")
    assert meta["status"] == "FAILED" and meta["error"]["type"] == "EnvironmentCheckFailed"
    mismatches = read_json(ws2.last_run("00_check_environment") / "metrics.json")["environment_mismatches"]
    assert {"gpu_model", "torch"} <= {m["field"] for m in mismatches}
    assert ws2.run("00_check_environment", "--no-strict").returncode == 0


@pytest.mark.parametrize("script", SCRIPTS)
def test_help_runs(script: str) -> None:
    res = subprocess.run([sys.executable, str(SRC / f"{script}.py"), "--help"], capture_output=True, text=True, encoding="utf-8", errors="replace")
    assert res.returncode == 0 and "--config" in res.stdout and "--set" in res.stdout


# ---------------------------- bloqueantes y validacion negativa ------------------------- #
def test_invalid_class_mapping_is_blocked(small_ws: Workspace) -> None:
    res = small_ws.run("01_validate_dataset", "--allow-count-mismatch", "--set", "classes.fruit_class_id=[0,1]")
    assert res.returncode == 1
    assert not (small_ws.artifacts / "dataset_validation.json").exists()

def test_missing_split_seed_blocks_split_but_not_validation(small_ws: Workspace) -> None:
    assert small_ws.run("01_validate_dataset", "--allow-count-mismatch", with_seed=False).returncode == 0
    res = small_ws.run("02_create_outer_split", "--set", "outer_split.seed=TODO_FIXTURE", with_seed=False)
    assert res.returncode == 1
    meta = read_json(small_ws.last_run("02_create_outer_split") / "metadata.json")
    assert meta["error"]["type"] in ("BlockingTodoError", "FixedDecisionViolation") and "outer_split.seed" in meta["error"]["message"]
    assert not (small_ws.artifacts / "split_manifest.csv").exists()


def test_downstream_requires_upstream(small_ws: Workspace) -> None:
    assert small_ws.run("02_create_outer_split").returncode == 1                       # sin validar
    assert small_ws.run("01_validate_dataset", "--allow-count-mismatch").returncode == 0
    assert small_ws.run("03_create_inner_folds").returncode == 1                       # sin split
    assert small_ws.run("02_create_outer_split").returncode == 0
    assert small_ws.run("05_build_classification_crops").returncode == 1               # sin folds
    assert small_ws.run("03_create_inner_folds").returncode == 0
    assert small_ws.run("04_build_detection_dataset").returncode == 0
    assert small_ws.run("05_build_classification_crops").returncode == 0
    # los crops de una corrida posterior sin --clean se detectan como obsoletos si cambiara el split: aqui es identico
    assert small_ws.run("05_build_classification_crops").returncode == 0


def _mutate_label(ws: Workspace, new_line: str) -> None:
    lab = sorted((ws.raw / "labels").glob("*.txt"))[0]
    lab.write_text(lab.read_text() + new_line + "\n")


@pytest.mark.parametrize("case, expected_code", [
    ("missing_label", "missing_label"), ("corrupt_image", "corrupt_image"), ("truncated_image", "corrupt_image"),
    ("unknown_class", "unknown_class"), ("out_of_range", "out_of_range"), ("zero_width", "non_positive_size"),
    ("bad_format", "bad_format"), ("duplicate_content", "duplicate_image_content"), ("duplicate_id_case", "duplicate_image_id"),
    ("degenerate_after_clip", "zero_area_after_clip"), ("non_integer_class", "bad_number"),
])
def test_validation_detects_problems(small_ws: Workspace, case: str, expected_code: str) -> None:
    images = sorted((small_ws.raw / "images").glob("*.jpg"))
    labels = sorted((small_ws.raw / "labels").glob("*.txt"))
    if case == "missing_label":
        labels[1].unlink()
    elif case == "corrupt_image":
        images[1].write_bytes(b"esto no es una imagen")
    elif case == "truncated_image":
        images[1].write_bytes(images[1].read_bytes()[:400])
    elif case == "unknown_class":
        _mutate_label(small_ws, "7 0.5 0.5 0.1 0.1")
    elif case == "out_of_range":
        _mutate_label(small_ws, "0 1.5 0.5 0.1 0.1")
    elif case == "zero_width":
        _mutate_label(small_ws, "0 0.5 0.5 0.0 0.1")
    elif case == "bad_format":
        _mutate_label(small_ws, "0 0.5 0.5 0.1")
    elif case == "duplicate_content":
        (small_ws.raw / "images" / "COPIA.jpg").write_bytes(images[0].read_bytes())
        (small_ws.raw / "labels" / "COPIA.txt").write_text(labels[0].read_text())
    elif case == "duplicate_id_case":
        (small_ws.raw / "images" / f"{images[0].stem.lower()}.png").write_bytes(b"x")
    elif case == "degenerate_after_clip":
        _mutate_label(small_ws, "0 1.0000009 0.5 0.000001 0.1")  # dentro de la tolerancia, pero fuera de la imagen tras el clip
    elif case == "non_integer_class":
        _mutate_label(small_ws, "0.5 0.5 0.5 0.1 0.1")
    res = small_ws.run("01_validate_dataset", "--allow-count-mismatch")
    assert res.returncode == 1, res.stdout[-1500:]
    report = read_json(small_ws.artifacts / "dataset_validation.json")
    assert report["passed"] is False
    issues = read_csv(small_ws.artifacts / "dataset_issues.csv")
    assert expected_code in set(issues.loc[issues["severity"] == "ERROR", "code"]), issues.to_dict("records")
    # y 02 se niega a continuar con un dataset no valido
    assert small_ws.run("02_create_outer_split").returncode == 1


def test_validation_warnings_do_not_fail(small_ws: Workspace) -> None:
    _mutate_label(small_ws, "0 0.5 0.5 0.001 0.001")  # valid geometry, descriptive tiny-box warning
    assert small_ws.run("01_validate_dataset", "--allow-count-mismatch").returncode == 0
    issues = read_csv(small_ws.artifacts / "dataset_issues.csv")
    assert "tiny_box" in set(issues.loc[issues["severity"] == "WARNING", "code"])
    assert read_json(small_ws.artifacts / "dataset_validation.json")["passed"] is True


def test_count_mismatch_fails_by_default(small_ws: Workspace) -> None:
    res = small_ws.run("01_validate_dataset")
    assert res.returncode == 1
    issues = read_csv(small_ws.artifacts / "dataset_issues.csv")
    assert "unexpected_image_count" in set(issues.loc[issues["severity"] == "ERROR", "code"])
