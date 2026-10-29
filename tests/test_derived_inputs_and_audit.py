"""Entradas derivadas para entrenamiento (folds, YAML, crops) y auditoria cientifica sobre un workspace real de 60 imagenes sinteticas."""

from __future__ import annotations

import shutil
import zipfile
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from conftest import Workspace
from orchestration.sync import pack_base, unpack, verify_zip
from utils.checks import MethodologyViolation
from utils.detector_data import detection_images_table, write_fold_yaml, write_outer_yaml
from utils.io import read_csv, read_yaml
from utils.prepared import load_prepared_data
from utils.scientific_audit import (
    _load_runs, check_detector_fruit_only, check_gt_crops, check_no_leakage, check_annotation_pairing, check_test_locked,
)
from utils.splits import TestSetLockedError


def test_classifier_folds_are_inner_folds_of_outer_train_and_share_no_image(cfg60: dict) -> None:
    pytest.importorskip("torch",reason="Live classifier data imports require PyTorch")
    pytest.importorskip("timm",reason="Classifier model module requires timm")
    from utils.classifier_data import load_crop_manifest
    from utils.classifier_train import load_fold_frames
    man = load_crop_manifest(cfg60)
    for fold in range(3):
        tr, va = load_fold_frames(man, cfg60, fold)
        assert set(tr["outer_split"]) == {"train"} and set(va["outer_split"]) == {"train"}
        assert not set(tr["image_id"]) & set(va["image_id"])
        assert set(va["inner_fold"].astype(int)) == {fold} and fold not in set(tr["inner_fold"].astype(int))
        assert set(tr["y"]) <= {0, 1} and (tr["y"] == (tr["label"] == "con_signos")).all()
    val_all = set().union(*(set(load_fold_frames(man, cfg60, f)[1]["image_id"]) for f in range(3)))
    assert val_all == set(man.loc[man["outer_split"] == "train", "image_id"])             # los 3 folds cubren exactamente outer train


def test_hpo_cannot_read_validation_or_test_crops(cfg60: dict) -> None:
    pytest.importorskip("torch",reason="Classifier data imports require PyTorch")
    from utils.classifier_data import crops_for, load_crop_manifest
    man = load_crop_manifest(cfg60)
    with pytest.raises(TestSetLockedError):
        crops_for(man, split="test", stage="hpo_trial", cfg=cfg60)
    with pytest.raises(TestSetLockedError, match="trials"):
        crops_for(man, split="validation", stage="hpo_trial", cfg=cfg60)
    assert len(crops_for(man, split="validation", stage="outer_validation", cfg=cfg60)) > 0
    with pytest.raises(MethodologyViolation, match="outer train"):
        crops_for(man, split="validation", stage="outer_validation", cfg=cfg60, exclude_fold=0)


def test_detection_fold_yaml_uses_only_outer_train_and_never_lists_test(cfg60: dict, tmp_path: Path) -> None:
    yml, counts = write_fold_yaml(cfg60, 1, tmp_path / "f1")
    data = read_yaml(yml)
    assert set(data) == {"train", "val", "nc", "names"} and data["nc"] == 1 and data["names"] == {0: "fruit"}     # sin clave 'test'
    from utils.detector_data import list_file_encoding

    tr = Path(data["train"]).read_text(encoding=list_file_encoding()).splitlines()
    va = Path(data["val"]).read_text(encoding=list_file_encoding()).splitlines()
    assert len(tr) == counts["n_train"] and len(va) == counts["n_val"] and not set(tr) & set(va)
    assert all("/images/train/" in p for p in tr + va)
    table = detection_images_table(cfg60)
    train_ids = set(table.loc[table["outer_split"] == "train", "image_id"])
    assert {Path(p).stem for p in tr + va} == train_ids
    with pytest.raises(TestSetLockedError):
        write_outer_yaml(cfg60, tmp_path / "o", val_split="test", stage="outer_train")
    assert read_yaml(write_outer_yaml(cfg60, tmp_path / "o", val_split="validation"))["val"].endswith("images/validation")


def test_ultralytics_reads_list_files_as_utf8_so_accented_paths_work() -> None:
    pytest.importorskip("ultralytics",reason="Ultralytics unavailable; its actual file reader cannot be inspected")
    import inspect

    from ultralytics.data.base import BaseDataset

    from utils.detector_data import list_file_encoding

    assert 'encoding="utf-8"' in inspect.getsource(BaseDataset.get_img_files) and list_file_encoding() == "utf-8"


def test_crop_dataset_loads_gt_crops_with_deterministic_eval_transform(cfg60: dict) -> None:
    pytest.importorskip("torch",reason="CropDataset and transforms require PyTorch")
    from utils.classifier_data import CropDataset, build_transforms, crops_for, load_crop_manifest
    man = load_crop_manifest(cfg60)
    va = crops_for(man, split="validation", stage="outer_validation", cfg=cfg60).head(6)
    norm = {"mean": (0.5, 0.5, 0.5), "std": (0.5, 0.5, 0.5)}
    root = Path(cfg60["paths"]["classification_dir"])
    ds = CropDataset(va, root, build_transforms(cfg60, norm, train=False))
    a, b = ds[0], ds[0]
    assert np.array_equal(a[0].numpy(), b[0].numpy()) and a[0].shape == (3, 224, 224) and a[1] in (0, 1)


def test_scientific_audit_passes_on_a_clean_workspace(ws60: Workspace, cfg60: dict) -> None:
    data = load_prepared_data(cfg60, with_folds=True)
    runs = _load_runs(ws60.runs)
    assert check_gt_crops(cfg60, data, runs).status == "PASS"
    assert check_annotation_pairing(cfg60, data).status == "PASS"
    assert check_detector_fruit_only(cfg60, data).status == "PASS"
    assert check_no_leakage(cfg60, data).status in ("PASS", "WARN")
    assert check_test_locked(cfg60, runs).status == "PASS"                                    # test sin abrir


def test_audit_detects_tampered_labels_and_leaking_labels(ws60: Workspace, cfg60: dict, tmp_path: Path) -> None:
    data = load_prepared_data(cfg60, with_folds=True)
    root = Path(cfg60["paths"]["classification_dir"])
    man_path = root / "manifest.csv"
    original = man_path.read_text(encoding="utf-8")
    try:
        man = read_csv(man_path)
        i = man.index[man["label"] == "sin_signos"][0]
        man.loc[i, "label"] = "con_signos"                                                    # etiqueta que no sigue el pairing GT
        man.to_csv(man_path, index=False, lineterminator="\n")
        assert check_annotation_pairing(cfg60, data).status == "FAIL"
        man2 = read_csv(man_path)
        man2.loc[0, "outer_split"] = "test" if man2.loc[0, "outer_split"] != "test" else "train"   # crop con split distinto al de su imagen
        man2.to_csv(man_path, index=False, lineterminator="\n")
        assert check_no_leakage(cfg60, data).status == "FAIL"
        man3 = read_csv(man_path)
        man3.loc[0, "x2"] = float(man3.loc[0, "x2"]) + 5                                      # caja del crop != caja GT
        man3.to_csv(man_path, index=False, lineterminator="\n")
        assert check_gt_crops(cfg60, data, []).status == "FAIL"
    finally:
        man_path.write_text(original, encoding="utf-8", newline="\n")
    lab = next((Path(cfg60["paths"]["detection_dir"]) / "labels" / "train").glob("*.txt"))
    old = lab.read_text(encoding="utf-8")
    try:
        lab.write_text(old + "1 0.5 0.5 0.1 0.1\n", encoding="utf-8")                        # una caja de signo (clase 1) en el dataset del detector
        assert check_detector_fruit_only(cfg60, data).status == "FAIL"
    finally:
        lab.write_text(old, encoding="utf-8", newline="\n")
    assert check_detector_fruit_only(cfg60, data).status == "PASS"


def test_audit_flags_identical_images_across_splits(ws60: Workspace, cfg60: dict) -> None:
    data = load_prepared_data(cfg60, with_folds=True)
    inv_path = Path(cfg60["paths"]["artifacts_dir"]) / "dataset_inventory.csv"
    original = inv_path.read_text(encoding="utf-8")
    try:
        inv = read_csv(inv_path)
        man = data.manifest.set_index("image_id")["outer_split"]   # type: ignore[union-attr]
        a = inv.index[inv["image_id"].map(man) == "train"][0]
        b = inv.index[inv["image_id"].map(man) == "validation"][0]
        inv.loc[b, "image_sha256"] = inv.loc[a, "image_sha256"]                               # misma imagen en train y validation
        inv.to_csv(inv_path, index=False, lineterminator="\n")
        res = check_no_leakage(cfg60, data)
        assert res.status == "FAIL" and "identicas" in res.summary
    finally:
        inv_path.write_text(original, encoding="utf-8", newline="\n")


# ---------------------------------------------- sync ------------------------------------------- #
def test_base_zip_roundtrip_detects_corruption_and_bad_hashes(ws60, cfg60, tmp_path):
    out=tmp_path/'base.zip'
    man=pack_base(cfg60,out)
    assert man['kind']=='base'
    names=set(zipfile.ZipFile(out).namelist())
    assert 'artifacts/split_manifest.csv' in names and 'artifacts/fleet_mapping.json' in names
    assert not any(n.startswith(('runs/','dataset/','data/')) for n in names)
    assert verify_zip(out)['n_files']==man['n_files']
    bad=tmp_path/'bad.zip';data=bytearray(out.read_bytes());data[len(data)//2]^=0xff;bad.write_bytes(data)
    shutil.copy(out.with_suffix('.zip.sha256'),bad.with_suffix('.zip.sha256'))
    with pytest.raises(ValueError):verify_zip(bad)


def test_base_requires_local_validated_dataset_and_preserves_queue_authority(ws60,cfg60,tmp_path,monkeypatch):
    import copy
    from orchestration import exchange as ex
    from orchestration.study_queue import StudyQueue
    from utils.config_loader import cfg_path
    src=cfg60;out=tmp_path/'base.zip';pack_base(src,out)
    fresh=tmp_path/'pc02';dst=copy.deepcopy(src)
    dst['paths'].update(artifacts_dir=str(fresh/'art'),queue_dir=str(fresh/'art/queue'),environments_dir=str(fresh/'art/environments'))
    # Raw/derived fixture bytes already exist locally. Base ZIP never transports images.
    art=cfg_path(dst,'paths.artifacts_dir');art.mkdir(parents=True)
    for name in ['dataset_validation.json','fleet_mapping.json']:
        shutil.copy(ws60.artifacts/name,art/name)
    monkeypatch.setattr(ex,'read_binding',lambda c:{'pc_id':'PC-02','machine_id':'i21314'})
    with pytest.raises(ValueError,match='--force'):ex.unpack(dst,out,fresh,runs_dir=fresh/'runs',force=True)
    result=ex.unpack(dst,out,fresh,runs_dir=fresh/'runs')
    assert result['kind']=='base' and (art/'split_manifest.csv').read_bytes()==(ws60.artifacts/'split_manifest.csv').read_bytes()
    q=StudyQueue.from_cfg(src);q.plan(src)
    monkeypatch.setattr(ex,'read_binding',lambda c:{'pc_id':'PC-01','machine_id':'i21315'})
    result=ex.pack_results(src,tmp_path/'results.zip','PC-01',ws60.runs,with_checkpoints=False)
    assert any(e['path']=='evidence/PC-01/queue.json' for e in result['files'])
    assert not any('queue.sqlite' in e['path'] for e in result['files'])


def test_results_zip_rejects_other_dataset_and_imports_idempotently(ws60,cfg60,tmp_path,monkeypatch):
    import json
    from orchestration import exchange as ex
    from utils.hashing import write_checksum_file
    runs=tmp_path/'runs_a';path=runs/'yolov8n/baseline/metrics.json';path.parent.mkdir(parents=True)
    path.write_text(json.dumps({'fixture':'SYNTHETIC_NO_TRAINING','x':1}))
    monkeypatch.setattr(ex,'read_binding',lambda c:{'pc_id':'PC-01','machine_id':'i21315'})
    ex.publish_architecture(cfg60,'PC-01','yolov8n',runs,stage='baseline')
    out=tmp_path/'results.zip';ex.pack_results(cfg60,out,'PC-01',runs,with_checkpoints=False)
    monkeypatch.setattr(ex,'read_binding',lambda c:{'pc_id':'PC-02','machine_id':'i21314'})
    dest=tmp_path/'runs_b';result=ex.unpack(cfg60,out,tmp_path,runs_dir=dest)
    assert (dest/'yolov8n/baseline/metrics.json').read_bytes()==path.read_bytes()
    assert result['queue_merge'].startswith('foreign evidence')
    assert ex.unpack(cfg60,out,tmp_path,runs_dir=dest)['idempotent']
    tampered=tmp_path/'other.zip'
    with zipfile.ZipFile(out) as zin,zipfile.ZipFile(tampered,'w') as zout:
        for item in zin.infolist():
            payload=zin.read(item.filename)
            if item.filename=='manifest.json':
                m=json.loads(payload);m['hashes']['dataset_hash']='0'*64;payload=json.dumps(m).encode()
            zout.writestr(item,payload)
    write_checksum_file(tampered,tampered.with_suffix('.zip.sha256'))
    with pytest.raises(ValueError,match='hashes diferentes'):ex.unpack(cfg60,tampered,tmp_path,runs_dir=dest)


def test_detection_table_matches_the_manifest(cfg60):
    t=detection_images_table(cfg60)
    man=pd.read_csv(Path(cfg60['paths']['artifacts_dir'])/'split_manifest.csv')
    assert len(t)==len(man)==60 and set(t['outer_split'])=={'train','validation','test'}
    assert t.loc[t['outer_split']!='train','inner_fold_det'].isna().all()
