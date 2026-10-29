"""Methodology tests using annotations; no real images or model checkpoints."""
import itertools
from pathlib import Path
import pandas as pd
import pytest
from utils.pairing import pair_image, maximum_assignment
from utils.labels import build_fruit_table
from utils.config_loader import load_config
from utils.checks import assert_pairing_labels_consistent, MethodologyViolation
from utils.semantics import semantic_fingerprint, derived_fingerprint
from utils.label_audit import audit_source, export_audit
from utils.metrics import pipeline_error_rates
from utils.pipeline_eval import summarize_from_table, DET_COLUMNS, MISS_COLUMNS


def rec(cid, ordinal=0, box=(0.,0.,.2,.2)):
    return dict(image_id="a",class_id=cid,ordinal=ordinal,x1=box[0],y1=box[1],x2=box[2],y2=box[3],image_w=1,image_h=1)


def test_three_classes_and_sign_independence():
    cfg=load_config()
    for extra in ([],[rec(2,box=(0,0,.2,.2))]):
        table=build_fruit_table(pd.DataFrame([rec(0),rec(1)]+extra),cfg)
        assert len(table)==1 and table.iloc[0]["label"]=="con_signos"
        assert_pairing_labels_consistent(table,cfg)
    table=build_fruit_table(pd.DataFrame([rec(0),rec(2)]),cfg)
    assert table.iloc[0]["label"]=="sin_signos"


def test_maximum_total_not_greedy_and_exact_ordinal_tie():
    assert maximum_assignment([[.90,.80],[.85,.10]])[0]==[1,0]
    assert maximum_assignment([[1.,1.],[1.,1.]])[0]==[0,1]
    # Near-equal weights cannot be overturned by an epsilon tie-break.
    assert maximum_assignment([[1.,1.],[1.,1.-1e-14]])[0]==[1,0]


@pytest.mark.parametrize("matrix",[[[.2,.5,.7],[.8,.1,.4]],[[.2],[.6]],[[0.,0.],[0.,0.]],[[.3,.3],[.3,.3]]])
def test_assignment_agrees_with_bruteforce(matrix):
    assignment,_=maximum_assignment(matrix)
    n,m=len(matrix),len(matrix[0])
    expected=max(sum(matrix[i][j] if j<m else 0 for i,j in enumerate(p)) for p in itertools.permutations(range(m+n),n))
    observed=sum(matrix[i][j] if j<m else 0 for i,j in enumerate(assignment))
    assert observed==pytest.approx(expected)


def test_order_independent_pairing_and_no_multiple_assignment():
    records=[rec(0),rec(0,1,(.5,.5,.8,.8)),rec(1,0,(.5,.5,.8,.8)),rec(1,1)]
    a=pair_image(records);b=pair_image(list(reversed(records)))
    assert a==b and not a[2]
    assert [r["fruit_ordinal"] for r in a[0]]==[1,0]


def test_threshold_restricts_optimization_before_assignment(monkeypatch):
    import utils.pairing as pairing
    # Unrestricted optimum (.6 + .49) would displace the best admissible .9.
    matrix = [[.9, .6], [.49, 0.]]
    monkeypatch.setattr(pairing, "box_iou", lambda s, f: matrix[int(s[0])][int(f[0])])
    records = [rec(0, i, (i, 0, i+1, 1)) for i in range(2)]
    records += [rec(1, i, (i, 0, i+1, 1)) for i in range(2)]
    decisions, candidates, issues = pair_image(records, .5)
    assert decisions[0]["fruit_ordinal"] == 0 and decisions[0]["iou"] == .9
    assert decisions[1]["fruit_ordinal"] is None
    assert decisions[1]["best_iou"] == .49
    assert any("insufficient_iou" in r["reason"] for r in issues)
    assert all(r["iou"] >= .5 for r in candidates if r["selected"])


@pytest.mark.parametrize("records,reason",[
    ([rec(1)],"no_candidate"),
    ([rec(0),rec(1,box=(.1,.1,.4,.4))],"insufficient_iou"),
    ([rec(0),rec(0,1),rec(1)],"duplicate_box"),
    ([rec(0),rec(1),rec(1,1)],"multiple_suspects_claim_fruit"),
    ([rec(0,box=(0,0,0,.3))],"degenerate_box"),
    ([rec(7)],"unknown_class"),
    ([rec(0),rec(0,1),rec(1)],"global_assignment_tie"),
])
def test_anomalies_are_never_silent_negatives(records,reason):
    _,_,issues=pair_image(records)
    assert any(reason in r["reason"] for r in issues)
    table=build_fruit_table(pd.DataFrame(records),load_config())
    assert table["label"].isna().all()
    if len(table):
        with pytest.raises(MethodologyViolation):assert_pairing_labels_consistent(table,load_config())


def test_label_audit_without_images_keeps_invalid_boxes_and_empty_labels(tmp_path):
    (tmp_path/"labels").mkdir()
    (tmp_path/"labels"/"a.txt").write_text("0 .5 .5 .2 .2\n1 .5 .5 .2 .2\n2 .5 .5 .1 .1\n")
    (tmp_path/"labels"/"b.txt").write_text("")
    (tmp_path/"labels"/"c.txt").write_text("0 1.2 .5 .1 .1\n7 .5 .5 .1 .1\n")
    result=audit_source(tmp_path)
    assert result[0]["labels_total"]==3 and result[0]["labels_empty"]==1
    assert result[0]["images_discovered"]==0 and result[0]["valid_pairs"]==1
    assert result[0]["anomalies_total"]>0 and result[0]["unknown_classes"]=={7:1}
    export_audit(result,tmp_path/"report")
    assert (tmp_path/"report"/"pairing_candidates.csv").is_file()


def test_semantic_and_dataset_split_changes_invalidate_derivatives():
    cfg=load_config(overrides=["outer_split.seed=67"])
    initial=semantic_fingerprint(cfg)
    for overrides in (["pairing.min_iou=.6"],["pairing.anomaly_policy=quarantine_images"]):
        assert semantic_fingerprint(load_config(overrides=["outer_split.seed=67"]+overrides))!=initial
    assert derived_fingerprint(cfg,"dataset1","split1")!=derived_fingerprint(cfg,"dataset2","split1")
    assert derived_fingerprint(cfg,"dataset1","split1")!=derived_fingerprint(cfg,"dataset1","split2")


@pytest.mark.parametrize("n_gt,n_miss,n_error",[(10,2,3),(5,0,5),(1,0,0),(20,15,4)])
def test_error_decomposition_identity(n_gt,n_miss,n_error):
    e=pipeline_error_rates(n_gt,n_miss,n_error,n_gt-n_miss)
    assert e["E_global"]==pytest.approx(e["E_det"]+(1-e["E_det"])*e["E_cls_cond"])
    assert e["E_global"]==pytest.approx(e["E_det"]+e["E_cls_abs"])


def test_error_empty_and_all_missed_denominators():
    e=pipeline_error_rates(0,0,0,0)
    assert e["E_global"] is None and e["E_cls_cond"] is None
    e=pipeline_error_rates(4,4,0,0)
    assert e["E_global"]==1 and e["E_cls_cond"] is None and e["E_cls_abs"]==0
    with pytest.raises(ValueError):pipeline_error_rates(10,4,2,9)


def test_pipeline_rejects_double_counting():
    d=pd.DataFrame([dict(image_id="a",det_id=i,matched_fruit_id=0,p_con_signos=.9,gt_label="con_signos") for i in range(2)],columns=DET_COLUMNS)
    with pytest.raises(ValueError,match="multiple"):
        summarize_from_table(d,pd.DataFrame(columns=MISS_COLUMNS),n_images=1,pos="con_signos",neg="sin_signos",op={"cls_threshold":.5})


def test_real_images_and_checkpoints_pending():
    # Explicit real-data check, separate from synthetic software fixtures.
    root=Path(__file__).resolve().parents[1]
    images=list((root/"data/raw/images").glob("*.png"))+list((root/"data/raw/images").glob("*.jpg"))
    if not images:pytest.skip("Project raw images absent; real dataset requires personal audit; official crops/checkpoints pending")
    if not list((root/"artifacts").rglob("*.pt")):pytest.skip("No real model checkpoints; inference and Grad-CAM pending")
    assert images


@pytest.mark.parametrize("policy", ["error", "quarantine_images"])
def test_prepared_aborts_or_quarantines_whole_ambiguous_image(tmp_path, monkeypatch, policy):
    """Exercise the actual preparation guard using annotation-only software fixtures."""
    import json
    import utils.prepared as prepared
    cfg = load_config(overrides=[f"paths.artifacts_dir={tmp_path}", f"pairing.anomaly_policy={policy}"])
    records = [rec(0), rec(1, box=(.7,.7,.9,.9)), {**rec(0), "image_id":"clean"}]
    images = pd.DataFrame([dict(image_id=i, image_w=1, image_h=1) for i in ("a", "clean")])
    monkeypatch.setattr(prepared, "require_validated_dataset", lambda *args, **kwargs: "fixture_hash")
    monkeypatch.setattr(prepared, "load_annotations", lambda *args, **kwargs: (images, pd.DataFrame(records)))
    if policy == "error":
        with pytest.raises(MethodologyViolation, match="no crops permitted"):
            prepared.load_prepared_data(cfg, with_split=False)
        assert not (tmp_path/"quarantine_manifest.json").exists()
    else:
        data = prepared.load_prepared_data(cfg, with_split=False)
        assert list(data.images["image_id"]) == ["clean"]
        assert list(data.fruits["image_id"]) == ["clean"]
        manifest = json.loads((tmp_path/"quarantine_manifest.json").read_text(encoding="utf-8"))
        assert manifest["explicit_policy"] == policy and manifest["image_ids"] == ["a"]
        assert manifest["excluded_physical_fruits"] == 1 and manifest["anomalies"]
    assert (tmp_path/"pairing_audit.csv").is_file()
    assert (tmp_path/"pairing_candidates.csv").is_file()


def test_gradcam_auxiliary_na_and_union_energy():
    import numpy as np
    from utils.spatial_audit import gradcam_sign_metrics
    cam=np.array([[1.,0.],[0.,3.]])
    assert gradcam_sign_metrics(cam,[],(0,0,2,2))["energy_inside_sign"] is None
    m=gradcam_sign_metrics(cam,[(1,1,2,2)],(0,0,2,2))
    assert m["energy_inside_sign"]==pytest.approx(.75) and m["peak_inside_sign"] is True
    # Overlapping auxiliary boxes cannot double-count activation energy.
    m=gradcam_sign_metrics(cam,[(0,0,2,2),(1,1,2,2)],(0,0,2,2))
    assert m["energy_inside_sign"]==1
