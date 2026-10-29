"""Content leakage, exhaustive exports, frozen roster and backend training safeguards."""
from pathlib import Path
from types import SimpleNamespace
import ast
import numpy as np
import pandas as pd
import pytest
from PIL import Image
from utils.architectures import get_arch, ArchSpec
from utils.checks import MethodologyViolation
from utils.config_loader import load_config
from utils.evidence import detection_evidence, classification_evidence, finish_evidence
from utils.hashing import sha256_file
from utils.io import read_csv
from utils.isolation import audit_source, audit_crops, seal_files, verify_files
from utils.splits import build_split_manifest


def cfg_at(tmp_path):
    return load_config(None, [f"paths.results_dir={tmp_path / 'results'}", f"paths.artifacts_dir={tmp_path / 'artifacts'}", f"test.freeze_manifest={tmp_path / 'freeze.json'}"])


def make_image(path, colour):
    path.parent.mkdir(parents=True, exist_ok=True)
    Image.new('RGB',(80,60),colour).save(path)
    return path


def test_detects_same_pixels_in_different_files_and_splits(tmp_path):
    a=make_image(tmp_path/'a.png','red'); b=tmp_path/'b.bmp'
    Image.open(a).save(b)
    images=pd.DataFrame({'image_id':['a','b'],'image_path':[str(a),str(b)]})
    with pytest.raises(MethodologyViolation,match='pixel_sha256'):
        audit_source(images,build_split_manifest({'a':'train','b':'test'}))


def test_crops_content_collision_and_inheritance(tmp_path):
    a=make_image(tmp_path/'train'/'a.png','green'); b=make_image(tmp_path/'test'/'b.png','green')
    man=build_split_manifest({'a':'train','b':'test'})
    rows=pd.DataFrame({'image_id':['a','b'],'fruit_id':[0,0],'outer_split':['train','test'],'crop_path':['train/a.png','test/b.png'],'label':['sin_signos','sin_signos']})
    with pytest.raises(MethodologyViolation,match='Fuga'):
        audit_crops(rows,tmp_path,man)
    rows.loc[1,'outer_split']='train'
    with pytest.raises(MethodologyViolation,match='hereda'):
        audit_crops(rows,tmp_path,man)


def test_integrity_rejects_modified_missing_or_stale_files(tmp_path):
    p=make_image(tmp_path/'validation'/'a.png','blue')
    seal_files(tmp_path,[p]); verify_files(tmp_path)
    make_image(tmp_path/'test'/'a.png','blue')
    with pytest.raises(MethodologyViolation,match='no registrados'):
        verify_files(tmp_path)
    (tmp_path/'test'/'a.png').unlink(); p.write_bytes(b'changed')
    with pytest.raises(MethodologyViolation,match='alterado'):
        verify_files(tmp_path)
    p.unlink()
    with pytest.raises(MethodologyViolation,match='ausente'):
        verify_files(tmp_path)


@pytest.mark.parametrize('name',['YOLOv8n','YOLO26l','RT-DETR-R50','other'])
def test_full_detector_export_includes_empty_images_duplicates_and_all_crops(tmp_path,name):
    cfg=cfg_at(tmp_path)
    spec=get_arch(name) if name!='other' else ArchSpec('Other Detector','other_detector','detection','other','','')
    man=build_split_manifest({'v1':'validation','v2':'validation','t':'test'})
    paths={'v1':make_image(tmp_path/'v1.png','blue'),'v2':make_image(tmp_path/'v2.png','red')}
    gt={'v1':{'boxes':np.array([[10,10,30,30]])},'v2':{'boxes':np.empty((0,4))}}
    pred=[(np.array([[10,10,30,30],[10,10,30,30],[50,40,70,55]]),np.array([.9,.8,.4])),(np.empty((0,4)),np.empty(0))]
    ev=detection_evidence(cfg,spec,'validation',man,paths,gt,pred)
    out=Path(ev['root']); tab=read_csv(out/'metrics'/'predictions.csv')
    assert ev['n_images']==2 and ev['n_crops']==3
    assert len(list((out/'visualizations').glob('*.png')))==2
    assert len(list((out/'crops').glob('*.png')))==3
    assert tab.iou_percent.tolist()==[100,100,0]
    assert tab.outcome_iou50.tolist()==['TP','FP','FP']
    assert tab.iou_role.tolist()==['one_to_one_match','best_overlap_only','best_overlap_only']
    assert set(tab.image_id)=={'v1'} and set(tab.split)=={'validation'}
    with pytest.raises(MethodologyViolation,match='exactamente'):
        detection_evidence(cfg,spec,'validation',man,{'v1':paths['v1']},gt,pred)
    with pytest.raises(Exception,match='bloqueado'):
        detection_evidence(cfg,spec,'test',man,{'t':paths['v1']},{'t':gt['v1']},pred[:1])


@pytest.mark.parametrize('name',['ResNet50','ViT-B/16','Swin-Tiny'])
def test_classifier_exports_only_actual_crops_with_labels_probabilities_and_outcomes(tmp_path,name):
    cfg=cfg_at(tmp_path); spec=get_arch(name)
    man=build_split_manifest({'v':'validation','t':'test'})
    paths=['validation/con_signos/v__f0.png','validation/sin_signos/v__f1.png','test/con_signos/t__f0.png']
    for i,p in enumerate(paths): make_image(tmp_path/'crops'/p,['red','blue','green'][i])
    crops=pd.DataFrame({'image_id':['v','v','t'],'fruit_id':[0,1,0],'crop_path':paths,'outer_split':['validation','validation','test']})
    preds=crops.iloc[:2].assign(y_true=[1,0],p_con_signos=[.8,.7])
    ev=classification_evidence(cfg,spec,'validation',man,crops,preds,tmp_path/'crops')
    out=Path(ev['root']); tab=read_csv(out/'metrics'/'predictions.csv')
    assert ev['n_crops']==2 and tab.correct.tolist()==[True,False]
    assert tab.true_label.tolist()==['con_signos','sin_signos'] and tab.confidence.tolist()==[.8,.7]
    for r in tab.itertuples():
        assert sha256_file(out/r.crop_path)==sha256_file(tmp_path/'crops'/r.source_crop)
        assert Image.open(out/r.crop_path).size==(80,60)
    assert len(list((out/'visualizations').glob('*.png')))==2
    with pytest.raises(MethodologyViolation,match='exactamente'):
        classification_evidence(cfg,spec,'validation',man,crops,preds.iloc[:1],tmp_path/'crops')


def load_function_without_ml(filename,name):
    tree=ast.parse((Path(__file__).parents[1]/'src'/filename).read_text())
    fn=next(n for n in tree.body if isinstance(n,ast.FunctionDef) and n.name==name)
    module=ast.Module(body=[ast.ImportFrom(module='__future__',names=[ast.alias(name='annotations')],level=0),fn],type_ignores=[])
    namespace={}; exec(compile(ast.fix_missing_locations(module),filename,'exec'),namespace)
    return namespace[name]


def test_fixed_epoch_detector_never_calls_backend_validation():
    from utils.detector_train import fixed_epoch_trainer
    cls=fixed_epoch_trainer(get_arch('YOLOv8n'))
    trainer=object.__new__(cls)
    assert trainer.validate()==({},0.0) and trainer.final_eval() is None
    text=(Path(__file__).parents[1]/'src'/'utils'/'detector_train.py').read_text()
    assert "args['trainer'] = fixed_epoch_trainer(spec)" in text
    assert 'chosen = last_pt if fixed_epochs' in text


def test_freeze_protocol_changes_and_nonwinner_weight_changes_rejected(tmp_path):
    from utils.comparison import verify_candidates,protocol
    from utils.hashing import sha256_json
    cfg=cfg_at(tmp_path)
    for task in ('detection','classification'):
        cfg['paths'][task+'_dir']=str(tmp_path/task)
        (tmp_path/task).mkdir(); (tmp_path/task/'content_integrity.json').write_text('{}')
    ck=tmp_path/'nonwinner.pt'; ck.write_bytes(b'weights'); va=tmp_path/'validation.json'; va.write_text('{}')
    comp={'protocol_sha256':sha256_json(protocol(cfg)), 'derived_content_sha256':{t:sha256_file(tmp_path/t/'content_integrity.json') for t in ('detection','classification')},
          'candidates':[{'architecture':'nonwinner','checkpoint':str(ck),'checkpoint_sha256':sha256_file(ck),'validation_file':str(va),'validation_sha256':sha256_file(va)}]}
    verify_candidates(cfg,comp)
    cfg['detector']['val_conf']=.7
    with pytest.raises(MethodologyViolation,match='Protocolo'):
        verify_candidates(cfg,comp)
    cfg['detector']['val_conf']=.001
    ck.write_bytes(b'modified')
    with pytest.raises(MethodologyViolation,match='Candidato'):
        verify_candidates(cfg,comp)


def test_near_duplicate_cross_split_requires_documented_review(tmp_path):
    from utils.isolation import require_near_duplicate_review
    from utils.io import write_csv,write_json
    cfg=cfg_at(tmp_path); art=tmp_path/'artifacts'
    man=build_split_manifest({'a':'train','b':'test'})
    write_csv(pd.DataFrame({'image_id':['a','b'],'dhash':['aabb0011aabb0011','aabb0011aabb0011']}),art/'dataset_inventory.csv')
    data=SimpleNamespace(manifest=man,artifacts_dir=art,dataset_hash='verified')
    with pytest.raises(MethodologyViolation,match='pares dHash'):
        require_near_duplicate_review(cfg,data)
    write_json(art/'near_duplicate_review.json',{'dataset_hash':'verified','pairs':[{'image_a':'a','image_b':'b','decision':'distinct_sources','evidence':'Inspection of capture IDs and original images confirms unrelated sources.'}]})
    require_near_duplicate_review(cfg,data)
    write_json(art/'near_duplicate_review.json',{'dataset_hash':'verified','pairs':[{'image_a':'a','image_b':'b','decision':'same_source','evidence':'Same capture.'}]})
    with pytest.raises(MethodologyViolation,match='Posible fuga'):
        require_near_duplicate_review(cfg,data)


def test_final_summary_verifies_counts_completeness_and_artifact_tampering(cfg60,tmp_path):
    from utils.prepared import load_prepared_data
    from utils.split_summary import export_split_summary
    cfg=dict(cfg60); cfg['paths']=dict(cfg60['paths'],results_dir=str(tmp_path/'results'))
    data=load_prepared_data(cfg,with_folds=True)
    spec=get_arch('RT-DETR-R50')
    ids=sorted(data.manifest.loc[data.manifest.outer_split=='validation','image_id'])
    paths={r.image_id:Path(r.image_path) for r in data.images.itertuples() if r.image_id in ids}
    gt={i:{'boxes':data.fruits.loc[data.fruits.image_id==i,['x1','y1','x2','y2']].to_numpy(float)} for i in ids}
    predictions=[(gt[i]['boxes'],np.full(len(gt[i]['boxes']),.95)) for i in ids]
    ev=detection_evidence(cfg,spec,'validation',data.manifest,paths,gt,predictions)
    finish_evidence({'architecture':spec.name,'split':'validation','dataset_hash':data.dataset_hash,'split_hash':data.split_hash,'evidence':ev,'map50_95':1.,'loss':None})
    summary=export_split_summary(cfg,data)
    assert {s:c['images'] for s,c in summary['counts'].items()}=={'train':42,'validation':12,'test':6}
    assert sum(c['crops'] for c in summary['counts'].values())==len(data.fruits)
    evaluated=[m for m in summary['models'] if m['status']=='evaluated']
    assert len(evaluated)==1 and evaluated[0]['split']=='validation'
    assert Path(cfg['paths']['results_dir'],'SUMMARY.md').exists()
    vis=next(Path(ev['root'],'visualizations').glob('*.png')); vis.write_bytes(b'changed')
    with pytest.raises(MethodologyViolation,match='Artefacto alterado'):
        export_split_summary(cfg,data)


def test_legacy_freeze_cannot_open_all_model_test(tmp_path):
    from utils.freeze import verify_freeze
    with pytest.raises(MethodologyViolation,match='Freeze anterior'):
        verify_freeze(cfg_at(tmp_path),{}, {}, {})
