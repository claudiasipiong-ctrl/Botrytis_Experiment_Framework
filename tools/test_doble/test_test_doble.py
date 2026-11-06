from pathlib import Path
import sys,importlib.util,hashlib,json,zipfile,copy
import pandas as pd
import pytest
p=Path(__file__).with_name('botrytis_test_doble_pc02.py');s=importlib.util.spec_from_file_location('dual',p);d=importlib.util.module_from_spec(s);s.loader.exec_module(d)

def test_positive_fn_includes_detector_misses_and_classifier_errors():
 det=pd.DataFrame([dict(matched_fruit_id=1,gt_label='con_signos',pred_label='con_signos'),dict(matched_fruit_id=2,gt_label='con_signos',pred_label='sin_signos'),dict(matched_fruit_id=3,gt_label='sin_signos',pred_label='con_signos'),dict(matched_fruit_id=None,gt_label=None,pred_label='con_signos')])
 miss=pd.DataFrame([dict(gt_label='con_signos'),dict(gt_label='sin_signos')]);a=d.positive_counts(det,miss)
 assert a['tp_positive']==1 and a['fn_positive']==2 and a['fp_positive']==2 and a['recall_positive']==1/3
 assert a['f2_positive']==5/15

def test_no_positives_reports_undefined_recall():
 det=pd.DataFrame(columns=['matched_fruit_id','gt_label','pred_label']);miss=pd.DataFrame(columns=['gt_label'])
 assert d.positive_counts(det,miss)['recall_positive'] is None

def test_pin_roundtrip_and_reject_modification(tmp_path):
 p=tmp_path/'freeze.json';v={'pairs':[list(x) for x in d.PAIRS]};d.pin_file(p,v);old=p.read_bytes();d.pin_file(p,v);assert p.read_bytes()==old
 with pytest.raises(RuntimeError):d.pin_file(p,{'changed':True})
 p.write_text('{}')
 with pytest.raises(RuntimeError):d.pin_file(p,{})

def test_zip_allowlist_rejects_traversal_before_writing(tmp_path):
 home=tmp_path/'home';(home/'Downloads').mkdir(parents=True);p=home/'Downloads/ResNet50_para_PC2.zip'
 with zipfile.ZipFile(p,'w') as z:z.writestr('../escape.pt','bad')
 p.with_suffix('.zip.sha256.txt').write_text(d.sha(p))
 with pytest.raises(RuntimeError,match='inesperado'):d.install_resnet(home,tmp_path/'dest')
 assert not (tmp_path/'escape.pt').exists()

def test_wrong_archive_hash_stops(tmp_path):
 home=tmp_path/'home';(home/'Downloads').mkdir(parents=True);p=home/'Downloads/ResNet50_para_PC2.zip';p.write_bytes(b'wrong');p.with_suffix('.zip.sha256.txt').write_text('0'*64)
 with pytest.raises(RuntimeError,match='SHA256'):d.install_resnet(home,tmp_path/'dest')

def test_reporting_does_not_require_completed_second_system(tmp_path):
 a=tmp_path/'A';a.mkdir();(a/'resultado.json').write_text(json.dumps({'positive':{'fn_positive':2},'summary':{'tp':1,'fp':2,'fn':3}}))
 d.save_comparison(tmp_path)
 assert 'resnet50' in (tmp_path/'comparacion_TEST.csv').read_text(encoding='utf-8-sig')


def test_native_freeze_guards_accept_explicit_three_candidates_and_require_real_release(tmp_path,monkeypatch):
 from utils.semantics import semantic_fingerprint
 from utils.hashing import sha256_json
 from utils.comparison import protocol
 from utils.freeze import verify_freeze,load_freeze
 from utils.splits import assert_split_allowed
 from utils.config_loader import load_config
 cfg=load_config('/tmp/botrytis-off-fix/configs/default.yaml');cfg=copy.deepcopy(cfg)
 art=tmp_path/'art';art.mkdir();cfg['paths']['artifacts_dir']=str(art);cfg['test']['freeze_manifest']=str(tmp_path/'freeze_manifest.json')
 derived={}
 for task in ['detection','classification']:
  root=tmp_path/task;root.mkdir();(root/'content_integrity.json').write_text('{}');cfg['paths'][task+'_dir']=str(root);derived[task]=d.sha(root/'content_integrity.json')
 candidates=[]
 for slug in ['yolo26l','resnet50','swin_tiny']:
  ck=tmp_path/(slug+'.pt');ck.write_bytes(slug.encode());v=tmp_path/(slug+'.json');v.write_text('{}');candidates.append(dict(architecture=slug,checkpoint=str(ck),checkpoint_sha256=d.sha(ck),validation_file=str(v),validation_sha256=d.sha(v)))
 c=dict(candidates=candidates,protocol_sha256=sha256_json(protocol(cfg)),review_sha256=None,derived_content_sha256=derived)
 f=dict(semantic_fingerprint=semantic_fingerprint(cfg),comparison=c,code={'code_hash':'test-code'},data={'dataset_hash':'data','split_hash':'split'},environment={},detector={'checkpoint':candidates[0]['checkpoint'],'checkpoint_hash':candidates[0]['checkpoint_sha256']},classifier={'checkpoint':candidates[1]['checkpoint'],'checkpoint_hash':candidates[1]['checkpoint_sha256']})
 path=Path(cfg['test']['freeze_manifest']);d.pin_file(path,f)
 verify_freeze(cfg,load_freeze(cfg),{'code_hash':'test-code','dataset_hash':'data','split_hash':'split'},{})
 with pytest.raises(RuntimeError):assert_split_allowed('test','final_test',cfg)
 (art/'manual_test_release.json').write_text(json.dumps({'state':'AUTORIZADO_PERSONALMENTE','operator':'synthetic fixture','evidence':'synthetic fixture only','freeze_sha256':d.sha(path)}))
 assert_split_allowed('test','final_test',cfg)
 Path(candidates[1]['checkpoint']).write_bytes(b'changed')
 with pytest.raises(Exception):assert_split_allowed('test','final_test',cfg)

def test_transfer_valid_allowlist_idempotent_and_no_source_overwrite(tmp_path):
 home=tmp_path/'home';(home/'Downloads').mkdir(parents=True);p=home/'Downloads/ResNet50_para_PC2.zip'
 names=['config.yaml','manifest.json','artifacts/checkpoints/resnet50/final.pt','artifacts/checkpoints/resnet50/final_meta.json','artifacts/best_configs/resnet50.json','artifacts/outer_validation/resnet50.json','artifacts/outer_validation/resnet50_predictions.csv']
 with zipfile.ZipFile(p,'w') as z:
  for n in names:z.writestr(n,n.encode())
 p.with_suffix('.zip.sha256.txt').write_text(d.sha(p));dest=tmp_path/'source'
 d.install_resnet(home,dest);d.install_resnet(home,dest)
 assert (dest/names[2]).read_bytes()==names[2].encode()
 (dest/names[2]).write_bytes(b'changed')
 with pytest.raises(RuntimeError,match='distinta'):d.install_resnet(home,dest)
 assert (dest/names[2]).read_bytes()==b'changed'


def test_no_model_training_or_threshold_search_in_executor():
 import ast
 tree=ast.parse(p.read_text())
 names=[n.func.attr if isinstance(n.func,ast.Attribute) else n.func.id if isinstance(n.func,ast.Name) else '' for n in ast.walk(tree) if isinstance(n,ast.Call)]
 assert not set(names)&{'train','fit','optimize','pick_threshold','threshold_sweep','create_round'}
 assert d.PAIRS==[('A','resnet50',.69),('B','swin_tiny',.85)]
