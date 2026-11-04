import contextlib
import importlib
import importlib.util
import logging
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest

SCRIPT=Path(__file__).with_name('botrytis_yolov8l_recuperar.py')
if not SCRIPT.is_file():SCRIPT=Path(__file__).resolve().parents[1]/'tools/botrytis_yolov8l_recuperar.py'
s=importlib.util.spec_from_file_location('y8l_close',SCRIPT);bt=importlib.util.module_from_spec(s);s.loader.exec_module(bt)
from utils.config_loader import load_config
from utils.experiments import fingerprint
from utils.architectures import get_arch
from utils.search_spaces import baseline_hyperparameters
from utils.io import write_csv,write_json
from orchestration.study_queue import StudyQueue


def fixture(tmp_path):
    c=load_config();c['project']['profile']='smoke';c['compute']['pc_id']='PC-01';c['experiment'].update(architecture=bt.MODEL,round_id='n_yolov8l')
    for key,name in [('artifacts_dir','a'),('runs_dir','r'),('results_dir','e'),('queue_dir','q')]:c['paths'][key]=str(tmp_path/name)
    f=tmp_path/'r/yolov8l/baseline';f.mkdir(parents=True)
    hp=baseline_hyperparameters(get_arch(bt.MODEL),c);exp=fingerprint(c)
    for fold in range(3):
        d=f/f'fold{fold}';d.mkdir();score=.82+.01*fold
        h=pd.DataFrame({'fold':[fold]*3,'epoch':[1,2,3],'train_loss':[None]*3,'val_loss':[None]*3,
            'train/box_loss':[1,.5,.25],'train/cls_loss':[.8,.4,.2],'train/dfl_loss':[1.4,1.,.8],
            'metrics/mAP50-95(B)':[.6,score,score-.01],'metrics/mAP50(B)':[.8,.95,.94],
            'metrics/precision(B)':[.7,.9,.89],'metrics/recall(B)':[.7,.9,.89]})
        write_csv(h,d/'completed_history.csv',float_format='%.12g');write_csv(h.drop(columns='fold'),d/'results.csv',float_format='%.12g')
        write_json(d/'completed_fold.json',dict(signature=bt.fold_signature(exp,bt.MODEL,hp,42,fold,100,20),history_sha256=bt.digest(d/'completed_history.csv'),
            row=dict(fold=fold,score=score,best_epoch=2,epochs_run=3,train_seconds=10.,vram_peak_mb=100.),params_m=43.6,gflops=164.8,vram_peak_mb=100.))
    meta=dict(checkpoints=[dict(role=f'fold{i}_best',sha256=str(i)*64,weights_discarded=True) for i in range(3)])
    return c,f,meta


def test_empty_aggregate_uses_authentic_components_and_never_fills_it(tmp_path):
    c,f,meta=fixture(tmp_path);before={p:bt.digest(p) for p in f.rglob('*') if p.is_file()}
    evidence,h=bt.verify_saved_folds(c,f)
    assert evidence['all_components_decreased'] and len(evidence['components'])==9 and h.train_loss.isna().all()
    assert before=={p:bt.digest(p) for p in f.rglob('*') if p.is_file()}


@pytest.mark.parametrize('damage',['missing_fold','altered_history','nan_train','inf_metric','bad_signature','different_raw','duplicate_epoch','wrong_best_epoch'])
def test_corrupt_or_incomplete_evidence_blocks_recovery(tmp_path,damage):
    c,f,meta=fixture(tmp_path);p=f/'fold1/completed_history.csv';h=pd.read_csv(p);r=bt.read(p.with_name('completed_fold.json'))
    if damage=='missing_fold':p.with_name('completed_fold.json').unlink()
    if damage=='nan_train':h.loc[0,'train/box_loss']=np.nan
    if damage=='inf_metric':h.loc[0,'metrics/mAP50-95(B)']=np.inf
    if damage=='duplicate_epoch':h.loc[1,'epoch']=1
    if damage=='different_raw':
        raw=pd.read_csv(p.with_name('results.csv'));raw.loc[0,'train/box_loss']=.12;write_csv(raw,p.with_name('results.csv'))
    if damage=='wrong_best_epoch':r['row']['best_epoch']=1
    if damage=='bad_signature':r['signature']='wrong'
    if damage in ('nan_train','inf_metric','duplicate_epoch','altered_history'):
        write_csv(h,p)
        if damage!='altered_history':r['history_sha256']=bt.digest(p)
    if damage!='missing_fold':write_json(p.with_name('completed_fold.json'),r)
    with pytest.raises((bt.Stop,ValueError)):bt.verify_saved_folds(c,f)


def test_component_increase_is_reported_not_falsely_passed(tmp_path):
    c,f,meta=fixture(tmp_path);_,h=bt.verify_saved_folds(c,f);h.loc[h.fold==1,'train/box_loss']=[.1,.2,.3]
    flags,details=bt.component_loss_checks(h)
    assert not all(flags) and any(not d['decreased'] for d in details)


def test_checkpoint_metadata_requires_real_retained_files_or_honest_discard_record(tmp_path):
    c,f,meta=fixture(tmp_path);bt.verify_checkpoint_metadata(meta,f)
    meta['checkpoints'][0]['weights_discarded']=False
    with pytest.raises(bt.Stop,match='conservado'):bt.verify_checkpoint_metadata(meta,f)


def test_only_the_exact_export_error_is_eligible():
    valid=dict(status='FAILED',error=dict(type='TypeError',message="float() argument must be a string or a real number, not 'NoneType'",traceback='06_run_baseline.py first_last.append(float(g["train_loss"].iloc[0]))'))
    assert bt.original_error_is_supported(valid)
    valid['error']['type']='CUDAError';assert not bt.original_error_is_supported(valid)


def test_native_06_reproduces_error_then_recovers_from_real_fold_cache_without_training(tmp_path,monkeypatch):
    import utils.hpo_runner as runner
    native=importlib.import_module('06_run_baseline')
    monkeypatch.setattr(native,'main',native.main)  # restore original main after the in-memory patch
    c,f,meta=fixture(tmp_path);evidence,h=bt.verify_saved_folds(c,f)
    monkeypatch.setattr(native,'preflight_training',lambda *a:SimpleNamespace(dataset_hash='dataset',split_hash='split'))
    monkeypatch.setattr(native,'GpuJobLock',lambda *a,**k:contextlib.nullcontext())
    monkeypatch.setattr(native,'save_training_figures',lambda *a:None)
    monkeypatch.setattr(runner,'detection_images_table',lambda *a:pd.DataFrame())
    class Ctx:
        def __init__(self):self.cfg=c;self.run_dir=f;self.pc_id='PC-01';self.logger=logging.getLogger('recovery');self.run_id='fixture';self.saved=[];self.extra={}
        def set_seed(self,*a):pass
        def add_metadata(self,**kw):self.extra.update(kw)
        def log_metrics(self,values):self.metrics=values
        def register_artifact(self,*a):pass
        def record_checkpoint(self,role,**kw):self.saved.append(dict(role=role,**kw))
        def note_split_access(self,*a):pytest.fail('No new fold should access data for training')
    # Exact original failure: in-memory callback history has Python None.
    from utils.hpo_runner import ConfigEvaluation
    def in_memory(*a,**kw):
        h['train_loss']=pd.Series([None]*len(h),dtype=object)
        return ConfigEvaluation([.82,.83,.84],.83,.01,[r['row'] for r in evidence['receipts']],30.,100.,43.6,164.8,h)
    with monkeypatch.context() as m:
        m.setattr(runner,'evaluate_config',in_memory)
        with pytest.raises(TypeError,match='NoneType'):native.main(Ctx(),SimpleNamespace(architecture=bt.MODEL,folds=None,force=False))
    original_evaluate=runner.evaluate_config
    monkeypatch.setattr(runner,'evaluate_config',original_evaluate);monkeypatch.setattr(runner,'train_detector',runner.train_detector);monkeypatch.setattr(runner,'train_classifier',runner.train_classifier)
    ctx=Ctx();before=evidence['hashes'].copy();bt.install_recovery(native,c,f,meta,evidence)
    native.main(ctx,SimpleNamespace(architecture=bt.MODEL,folds=None,force=False))
    result=bt.read(f/'summary.json')
    assert result['score_mean']==pytest.approx(.83) and result['train_seconds']==30.
    assert result['convergence_checks']['train_loss_decreased_in_all_folds'] is True
    assert len(ctx.saved)==3 and ctx.extra['recovery_executor_sha256']==bt.EXECUTOR_SHA256
    assert before==bt.verify_saved_folds(c,f)[0]['hashes']
    with pytest.raises(bt.Stop,match='PROHIBIDO_ENTRENAR'):runner.train_detector()


def test_missing_cache_after_install_never_falls_back_to_training(tmp_path,monkeypatch):
    import utils.hpo_runner as runner
    native=importlib.import_module('06_run_baseline');monkeypatch.setattr(native,'main',native.main)
    c,f,meta=fixture(tmp_path);evidence,h=bt.verify_saved_folds(c,f)
    monkeypatch.setattr(runner,'evaluate_config',runner.evaluate_config);monkeypatch.setattr(runner,'train_detector',runner.train_detector);monkeypatch.setattr(runner,'train_classifier',runner.train_classifier)
    bt.install_recovery(native,c,f,meta,evidence);(f/'fold0/completed_fold.json').unlink()
    with pytest.raises(bt.Stop,match='ausente'):runner.evaluate_config(get_arch(bt.MODEL),{},c,SimpleNamespace(),f,folds=[0,1,2],trial=None)


def test_native_queue_recovery_requires_review_and_hpo_stays_pending(tmp_path):
    c,f,meta=fixture(tmp_path);q=StudyQueue.from_cfg(c);q.plan(c,[get_arch(bt.MODEL)])
    baseline=next(j for j in q.jobs() if j.kind=='baseline');hpo=next(j for j in q.jobs() if j.kind=='hpo')
    q.finish(baseline.job_id,'FAILED');q.finish(baseline.job_id,'PENDING')
    job=q.claim_next('PC-01',slugs=[bt.MODEL],kinds=['baseline']);assert job
    q.computed(job.job_id,published=True)
    assert q.get(job.job_id).status=='PENDING_REVIEW' and q.get(hpo.job_id).status=='PENDING'
    assert q.claim_next('PC-01',slugs=[bt.MODEL],kinds=['hpo']) is None
    with q._conn() as con:assert con.execute('SELECT count(*) FROM reviews').fetchone()[0]==0
