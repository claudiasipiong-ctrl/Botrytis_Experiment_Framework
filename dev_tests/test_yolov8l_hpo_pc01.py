import contextlib
import importlib
import importlib.util
import logging
from pathlib import Path
import sys
from types import SimpleNamespace

import numpy as np
import optuna
import pandas as pd
import pytest
import torch

SCRIPT=Path(__file__).with_name('botrytis_yolov8l_hpo_pc01.py')
if not SCRIPT.is_file():SCRIPT=Path(__file__).resolve().parents[1]/'tools/botrytis_yolov8l_hpo_pc01.py'
spec_=importlib.util.spec_from_file_location('swin_controller',SCRIPT);bt=importlib.util.module_from_spec(spec_);spec_.loader.exec_module(bt)
from utils.config_loader import load_config,cfg_path
from utils.architectures import get_arch
from utils.experiments import fingerprint
from utils.io import write_csv,write_json,write_yaml
from utils.search_spaces import baseline_hyperparameters
from orchestration.study_queue import StudyQueue

torch.set_num_threads(2)


def config(tmp_path):
    c=load_config()
    for key,value in dict(runs_dir='r',artifacts_dir='a',queue_dir='q',optuna_dir='o',results_dir='e',best_configs_dir='best',classification_dir='classification').items():c['paths'][key]=str(tmp_path/value)
    c['compute']['pc_id']='PC-01';c['experiment'].update(round_id='n_yolov8l',experiment_id='botrytis_v4',architecture='yolov8l')
    return c




def test_valid_baseline_three_folds_is_read_only(tmp_path):
    c,spec,f,exp=baseline_fixture(tmp_path);before=bt.baseline_evidence_manifest(f)
    result=bt.inspect_baseline(c,spec,f,'test_code','test_commit',exp)
    assert result['mean']==pytest.approx(.83888) and len(result['folds'])==3
    assert result['human_visual_review'] is False and before==bt.baseline_evidence_manifest(f)


@pytest.mark.parametrize('damage',['history_hash','epoch_duplicate','nan_loss','nan_ap','missing_fold','checkpoint_hash','wrong_code','wrong_summary','different_config'])
def test_invalid_baseline_stops_without_retraining_or_modification(tmp_path,damage):
    c,spec,f,exp=baseline_fixture(tmp_path)
    if damage in ('history_hash','epoch_duplicate','nan_loss','nan_ap'):
        p=f/'fold0/completed_history.csv';h=pd.read_csv(p)
        if damage=='epoch_duplicate':h.loc[1,'epoch']=1
        if damage=='nan_loss':h.loc[1,'train/box_loss']=np.nan
        if damage=='nan_ap':h.loc[1,'metrics/mAP50-95(B)']=np.inf
        write_csv(h,p)
        if damage=='history_hash':p.write_bytes(p.read_bytes()+b'\n')
        if damage!='history_hash':
            r=bt.read(p.with_name('completed_fold.json'));r['history_sha256']=bt.digest(p);write_json(p.with_name('completed_fold.json'),r)
    if damage=='missing_fold':(f/'fold2/completed_fold.json').unlink()
    if damage=='checkpoint_hash':
        r=bt.read(f/'checkpoint_metadata.json');r['checkpoints'][0]['sha256']='bad';write_json(f/'checkpoint_metadata.json',r)
    if damage=='wrong_code':
        r=bt.read(f/'metadata.json');r['code_hash']='other';write_json(f/'metadata.json',r)
    if damage=='wrong_summary':
        r=bt.read(f/'summary.json');r['score_mean']=.99;write_json(f/'summary.json',r)
    if damage=='different_config':c['detector']['baseline']['lr0']=.01
    before=bt.baseline_evidence_manifest(f)
    with pytest.raises((bt.Stop,ValueError,KeyError)):bt.inspect_baseline(c,spec,f,'test_code','test_commit',exp)
    assert bt.baseline_evidence_manifest(f)==before




@pytest.mark.parametrize('answer',['','NO','aprobar'])
def test_no_human_acceptance_no_queue_transition(tmp_path,monkeypatch,answer):
    c=config(tmp_path);q=StudyQueue.from_cfg(c);q.plan(c,[get_arch(bt.MODEL)])
    job=next(j for j in q.jobs() if j.kind=='baseline');q.finish(job.job_id,'PENDING_REVIEW')
    monkeypatch.setattr(sys.stdin,'isatty',lambda:True)
    audit=dict(baseline=dict(mean=.8,sd=.1),fold_range=.2,curves=[],isolation=dict(status='NO_DETECTED_OVERLAP'))
    with pytest.raises(bt.Stop,match='Sin aprobacion'):bt.human_approve(q,q.get(job.job_id),tmp_path,audit,input_fn=lambda _:answer)
    assert q.get(job.job_id).status=='PENDING_REVIEW'
    with q._conn() as con:assert con.execute('SELECT count(*) FROM reviews').fetchone()[0]==0


def test_native_approval_unlocks_only_hpo_and_never_fakes_hpo_review(tmp_path,monkeypatch):
    c,spec,f,exp=baseline_fixture(tmp_path);q=StudyQueue.from_cfg(c);q.plan(c,[spec]);job=next(j for j in q.jobs() if j.kind=='baseline')
    q.finish(job.job_id,'PENDING_REVIEW');monkeypatch.setattr(sys.stdin,'isatty',lambda:True)
    monkeypatch.setenv('USERNAME','I21315')
    audit=dict(baseline=dict(mean=.9,sd=.05),fold_range=.1,curves=[],isolation=dict(status='NO_DETECTED_OVERLAP'),
        baseline_dir=str(f),baseline_evidence=bt.baseline_evidence_manifest(f),experiment_fingerprint=exp)
    bt.write_report(tmp_path,audit)
    bt.human_approve(q,q.get(job.job_id),tmp_path,audit,input_fn=lambda _:'APROBAR')
    assert q.get(job.job_id).status=='COMPLETE'
    claimed=q.claim_next('PC-01',kinds=['hpo'],slugs=['yolov8l'])
    assert claimed.kind=='hpo' and claimed.depends_on==[job.job_id]
    assert bt.read(tmp_path/'aprobacion.json')['scope']=='Solo baseline YOLOv8l -> HPO15x3'
    approved=bt.read(tmp_path/'aprobacion.json');audit['status']='HPO_TERMINADO';bt.write_report(tmp_path,audit)
    assert bt.digest(approved['report'])==approved['report_sha256'] and bt.digest(approved['evidence_json'])==approved['evidence_sha256']
    bt.human_approve(q,q.get(job.job_id),tmp_path,audit,input_fn=lambda _:pytest.fail('Duplicate approval'))


def test_changed_evidence_during_prompt_blocks_approval(tmp_path,monkeypatch):
    c,spec,f,exp=baseline_fixture(tmp_path);q=StudyQueue.from_cfg(c);q.plan(c,[spec]);job=next(j for j in q.jobs() if j.kind=='baseline')
    q.finish(job.job_id,'PENDING_REVIEW');monkeypatch.setattr(sys.stdin,'isatty',lambda:True)
    audit=dict(baseline=dict(mean=.9,sd=.05),fold_range=.1,curves=[],isolation=dict(status='NO_DETECTED_OVERLAP'),baseline_dir=str(f),baseline_evidence=bt.baseline_evidence_manifest(f))
    def decide(_):(f/'summary.json').write_text('{}');return 'APROBAR'
    with pytest.raises(bt.Stop,match='cambiaron'):bt.human_approve(q,q.get(job.job_id),tmp_path,audit,input_fn=decide)
    assert q.get(job.job_id).status=='PENDING_REVIEW'


def test_existing_completed_hpo_is_not_launched(tmp_path,monkeypatch):
    c=config(tmp_path);q=StudyQueue.from_cfg(c);q.plan(c,[get_arch(bt.MODEL)]);job=next(j for j in q.jobs() if j.kind=='hpo');q.finish(job.job_id,'PENDING_REVIEW')
    monkeypatch.setattr('utils.optuna_storage.load_study',lambda *a:SimpleNamespace(trials=[],best_trial=SimpleNamespace(number=1),best_value=.9))
    monkeypatch.setattr('utils.optuna_storage.budget_used',lambda *a:15)
    monkeypatch.setattr(bt,'verify_trial_evidence',lambda *a:[])
    monkeypatch.setattr(bt.importlib,'import_module',lambda *a:pytest.fail('No HPO launcher'))
    assert bt.run_hpo(tmp_path,tmp_path/'config',c,[],tmp_path,q,q.get(job.job_id),tmp_path)==0


@pytest.mark.parametrize('alive',[True,False])
def test_recover_job_requires_matching_dead_owner(tmp_path,monkeypatch,alive):
    c=config(tmp_path);q=StudyQueue.from_cfg(c);q.plan(c,[get_arch(bt.MODEL)]);base=next(j for j in q.jobs() if j.kind=='baseline');q.finish(base.job_id,'COMPLETE')
    job=q.claim_next('PC-01',kinds=['hpo'],slugs=[bt.MODEL])
    bt.atomic_json(tmp_path/'owner.json',dict(job_id=job.job_id,attempt=job.attempt,executor_sha256=bt.EXECUTOR_SHA256,process=dict(pid=1)))
    monkeypatch.setattr(bt,'owner_alive',lambda _:alive)
    if alive:
        with pytest.raises(bt.Stop,match='activo'):bt.recover_hpo_job(q,job,tmp_path)
        assert q.get(job.job_id).status=='RUNNING'
    else:bt.recover_hpo_job(q,job,tmp_path);assert q.get(job.job_id).status=='PENDING'


def test_saved_sampler_exact_continuation_and_tampering_blocks(tmp_path):
    s=optuna.create_study(sampler=optuna.samplers.TPESampler(seed=42));t=s.ask();t.suggest_float('x',.01,1);s.tell(t,.5)
    state=bt.OptimizerState(tmp_path,'identity');state.save(s)
    sampler=__import__('pickle').loads(__import__('pickle').dumps(s.sampler));expected=optuna.create_study(sampler=sampler)
    e=expected.ask().suggest_float('x',.01,1)
    s.sampler=optuna.samplers.TPESampler(seed=999);assert state.restore(s)
    assert s.ask().suggest_float('x',.01,1)==e
    m=bt.read(tmp_path/'state.json');(tmp_path/m['file']).write_bytes(b'bad')
    with pytest.raises(bt.Stop,match='alterado'):state.restore(s)


def test_sampler_unsaved_parameter_draw_is_detected(tmp_path):
    s=optuna.create_study();state=bt.OptimizerState(tmp_path,'identity');state.save(s)
    s.ask().suggest_float('x',.1,.9)
    assert state.restore(s) is False








def baseline_fixture(tmp_path):
    c=config(tmp_path);spec=get_arch(bt.MODEL);f=tmp_path/'r/yolov8l/baseline';f.mkdir(parents=True)
    hp=baseline_hyperparameters(spec,c);exp=fingerprint(c);scores=[.85742,.83619,.82303]
    for fold,score in enumerate(scores):
        folder=f/f'fold{fold}';folder.mkdir()
        h=pd.DataFrame({'fold':[fold]*3,'epoch':[1,2,3],'train_loss':[None]*3,'val_loss':[None]*3,
            'train/box_loss':[1,.5,.2],'train/cls_loss':[1,.4,.1],'train/dfl_loss':[1.4,1.,.8],
            'val/box_loss':[1,.6,.8], 'metrics/mAP50-95(B)':[.5,score,score-.01],
            'metrics/mAP50(B)':[.7,.95,.94], 'metrics/precision(B)':[.5,.9,.88],'metrics/recall(B)':[.5,.9,.88]})
        write_csv(h,folder/'completed_history.csv',float_format='%.12g');write_csv(h.drop(columns='fold'),folder/'results.csv',float_format='%.12g')
        write_json(folder/'completed_fold.json',dict(signature=bt.fold_signature(exp,bt.MODEL,hp,42,fold,100,20),history_sha256=bt.digest(folder/'completed_history.csv'),
            row=dict(fold=fold,score=score,best_epoch=2,epochs_run=3,train_seconds=1)))
    write_json(f/'metadata.json',dict(status='COMPLETE',git_commit='test_commit',code_hash='test_code',pc_id='PC-01'))
    write_yaml(f/'config.yaml',c)
    write_json(f/'summary.json',dict(slug=bt.MODEL,folds=[0,1,2],objective='map50_95',hp=hp,seed=42,max_epochs=100,patience=20,fold_scores=scores,
        score_mean=float(np.mean(scores)),score_sd=float(np.std(scores,ddof=1)),dataset_hash=None,split_hash=None,pc_id='PC-01'))
    write_json(f/'checkpoint_metadata.json',dict(git_commit='test_commit',code_hash='test_code',dataset_hash=None,split_hash=None,
        checkpoints=[dict(role=f'fold{i}_best',sha256=str(i)*64,weights_discarded=True) for i in range(3)]))
    return c,spec,f,exp


def test_yolo_optional_validation_losses_and_empty_aggregate_remain_explicit(tmp_path):
    c,spec,f,exp=baseline_fixture(tmp_path)
    p=f/'fold0/results.csv';h=pd.read_csv(p);h=h.drop(columns='val/box_loss');h['val/cls_loss']=['nan']*3;write_csv(h,p)
    original=p.read_bytes();result=bt.curve_evidence(p,True,2)
    assert result['loss_availability']['val/box_loss']['status']=='ND'
    assert result['loss_availability']['val/cls_loss']['n_nan']==3
    assert result['warnings'] and result['train_validation_metric_gap'] is None
    assert result['best_value']==pytest.approx(.85742) and p.read_bytes()==original


def test_real_native_07_fifteen_trials_three_folds_retains_weights_and_stops(tmp_path,monkeypatch):
    """Native 07/Optuna/queue/cache with a tiny explicit detector fixture, no GPU."""
    import utils.hpo_runner as runner
    import utils.detector_train as dt
    from utils.resume import receipt
    native=importlib.import_module('07_run_hpo_study')
    c=config(tmp_path);q=StudyQueue.from_cfg(c);spec=get_arch(bt.MODEL);q.plan(c,[spec])
    baseline=next(j for j in q.jobs() if j.kind=='baseline');q.finish(baseline.job_id,'PENDING_REVIEW');q.approve(baseline.job_id,operator='TEST',evidence='fixture')
    job=next(j for j in q.jobs() if j.kind=='hpo')
    old=tmp_path/'r/yolov8l/baseline/DO_NOT_CHANGE';old.parent.mkdir(parents=True);old.write_bytes(b'BASELINE')
    accesses=[];calls=[];launches=[]
    class Run:
        def __init__(self,path):
            self.cfg=c;self.run_dir=Path(path);self.run_dir.mkdir(parents=True,exist_ok=True);self.runs_base=cfg_path(c,'paths.runs_dir')
            self.pc_id='PC-01';self.logger=logging.getLogger('hpo_test');self.run_id='tiny-fixture';self.env={'git_commit':'fixture'};self.hashes={};self.entries=[]
        def __enter__(self):return self
        def __exit__(self,*args):return False
        def set_seed(self,*args):pass
        def add_metadata(self,**kwargs):pass
        def note_split_access(self,*args):accesses.append(args)
        def log_metrics(self,values):write_json(self.run_dir/'metrics.json',values)
        def save_figure(self,fig,*args,**kwargs):
            import matplotlib.pyplot as plt
            plt.close(fig)
        def record_checkpoint(self,role,**kwargs):
            self.entries.append(dict(role=role,**kwargs));write_json(self.run_dir/'checkpoint_metadata.json',dict(checkpoints=self.entries))
    class TinyDetector:
        def __init__(self):self.callbacks={}
        def add_callback(self,name,fn):self.callbacks.setdefault(name,[]).append(fn)
    monkeypatch.setattr(dt,'_new_model',lambda *a:(TinyDetector(),'f'*64))
    def fake_train(spec,hp,cfg,**opts):
        assert opts['keep_weights'] is True
        calls.append((opts['name'],dict(hp)));out=Path(opts['project_dir'])/opts['name'];weights=out/'weights';weights.mkdir(parents=True)
        model,_=dt._new_model(spec,cfg)
        best=weights/'best.pt';last=weights/'last.pt';torch.save(dict(model='SYNTHETIC',optimizer=None),best);torch.save(dict(optimizer={'state':{'SYNTHETIC':1}}),last)
        tr=SimpleNamespace(last=str(last),epoch=2)
        for fn in model.callbacks.get('on_model_save',[]):fn(tr)
        receipt(last,spec,cfg,dict(seed=opts['seed'],hp=dict(hp),epoch=3))
        h=pd.DataFrame({'epoch':[1,2,3],'train_loss':[None]*3,'val_loss':[None]*3,'train/box_loss':[1.,.5,.2],
            'train/cls_loss':[1.,.4,.1],'train/dfl_loss':[1.4,1.,.8],'metrics/mAP50-95(B)':[.6,.8,.79],
            'metrics/mAP50(B)':[.8,.9,.89],'metrics/precision(B)':[.7,.9,.89],'metrics/recall(B)':[.7,.9,.89],
            'map50_95':[.6,.8,.79],'map50':[.8,.9,.89]})
        write_csv(h,out/'results.csv',float_format='%.12g')
        for _,row in h.iterrows():
            if opts.get('epoch_callback'):opts['epoch_callback'](opts.get('step_offset',0)+int(row.epoch)-1,row)
        return dt.DetectorRun(score=.8,best_epoch=2,epochs_run=3,stopped_early=True,history=h,
            final_metrics=dict(map50_95=.8,map50=.9,precision=.9,recall=.9,f1=.9),train_seconds=.01,vram_peak_mb=None,
            micro_batch=16,params_m=.1,gflops=.1,weights_path=str(best),weights_sha256=bt.digest(best),pretrained_sha256='f'*64)
    monkeypatch.setattr(runner,'detection_images_table',lambda cfg:pd.DataFrame())
    monkeypatch.setattr(runner,'write_fold_yaml',lambda cfg,fold,out,**kw:(Path(out)/'data.yaml',dict(n_train=2,n_val=1)))
    monkeypatch.setattr(runner,'train_detector',fake_train)
    # Undo hook substitutions when this fixture finishes.
    monkeypatch.setattr(runner,'evaluate_config',runner.evaluate_config);monkeypatch.setattr(runner,'shutil',runner.shutil)
    monkeypatch.setattr(native,'create_or_load_study',native.create_or_load_study)
    monkeypatch.setattr(native,'preflight_training',lambda *a:SimpleNamespace(dataset_hash=None,split_hash=None))
    monkeypatch.setattr(native,'GpuJobLock',lambda *a,**k:contextlib.nullcontext())
    monkeypatch.setattr(native,'nested_run',lambda ctx,spec,name,**kw:Run(cfg_path(c,'paths.runs_dir')/spec.slug/name))
    monkeypatch.setattr(native,'save_training_figures',lambda *a,**k:None)
    def cli(script,main,**kwargs):
        launches.append(kwargs['argv']);assert '--trial-block-size' in kwargs['argv'] and kwargs['argv'][-1]=='15'
        claimed=q.claim_next('PC-01',kinds=['hpo'],slugs=[bt.MODEL]);assert claimed
        ctx=Run(tmp_path/'r/yolov8l/hpo/study');args=SimpleNamespace(architecture=bt.MODEL,n_trials=None,trial_block_size=15,allow_budget_override=False)
        main(ctx,args);q.computed(job.job_id,published=True)
        from orchestration.worker import top_config_ids
        q.expand_after_hpo(c,spec,top_config_ids(c,spec.slug));return 0
    monkeypatch.setattr(native,'run_cli',cli)
    assert bt.run_hpo(tmp_path,tmp_path/'config',c,[],tmp_path,q,q.get(job.job_id),tmp_path/'audit')==0
    assert len(calls)==45 and len(launches)==1 and old.read_bytes()==b'BASELINE'
    assert q.get(job.job_id).status=='PENDING_REVIEW'
    assert len([j for j in q.jobs() if j.kind in ('repeat','aggregate','outer_train','outer_eval','thresholds')])==13
    assert all(j.status=='PENDING' for j in q.jobs() if j.kind not in ('baseline','hpo'))
    assert accesses and all(a[0]=='train' for a in accesses)
    retained=list((tmp_path/'r/yolov8l/hpo').glob('trial_*/fold*/weights/best.pt'))
    snapshots=list((tmp_path/'r/yolov8l/hpo').glob('trial_*/fold*/recovery/epoch.pt'))
    assert len(retained)==45 and len(snapshots)==45
    assert bt.run_hpo(tmp_path,tmp_path/'config',c,[],tmp_path,q,q.get(job.job_id),tmp_path/'audit')==0
    assert len(calls)==45 and len(launches)==1


def test_script_only_launches_native_07_and_never_06_or_08():
    import ast
    tree=ast.parse(SCRIPT.read_text());calls=[]
    for n in ast.walk(tree):
        if isinstance(n,ast.Call) and isinstance(n.func,ast.Attribute) and n.func.attr=='import_module' and n.args and isinstance(n.args[0],ast.Constant):calls.append(n.args[0].value)
    assert calls==['07_run_hpo_study'] and '--force' not in SCRIPT.read_text()


def test_detector_partial_resume_preserves_prior_optimizer_and_complete_csv(tmp_path):
    from utils.resume import receipt
    from utils.detector_train import DetectorRun
    c=config(tmp_path);spec=get_arch(bt.MODEL);hp=dict(optimizer='AdamW',lr0=.001)
    source=tmp_path/'r/yolov8l/hpo/trial_000';fold=source/'fold0';weights=fold/'weights';weights.mkdir(parents=True)
    cp=weights/'last.pt';torch.save(dict(optimizer={'state':{'SYNTHETIC':1}}),cp)
    receipt(cp,spec,c,dict(seed=42,hp=hp,epoch=12,context=dict(fold=0)))
    old_bytes=cp.read_bytes()
    raw=pd.DataFrame({'epoch':range(1,13),'metrics/mAP50-95(B)':np.linspace(.1,.3,12),'metrics/mAP50(B)':np.linspace(.2,.4,12)})
    write_csv(raw,fold/'results.csv')
    hooks=bt.RecoveryHooks(c,spec,tmp_path/'state');destination=tmp_path/'r/yolov8l/hpo/trial_001'
    hooks.context=dict(source=source,run_dir=destination,options=dict(seed=42,max_epochs=60,patience=20))
    def resumed(spec,hp,cfg,**kwargs):
        assert cfg['detector']['resume_checkpoint']==str(cp)
        assert Path(kwargs['project_dir'])/kwargs['name']==fold and kwargs['keep_weights']
        complete=pd.DataFrame({'epoch':range(1,16),'metrics/mAP50-95(B)':np.linspace(.1,.4,15),'metrics/mAP50(B)':np.linspace(.2,.5,15)})
        write_csv(complete,fold/'results.csv');best=weights/'best.pt';torch.save(dict(model='SYNTHETIC_RESUMED'),best)
        return DetectorRun(score=.4,best_epoch=15,epochs_run=3,stopped_early=True,history=complete.tail(3),final_metrics=dict(map50_95=.4),train_seconds=1.,
            vram_peak_mb=None,micro_batch=16,params_m=.1,gflops=.1,weights_path=str(best),weights_sha256=bt.digest(best),pretrained_sha256='f'*64)
    result=hooks.detector(resumed,spec,hp,c,data_yaml=destination/'fold0/data.yaml',project_dir=destination,name='fold0',seed=42,epochs=60)
    archived=list((tmp_path/'state/archive').glob('*/weights/last.pt'))
    assert len(archived)==1 and archived[0].read_bytes()==old_bytes
    assert result.epochs_run==15 and len(result.history)==15 and result.best_epoch==15
    assert len(pd.read_csv(destination/'fold0/results.csv'))==15
