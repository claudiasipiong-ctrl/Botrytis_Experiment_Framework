import contextlib
import copy
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

SCRIPT=Path(__file__).with_name('botrytis_swin_hpo_pc02.py')
if not SCRIPT.is_file():SCRIPT=Path(__file__).resolve().parents[1]/'tools/botrytis_swin_hpo_pc02.py'
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
    c['compute']['pc_id']='PC-02';c['experiment'].update(round_id='n_swin_tiny',experiment_id='botrytis_v4',architecture='swin_tiny')
    return c


def baseline_fixture(tmp_path):
    c=config(tmp_path);spec=get_arch('swin_tiny');folder=tmp_path/'r/swin_tiny/baseline';folder.mkdir(parents=True)
    hp=baseline_hyperparameters(spec,c);exp=fingerprint(c);scores=[.95,.89,.85]
    for fold,score in enumerate(scores):
        f=folder/f'fold{fold}';f.mkdir()
        h=pd.DataFrame(dict(fold=[fold]*3,epoch=[1,2,3],train_loss=[1,.3,.1],val_loss=[1,.2,.3],val_ap=[.4,score,score-.01],val_f1=[.4,.8,.7],val_precision=[.5,.8,.7],val_recall=[.5,.8,.7],val_roc_auc=[.5,.9,.8]))
        write_csv(h,f/'completed_history.csv',float_format='%.12g')
        write_json(f/'completed_fold.json',dict(signature=bt.fold_signature(exp,spec.slug,hp,42,fold,50,15),history_sha256=bt.digest(f/'completed_history.csv'),
            row=dict(fold=fold,score=score,epochs_run=3,best_epoch=2,train_seconds=1)))
    write_json(folder/'metadata.json',dict(status='COMPLETE',git_commit='test_commit',code_hash='test_code',pc_id='PC-02'))
    write_yaml(folder/'config.yaml',c)
    write_json(folder/'summary.json',dict(slug=spec.slug,folds=[0,1,2],objective='average_precision_positive',hp=hp,seed=42,max_epochs=50,patience=15,
        fold_scores=scores,score_mean=float(np.mean(scores)),score_sd=float(np.std(scores,ddof=1)),dataset_hash=None,split_hash=None,pc_id='PC-02'))
    write_json(folder/'checkpoint_metadata.json',dict(git_commit='test_commit',code_hash='test_code',dataset_hash=None,split_hash=None,
        checkpoints=[dict(role=f'fold{i}_best',sha256=str(i)*64,weights_discarded=True) for i in range(3)]))
    return c,spec,folder,exp


def test_valid_baseline_three_folds_is_read_only(tmp_path):
    c,spec,f,exp=baseline_fixture(tmp_path);before=bt.baseline_evidence_manifest(f)
    result=bt.inspect_baseline(c,spec,f,'test_code','test_commit',exp)
    assert result['mean']==pytest.approx(.8966666666666666) and len(result['folds'])==3
    assert result['human_visual_review'] is False and before==bt.baseline_evidence_manifest(f)


@pytest.mark.parametrize('damage',['history_hash','epoch_duplicate','nan_loss','nan_ap','missing_fold','checkpoint_hash','wrong_code','wrong_summary','different_config'])
def test_invalid_baseline_stops_without_retraining_or_modification(tmp_path,damage):
    c,spec,f,exp=baseline_fixture(tmp_path)
    if damage in ('history_hash','epoch_duplicate','nan_loss','nan_ap'):
        p=f/'fold0/completed_history.csv';h=pd.read_csv(p)
        if damage=='epoch_duplicate':h.loc[1,'epoch']=1
        if damage=='nan_loss':h.loc[1,'train_loss']=np.nan
        if damage=='nan_ap':h.loc[1,'val_ap']=np.inf
        write_csv(h,p)
        if damage!='history_hash':
            r=bt.read(p.with_name('completed_fold.json'));r['history_sha256']=bt.digest(p);write_json(p.with_name('completed_fold.json'),r)
    if damage=='missing_fold':(f/'fold2/completed_fold.json').unlink()
    if damage=='checkpoint_hash':
        r=bt.read(f/'checkpoint_metadata.json');r['checkpoints'][0]['sha256']='bad';write_json(f/'checkpoint_metadata.json',r)
    if damage=='wrong_code':
        r=bt.read(f/'metadata.json');r['code_hash']='other';write_json(f/'metadata.json',r)
    if damage=='wrong_summary':
        r=bt.read(f/'summary.json');r['score_mean']=.99;write_json(f/'summary.json',r)
    if damage=='different_config':c['classifier']['baseline']['lr_transformers']=.01
    before=bt.baseline_evidence_manifest(f)
    with pytest.raises((bt.Stop,ValueError,KeyError)):bt.inspect_baseline(c,spec,f,'test_code','test_commit',exp)
    assert bt.baseline_evidence_manifest(f)==before


def test_classification_curves_flag_late_loss_rise_and_do_not_invent_map(tmp_path):
    c,spec,f,exp=baseline_fixture(tmp_path)
    result=bt._classification_curve_evidence(f/'fold0/completed_history.csv',False,2)
    assert result['flags'][0]['kind']=='POSSIBLE_LATE_OVERFIT'
    assert result['metrics_at_same_best_epoch']['average_precision_positive']==.95
    assert result['metrics_at_same_best_epoch']['map50'] is None and result['train_validation_metric_gap'] is None


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
    monkeypatch.setenv('USERNAME','I21314')
    audit=dict(baseline=dict(mean=.9,sd=.05),fold_range=.1,curves=[],isolation=dict(status='NO_DETECTED_OVERLAP'),
        baseline_dir=str(f),baseline_evidence=bt.baseline_evidence_manifest(f),experiment_fingerprint=exp)
    bt.write_report(tmp_path,audit)
    bt.human_approve(q,q.get(job.job_id),tmp_path,audit,input_fn=lambda _:'APROBAR')
    assert q.get(job.job_id).status=='COMPLETE'
    claimed=q.claim_next('PC-02',kinds=['hpo'],slugs=['swin_tiny'])
    assert claimed.kind=='hpo' and claimed.depends_on==[job.job_id]
    assert bt.read(tmp_path/'aprobacion.json')['scope']=='Solo baseline Swin-Tiny -> HPO20x3'
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
    monkeypatch.setattr('utils.optuna_storage.budget_used',lambda *a:20)
    monkeypatch.setattr(bt,'verify_trial_evidence',lambda *a:[])
    monkeypatch.setattr(bt.importlib,'import_module',lambda *a:pytest.fail('No HPO launcher'))
    assert bt.run_hpo(tmp_path,tmp_path/'config',c,[],tmp_path,q,q.get(job.job_id),tmp_path)==0


@pytest.mark.parametrize('alive',[True,False])
def test_recover_job_requires_matching_dead_owner(tmp_path,monkeypatch,alive):
    c=config(tmp_path);q=StudyQueue.from_cfg(c);q.plan(c,[get_arch(bt.MODEL)]);base=next(j for j in q.jobs() if j.kind=='baseline');q.finish(base.job_id,'COMPLETE')
    job=q.claim_next('PC-02',kinds=['hpo'],slugs=[bt.MODEL])
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


def test_partial_checkpoint_resume_matches_uninterrupted_native_training(tmp_path,monkeypatch):
    import utils.classifier_train as ct
    from torch import nn
    c=config(tmp_path);c['project']['profile']='smoke';c['classifier'].update(batch_size=2,workers=0,pretrained=False,precision='fp32',image_size=4)
    class Dataset(torch.utils.data.Dataset):
        def __init__(self,frame,*args):self.frame=frame;self.y=frame.y.to_numpy()
        def __len__(self):return len(self.frame)
        def __getitem__(self,index):
            y=int(self.y[index]);return torch.full((3,4,4),float(y)+.1*index),torch.tensor(y),index
    def model(*a,**k):return nn.Sequential(nn.Flatten(),nn.Dropout(.1),nn.Linear(48,2)),dict(mean=[0,0,0],std=[1,1,1])
    monkeypatch.setattr(ct,'build_classifier',model);monkeypatch.setattr('utils.classifier_models.build_classifier',model)
    monkeypatch.setattr(ct,'classifier_stats',lambda *a:dict(params_m=.0001,gflops=.001));monkeypatch.setattr('utils.model_stats.classifier_stats',lambda *a:dict(params_m=.0001,gflops=.001))
    monkeypatch.setattr(ct,'CropDataset',Dataset);monkeypatch.setattr(ct,'build_transforms',lambda *a,**k:None);monkeypatch.setattr(ct,'assert_deterministic_transform',lambda *a:None)
    monkeypatch.setattr(ct,'resolve_device',lambda *a:'cpu')
    def loader(ds,*,batch_size,shuffle,seed,workers,drop_last=False):return torch.utils.data.DataLoader(ds,batch_size=batch_size,shuffle=shuffle,drop_last=drop_last,generator=torch.Generator().manual_seed(seed))
    monkeypatch.setattr(ct,'make_loader',loader)
    hp=dict(optimizer='AdamW',lr=.01,weight_decay=.0001,warmup_epochs=0,label_smoothing=0.)
    train=pd.DataFrame(dict(image_id=[f't{i}' for i in range(6)],fruit_id=list(range(6)),y=[0,1]*3));val=train.copy();val.image_id=['v'+str(i) for i in range(6)]
    spec=get_arch(bt.MODEL);options=dict(seed=42,max_epochs=4,patience=15,keep_state=True,log=lambda *a:None)
    full=ct.train_classifier(spec,hp,c,train,val,**options)
    partial=copy.deepcopy(c);cp=tmp_path/'partial.pt';partial['classifier'].update(resume_output=str(cp),_resume_context=dict(fold=0))
    def interrupt(step,row):
        if step==1:raise KeyboardInterrupt()
    with pytest.raises(KeyboardInterrupt):ct.train_classifier(spec,hp,partial,train,val,**options,epoch_callback=interrupt)
    partial['classifier']['resume_checkpoint']=str(cp)
    resumed=ct.train_classifier(spec,hp,partial,train,val,**options)
    assert full.state_sha256==resumed.state_sha256
    pd.testing.assert_frame_equal(full.history.drop(columns='epoch_seconds'),resumed.history.drop(columns='epoch_seconds'))
    saved=bt.load_saved_fold(cp,spec,hp,partial,train,val,options)
    finished=bt.finished_classifier(saved,spec,partial,dict(options,hp=hp))
    assert finished.state_sha256==full.state_sha256 and finished.score==full.score and finished.epochs_run==4


def test_only_native_07_is_executable_and_no_baseline_repeat_or_robustness():
    import ast
    tree=ast.parse(SCRIPT.read_text());calls=[]
    for n in ast.walk(tree):
        if isinstance(n,ast.Call) and isinstance(n.func,ast.Attribute) and n.func.attr=='import_module' and n.args and isinstance(n.args[0],ast.Constant):calls.append(n.args[0].value)
    assert calls==['07_run_hpo_study']
    text=SCRIPT.read_text();assert "'--trial-block-size','20'" in text and '--force' not in text
    assert "'baseline_reexecuted'" not in text or 'baseline_reexecuted=False' in text


def test_real_native_hpo_20_trials_three_folds_and_second_invocation_reuses(tmp_path,monkeypatch):
    """Real 07/Optuna/queue/fold cache, explicit tiny trainer fixture; no GPU claims."""
    import utils.hpo_runner as runner
    import utils.classifier_train as ct
    from utils.metrics import binary_classification_report
    from utils.training_common import state_dict_sha256
    native=importlib.import_module('07_run_hpo_study')
    c=config(tmp_path);q=StudyQueue.from_cfg(c);spec=get_arch(bt.MODEL);q.plan(c,[spec])
    baseline=next(j for j in q.jobs() if j.kind=='baseline');q.finish(baseline.job_id,'PENDING_REVIEW');q.approve(baseline.job_id,operator='TEST',evidence='fixture')
    job=next(j for j in q.jobs() if j.kind=='hpo')
    old=tmp_path/'r/swin_tiny/baseline/DO_NOT_CHANGE';old.parent.mkdir(parents=True);old.write_bytes(b'BASELINE')
    accesses=[];calls=[];launches=[]
    class Run:
        def __init__(self,path):
            self.cfg=c;self.run_dir=Path(path);self.run_dir.mkdir(parents=True,exist_ok=True)
            self.runs_base=cfg_path(c,'paths.runs_dir');self.pc_id='PC-02';self.logger=logging.getLogger('hpo_test');self.run_id='tiny-fixture';self.env={'git_commit':'fixture'};self.hashes={};self.entries=[]
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
    def fake_train(spec,hp,cfg,tr,va,**opts):
        calls.append((Path(cfg['classifier']['resume_output']).parent.name,dict(hp)))
        true=np.array([0,1]);prob=np.array([.1,.9]);rep=binary_classification_report(true,prob,threshold=.5)
        h=pd.DataFrame(dict(epoch=[1,2],train_loss=[1,.2],val_loss=[1,.3],val_ap=[.5,1.],val_precision=[.5,1.],val_recall=[1.,1.],val_f1=[.6,1.],val_roc_auc=[.5,1.]))
        for i,row in h.iterrows():
            if opts.get('epoch_callback'):opts['epoch_callback'](opts.get('step_offset',0)+int(row.epoch)-1,row)
        state={'weight':torch.ones((2,2))}
        return ct.ClassifierRun(score=1.,best_epoch=2,epochs_run=2,stopped_early=True,history=h,val_true=true,val_prob=prob,val_index=np.array([0,1]),metrics=rep,train_seconds=.01,vram_peak_mb=None,micro_batch=32,accum_steps=1,params_m=.1,gflops=.1,state_sha256=state_dict_sha256(state),state=state,norm=dict(mean=[0,0,0],std=[1,1,1]))
    records=pd.DataFrame(dict(image_id=['a','b','c','d','e','f'],fruit_id=[0]*6,y=[0,1]*3,outer_split=['train']*6,inner_fold=[0,0,1,1,2,2]))
    write_csv(records,cfg_path(c,'paths.classification_dir')/c['classification_dataset']['manifest_name'])
    monkeypatch.setattr(runner,'load_crop_manifest',lambda cfg:records)
    monkeypatch.setattr(runner,'load_fold_frames',lambda m,cfg,fold:(m[m.inner_fold!=fold],m[m.inner_fold==fold]))
    monkeypatch.setattr(runner,'train_classifier',fake_train)
    monkeypatch.setattr(native,'preflight_training',lambda *a:SimpleNamespace(dataset_hash=None,split_hash=None))
    monkeypatch.setattr(native,'GpuJobLock',lambda *a,**k:contextlib.nullcontext())
    monkeypatch.setattr(native,'nested_run',lambda ctx,spec,name,**kw:Run(cfg_path(c,'paths.runs_dir')/spec.slug/name))
    monkeypatch.setattr(native,'save_training_figures',lambda *a,**k:None)
    def cli(script,main,**kwargs):
        launches.append(kwargs['argv'])
        claimed=q.claim_next('PC-02',kinds=['hpo'],slugs=[bt.MODEL]);assert claimed
        ctx=Run(tmp_path/'r/swin_tiny/hpo/study')
        args=SimpleNamespace(architecture=bt.MODEL,n_trials=None,trial_block_size=20,allow_budget_override=False)
        main(ctx,args)
        q.computed(job.job_id,published=True)
        from orchestration.worker import top_config_ids
        q.expand_after_hpo(c,spec,top_config_ids(c,spec.slug))
        return 0
    monkeypatch.setattr(native,'run_cli',cli)
    assert bt.run_hpo(tmp_path,tmp_path/'config',c,[],tmp_path,q,q.get(job.job_id),tmp_path/'audit')==0
    assert len(calls)==60 and len(launches)==1
    assert old.read_bytes()==b'BASELINE' and q.get(baseline.job_id).status=='COMPLETE'
    assert q.get(job.job_id).status=='PENDING_REVIEW'
    assert len([j for j in q.jobs() if j.kind in ('repeat','aggregate','outer_train','outer_eval','thresholds')])==13
    assert all(j.status=='PENDING' for j in q.jobs() if j.kind not in ('baseline','hpo'))
    assert accesses and all(a[0]=='train' for a in accesses)
    assert bt.run_hpo(tmp_path,tmp_path/'config',c,[],tmp_path,q,q.get(job.job_id),tmp_path/'audit')==0
    assert len(calls)==60 and len(launches)==1
