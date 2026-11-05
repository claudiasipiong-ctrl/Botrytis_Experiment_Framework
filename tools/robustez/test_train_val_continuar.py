"""CPU-only orchestration tests; native queue and explicit fake neural execution."""
import argparse
import contextlib
import importlib
import importlib.util
import json
import os
from pathlib import Path
from types import SimpleNamespace
import sys

import pytest

ROOT=Path(__file__).parent
s=importlib.util.spec_from_file_location('train_val_driver',ROOT/'botrytis_train_val_continuar.py')
v=importlib.util.module_from_spec(s);s.loader.exec_module(v);bt=v.bt
from utils.config_loader import load_config,cfg_path
from utils.architectures import get_arch
from orchestration.study_queue import StudyQueue
from utils.experiments import fingerprint
from utils.hpo_results import best_config_path,stage_json_path


def config(tmp_path):
    c=load_config()
    for key,value in dict(runs_dir='r',artifacts_dir='a',queue_dir='q',optuna_dir='o',results_dir='e',best_configs_dir='best').items():
        c['paths'][key]=str(tmp_path/value)
    c['compute']['pc_id']='PC-02'
    c['experiment'].update(round_id='fixture',experiment_id='botrytis_v4',architecture='yolo26l',registry_root=str(tmp_path/'registry'))
    return c


def planned(tmp_path):
    c=config(tmp_path);sp=get_arch('yolo26l');q=StudyQueue.from_cfg(c);q.plan(c,[sp])
    for j in q.jobs():q.finish(j.job_id,'COMPLETE')
    q.expand_after_hpo(c,sp,['cfg_a','cfg_b','cfg_c'])
    for j in q.jobs():
        if j.kind=='repeat':q.finish(j.job_id,'COMPLETE')
        if j.kind=='aggregate':q.finish(j.job_id,'PENDING_REVIEW')
    return c,sp,q


def paused(tmp_path):
    c,sp,q=planned(tmp_path)
    old=[bt.pure_job(j) for j in q.jobs() if j.kind in ('outer_train','outer_eval','thresholds')]
    p=tmp_path/'pause.json';marker='POSPUESTO_POR_PRIORIDAD '+str(p)
    bt.atomic_json(p,dict(database=str(q.path),architecture=sp.slug,marker=marker,postponed=old))
    for j in old:q.finish(j['job_id'],'BLOCKED',error=marker)
    return c,sp,q,p


def item(tmp_path):
    c,sp,q=planned(tmp_path);best=dict(config_id='cfg_a',hp={'lr0':.001},final_epochs=46,score_mean=.89,score_sd=.002)
    p=best_config_path(c,sp);bt.atomic_json(p,best)
    evidence=tmp_path/'evidence';evidence.mkdir();bt.atomic_json(evidence/'revision.json',dict(fixture=True))
    record=dict(model=sp.slug,selected=best,fingerprint=fingerprint(c),best_config_path=str(p),best_config_sha256=bt.digest(p),
                baseline={'mean':.88},hpo_best=.90,curve_warnings=[],isolation={'status':'NOT_PROVEN_BIOLOGICALLY_INDEPENDENT'})
    return dict(model=sp.slug,cfg=c,spec=sp,queue=q,jobs=[bt.pure_job(j) for j in q.jobs()],record=record,evidence=evidence,
                ledger=None,pause_paths=[],state=tmp_path/'state',project=tmp_path,config=tmp_path/'config.yaml',sets=[],cands=[])


def test_restore_only_09_10_exact_priority_pause_and_idempotence(tmp_path):
    c,sp,q,p=paused(tmp_path)
    assert len(v.restore_outer_pauses(q.path,sp.slug,[p]))==2
    assert all(j.status=='BLOCKED' for j in q.jobs() if j.kind in ('outer_train','outer_eval','thresholds'))
    assert len(v.restore_outer_pauses(q.path,sp.slug,[p],apply=True))==2
    assert all(j.status=='PENDING' for j in q.jobs() if j.kind in ('outer_train','outer_eval'))
    assert next(j for j in q.jobs() if j.kind=='thresholds').status=='BLOCKED'
    assert next(j for j in q.jobs() if j.kind=='aggregate').status=='PENDING_REVIEW'
    assert v.restore_outer_pauses(q.path,sp.slug,[p],apply=True)==[]


def test_foreign_pause_and_unrelated_block_not_removed(tmp_path):
    c,sp,q,p=paused(tmp_path);job=next(j for j in q.jobs() if j.kind=='outer_train')
    q.finish(job.job_id,'BLOCKED',error='REAL_METHODOLOGY_BLOCK')
    v.restore_outer_pauses(q.path,sp.slug,[p],apply=True)
    assert q.get(job.job_id).error=='REAL_METHODOLOGY_BLOCK'
    assert v.restore_outer_pauses(q.path,'swin_tiny',[p],apply=True)==[]


def test_changed_pause_identity_blocks(tmp_path):
    c,sp,q,p=paused(tmp_path)
    with q._conn() as con:con.execute("UPDATE jobs SET attempt=1 WHERE kind='outer_train'")
    with pytest.raises(bt.Stop,match='Identidad'):v.restore_outer_pauses(q.path,sp.slug,[p],apply=True)
    assert next(j for j in q.jobs() if j.kind=='outer_train').status=='BLOCKED'


def test_real_manual_approval_and_external_predeclaration_reused(tmp_path,monkeypatch):
    x=item(tmp_path);q=x['queue'];out=tmp_path/'approval';out.mkdir()
    import utils.experiments as exp
    monkeypatch.setattr(exp,'verify_round',lambda *a,**k:None)
    monkeypatch.setenv('USERNAME','I21314');monkeypatch.setattr(sys.stdin,'isatty',lambda:True)
    prompts=[];monkeypatch.setattr('builtins.input',lambda prompt:prompts.append(prompt) or 'APROBAR')
    v.authorize_ready([x],out)
    agg=next(j for j in q.jobs() if j.kind=='aggregate');assert agg.status=='COMPLETE'
    row=v.ledger_entry(x['cfg']);assert row['fingerprint']==fingerprint(x['cfg']) and row['uses']=={}
    proof=bt.read(out/'aprobacion.json');assert proof['human_visual_inspection_claimed'] is False
    x['ledger']=row;x['jobs']=[bt.pure_job(j) for j in q.jobs()]
    v.authorize_ready([x],out)
    assert len(prompts)==1
    with q._conn() as con:assert con.execute('SELECT count(*) FROM reviews').fetchone()[0]==1


@pytest.mark.parametrize('answer',['NO','',None])
def test_declined_or_noninteractive_never_approves(tmp_path,monkeypatch,answer):
    x=item(tmp_path);out=tmp_path/'approval';out.mkdir()
    monkeypatch.setattr(sys.stdin,'isatty',lambda:answer is not None)
    monkeypatch.setattr('builtins.input',lambda prompt:answer)
    with pytest.raises(bt.Stop):v.authorize_ready([x],out)
    assert next(j for j in x['queue'].jobs() if j.kind=='aggregate').status=='PENDING_REVIEW'
    assert v.ledger_entry(x['cfg']) is None


def test_candidate_changed_during_prompt_cannot_be_approved(tmp_path,monkeypatch):
    x=item(tmp_path);out=tmp_path/'approval';out.mkdir();monkeypatch.setattr(sys.stdin,'isatty',lambda:True)
    def reply(prompt):Path(x['record']['best_config_path']).write_text('changed');return 'APROBAR'
    monkeypatch.setattr('builtins.input',reply)
    with pytest.raises(bt.Stop,match='cambio'):v.authorize_ready([x],out)
    assert next(j for j in x['queue'].jobs() if j.kind=='aggregate').status=='PENDING_REVIEW'


def test_external_outputs_without_predeclared_use_not_retroactively_accepted(tmp_path,monkeypatch):
    c,sp,q=planned(tmp_path);p=stage_json_path(c,'outer_validation',sp);bt.atomic_json(p,{'fixture':True})
    with pytest.raises(bt.Stop,match='retroactivamente'):v.verify_external_state(c,sp,list(map(bt.pure_job,q.jobs())))


@pytest.mark.parametrize('state,output,jobstatus,expected',[('COMPLETE',True,'COMPLETE','verified'),('COMPLETE',False,'COMPLETE','verified'),
    ('STARTED',True,'FAILED','verified'),('STARTED',False,'FAILED','recoverable'),('STARTED',False,'PENDING','blocked')])
def test_external_consumption_is_checked_before_training(tmp_path,monkeypatch,state,output,jobstatus,expected):
    c,sp,q=planned(tmp_path)
    monkeypatch.setattr(v,'ledger_entry',lambda c:{'uses':{'outer_validation':{'state':state,'checkpoint_sha256':'sha'}}})
    monkeypatch.setattr(bt,'validate_final',lambda *a:{'sha256':'sha'})
    calls=[];monkeypatch.setattr(bt,'validate_external',lambda *a:calls.append(a) or {'valid':True})
    if output:bt.atomic_json(stage_json_path(c,'outer_validation',sp),{'fixture':True})
    job=next(j for j in q.jobs() if j.kind=='outer_eval');q.finish(job.job_id,jobstatus)
    if expected=='blocked':
        with pytest.raises(bt.Stop,match='uso VAL20|Uso VAL20'):v.verify_external_state(c,sp,list(map(bt.pure_job,q.jobs())))
    else:
        v.verify_external_state(c,sp,list(map(bt.pure_job,q.jobs())))
        assert bool(calls)==(expected=='verified')


def test_native_queue_sequence_09_then_10_and_idempotent_completion(tmp_path,monkeypatch):
    x=item(tmp_path);q=x['queue'];agg=next(j for j in q.jobs() if j.kind=='aggregate');q.approve(agg.job_id,operator='TEST',evidence='explicit fixture')
    calls=[]
    monkeypatch.setattr(bt,'python_processes',lambda:[])
    monkeypatch.setattr(bt,'ensure_partial_resume',lambda *a:None)
    monkeypatch.setattr(bt,'recovery_or_reuse',lambda *a:False)
    def launch(project,cfg,spec,config,sets,kind,state,evidence,target_job=None):
        calls.append(kind);q.finish(target_job,'COMPLETE')
    monkeypatch.setattr(bt,'launch_visible',launch)
    train={'checkpoint_sha256':'sha','map50_95':.95}
    final=dict(sha256='sha',train_evaluation=train)
    monkeypatch.setattr(bt,'technical_job_result',lambda *a:final)
    monkeypatch.setattr(bt,'validate_external',lambda *a:dict(path='val.json',metrics={'map50_95':.89}))
    monkeypatch.setattr(v,'ledger_entry',lambda c:{'uses':{'outer_validation':{'state':'COMPLETE','checkpoint_sha256':'sha'}}})
    result=v.complete_model(x)
    assert calls==['outer_train','outer_eval'] and result['gap_train_validation']==pytest.approx(.06)
    assert result['test_opened'] is False and result['thresholds_launched'] is False
    v.complete_model(x);assert calls==['outer_train','outer_eval']
    assert next(j for j in q.jobs() if j.kind=='thresholds').status=='PENDING'


def test_review_dependency_prevents_any_final_training(tmp_path,monkeypatch):
    x=item(tmp_path);calls=[];monkeypatch.setattr(bt,'launch_visible',lambda *a,**k:calls.append(a))
    with pytest.raises(v.NotReady,match='Dependencias'):v.complete_model(x)
    assert calls==[]


def test_child_dispatch_whitelist_and_executor_integrity(tmp_path,monkeypatch):
    calls=[];monkeypatch.setattr(bt,'child_entry',lambda *a:calls.append(a) or 0)
    base=[str(v.SELF),'--executor-sha',v.SHA,'--_stage','worker','--project',str(tmp_path),'--state-dir',str(tmp_path)]
    monkeypatch.setattr(sys,'argv',base+['--','--pc-id','PC-02','--architectures','yolo26l','--kinds','outer_train'])
    assert v.main()==0 and calls[-1][0]=='worker'
    for tail in (['--pc-id','PC-03','--architectures','rtdetr_r50','--kinds','outer_train'],
                 ['--pc-id','PC-02','--architectures','yolo26l','--kinds','hpo']):
        monkeypatch.setattr(sys,'argv',base+['--',*tail])
        with pytest.raises(bt.Stop,match='fuera de alcance'):v.main()
    assert len(calls)==1


def test_classifier_outer_resume_uses_verified_rng_hook(tmp_path,monkeypatch):
    calls=[];module=SimpleNamespace(train_classifier=lambda *a,**k:calls.append('train') or 'resumed')
    monkeypatch.setattr(v,'_install_outer',lambda *a:None)
    @contextlib.contextmanager
    def hook(event):calls.append('install');yield;calls.append('restore')
    monkeypatch.setattr(bt,'classifier_rng_restore',hook)
    v.install_outer(module,{},SimpleNamespace(is_detector=False),tmp_path,SimpleNamespace(event=lambda *a,**k:None))
    assert module.train_classifier()=='resumed' and calls==['install','train','restore']


def test_incomplete_yolov8n_report_does_not_prevent_other_ready_model(tmp_path,monkeypatch):
    class LocalPath(type(tmp_path)):
        @classmethod
        def home(cls):return cls(tmp_path)
    project=tmp_path/'project';py=project/'.venv/Scripts/python.exe';py.parent.mkdir(parents=True);py.symlink_to(sys.executable)
    import utils.config_loader as cl
    import orchestration.machine as machine
    monkeypatch.setattr(cl,'load_config',lambda p:{'compute':{'pc_id':'PC-01'}})
    monkeypatch.setattr(machine,'read_binding',lambda c:{'pc_id':'PC-01','machine_id':'i21315'})
    monkeypatch.setattr(v,'Path',LocalPath);monkeypatch.setattr(v,'os',SimpleNamespace(name='nt',environ={'USERNAME':'i21315'},chdir=lambda p:None))
    monkeypatch.setattr(bt,'find_project',lambda p:project);monkeypatch.setattr(bt,'read',lambda p:{})
    monkeypatch.setattr(v.r,'controller_lock',lambda *a:contextlib.nullcontext());monkeypatch.setattr(v.r,'wait_idle',lambda *a:contextlib.nullcontext())
    calls=[];reports=[]
    def check(project,pc,model,index,out):
        calls.append(model)
        if model=='yolov8n':raise v.NotReady('repeat cfg_x s73 RUNNING')
        return dict(model=model,record={'model':model})
    monkeypatch.setattr(v,'verify_model',check);monkeypatch.setattr(v,'authorize_ready',lambda items,out:calls.append('APROBAR'))
    monkeypatch.setattr(v,'complete_model',lambda x:calls.append('09->10 '+x['model']) or {'model':x['model']})
    monkeypatch.setattr(v,'write_report',lambda *a:reports.append(a))
    monkeypatch.setattr(sys,'argv',[str(v.SELF),'--expected-pc','PC-01'])
    assert v.main()==2 and calls==['yolov8n','yolov8l','APROBAR','09->10 yolov8l']
    assert len(reports[0][-1])==1 and reports[0][-1][0]['model']=='yolov8n'


def test_actual_native_09_then_10_with_fake_neural_execution(tmp_path,monkeypatch):
    """Execute native main functions, fake only inference/training/data/GPU guards."""
    import pandas as pd
    from utils.logging_utils import ACTIVE_CONFIG
    import utils.experiments as exp
    native09=importlib.import_module('09_train_outer_candidate');native10=importlib.import_module('10_evaluate_outer_validation')
    c=config(tmp_path);sp=get_arch('yolo26l')
    best=dict(config_id='cfg_test',hp={'optimizer':'AdamW','lr0':.001},final_epochs=46,score_mean=.89,score_sd=.002)
    from utils.semantics import semantic_fingerprint
    best['semantic_fingerprint']=semantic_fingerprint(c)
    best['experiment_fingerprint']=fingerprint(c)
    bt.atomic_json(best_config_path(c,sp),best)
    data=SimpleNamespace(dataset_hash='data',split_hash='splits');calls=[];access=[]
    class Run:
        def __init__(self):self.cfg=c;self.pc_id='PC-02';self.run_dir=tmp_path/'run';self.run_dir.mkdir(exist_ok=True);self.hashes={'code_hash':'code'};self.env={};self.logger=SimpleNamespace(info=lambda *a,**k:None)
        def set_seed(self,*a):pass
        def add_metadata(self,**kw):pass
        def note_split_access(self,*a):access.append(a)
        def set_checkpoint(self,*a,**k):pass
        def log_metrics(self,*a):pass
        def register_artifact(self,*a):pass
    for module in (native09,native10):
        monkeypatch.setattr(module,'guard_not_frozen',lambda *a:None)
        monkeypatch.setattr(module,'GpuJobLock',lambda *a,**k:contextlib.nullcontext())
    monkeypatch.setattr(native09,'preflight_training',lambda *a:data)
    monkeypatch.setattr(native09,'record_training_source',lambda *a:None)
    monkeypatch.setattr(native09,'write_outer_yaml',lambda *a,**k:tmp_path/'train.yaml')
    monkeypatch.setattr(native09,'save_training_figures',lambda *a,**k:None)
    weights=tmp_path/'last.pt';weights.write_bytes(b'EXPLICIT_FAKE_MODEL')
    def train(*a,**k):
        assert k['fixed_epochs'] is True and k['epochs']==46
        calls.append('train')
        return SimpleNamespace(weights_path=str(weights),vram_peak_mb=None,train_seconds=0,params_m=1,gflops=1,
            pretrained_sha256='pretrained',history=pd.DataFrame({'epoch':[1,46],'train_loss':[1,.1]}),micro_batch=16,extra={})
    monkeypatch.setattr(native09,'train_detector',train)
    def evaluate(ctx,spec,data,best,ckpt,split='validation',**kwargs):
        calls.append('evaluate_'+split)
        assert ckpt.read_bytes()==b'EXPLICIT_FAKE_MODEL'
        return {'checkpoint_sha256':bt.digest(ckpt),'map50_95':.95 if split=='train' else .89,'evaluated_on':split}
    monkeypatch.setattr(native10,'evaluate_detector',evaluate)
    monkeypatch.setattr(native10,'prepare_for_inference',lambda *a:data)
    monkeypatch.setattr(exp,'use_external',lambda cfg,stage,cp:calls.append('consume_'+stage))
    import utils.split_summary as summary
    monkeypatch.setattr(summary,'export_split_summary',lambda *a:None)
    args=argparse.Namespace(architecture='yolo26l',force=False,no_gradcam=False)
    ctx=Run();token=ACTIVE_CONFIG.set(c)
    try:
        native09.main(ctx,args)
        native09.main(ctx,args)  # Valid final.pt: never train it twice.
        native10.main(ctx,args)
    finally:ACTIVE_CONFIG.reset(token)
    assert calls==['train','evaluate_train','consume_outer_validation','evaluate_validation']
    assert ('train','outer_train') in access and ('validation','outer_validation') in access
    assert not any(a[0]=='test' for a in access)
    assert (ctx.run_dir/'train_evaluation.json').is_file()
    assert bt.read(stage_json_path(c,'outer_validation',sp))['map50_95']==.89


@pytest.mark.parametrize('pending',[True,False])
def test_verify_model_checks_nine_queue_jobs_before_issuing_verified(tmp_path,monkeypatch,pending):
    c,sp,q=planned(tmp_path);p=tmp_path/'config.yaml';p.write_text('fixture')
    import utils.config_loader as cl
    import utils.experiments as exp
    import utils.manual_gates as gates
    import utils.optuna_storage as storage
    import utils.hpo_results as hpo
    from orchestration.pc_registry import PCRegistry
    monkeypatch.setattr(cl,'load_config',lambda *a:c)
    monkeypatch.setattr(exp,'verify_round',lambda *a,**k:None)
    monkeypatch.setattr(gates,'require_gate',lambda *a:None)
    monkeypatch.setattr(PCRegistry,'owner',lambda *a:'PC-02')
    monkeypatch.setattr(bt,'make_short_paths',lambda *a:(tmp_path,tmp_path/'r',tmp_path/'e',tmp_path/'s'))
    monkeypatch.setattr(v.r.audit,'readonly_code',lambda p:dict(code_hash=v.r.CODE,git_commit=v.r.COMMIT))
    monkeypatch.setattr(v.r.audit,'readonly_prepared',lambda *a:(c,SimpleNamespace(dataset_hash=v.r.DATA,split_hash=v.r.SPLIT)))
    db=tmp_path/'study.sqlite';db.touch();monkeypatch.setattr(storage,'study_db_path',lambda *a:db)
    monkeypatch.setattr(storage,'load_study',lambda *a:SimpleNamespace(best_value=.9))
    cands=[dict(config_id='cfg_'+k,hp={'fixture':k},trial_number=i) for i,k in enumerate('abc')]
    monkeypatch.setattr(hpo,'top_configs',lambda *a:cands)
    best=dict(config_id='cfg_a',hp={'fixture':'a'},final_epochs=46,score_mean=.89,score_sd=.002)
    bp=best_config_path(c,sp);bt.atomic_json(bp,best)
    checked=[];emitted=[]
    monkeypatch.setattr(v.r,'verify_closed_hpo',lambda *a:checked.append('hpo') or {'trials':[]})
    monkeypatch.setattr(bt,'validate_aggregate',lambda *a:checked.append('aggregate') or {'selected':best,'repeat_evidence':[]})
    monkeypatch.setattr(v.r,'baseline_evidence',lambda *a:{'mean':.88})
    monkeypatch.setattr(v.r.audit,'partition_evidence',lambda *a:{'confirmed_relevant_conflicts':[],'status':'UNKNOWN_BIOLOGICAL_GROUPS'})
    monkeypatch.setattr(bt,'protect_existing_final',lambda *a:None)
    monkeypatch.setattr(bt,'emit',lambda name,**kwargs:emitted.append((name,kwargs)))
    if pending:
        repeat=next(j for j in q.jobs() if j.kind=='repeat')
        with q._conn() as con:con.execute("UPDATE jobs SET status='RUNNING' WHERE job_id=?",(repeat.job_id,))
        with pytest.raises(v.NotReady,match='Robustez incompleta'):v.verify_model(tmp_path,'PC-02',sp.slug,{sp.slug:str(p)},tmp_path)
        assert checked==[] and emitted[-1][0]=='ROBUSTEZ_PENDIENTE'
        assert emitted[-1][1]['jobs'][0]['job']==repeat.job_id
    else:
        x=v.verify_model(tmp_path,'PC-02',sp.slug,{sp.slug:str(p)},tmp_path)
        assert checked==['hpo','aggregate'] and x['record']['robustness_folds']==27
        assert emitted[-1][0]=='ROBUSTEZ_VERIFICADA'
        assert next(j for j in q.jobs() if j.kind=='aggregate').status=='PENDING_REVIEW'
