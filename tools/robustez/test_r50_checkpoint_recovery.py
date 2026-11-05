"""Recovery from an interrupted publication; all model states here are explicit CPU fixtures."""
import importlib.util
from pathlib import Path
from types import SimpleNamespace
import numpy as np
import optuna
import pandas as pd
import pytest
import torch

ROOT=Path(__file__).parent
spec_=importlib.util.spec_from_file_location('r50_checkpoint_fix',ROOT/'botrytis_r50_hpo_pc03.py')
bt=importlib.util.module_from_spec(spec_);spec_.loader.exec_module(bt)
from utils.config_loader import load_config,cfg_path
from utils.architectures import get_arch
from utils.experiments import fingerprint
from utils.resume import receipt
from utils.search_spaces import config_id


def config(tmp):
    cfg=load_config()
    for key,value in dict(runs_dir='r',artifacts_dir='a',queue_dir='q',optuna_dir='o',results_dir='e').items():cfg['paths'][key]=str(tmp/value)
    cfg['compute']['pc_id']='PC-03'
    cfg['experiment'].update(round_id='r50vd_01',experiment_id='botrytis_v4',architecture='rtdetr_r50')
    return cfg


def history(epoch):
    values=[.1,.6]+[.59]*max(0,epoch-2)
    values=values[:epoch]
    h=pd.DataFrame(dict(epoch=range(1,epoch+1),optimization_loss_total=[3.]*epoch,
        train_loss=[2.]*epoch,val_loss=[2.1]*epoch,map50_95=values,map50=[v+.1 for v in values],
        precision=[.7]*epoch,recall=[.7]*epoch))
    for role,total in [('train',2.),('val',2.1)]:
        for key,value in [('vfl',.5),('bbox',.5),('giou',total-1)]:h[role+'/'+key+'_loss']=value
    return h


def checkpoint(out,cfg,hp,epoch,*,fold=0):
    adapter=bt.audit.load_adapter();spec=get_arch('rtdetr_r50');path=out/'cp'/f'e{epoch:04d}.pt';path.parent.mkdir(parents=True,exist_ok=True)
    signature=adapter.canonical(dict(experiment=fingerprint(cfg),hp=hp,seed=42,fold=fold,epochs=60,patience=20,fixed=False))
    h=history(epoch)
    state=dict(format=adapter.FORMAT,signature=signature,seed=42,hp=hp,fold=fold,epoch=epoch,
        model={'fixture':torch.zeros(1)},ema={},optimizer={'fixture':True},scheduler={},scaler={},gradients={},rng={},
        history=h.to_dict('records'),best_score=float(h.map50_95.max()),best_epoch=min(2,epoch),best_state={},best_metrics={},
        ema_updates=epoch,last_opt_step=0,elapsed=0,stopper_epoch=min(2,epoch),stopper_fitness=float(h.map50_95.max()),stopped=False)
    torch.save(state,path)
    receipt(path,spec,cfg,dict(seed=42,hp=hp,context=dict(fold=fold),epoch=epoch,adapter_sha256=bt.audit.ADAPTER_SHA))
    return path


def trial(study,cfg,hp,*,parent=None,final='FAIL'):
    t=study.ask();t.suggest_float('lr0',1e-5,1e-5)
    folder=cfg_path(cfg,'paths.runs_dir')/'rtdetr_r50/hpo'/f'trial_{t.number:03d}';folder.mkdir(parents=True)
    for key,value in dict(hp=hp,run_dir=str(folder.resolve()),config_id=config_id(hp),pc_id='PC-03').items():t.set_user_attr(key,value)
    if parent is not None:
        t.set_user_attr('bt_retry_of',parent.number);t.set_user_attr('bt_recovery_source',parent.user_attrs['run_dir'])
    if final=='COMPLETE':study.tell(t,.6)
    elif final=='PRUNED':study.tell(t,state=optuna.trial.TrialState.PRUNED)
    else:study.tell(t,state=optuna.trial.TrialState.FAIL)
    return study.trials[t.number],folder


def test_unsigned_newest_is_preserved_and_last_signed_state_selected(tmp_path):
    cfg=config(tmp_path);hp={'lr0':1e-5};out=tmp_path/'fold0'
    saved=checkpoint(out,cfg,hp,3)
    bt.atomic_json(out/'resume.json',dict(path=str(saved),sha256=bt.digest(saved),epoch=3))
    orphan=out/'cp/e0004.pt';orphan.write_bytes(b'POWER_LOSS_NO_RECEIPT_DO_NOT_LOAD')
    original=bt.audit.frozen_inventory(out)
    selected=bt.select_partial_checkpoint(out,cfg,get_arch('rtdetr_r50'),hp,42,0,60,20)
    assert selected['path']==str(saved) and selected['epoch']==3
    assert selected['unpublished'][0]['path']==str(orphan)
    assert bt.audit.frozen_inventory(out)==original


@pytest.mark.parametrize('damage',['committed_missing_receipt','corrupt_signed','wrong_hp','missing_optimizer','bad_history','wrong_adapter'])
def test_real_integrity_failure_never_falls_back(tmp_path,damage):
    cfg=config(tmp_path);hp={'lr0':1e-5};out=tmp_path/'fold0'
    checkpoint(out,cfg,hp,2);latest=checkpoint(out,cfg,hp,3)
    bt.atomic_json(out/'resume.json',dict(path=str(latest),sha256=bt.digest(latest),epoch=3))
    side=latest.with_suffix('.pt.resume.json')
    if damage=='committed_missing_receipt':side.unlink()
    if damage=='corrupt_signed':latest.write_bytes(b'CORRUPT')
    if damage=='wrong_hp':
        row=bt.read(side);row['extra']['hp']={'lr0':.1};bt.atomic_json(side,row)
    if damage=='wrong_adapter':
        row=bt.read(side);row['extra']['adapter_sha256']='other';bt.atomic_json(side,row)
    if damage in ('missing_optimizer','bad_history'):
        state=torch.load(latest,weights_only=False)
        if damage=='missing_optimizer':state.pop('optimizer')
        else:state['history'].pop()
        torch.save(state,latest)
        row=bt.read(side);row['sha256']=bt.digest(latest);bt.atomic_json(side,row)
        bt.atomic_json(out/'resume.json',dict(path=str(latest),sha256=bt.digest(latest),epoch=3))
    with pytest.raises((RuntimeError,bt.Stop)):
        bt.select_partial_checkpoint(out,cfg,get_arch('rtdetr_r50'),hp,42,0,60,20)


def test_failed_empty_retry_recovers_ancestor_without_copying_orphan(tmp_path):
    cfg=config(tmp_path);hp={'lr0':1e-5};sp=get_arch('rtdetr_r50');study=optuna.create_study()
    root,source=trial(study,cfg,hp);saved=checkpoint(source/'fold0',cfg,hp,3)
    bt.atomic_json(source/'fold0/resume.json',dict(path=str(saved),sha256=bt.digest(saved),epoch=3))
    orphan=source/'fold0/cp/e0004.pt';orphan.write_bytes(b'UNPUBLISHED')
    # Exact situation of the failed retry: its own folder contains no new work.
    failed,dest=trial(study,cfg,hp,parent=root)
    original=bt.audit.frozen_inventory(source)
    plan=bt.recovery_plan(study,failed,cfg,sp)
    assert plan['family']==[1,0] and plan['folds'][0]['epoch']==3
    assert plan['folds'][1]['kind']=='NEW'
    actions=bt.schedule_retries(study,cfg,sp,tmp_path/'audit')
    assert [a['trial'] for a in actions]==[1] and len(study.trials)==3
    bt.schedule_retries(study,cfg,sp,tmp_path/'audit');assert len(study.trials)==3
    preview=bt.preview_recovery(study,cfg,sp,tmp_path/'audit')
    assert preview[0]['waiting_trial']==2 and preview[0]['folds'][0]['epoch']==3
    retry=study.ask();output=cfg_path(cfg,'paths.runs_dir')/'rtdetr_r50/hpo/trial_002';output.mkdir()
    controller=bt.R50Recovery(cfg,sp,tmp_path/'controller');controller.study=study
    def cached_and_resume(spec,hp,cfg,ctx,folder,**kw):
        cp=folder/'fold0/cp/e0003.pt'
        assert cp.read_bytes()==saved.read_bytes()
        assert cp.with_suffix('.pt.resume.json').read_bytes()==saved.with_suffix('.pt.resume.json').read_bytes()
        assert not (folder/'fold0/cp/e0004.pt').exists()
        adapter=bt.audit.load_adapter()
        sig=adapter.canonical(dict(experiment=fingerprint(cfg),hp=hp,seed=42,fold=0,epochs=60,patience=20,fixed=False))
        assert adapter.load_resume(folder/'fold0',sig,sp,cfg)['epoch']==3
        return 'RESUME_FROM_3'
    assert controller.evaluate(cached_and_resume,sp,hp,cfg,object(),output,trial=retry,folds=[0,1,2],seed=42,max_epochs=60,patience=20)=='RESUME_FROM_3'
    assert original==bt.audit.frozen_inventory(source)
    assert study.trials[0].state.name==study.trials[1].state.name=='FAIL'


def test_no_verified_state_stops_before_enqueuing_or_claiming(tmp_path):
    cfg=config(tmp_path);hp={'lr0':1e-5};sp=get_arch('rtdetr_r50');study=optuna.create_study()
    root,source=trial(study,cfg,hp)
    orphan=source/'fold0/cp/e0001.pt';orphan.parent.mkdir(parents=True);orphan.write_bytes(b'UNSIGNED')
    with pytest.raises(bt.Stop,match='No hay estado'):
        bt.preview_recovery(study,cfg,sp,tmp_path/'audit')
    assert len(study.trials)==1
    with pytest.raises(bt.Stop,match='No hay estado'):
        bt.schedule_retries(study,cfg,sp,tmp_path/'audit')
    assert len(study.trials)==1 and orphan.read_bytes()==b'UNSIGNED'


def test_interrupted_copy_can_use_genuine_receipt_in_its_ancestor(tmp_path):
    cfg=config(tmp_path);hp={'lr0':1e-5};sp=get_arch('rtdetr_r50');study=optuna.create_study()
    root,source=trial(study,cfg,hp);saved=checkpoint(source/'fold0',cfg,hp,3)
    failed,copy=trial(study,cfg,hp,parent=root)
    unsigned=copy/'fold0/cp/e0003.pt';unsigned.parent.mkdir(parents=True);unsigned.write_bytes(saved.read_bytes())
    before=bt.audit.frozen_inventory(copy)
    plan=bt.recovery_plan(study,failed,cfg,sp)
    assert plan['folds'][0]['source_trial']==root.number
    assert plan['folds'][0]['path']==str(saved)
    assert plan['folds'][0]['unrecoverable_copies'][0]['folder']==str(copy/'fold0')
    assert before==bt.audit.frozen_inventory(copy)


def test_callback_failure_trial_mirror_is_reconciled_without_false_completion(tmp_path):
    from orchestration.study_queue import StudyQueue
    cfg=config(tmp_path);hp={'lr0':1e-5};sp=get_arch('rtdetr_r50');study=optuna.create_study()
    root,source=trial(study,cfg,hp);checkpoint(source/'fold0',cfg,hp,3)
    q=StudyQueue.from_cfg(cfg)
    q.record_trial(study.study_name,root.number,'RUNNING',pc_id='PC-03',run_dir=str(source))
    before=q.trials_table().iloc[0].to_dict()
    bt.schedule_retries(study,cfg,sp,tmp_path/'audit')
    after=q.trials_table().iloc[0].to_dict()
    assert after['status']=='FAILED' and study.trials[0].state.name=='FAIL'
    for key in ('value','started_at','finished_at','run_dir','job_id'):assert after[key]==before[key]


def test_recovered_failures_do_not_become_complete_and_unlinked_fail_blocks(tmp_path):
    cfg=config(tmp_path);hp={'lr0':1e-5};sp=get_arch('rtdetr_r50');study=optuna.create_study()
    root,_=trial(study,cfg,hp);failed,_=trial(study,cfg,hp,parent=root)
    closed,_=trial(study,cfg,hp,parent=failed,final='COMPLETE')
    checks=bt.recovered_failures(study,cfg,sp)
    assert checks==[dict(failed_trial=0,recovered_by=2,original_state_preserved='FAIL'),dict(failed_trial=1,recovered_by=2,original_state_preserved='FAIL')]
    assert study.trials[0].state.name=='FAIL'
    trial(study,cfg,hp)
    with pytest.raises(bt.Stop,match='sin recuperacion'):bt.recovered_failures(study,cfg,sp)


def test_altered_retry_lineage_rejected(tmp_path):
    cfg=config(tmp_path);hp={'lr0':1e-5};sp=get_arch('rtdetr_r50');study=optuna.create_study()
    root,_=trial(study,cfg,hp);failed,_=trial(study,cfg,hp,parent=root)
    changed=SimpleNamespace(number=failed.number,user_attrs=dict(failed.user_attrs,bt_recovery_source='invented'),params=failed.params,state=failed.state)
    with pytest.raises(bt.Stop,match='Linaje'):bt.recovery_family(SimpleNamespace(trials=[root,changed]),changed,cfg,sp)


def test_preflight_failure_does_not_consume_another_hpo_attempt(tmp_path,monkeypatch):
    from orchestration.study_queue import StudyQueue
    import utils.optuna_storage as storage
    cfg=config(tmp_path);sp=get_arch('rtdetr_r50');hp={'lr0':1e-5}
    q=StudyQueue.from_cfg(cfg);q.plan(cfg,[sp])
    baseline=next(j for j in q.jobs() if j.kind=='baseline');q.finish(baseline.job_id,'COMPLETE')
    job=q.claim_next('PC-03',kinds=['hpo'],slugs=['rtdetr_r50']);q.finish(job.job_id,'PENDING')
    job=q.claim_next('PC-03',kinds=['hpo'],slugs=['rtdetr_r50']);q.finish(job.job_id,'FAILED')
    assert job.attempt==2
    study=optuna.create_study();_,source=trial(study,cfg,hp)
    orphan=source/'fold0/cp/e0001.pt';orphan.parent.mkdir(parents=True);orphan.write_bytes(b'NO_RECEIPT')
    db=storage.study_db_path(cfg,sp);db.parent.mkdir(parents=True,exist_ok=True);db.write_bytes(b'EXPLICIT_STUB')
    monkeypatch.setattr(storage,'load_study',lambda *a:study)
    monkeypatch.setattr(bt,'snapshot_sqlite',lambda *a:None)
    monkeypatch.setattr(bt,'verify_trials',lambda *a:[])
    monkeypatch.setattr(bt,'recover_owned_job',lambda *a:pytest.fail('Recovery mutated job before input proof'))
    with pytest.raises(bt.Stop,match='No hay estado'):
        bt.run_hpo(tmp_path,tmp_path,tmp_path/'config',cfg,q,q.get(job.job_id),None,tmp_path,tmp_path/'audit')
    assert q.get(job.job_id).attempt==2 and q.get(job.job_id).status=='FAILED'


def test_driver_accepts_historical_fail_only_after_closed_linked_recovery(tmp_path,monkeypatch):
    spec_=importlib.util.spec_from_file_location('driver_checkpoint_test',ROOT/'botrytis_robustez_reanudar.py')
    driver=importlib.util.module_from_spec(spec_);spec_.loader.exec_module(driver)
    cfg=config(tmp_path);sp=get_arch('rtdetr_r50');study=optuna.create_study();hp={'lr0':1e-5}
    root,_=trial(study,cfg,hp);retry,_=trial(study,cfg,hp,parent=root,final='COMPLETE')
    for _ in range(14):trial(study,cfg,hp,final='COMPLETE')
    fake=SimpleNamespace(recovered_failures=lambda st,c,s:bt.recovered_failures(st,c,s),verify_trials=lambda *a:[])
    monkeypatch.setattr(driver,'official',lambda:fake)
    # Curves/folds are verified by the production routine separately; here test
    # the closure gate in isolation before it reads those synthetic folders.
    monkeypatch.setattr(driver.bt,'allowed_trial_folder',lambda *a: (_ for _ in ()).throw(RuntimeError('REACHED_VERIFIED_CURVE_GATE')))
    with pytest.raises(RuntimeError,match='REACHED_VERIFIED_CURVE_GATE'):
        driver.verify_closed_hpo(study,cfg,sp,dict(status='PENDING_REVIEW',payload={}))
    trial(study,cfg,hp)
    with pytest.raises(bt.Stop,match='sin recuperacion'):
        driver.verify_closed_hpo(study,cfg,sp,dict(status='PENDING_REVIEW',payload={}))
