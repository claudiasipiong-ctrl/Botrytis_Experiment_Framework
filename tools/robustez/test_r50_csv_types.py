"""Typed R50 CSV comparison: no GPU, original artifacts remain byte-identical."""
import importlib.util
import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest

ROOT=Path(__file__).parent
spec_=importlib.util.spec_from_file_location('r50_csv_types',ROOT/'botrytis_r50_hpo_pc03.py')
bt=importlib.util.module_from_spec(spec_);spec_.loader.exec_module(bt)
from utils.config_loader import load_config,cfg_path
from utils.architectures import get_arch
from orchestration.study_queue import StudyQueue


def frame():
    return pd.DataFrame(dict(epoch=[1,2],train_loss=[2.,1.],val_loss=[2.1,1.1],
        map50_95=[.5,.6],map50=[.6,.7],precision=[.8,.9],recall=[.7,.8],
        ap_per_iou=[np.linspace(.6,.4,10).tolist(),np.linspace(.7,.5,10).tolist()]))


def csv_pair(tmp,history=None):
    raw=frame();signed=raw.copy(deep=True) if history is None else history
    raw.to_csv(tmp/'results.csv',index=False)
    signed.to_csv(tmp/'completed_history.csv',index=False,float_format='%.12g')
    return pd.read_csv(tmp/'results.csv'),pd.read_csv(tmp/'completed_history.csv')


def test_reproduces_original_string_subtraction_and_fixes_it(tmp_path):
    raw,signed=csv_pair(tmp_path)
    originals={p:p.read_bytes() for p in tmp_path.iterdir()}
    with pytest.raises(TypeError,match='str'):
        bt.audit.compare_histories(raw,signed,list(raw))
    result=bt.compare_training_histories(raw,signed)
    assert result['ap_vectors_compared']==2 and result['originals_modified'] is False
    assert all(p.read_bytes()==data for p,data in originals.items())


def test_vector_formats_and_numeric_scalars_are_parsed_without_mutation(tmp_path):
    raw,signed=csv_pair(tmp_path)
    signed['ap_per_iou']=[json.dumps(v,indent=1) for v in frame().ap_per_iou]
    signed['train_loss']=signed.train_loss.astype(str)
    before=signed.copy(deep=True)
    assert bt.compare_training_histories(raw,signed)['ap_vectors_compared']==2
    pd.testing.assert_frame_equal(signed,before)


@pytest.mark.parametrize('damage',['different','short','long','nan','inf','out_of_range','nested','bool','text','code','missing_one'])
def test_invalid_vector_is_not_ignored(tmp_path,damage):
    raw,signed=csv_pair(tmp_path)
    vector=frame().ap_per_iou.iloc[0]
    if damage=='different':vector[4]=.1
    if damage=='short':vector=vector[:9]
    if damage=='long':vector=vector+[.1]
    if damage=='nan':vector[4]=float('nan')
    if damage=='inf':vector[4]=float('inf')
    if damage=='out_of_range':vector[4]=1.1
    if damage=='nested':vector=[vector]
    if damage=='bool':vector[4]=True
    if damage=='text':vector[4]='0.1'
    if damage=='code':vector="__import__('os').system('should_never_run')"
    if damage=='missing_one':vector=None
    signed.loc[0,'ap_per_iou']=repr(vector) if vector is not None else np.nan
    with pytest.raises(bt.Stop,match='ap_per_iou'):
        bt.compare_training_histories(raw,signed)


@pytest.mark.parametrize('damage',['length','epoch','scalar','nan','inf','text','duplicate','missing_vector_column'])
def test_scalar_and_structure_checks_are_retained(tmp_path,damage):
    raw,signed=csv_pair(tmp_path)
    if damage=='length':signed=signed.iloc[:1]
    if damage=='epoch':signed.loc[0,'epoch']=2
    if damage=='scalar':signed.loc[0,'map50_95']=.1
    if damage=='nan':signed.loc[0,'val_loss']=np.nan
    if damage=='inf':signed.loc[0,'val_loss']=np.inf
    if damage=='text':signed['val_loss']=['oops','1.1']
    if damage=='duplicate':signed.columns=['epoch','epoch',*signed.columns[2:]]
    if damage=='missing_vector_column':signed=signed.drop(columns='ap_per_iou')
    with pytest.raises(bt.Stop):bt.compare_training_histories(raw,signed)


def test_optional_vector_unavailable_in_both_is_explicitly_reported(tmp_path):
    raw,signed=csv_pair(tmp_path)
    raw.loc[0,'ap_per_iou']=np.nan;signed.loc[0,'ap_per_iou']=np.nan
    result=bt.compare_training_histories(raw,signed)
    assert result['ap_vectors_compared']==1 and result['ap_vector_unavailable_epochs']==[1]


@pytest.mark.parametrize('alive',[False,True])
def test_known_predecessor_owner_requires_real_dead_process(tmp_path,monkeypatch,alive):
    cfg=load_config()
    cfg['paths']['queue_dir']=str(tmp_path/'q')
    cfg['compute']['pc_id']='PC-03'
    cfg['experiment'].update(round_id='r50vd_01',experiment_id='botrytis_v4',architecture='rtdetr_r50')
    q=StudyQueue.from_cfg(cfg);q.plan(cfg,[get_arch('rtdetr_r50')])
    baseline=next(j for j in q.jobs() if j.kind=='baseline');q.finish(baseline.job_id,'COMPLETE')
    job=q.claim_next('PC-03',kinds=['hpo'],slugs=['rtdetr_r50'])
    owner=dict(job_id=job.job_id,attempt=job.attempt,executor_sha256=bt.LEGACY_EXECUTOR_SHAS[0],process=dict(pid=123))
    bt.atomic_json(tmp_path/'owner.json',owner);before=(tmp_path/'owner.json').read_bytes()
    monkeypatch.setattr(bt,'owner_alive',lambda process:alive)
    if alive:
        with pytest.raises(bt.Stop,match='activo'):bt.recover_owned_job(q,job,tmp_path)
        assert q.get(job.job_id).status=='RUNNING'
    else:
        bt.recover_owned_job(q,job,tmp_path)
        assert q.get(job.job_id).status=='PENDING'
    assert (tmp_path/'owner.json').read_bytes()==before


def test_foreign_owner_is_never_accepted(tmp_path,monkeypatch):
    owner=dict(job_id='job',attempt=1,executor_sha256='0'*64,process=dict(pid=123))
    bt.atomic_json(tmp_path/'owner.json',owner)
    job=SimpleNamespace(status='RUNNING',job_id='job',attempt=1)
    monkeypatch.setattr(bt,'owner_alive',lambda process:False)
    q=SimpleNamespace(finish=lambda *a,**k:pytest.fail('Foreign queue changed'))
    with pytest.raises(bt.Stop,match='otra ejecucion'):bt.recover_owned_job(q,job,tmp_path)


def test_previous_baseline_audit_stays_compatible_without_repeating_qa():
    code=dict(git_commit='commit',code_hash='code')
    prior=dict(decision='B_APTO_CON_OBSERVACIONES',executor_sha256='432881891c91bc6e20becc7fec97d245a657ce5c7e6f1677fb99c224d344149e',
        fingerprint='fp',config_sha256='cfg',code=code,baseline_inventory={'file':'sha'},
        baseline_unchanged=True,prepared_artifacts_unchanged=True)
    assert bt.cached_audit_compatible(prior,'fp','cfg',code,{'file':'sha'})
    assert not bt.cached_audit_compatible(prior,'fp','cfg',code,{'file':'changed'})
