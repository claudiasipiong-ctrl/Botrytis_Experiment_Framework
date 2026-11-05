"""Controller tests: native queue/Optuna/07 with explicit fake backend; no GPU training."""
import ast,contextlib,importlib,importlib.util,json,logging,os,pickle,sys
from pathlib import Path
from types import SimpleNamespace
import numpy as np
import optuna
import pandas as pd
import pytest
import torch
ROOT=Path(__file__).parent
spec_=importlib.util.spec_from_file_location('r50_hpo_controller',ROOT/'botrytis_r50_hpo_pc03.py');bt=importlib.util.module_from_spec(spec_);spec_.loader.exec_module(bt)
from utils.config_loader import load_config,cfg_path
from utils.architectures import get_arch
from utils.experiments import fingerprint
from utils.io import write_json,write_csv
from utils.search_spaces import baseline_hyperparameters,suggest_hyperparameters,load_search_spaces,config_id
from utils.resume import receipt
from orchestration.study_queue import StudyQueue

def config(tmp_path):
    c=load_config()
    for key,value in dict(runs_dir='r',artifacts_dir='a',queue_dir='q',optuna_dir='o',results_dir='e',best_configs_dir='best',classification_dir='cls').items():c['paths'][key]=str(tmp_path/value)
    c['compute']['pc_id']='PC-03';c['experiment'].update(round_id='r50vd_01',experiment_id='botrytis_v4',architecture='rtdetr_r50')
    return c

def queue(tmp_path):
    c=config(tmp_path);q=StudyQueue.from_cfg(c);q.plan(c,[get_arch(bt.MODEL)])
    b=next(j for j in q.jobs() if j.kind=='baseline');q.finish(b.job_id,'PENDING_REVIEW')
    return c,q,q.get(b.job_id)

def review(tmp_path):
    baseline=tmp_path/'baseline';baseline.mkdir();(baseline/'metrics.json').write_text('ORIGINAL')
    r=dict(decision='B_APTO_CON_OBSERVACIONES',baseline_inventory=bt.audit.frozen_inventory(baseline),limitations=['Overfitting no descartado'])
    bt.atomic_json(tmp_path/'baseline-audit.json',r);bt.atomic_json(tmp_path/'critical-inputs.json',dict(training=False,fixture=True))
    return baseline,r

def test_explicit_flag_approval_only_baseline(tmp_path,monkeypatch):
    c,q,b=queue(tmp_path);baseline,r=review(tmp_path);monkeypatch.setenv('USERNAME','I21313')
    bt.approve_baseline(q,b,r,tmp_path,baseline,True)
    assert q.get(b.job_id).status=='COMPLETE'
    hpo=next(j for j in q.jobs() if j.kind=='hpo');assert hpo.status=='PENDING'
    p=bt.read(tmp_path/'aprobacion.json');assert p['absence_of_overfitting_certified'] is False and p['human_visual_inspection_claimed'] is False
    original=(tmp_path/'aprobacion.json').read_bytes();bt.approve_baseline(q,q.get(b.job_id),r,tmp_path,baseline,True)
    assert (tmp_path/'aprobacion.json').read_bytes()==original
    with q._conn() as con:assert con.execute('SELECT count(*) FROM reviews').fetchone()[0]==1

@pytest.mark.parametrize('failure',['no_authorization','missing_evidence','changed_baseline'])
def test_approval_cannot_bypass_checks(tmp_path,failure):
    c,q,b=queue(tmp_path);baseline,r=review(tmp_path)
    if failure=='missing_evidence':r['decision']='D_EVIDENCIA_INSUFICIENTE'
    if failure=='changed_baseline':(baseline/'metrics.json').write_text('CHANGED')
    with pytest.raises(bt.Stop):bt.approve_baseline(q,b,r,tmp_path,baseline,failure!='no_authorization')
    assert q.get(b.job_id).status=='PENDING_REVIEW'
    with q._conn() as con:assert con.execute('SELECT count(*) FROM reviews').fetchone()[0]==0

@pytest.mark.parametrize('alive',[True,False])
def test_only_matching_dead_owner_recovers(tmp_path,monkeypatch,alive):
    c,q,b=queue(tmp_path);q.approve(b.job_id,operator='TEST',evidence='fixture')
    job=q.claim_next('PC-03',kinds=['hpo'],slugs=[bt.MODEL])
    bt.atomic_json(tmp_path/'owner.json',dict(job_id=job.job_id,attempt=job.attempt,executor_sha256=bt.EXECUTOR_SHA256,process=dict(pid=123)))
    monkeypatch.setattr(bt,'owner_alive',lambda p:alive)
    if alive:
        with pytest.raises(bt.Stop):bt.recover_owned_job(q,job,tmp_path)
        assert q.get(job.job_id).status=='RUNNING'
    else:
        bt.recover_owned_job(q,job,tmp_path);assert q.get(job.job_id).status=='PENDING'


def test_sampler_restore_and_tamper_detection(tmp_path):
    study=optuna.create_study(sampler=optuna.samplers.TPESampler(seed=42));t=study.ask();t.suggest_float('x',.01,1);study.tell(t,.5)
    store=bt.OptimizerState(tmp_path,'fp');store.save(study)
    expected=optuna.create_study(sampler=pickle.loads(pickle.dumps(study.sampler))).ask().suggest_float('x',.01,1)
    study.sampler=optuna.samplers.TPESampler(seed=999);assert store.restore(study)
    assert study.ask().suggest_float('x',.01,1)==expected
    meta=bt.read(tmp_path/'state.json');(tmp_path/meta['file']).write_bytes(b'corrupt')
    with pytest.raises(bt.Stop):store.restore(study)


def history(score=.6,fold=None):
    scores=[.1,score]+[score-.01]*(20)
    data=dict(epoch=list(range(1,23)),optimization_loss_total=[5.]*22,
        train_loss=[2.]*22,val_loss=[2.1]*22,map50_95=scores,map50=[min(1,x+.1) for x in scores],precision=[.7]*22,recall=[.7]*22,f1=[.7]*22)
    for role,total in [('train',2.),('val',2.1)]:
        data[role+'/vfl_loss']=[.5]*22;data[role+'/bbox_loss']=[.5]*22;data[role+'/giou_loss']=[total-1]*22
    h=pd.DataFrame(data)
    for col,alias in [('metrics/mAP50-95(B)','map50_95'),('metrics/mAP50(B)','map50'),('metrics/precision(B)','precision'),('metrics/recall(B)','recall')]:h[col]=h[alias]
    h['ap_per_iou']=[np.linspace(v+.1,v-.1,10).tolist() for v in h.map50_95]
    if fold is not None:h.insert(0,'fold',fold)
    return h

@pytest.mark.parametrize('damage',['nan','absent','duplicate','inconsistent_sum'])
def test_invalid_history_stops(tmp_path,damage):
    h=history()
    if damage=='nan':h.loc[0,'val_loss']=np.inf
    if damage=='absent':h=h.drop(columns='val/vfl_loss')
    if damage=='duplicate':h.loc[1,'epoch']=1
    if damage=='inconsistent_sum':h.loc[0,'train_loss']=.3
    p=tmp_path/'results.csv';h.to_csv(p,index=False)
    with pytest.raises(bt.Stop):bt.inspect_curve(p)


def test_completed_hpo_never_launched_again(tmp_path,monkeypatch):
    c,q,b=queue(tmp_path);q.approve(b.job_id,operator='TEST',evidence='fixture');j=next(j for j in q.jobs() if j.kind=='hpo');q.finish(j.job_id,'PENDING_REVIEW')
    import utils.optuna_storage as storage
    db=storage.study_db_path(c,get_arch(bt.MODEL));db.parent.mkdir(parents=True,exist_ok=True);db.write_bytes(b'fixture')
    monkeypatch.setattr(bt,'snapshot_sqlite',lambda *a:None)
    study=SimpleNamespace(trials=[],best_trial=SimpleNamespace(number=4),best_value=.8)
    monkeypatch.setattr(storage,'load_study',lambda *a:study);monkeypatch.setattr(storage,'budget_used',lambda *a:15);monkeypatch.setattr(bt,'verify_trials',lambda *a:[])
    monkeypatch.setattr(bt.importlib,'import_module',lambda *a:pytest.fail('HPO launched twice'))
    assert bt.run_hpo(tmp_path,tmp_path,tmp_path/'config',c,q,q.get(j.job_id),None,tmp_path,tmp_path)==study


def test_partial_and_complete_fold_recovery_preserves_originals(tmp_path):
    c=config(tmp_path);sp=get_arch(bt.MODEL);hp=baseline_hyperparameters(sp,c)
    study=optuna.create_study();t=study.ask();t.suggest_float('x',.01,1)
    source=cfg_path(c,'paths.runs_dir')/bt.MODEL/'hpo/trial_000';source.mkdir(parents=True)
    for k,v in dict(hp=hp,run_dir=str(source.resolve()),pc_id='PC-03',config_id=config_id(hp)).items():t.set_user_attr(k,v)
    fold0=source/'fold0';fold0.mkdir();h=history(fold=0);write_csv(h,fold0/'completed_history.csv',float_format='%.12g');write_csv(h.drop(columns='fold'),fold0/'results.csv',float_format='%.12g')
    sig=bt.fold_signature(fingerprint(c),bt.MODEL,hp,42,0,60,20)
    write_json(fold0/'completed_fold.json',dict(signature=sig,history_sha256=bt.digest(fold0/'completed_history.csv'),row=dict(fold=0,score=.6,best_epoch=2,epochs_run=22)))
    best=fold0/'best.pt';torch.save(dict(format='SYNTHETIC_CONTROLLER_FIXTURE'),best);receipt(best,sp,c,dict(seed=42,hp=hp,context=dict(fold=0)))
    adapter=bt.audit.load_adapter();partial=source/'fold1/cp/e0003.pt';partial.parent.mkdir(parents=True)
    partial_sig=adapter.canonical(dict(experiment=fingerprint(c),hp=hp,seed=42,fold=1,epochs=60,patience=20,fixed=False))
    torch.save(dict(format=adapter.FORMAT,signature=partial_sig,epoch=3,seed=42,hp=hp,fold=1,history=history().iloc[:3].to_dict('records'),model={},ema={},optimizer={},scheduler={},scaler={},gradients={},rng={},best_state={},best_metrics={},ema_updates=0,best_epoch=2,best_score=.6,last_opt_step=0,elapsed=0,stopper_epoch=2,stopper_fitness=.6,stopped=False),partial);receipt(partial,sp,c,dict(seed=42,hp=hp,context=dict(fold=1),epoch=3,adapter_sha256=bt.audit.ADAPTER_SHA))
    t.report(.1,0);study.tell(t,state=optuna.trial.TrialState.FAIL)
    before=bt.audit.frozen_inventory(source);bt.schedule_retries(study,c,sp,tmp_path)
    assert len(study.trials)==2
    bt.schedule_retries(study,c,sp,tmp_path);assert len(study.trials)==2
    retry=study.ask();dest=cfg_path(c,'paths.runs_dir')/bt.MODEL/'hpo/trial_001'
    controller=bt.R50Recovery(c,sp,tmp_path/'control');controller.study=study
    def check(spec,hp,cfg,ctx,out,**kw):
        assert (out/'fold0/completed_fold.json').is_file()
        assert adapter.load_resume(out/'fold1',partial_sig,spec,cfg)['epoch']==3
        return 'CACHED_PLUS_RESUME_ONLY'
    assert controller.evaluate(check,sp,hp,c,object(),dest,trial=retry,folds=[0,1,2],seed=42,max_epochs=60,patience=20)=='CACHED_PLUS_RESUME_ONLY'
    assert before==bt.audit.frozen_inventory(source)
    assert retry.user_attrs['bt_retry_of']==0


def test_real_native_07_fifteen_trials_three_folds_and_idempotence(tmp_path,monkeypatch):
    """Real native HPO/Optuna/queue with fake R50 data, no model training or GPU."""
    import utils.hpo_runner as runner
    import utils.detector_train as dt
    native=importlib.import_module('07_run_hpo_study');adapter=bt.audit.load_adapter()
    c,q,b=queue(tmp_path);q.approve(b.job_id,operator='TEST',evidence='explicit fixture')
    j=next(j for j in q.jobs() if j.kind=='hpo');sp=get_arch(bt.MODEL)
    original=tmp_path/'r/rtdetr_r50/baseline/DO_NOT_CHANGE';original.parent.mkdir(parents=True);original.write_bytes(b'BASELINE')
    baseline_cp=original.parent/'fold0/cp/e0100.pt';baseline_cp.parent.mkdir(parents=True);torch.save(dict(format='SYNTHETIC_SIZE_FIXTURE'),baseline_cp)
    torch.save(dict(format='SYNTHETIC_SIZE_FIXTURE'),baseline_cp.parent.parent/'best.pt')
    calls=[];launches=[];access=[]
    class Run:
        def __init__(self,path):
            self.cfg=c;self.run_dir=Path(path);self.run_dir.mkdir(parents=True,exist_ok=True);self.runs_base=cfg_path(c,'paths.runs_dir')
            self.pc_id='PC-03';self.logger=logging.getLogger('test_hpo');self.run_id='fixture';self.env={'git_commit':'fixture'};self.hashes={};self.checkpoints=[]
        def __enter__(self):return self
        def __exit__(self,*a):return False
        def set_seed(self,*a):pass
        def add_metadata(self,**kw):pass
        def note_split_access(self,*a):access.append(a)
        def log_metrics(self,v):write_json(self.run_dir/'metrics.json',v)
        def save_figure(self,fig,*a,**kw):
            import matplotlib.pyplot as plt
            plt.close(fig)
        def record_checkpoint(self,role,**kwargs):
            p=self.run_dir/f"fold{kwargs['fold']}"/'best.pt';kwargs.update(weights_discarded=False,retained_path=str(p))
            self.checkpoints.append(dict(role=role,**kwargs));write_json(self.run_dir/'checkpoint_metadata.json',dict(checkpoints=self.checkpoints))
    def fake_train(spec,hp,cfg,**opts):
        calls.append((opts['name'],dict(hp)));out=Path(opts['project_dir'])/opts['name'];out.mkdir(parents=True,exist_ok=True)
        trial=int(out.parent.name.split('_')[1]);fold=int(opts['name'][4:]);score=.5+trial*.001+fold*.001
        h=history(score);write_csv(h,out/'results.csv',float_format='%.12g')
        best=out/'best.pt';torch.save(dict(format='EXPLICIT_FAKE_BACKEND'),best);receipt(best,spec,cfg,dict(seed=42,hp=hp,context=dict(fold=fold)))
        cp=out/'cp/e0022.pt';cp.parent.mkdir();torch.save(dict(format='FAKE_STATE_NO_OPTIMIZER_STEPS'),cp);receipt(cp,spec,cfg,dict(seed=42,hp=hp,context=dict(fold=fold),epoch=22))
        for _,row in h.iterrows():opts['epoch_callback'](opts['step_offset']+int(row.epoch)-1,row)
        return dt.DetectorRun(score=score,best_epoch=2,epochs_run=22,stopped_early=True,history=h,final_metrics=dict(map50_95=score,map50=score+.1,precision=.7,recall=.7,f1=.7),train_seconds=0,vram_peak_mb=None,micro_batch=16,params_m=42.729027,gflops=136.7,weights_path=str(best),weights_sha256=bt.digest(best),pretrained_sha256='f'*64)
    # Test replacement is explicit. Production installs the fixed official adapter.
    monkeypatch.setattr(adapter,'train_detector',fake_train)
    for module,name in [(runner,'train_detector'),(runner,'evaluate_config'),(native,'create_or_load_study')]:monkeypatch.setattr(module,name,getattr(module,name))
    monkeypatch.setattr(runner,'record_training_source',lambda *a:None)
    monkeypatch.setattr(runner,'detection_images_table',lambda cfg:pd.DataFrame())
    monkeypatch.setattr(runner,'write_fold_yaml',lambda cfg,fold,out,**kw:(Path(out)/'data.yaml',dict(n_train=2,n_val=1)))
    monkeypatch.setattr(native,'preflight_training',lambda *a:SimpleNamespace(dataset_hash=None,split_hash=None))
    monkeypatch.setattr(native,'GpuJobLock',lambda *a,**kw:contextlib.nullcontext())
    monkeypatch.setattr(native,'nested_run',lambda ctx,spec,name,**kw:Run(cfg_path(c,'paths.runs_dir')/spec.slug/name))
    monkeypatch.setattr(native,'save_training_figures',lambda *a,**kw:None)
    import utils.inference as inference
    from utils.logging_utils import RunContext
    for module,name in [(dt,'train_detector'),(inference,'load_detector'),(inference,'predict_detector'),(RunContext,'record_checkpoint')]:monkeypatch.setattr(module,name,getattr(module,name))
    def cli(script,main,**kwargs):
        launches.append(kwargs['argv']);assert kwargs['argv'][kwargs['argv'].index('--architecture')+1]=='rtdetr_r50'
        claimed=q.claim_next('PC-03',kinds=['hpo'],slugs=[bt.MODEL]);assert claimed
        main(Run(tmp_path/'r/rtdetr_r50/hpo/study'),SimpleNamespace(architecture=bt.MODEL,n_trials=None,trial_block_size=15,allow_budget_override=False))
        q.computed(j.job_id,published=True)
        from orchestration.worker import top_config_ids
        q.expand_after_hpo(c,sp,top_config_ids(c,sp.slug));return 0
    monkeypatch.setattr(native,'run_cli',cli)
    state=tmp_path/'control';attempt=state/'attempt';attempt.mkdir(parents=True)
    study=bt.run_hpo(tmp_path,tmp_path,tmp_path/'config',c,q,q.get(j.job_id),adapter,state,attempt)
    assert len(calls)==45 and len(launches)==1 and original.read_bytes()==b'BASELINE'
    assert len(study.trials)==15 and sum(t.state.name=='COMPLETE' for t in study.trials)==15
    assert q.get(j.job_id).status=='PENDING_REVIEW'
    post=[j for j in q.jobs() if j.kind not in ('baseline','hpo')];assert len(post)==13 and all(j.status=='PENDING' for j in post)
    assert access and all(x[0]=='train' for x in access)
    bt.run_hpo(tmp_path,tmp_path,tmp_path/'config',c,q,q.get(j.job_id),adapter,state,attempt)
    assert len(calls)==45 and len(launches)==1


def test_only_native_07_is_launched():
    source=(ROOT/'botrytis_r50_hpo_pc03.py').read_text();tree=ast.parse(source)
    imports=[n.args[0].value for n in ast.walk(tree) if isinstance(n,ast.Call) and isinstance(n.func,ast.Attribute) and n.func.attr=='import_module' and n.args and isinstance(n.args[0],ast.Constant)]
    assert imports==['07_run_hpo_study']
    assert 'robustness_started=False' in source and "winner='NO_DECLARADO'" in source


def test_equivalent_index_refresh_does_not_invalidate_baseline_audit():
    code=dict(git_commit='commit',code_hash='hash',eol_equivalent=[])
    report=dict(decision='B_APTO_CON_OBSERVACIONES',executor_sha256=bt.AUDIT_SHA,fingerprint='fp',config_sha256='config',code=dict(code,eol_equivalent=['src/file.py']),baseline_inventory={'a':'b'},baseline_unchanged=True,prepared_artifacts_unchanged=True)
    assert bt.cached_audit_compatible(report,'fp','config',code,{'a':'b'})
    report['code']['code_hash']='modified'
    assert not bt.cached_audit_compatible(report,'fp','config',code,{'a':'b'})


@pytest.mark.parametrize('rc',[0,2])
def test_initial_audit_preserves_decision_and_does_not_ask_approval_twice(monkeypatch,rc):
    events=[];original=bt.audit.emit
    def perform(*args):
        bt.audit.emit('AUDITORIA_FINALIZADA',decision='B' if rc==0 else 'D')
        bt.audit.emit('APROBACION_PENDIENTE')
        return rc
    monkeypatch.setattr(bt.audit,'perform_audit',perform);monkeypatch.setattr(bt,'emit',lambda name,**kw:events.append(name))
    assert bt.run_initial_audit(None,None,None,None,None)==rc
    assert bt.audit.emit is original and 'AUTORIZACION_EXPLICITA_RECIBIDA' in events


def test_missing_optuna_with_prior_weights_never_creates_new_study(tmp_path):
    c,q,b=queue(tmp_path);q.approve(b.job_id,operator='TEST',evidence='fixture');job=next(j for j in q.jobs() if j.kind=='hpo')
    old=cfg_path(c,'paths.runs_dir')/bt.MODEL/'hpo/trial_000/fold0/cp/e0001.pt';old.parent.mkdir(parents=True);old.write_bytes(b'ORIGINAL')
    with pytest.raises(bt.Stop,match='Falta SQLite Optuna'):bt.run_hpo(tmp_path,tmp_path,tmp_path/'config',c,q,job,None,tmp_path,tmp_path)
    assert old.read_bytes()==b'ORIGINAL' and not list(cfg_path(c,'paths.optuna_dir').glob('*.db'))


def test_unlinked_existing_trial_folder_stops_before_training(tmp_path):
    c=config(tmp_path);sp=get_arch(bt.MODEL);hp=baseline_hyperparameters(sp,c);study=optuna.create_study();trial=study.ask()
    folder=tmp_path/'trial/fold0';folder.mkdir(parents=True);(folder/'results.csv').write_text('ORIGINAL')
    recovery=bt.R50Recovery(c,sp,tmp_path/'state');recovery.study=study
    with pytest.raises(bt.RecoveryStop):recovery.evaluate(lambda *a,**k:pytest.fail('Training repeated'),sp,hp,c,object(),folder.parent,trial=trial,folds=[0,1,2],seed=42,max_epochs=60,patience=20)


@pytest.mark.parametrize('condition',['valid','hash_changed','source_conflict'])
def test_current_inputs_must_pass_before_approval(tmp_path,monkeypatch,condition):
    import utils.detector_data
    data=SimpleNamespace(dataset_hash=bt.audit.EXPECTED_DATA,split_hash=bt.audit.EXPECTED_SPLIT)
    if condition=='hash_changed':data.dataset_hash='changed'
    monkeypatch.setattr(bt.audit,'readonly_prepared',lambda cfg,dest:(cfg,data))
    monkeypatch.setattr(utils.detector_data,'detection_images_table',lambda cfg:[])
    monkeypatch.setattr(bt.audit,'verify_original_lists',lambda *args:['verified'])
    monkeypatch.setattr(bt.audit,'partition_evidence',lambda *args:dict(confirmed_relevant_conflicts=['conflict'] if condition=='source_conflict' else []))
    if condition=='valid':
        result=bt.fresh_critical_inputs({},tmp_path,tmp_path);assert result['training'] is False and (tmp_path/'critical-inputs.json').is_file()
    else:
        with pytest.raises(bt.Stop):bt.fresh_critical_inputs({},tmp_path,tmp_path)
        assert not (tmp_path/'critical-inputs.json').exists()
