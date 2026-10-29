"""CPU-only control-plane regressions. No optimizer training, HPO or real test access."""
from __future__ import annotations
import copy
import json
from pathlib import Path
from types import SimpleNamespace
import pytest
from utils.config_loader import load_config,cfg_path,FixedDecisionViolation
from utils.experiments import create_round,verify_round,fingerprint
from utils.durable import atomic_json
from utils.architectures import archs,get_arch,trials_for
from orchestration.study_queue import StudyQueue,COMPLETE,PENDING_REVIEW,PUBLICATION_PENDING
from orchestration.pc_registry import PCRegistry


def config_at(p):
    return load_config(overrides=[f'paths.artifacts_dir={p}/art',f'paths.queue_dir={p}/queue',f'paths.runs_dir={p}/runs',f'paths.optuna_dir={p}/optuna'])


def test_seven_active_models_and_budgets():
    c=load_config()
    assert {a.name for a in archs()}=={'YOLOv8n','YOLOv8l','YOLO26l','RT-DETR-R50','ResNet50','ViT-B/16','Swin-Tiny'}
    for name in ['YOLOv8m','YOLO26n','YOLO26s']:
        with pytest.raises(KeyError):get_arch(name)
    assert sum(trials_for(a,c) for a in archs('detection'))==60
    assert sum(trials_for(a,c) for a in archs('classification'))==60
    assert c['robustness']['top_k_configs']==3 and c['robustness']['seeds']==[17,42,73]
    assert c['inner_hpo']['folds']==3
    reg=PCRegistry(c)
    assert [len(reg.owned_slugs(pc)) for pc in reg.pcs]==[3,2,2]
    assert len(reg.ownership())==7


def test_three_independent_queues_never_claim_foreign_models(tmp_path):
    results=[]
    for pc in ('PC-01','PC-02','PC-03'):
        c=config_at(tmp_path/pc);q=StudyQueue.from_cfg(c);q.plan(c)
        taken=[]
        while job:=q.claim_next(pc,kinds=['baseline']):
            assert q.claim_next(pc) is None # at most one local job
            taken.append(job.architecture);q.finish(job.job_id,COMPLETE)
        results.append(set(taken))
    assert all(not(a&b) for i,a in enumerate(results) for b in results[i+1:])
    assert set.union(*results)=={a.slug for a in archs()}


def test_reviews_pause_dependencies_not_independent_models(tmp_path):
    c=config_at(tmp_path);q=StudyQueue.from_cfg(c);q.plan(c,[get_arch('YOLOv8n'),get_arch('ResNet50')])
    first=q.claim_next('PC-01',slugs=['yolov8n']);q.computed(first.job_id,published=True)
    assert q.get(first.job_id).status==PENDING_REVIEW
    assert q.claim_next('PC-01',slugs=['yolov8n']) is None
    other=q.claim_next('PC-01',slugs=['resnet50']);assert other
    q.finish(other.job_id,COMPLETE)
    with pytest.raises(ValueError):q.approve(first.job_id,operator='',evidence='fixture')
    q.approve(first.job_id,operator='SYNTHETIC_OPERATOR',evidence='fixture.json')
    hpo=q.claim_next('PC-01',slugs=['yolov8n']);assert hpo.kind=='hpo'
    q.computed(hpo.job_id,published=True)
    ids=['a','b','c'];assert q.expand_after_hpo(c,get_arch('YOLOv8n'),ids)==13
    assert q.claim_next('PC-01',slugs=['yolov8n']) is None
    q.approve(hpo.job_id,operator='SYNTHETIC_OPERATOR',evidence='fixture-hpo')
    for _ in range(9):
        repeat=q.claim_next('PC-01',slugs=['yolov8n']);assert repeat.kind=='repeat';q.finish(repeat.job_id,COMPLETE)
    aggregate=q.claim_next('PC-01',slugs=['yolov8n']);assert aggregate.kind=='aggregate'
    q.computed(aggregate.job_id,published=True)
    assert q.claim_next('PC-01',slugs=['yolov8n']) is None
    q.approve(aggregate.job_id,operator='SYNTHETIC_OPERATOR',evidence='fixture-robustness')
    assert q.claim_next('PC-01',slugs=['yolov8n']).kind=='outer_train'


def test_publication_failure_never_approves_computation(tmp_path):
    c=config_at(tmp_path);q=StudyQueue.from_cfg(c);q.plan(c,[get_arch('YOLOv8n')])
    j=q.claim_next('PC-01');q.computed(j.job_id)
    assert q.get(j.job_id).status==PUBLICATION_PENDING
    with pytest.raises(ValueError):q.approve(j.job_id,operator='x',evidence='x')
    assert q.claim_next('PC-01') is None


def test_round_immutable_isolated_preserves_signed_split(tmp_path):
    c=config_at(tmp_path/'base');art=cfg_path(c,'paths.artifacts_dir');art.mkdir(parents=True)
    (art/'split_manifest.csv').write_text('SYNTHETIC_SPLIT_UNCHANGED')
    path=create_round(c,round_id='r1',architecture='YOLOv8n',operator='SYNTHETIC',reason='fixture')
    r1=load_config(path);verify_round(r1)
    assert (cfg_path(r1,'paths.artifacts_dir')/'split_manifest.csv').read_bytes()==(art/'split_manifest.csv').read_bytes()
    path2=create_round(r1,round_id='r2',architecture='YOLOv8n',operator='SYNTHETIC',reason='fixture',parent='r1',overrides=['detector.baseline.lr0=0.002'])
    r2=load_config(path2)
    assert cfg_path(r1,'paths.runs_dir')!=cfg_path(r2,'paths.runs_dir')
    assert fingerprint(r1)!=fingerprint(r2)
    with pytest.raises(FileExistsError):create_round(c,round_id='r1',architecture='YOLOv8n',operator='x',reason='x')
    altered=copy.deepcopy(r1);altered['detector']['baseline']['lr0']=.003
    with pytest.raises(RuntimeError,match='incompatible'):verify_round(altered)
    q1=StudyQueue.from_cfg(r1);q1.plan(r1,[get_arch('YOLOv8n')])
    q2=StudyQueue.from_cfg(r2);q2.plan(r2,[get_arch('YOLOv8n')])
    assert {j.job_id for j in q1.jobs()}.isdisjoint(j.job_id for j in q2.jobs())


def test_queue_rejects_configuration_change_and_empty_selection(tmp_path):
    c=config_at(tmp_path);q=StudyQueue.from_cfg(c);assert q.plan(c,[])==0
    q.plan(c,[get_arch('YOLOv8n')]);other=copy.deepcopy(c);other['detector']['baseline']['lr0']=.5
    with pytest.raises(RuntimeError,match='incompatible'):StudyQueue.from_cfg(other)


def test_inference_with_requires_grad_parameters():
    import torch
    from torch.utils.data import DataLoader,TensorDataset
    from utils.classifier_train import predict_probs
    model=torch.nn.Linear(3,2)
    x=torch.tensor([[1.,2.,3.],[3.,2.,1.]])
    loader=DataLoader(TensorDataset(x,torch.tensor([0,1]),torch.arange(2)),batch_size=2)
    assert all(p.requires_grad for p in model.parameters())
    p,y,i,loss=predict_probs(model,loader,'cpu',False)
    assert p.shape==(2,) and (p>=0).all() and (p<=1).all() and loss>0
    assert y.tolist()==[0,1] and i.tolist()==[0,1]
    assert all(w.grad is None for w in model.parameters())


@pytest.mark.parametrize('case',['valid','absent','wrong_hash','modified','stale_dict'])
def test_test_release_uses_real_freeze_bytes(tmp_path,case):
    from utils.manual_gates import require_test_release
    from utils.hashing import sha256_file
    c=config_at(tmp_path);c['test']['freeze_manifest']=str(tmp_path/'freeze.json')
    path=Path(c['test']['freeze_manifest']);path.write_text('{"fixture":"SYNTHETIC_NO_SCIENTIFIC_TEST"}')
    freeze=json.loads(path.read_text());h=sha256_file(path)
    if case!='absent':atomic_json(cfg_path(c,'paths.artifacts_dir')/'manual_test_release.json',dict(state='AUTORIZADO_PERSONALMENTE',freeze_sha256=h if case!='wrong_hash' else '0'*64,operator='SYNTHETIC_UNIT',evidence='fixture'))
    if case=='modified':path.write_text('{"fixture":"CHANGED"}');freeze=json.loads(path.read_text())
    if case=='stale_dict':freeze['fixture']='stale'
    if case=='valid':require_test_release(c,freeze)
    else:
        with pytest.raises(RuntimeError):require_test_release(c,freeze)


def test_study_storage_identity_without_running_hpo(tmp_path):
    from utils.optuna_storage import create_or_load_study,load_study
    c=config_at(tmp_path);spec=get_arch('YOLOv8n')
    study=create_or_load_study(spec,c,42)
    assert not study.trials
    altered=copy.deepcopy(c);altered['detector']['baseline']['lr0']=.321
    with pytest.raises(ValueError,match='incompatible'):create_or_load_study(spec,altered,42)
    assert load_study(spec,c).study_name==study.study_name


def test_external_validation_cannot_be_used_for_round_exploration(tmp_path):
    from utils.experiments import authorize_external,use_external
    c=config_at(tmp_path/'base');p=create_round(c,round_id='r1',architecture='YOLOv8n',operator='SYNTHETIC',reason='fixture')
    r=load_config(p);checkpoint=tmp_path/'synthetic-not-model.pt';checkpoint.write_bytes(b'SYNTHETIC')
    with pytest.raises(RuntimeError):use_external(r,'validation',checkpoint)
    authorize_external(r,operator='SYNTHETIC',evidence='fixture')
    use_external(r,'validation',checkpoint)
    with pytest.raises(RuntimeError):use_external(r,'validation',checkpoint)
    with pytest.raises(RuntimeError):authorize_external(r,operator='SYNTHETIC',evidence='fixture')


def test_transfer_requires_source_fence_receipt_and_confirmation(tmp_path,monkeypatch):
    from orchestration import transfer as t
    from utils.hashing import write_checksum_file
    from utils.hashing import sha256_file
    src=config_at(tmp_path/'PC-01');dst=config_at(tmp_path/'PC-02')
    # Snapshot writer/importer are separately integration-tested with real ZIP files.
    # This fixture isolates the offline handoff state machine, without real datasets.
    def pack(c,path,pc,runs):
        Path(path).write_bytes(b'SYNTHETIC_SNAPSHOT');return {}
    monkeypatch.setattr(t,'pack_results',pack)
    monkeypatch.setattr(t,'unpack',lambda *args,**kwargs: {})
    bundle=tmp_path/'snapshot.zip';receipt=tmp_path/'receipt.json';confirmation=tmp_path/'confirmation.json'
    row=t.offer(src,'yolov8n','PC-01','PC-02',bundle,operator='SYNTHETIC',evidence='fixture')
    assert PCRegistry(src).owner('yolov8n')=='__TRANSFER_PENDING__'
    qs=StudyQueue.from_cfg(src);qs.plan(src,[get_arch('YOLOv8n')]);assert qs.claim_next('PC-01') is None
    t.receive(dst,'PC-02',bundle,bundle.with_suffix('.offer.json'),receipt,operator='SYNTHETIC')
    qd=StudyQueue.from_cfg(dst);qd.plan(dst,[get_arch('YOLOv8n')]);assert qd.claim_next('PC-02') is None
    t.confirm(src,'yolov8n',receipt,confirmation,operator='SYNTHETIC')
    assert PCRegistry(src).owner('yolov8n')=='PC-02'
    t.activate(dst,'yolov8n',receipt,confirmation,operator='SYNTHETIC')
    assert qd.claim_next('PC-02').architecture=='yolov8n'
    assert qs.claim_next('PC-01') is None
    with pytest.raises(ValueError):t.activate(dst,'yolov8n',receipt,confirmation,operator='SYNTHETIC')


def test_transfer_rejects_running_and_wrong_receipt(tmp_path,monkeypatch):
    from orchestration import transfer as t
    src=config_at(tmp_path);q=StudyQueue.from_cfg(src);q.plan(src,[get_arch('YOLOv8n')]);job=q.claim_next('PC-01')
    with pytest.raises(ValueError,match='RUNNING'):
        t.offer(src,'yolov8n','PC-01','PC-02',tmp_path/'bundle.zip',operator='SYNTHETIC',evidence='fixture')


def test_foreign_queue_state_never_advances_local_jobs(tmp_path):
    c=config_at(tmp_path);q=StudyQueue.from_cfg(c);q.plan(c,[get_arch('YOLOv8n')])
    p=q.export_state(tmp_path/'state.json');data=json.loads(p.read_text())
    for j in data['jobs']:j['status']='COMPLETE'
    p.write_text(json.dumps(data));q.merge_state(p)
    assert all(j.status=='PENDING' for j in q.jobs())


def test_optimizer_resources_and_lazy_windows_dataset(tmp_path):
    import torch
    import pandas as pd
    from PIL import Image
    from utils.classifier_data import CropDataset,make_loader
    from torchvision.transforms import ToTensor
    Image.new('RGB',(8,8),'red').save(tmp_path/'fixture.png')
    ds=CropDataset(pd.DataFrame([{'crop_path':'fixture.png','y':1}]),tmp_path,ToTensor())
    assert not hasattr(ds,'images') and ds[0][0].shape==(3,8,8)
    loader=make_loader(ds,batch_size=1,shuffle=False,seed=42,workers=0)
    assert loader.persistent_workers is False
    from utils.detector_train import fixed_epoch_trainer
    cls=fixed_epoch_trainer(get_arch('YOLOv8n'))
    trainer=object.__new__(cls)
    assert trainer.validate()==({},0.) and trainer.final_eval() is None


def test_hpo_blocks_pause_without_unlocking_robustness(tmp_path):
    c=config_at(tmp_path);q=StudyQueue.from_cfg(c);q.plan(c,[get_arch('YOLOv8n')])
    base=q.claim_next('PC-01');q.finish(base.job_id,COMPLETE)
    hpo=q.claim_next('PC-01');q.mark_partial_block(hpo.job_id);q.computed(hpo.job_id,published=True)
    assert q.get(hpo.job_id).status==PENDING_REVIEW
    assert q.claim_next('PC-01') is None
    q.approve(hpo.job_id,operator='SYNTHETIC',evidence='fixture block')
    assert q.get(hpo.job_id).status=='PENDING'
    assert q.claim_next('PC-01').job_id==hpo.job_id


def test_assemble_requires_seven_verified_candidates_and_preserves_sources(tmp_path):
    from utils.experiments import assemble_cohort
    from utils.hashing import sha256_file
    c=config_at(tmp_path/'base');configs=[];snapshots={}
    for spec in archs():
        p=create_round(c,round_id=spec.slug+'_r1',architecture=spec.name,operator='SYNTHETIC',reason='fixture')
        configs.append(p);r=load_config(p);art=cfg_path(r,'paths.artifacts_dir')
        ck=art/'checkpoints'/spec.slug/'final.pt';ck.parent.mkdir(parents=True)
        ck.write_bytes(b'SYNTHETIC_NOT_MODEL_'+spec.slug.encode());snapshots[str(ck)]=sha256_file(ck)
        atomic_json(ck.with_name('final_meta.json'),dict(sha256=sha256_file(ck)))
        from utils.semantics import semantic_fingerprint
        atomic_json(art/'best_configs'/(spec.slug+'.json'),dict(config_id='fixture',experiment_fingerprint=fingerprint(r),semantic_fingerprint=semantic_fingerprint(r)))
        for stage in ['outer_validation','inference_thresholds']:
            atomic_json(art/stage/(spec.slug+'.json'),dict(checkpoint_sha256=sha256_file(ck)))
    with pytest.raises(ValueError):assemble_cohort(configs[:-1],cohort_id='missing',operator='SYNTHETIC',reason='fixture')
    p=assemble_cohort(configs,cohort_id='cohort1',operator='SYNTHETIC',reason='fixture')
    cohort=load_config(p);assert len(verify_round(cohort)['cohort'])==7
    from utils.hpo_results import load_best_config
    for spec in archs():assert load_best_config(cohort,spec)['config_id']=='fixture'
    assert all(sha256_file(Path(p))==digest for p,digest in snapshots.items())


def test_cli_parser_and_unicode_working_directory(tmp_path):
    from experiment_control import main,parser
    c=config_at(tmp_path/'ruta con espacios ñ');from utils.io import write_yaml
    path=tmp_path/'config con espacio ñ.yaml';write_yaml(path,c)
    created=main(['--config',str(path),'new-round','--round-id','r1','--architecture','ViT-B/16','--operator','SYNTHETIC','--reason','fixture'])
    result=main(['--config',created,'status'])
    assert result['round']['architecture']=='vit_b16'
    assert parser().parse_args(['compare','--configs',created,'--output',str(tmp_path/'comparación')]).action=='compare'


def test_figure_export_closes_even_when_save_fails(tmp_path,monkeypatch):
    import matplotlib.pyplot as plt
    import utils.visualization
    from utils.logging_utils import RunContext
    c=config_at(tmp_path);ctx=RunContext(c,'fixture',runs_dir=tmp_path/'runs',env_override={})
    fig,ax=plt.subplots();number=fig.number
    def fail(*args,**kwargs):raise OSError('SYNTHETIC_DISK_FAILURE')
    monkeypatch.setattr(utils.visualization,'save_figure',fail)
    try:
        with pytest.raises(OSError):ctx.save_figure(fig,'fixture')
        assert not plt.fignum_exists(number)
    finally:ctx._run_lock.__exit__(None,None,None)


def test_rtdetr_block_does_not_block_vit_jobs(tmp_path):
    from orchestration.worker import local_inputs_ready
    c=config_at(tmp_path);q=StudyQueue.from_cfg(c);q.plan(c,[get_arch('RT-DETR-R50'),get_arch('ViT-B/16')])
    job=q.claim_next('PC-03',ready_check=lambda j:local_inputs_ready(c,j,tmp_path/'runs'))
    assert job.architecture=='vit_b16'
    assert q.get('baseline:rtdetr_r50').status=='PENDING'


def test_force_baseline_rejects_overwrite_without_training(tmp_path,monkeypatch):
    import importlib,logging
    baseline=importlib.import_module('06_run_baseline')
    c=config_at(tmp_path);run=tmp_path/'run';run.mkdir();(run/'summary.json').write_text('{"fixture":"PRESERVE"}')
    ctx=SimpleNamespace(cfg=c,logger=logging.getLogger('fixture'),run_dir=run,set_seed=lambda s:None)
    monkeypatch.setattr(baseline,'preflight_training',lambda *a,**k:SimpleNamespace(dataset_hash='D',split_hash='S'))
    monkeypatch.setattr(baseline,'guard_not_frozen',lambda *a:None)
    with pytest.raises(Exception,match='--force no sobrescribe'):
        baseline.main(ctx,SimpleNamespace(architecture='YOLOv8n',folds=None,force=True))
    assert json.loads((run/'summary.json').read_text())=={'fixture':'PRESERVE'}


def test_resume_receipt_rejects_modified_checkpoint_or_protocol(tmp_path):
    from utils.resume import receipt,require_checkpoint
    c=config_at(tmp_path);checkpoint=tmp_path/'synthetic.pt';checkpoint.write_bytes(b'SYNTHETIC_NO_MODEL')
    spec=get_arch('YOLOv8n');receipt(checkpoint,spec,c)
    assert require_checkpoint(checkpoint,spec,c)==checkpoint.resolve()
    different=copy.deepcopy(c);different['detector']['baseline']['lr0']=.09
    with pytest.raises(RuntimeError):require_checkpoint(checkpoint,spec,different)
    checkpoint.write_bytes(b'ALTERED')
    with pytest.raises(RuntimeError):require_checkpoint(checkpoint,spec,c)


def test_publication_error_has_distinct_state_and_does_not_repeat_compute(tmp_path,monkeypatch):
    from orchestration import worker,exchange
    import logging
    c=config_at(tmp_path);q=StudyQueue.from_cfg(c);q.plan(c,[get_arch('YOLOv8n')]);job=q.claim_next('PC-01')
    calls=[]
    class Child:
        def __init__(self,*a,**k):calls.append('compute')
        def wait(self,timeout=None):return 0
    monkeypatch.setattr(worker.subprocess,'Popen',Child)
    def fail(*args,**kwargs):raise OSError('SYNTHETIC_PUBLICATION_FAILURE')
    monkeypatch.setattr(exchange,'publish_architecture',fail)
    ctx=SimpleNamespace(cfg=c,pc_id='PC-01',runs_base=tmp_path/'runs',logger=logging.getLogger('fixture'))
    args=SimpleNamespace(config='configs/default.yaml',set=[],runs_dir=None)
    assert worker._run_job(ctx,q,job,args,tmp_path)==4
    assert q.get(job.job_id).status==PUBLICATION_PENDING and calls==['compute']


def test_completed_folds_are_reused_only_with_matching_identity(tmp_path,monkeypatch):
    from utils import hpo_runner as h
    import pandas as pd,logging
    calls=[];c=config_at(tmp_path)
    monkeypatch.setattr(h,'record_training_source',lambda *a:None)
    monkeypatch.setattr(h,'load_crop_manifest',lambda c:pd.DataFrame())
    frame=pd.DataFrame({'y':[0,1]})
    monkeypatch.setattr(h,'load_fold_frames',lambda *a:(frame,frame))
    def compute(*args,**kwargs):
        calls.append(kwargs['seed'])
        return SimpleNamespace(score=.5,best_epoch=1,epochs_run=1,stopped_early=False,metrics={},
            ap_undefined=False,params_m=1.,gflops=1.,vram_peak_mb=None,micro_batch=2,
            train_seconds=0.,state_sha256='0'*64,state=None,history=pd.DataFrame({'epoch':[1],'train_loss':[1.]}))
    monkeypatch.setattr(h,'train_classifier',compute)
    ctx=SimpleNamespace(logger=logging.getLogger('fixture'),note_split_access=lambda *a:None,record_checkpoint=lambda *a,**k:None,run_id='SYNTHETIC')
    spec=get_arch('ResNet50');path=tmp_path/'run'
    first=h.evaluate_config(spec,{},c,ctx,path,folds=[0,1,2],seed=42,max_epochs=1,patience=1)
    again=h.evaluate_config(spec,{},c,ctx,path,folds=[0,1,2],seed=42,max_epochs=1,patience=1)
    assert calls==[42,42,42] and again.fold_scores==first.fold_scores==[.5,.5,.5]
    with pytest.raises(Exception,match='incompatible'):
        h.evaluate_config(spec,{'lr':.5},c,ctx,path,folds=[0,1,2],seed=42,max_epochs=1,patience=1)


def test_transfer_adopts_completed_stages_and_reviews_without_retraining(tmp_path,monkeypatch):
    from orchestration import transfer as t
    src=config_at(tmp_path/'src');dst=config_at(tmp_path/'dst')
    q=StudyQueue.from_cfg(src);q.plan(src,[get_arch('YOLOv8n')]);j=q.claim_next('PC-01');q.computed(j.job_id,published=True);q.approve(j.job_id,operator='SYNTHETIC',evidence='baseline fixture')
    monkeypatch.setattr(t,'pack_results',lambda c,p,*a,**k:Path(p).write_bytes(b'SYNTHETIC_SNAPSHOT'))
    monkeypatch.setattr(t,'unpack',lambda *a,**k:{})
    bundle=tmp_path/'x.zip';receipt=tmp_path/'receipt.json';confirmation=tmp_path/'confirmation.json'
    t.offer(src,'yolov8n','PC-01','PC-02',bundle,operator='SYNTHETIC',evidence='fixture')
    t.receive(dst,'PC-02',bundle,bundle.with_suffix('.offer.json'),receipt,operator='SYNTHETIC')
    # Recovery before activation is idempotent, including reviews.
    t.receive(dst,'PC-02',bundle,bundle.with_suffix('.offer.json'),receipt,operator='SYNTHETIC')
    t.confirm(src,'yolov8n',receipt,confirmation,operator='SYNTHETIC')
    t.confirm(src,'yolov8n',receipt,confirmation,operator='SYNTHETIC')
    t.activate(dst,'yolov8n',receipt,confirmation,operator='SYNTHETIC')
    target=StudyQueue.from_cfg(dst)
    assert target.get(j.job_id).status==COMPLETE
    assert target.claim_next('PC-02').kind=='hpo'
    with target._conn() as c:assert c.execute('SELECT count(*) FROM reviews').fetchone()[0]==1


def test_three_cpu_machine_simulations_really_overlap_without_gpu(tmp_path):
    from concurrent.futures import ThreadPoolExecutor
    from threading import Barrier,Event
    from utils.durable import FileLock
    from utils.guards import gpu_lock_path
    pairs=[('PC-01','YOLOv8n'),('PC-02','YOLO26l'),('PC-03','ViT-B/16')]
    queues=[]
    for pc,arch in pairs:
        c=config_at(tmp_path/pc);q=StudyQueue.from_cfg(c);q.plan(c,[get_arch(arch)]);queues.append((pc,c,q))
    barrier=Barrier(4);release=Event()
    def simulate(pc,c,q):
        with FileLock(gpu_lock_path(c,pc).with_suffix('.worker.guard')):
            job=q.claim_next(pc)
            barrier.wait(timeout=30)
            if not release.wait(timeout=30):raise RuntimeError('Fixture coordinator did not release')
            q.computed(job.job_id,published=True)
            return job.architecture
    with ThreadPoolExecutor(max_workers=3) as pool:
        futures=[pool.submit(simulate,*row) for row in queues]
        try:
            barrier.wait(timeout=30)
            assert sum(q.counts()['RUNNING'] for _,_,q in queues)==3
            assert all(q.claim_next(pc) is None for pc,_,q in queues)
        finally:release.set()
        assert {f.result(timeout=30) for f in futures}=={'yolov8n','yolo26l','vit_b16'}
    assert all(q.counts()[PENDING_REVIEW]==1 for _,_,q in queues)


def test_mirror_round_preserves_identity_across_unicode_local_paths(tmp_path):
    from utils.experiments import mirror_round,round_root
    source=config_at(tmp_path/'origen ñ');local=config_at(tmp_path/'destino con espacios')
    p=create_round(source,round_id='r1',architecture='YOLOv8n',operator='SYNTHETIC_SOURCE',reason='fixture',initial_weights={'yolov8n':'a'*64})
    original=load_config(p)
    mirrored=mirror_round(local,p,round_root(original)/'manifest.json',operator='SYNTHETIC_TARGET',reason='fixture mirror')
    target=load_config(mirrored)
    assert fingerprint(original)==fingerprint(target)
    assert cfg_path(original,'paths.artifacts_dir')!=cfg_path(target,'paths.artifacts_dir')
    assert target['experiment']['initial_weights']=={'yolov8n':'a'*64}


def test_ownership_or_mode_changes_cannot_bypass_round_identity(tmp_path):
    c=config_at(tmp_path);p=create_round(c,round_id='r1',architecture='YOLOv8n',operator='SYNTHETIC',reason='fixture');r=load_config(p)
    altered=copy.deepcopy(r);altered['pc_registry']['first_batch']['PC-01'].remove('YOLOv8n');altered['pc_registry']['first_batch']['PC-02'].append('YOLOv8n')
    with pytest.raises(RuntimeError):verify_round(altered)
    altered=copy.deepcopy(r);altered['experiment']['mode']='automatic'
    with pytest.raises(RuntimeError):verify_round(altered)


def test_worker_writes_scoped_round_job_log_without_path_errors(tmp_path,monkeypatch):
    from orchestration import worker,exchange
    import logging
    c=config_at(tmp_path);path=create_round(c,round_id='r1',architecture='YOLOv8n',operator='SYNTHETIC',reason='fixture');c=load_config(path)
    q=StudyQueue.from_cfg(c);q.plan(c,[get_arch('YOLOv8n')]);job=q.claim_next('PC-01');assert '/' in job.job_id
    class Child:
        def __init__(self,*a,**k):assert k['env']['BOTRYTIS_WORKER_JOB_ID']==job.job_id
        def wait(self,timeout=None):return 0
    monkeypatch.setattr(worker.subprocess,'Popen',Child)
    monkeypatch.setattr(exchange,'publish_architecture',lambda *a,**k:{})
    ctx=SimpleNamespace(cfg=c,pc_id='PC-01',runs_base=cfg_path(c,'paths.runs_dir'),logger=logging.getLogger('fixture'))
    args=SimpleNamespace(config=str(path),set=[],runs_dir=None)
    logs=tmp_path/'logs';logs.mkdir()
    assert worker._run_job(ctx,q,job,args,logs)==0
    assert len(list(logs.glob('*.log')))==1 and q.get(job.job_id).status==PENDING_REVIEW


def test_comparison_preserves_historical_identity_and_separates_objectives(tmp_path,monkeypatch):
    from experiment_control import compare
    from utils.experiments import round_root
    from utils.hashing import sha256_json
    configs=[];expected=[]
    for arch,rid,objective in [('YOLOv8n','det','SYNTHETIC_MAP'),('ResNet50','cls','SYNTHETIC_AP')]:
        c=config_at(tmp_path/rid);path=create_round(c,round_id=rid,architecture=arch,operator='SYNTHETIC',reason='fixture');configs.append(path)
        r=load_config(path);manifest=json.loads((round_root(r)/'manifest.json').read_text());expected.append(sha256_json(manifest['identity']))
        atomic_json(cfg_path(r,'paths.runs_dir')/r['experiment']['architecture']/'baseline/summary.json',dict(objective=objective,score_mean=.5,score_sd=.1,train_seconds=1,vram_peak_mb=None,fixture='SYNTHETIC_NOT_MODEL_METRICS'))
    # Changing current runtime code identity must not relabel historical fingerprints.
    from utils import experiments
    monkeypatch.setattr(experiments,'identity',lambda c:{'different_runtime':'SYNTHETIC'})
    out=tmp_path/'comparison';compare(configs,out)
    report=json.loads((out/'comparison.json').read_text())
    assert not report['methodologically_equivalent']
    assert [r['fingerprint'] for r in report['rows']]==expected
    assert all(not r['runtime_compatible'] for r in report['rows'])
    assert {r['objective'] for r in report['rows']}=={'SYNTHETIC_MAP','SYNTHETIC_AP'}
    assert (out/'comparison.png').stat().st_size>100
