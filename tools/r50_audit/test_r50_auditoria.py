"""Read-only audit regressions. No training, backward or optimizer.step."""
import ast,copy,hashlib,importlib.util,json,sqlite3,os
from pathlib import Path
from types import SimpleNamespace
import numpy as np
import pandas as pd
import pytest
ROOT=Path(__file__).parent
spec=importlib.util.spec_from_file_location('r50_audit',ROOT/'botrytis_r50_auditoria.py')
a=importlib.util.module_from_spec(spec);spec.loader.exec_module(a)

@pytest.fixture
def evidence():
    return json.loads((ROOT/'fixtures/metrics.json').read_text()),pd.read_csv(ROOT/'fixtures/fold_metrics.csv'),pd.read_csv(ROOT/'fixtures/history.csv')

def test_real_evidence(evidence):
    result=a.tables_audit(*evidence)
    assert result['score_mean']==pytest.approx(.4724513698899003)
    assert result['score_sd']==pytest.approx(.3253664113629352)
    assert [r['best_epoch'] for r in result['folds']]==[3,100,93]
    assert [r['epochs'] for r in result['folds']]==[23,100,100]
    assert result['folds'][0]['early_stopping_consistent']
    assert result['folds'][2]['zero_detection_metric_epochs']==[1,2,3]
    assert result['variable_fold_count']['after23']==2
    assert result['variable_fold_count']['last_available_fold_mean']==pytest.approx(.630283835)
    assert result['truncated100_curve_mean_before60']==pytest.approx(.3746801066666667)
    assert result['truncation_is_not_60_epoch_training'] is True

@pytest.mark.parametrize('value',[np.nan,np.inf,-np.inf,-1])
def test_nonfinite_or_negative_primary_loss_blocks(evidence,value):
    evidence[2].loc[0,'val/bbox_loss']=value
    with pytest.raises(a.Stop):a.tables_audit(*evidence)

@pytest.mark.parametrize('column',['val/bbox_loss','train/vfl_loss','metrics/mAP50-95(B)'])
def test_missing_guaranteed_official_columns_are_not_imputed(evidence,column):
    with pytest.raises(a.MissingEvidence):a.tables_audit(evidence[0],evidence[1],evidence[2].drop(columns=column))

def test_high_or_low_metric_is_not_a_failure(evidence):
    result=a.tables_audit(*evidence)
    assert result['folds'][0]['last_mAP50_95']<.05
    assert result['numeric_integrity']

def test_component_losses_not_forced_to_decrease(evidence):
    result=a.tables_audit(*evidence)
    vfl=result['folds'][0]['losses']['train/vfl_loss']
    assert vfl['last']>vfl['first']

def test_truncated_history_blocks(evidence):
    with pytest.raises(a.Stop):a.tables_audit(evidence[0],evidence[1],evidence[2].iloc[:-1])

def test_duplicate_epoch_blocks(evidence):
    h=pd.concat([evidence[2],evidence[2].iloc[:1]],ignore_index=True)
    with pytest.raises(a.Stop):a.tables_audit(evidence[0],evidence[1],h)

def test_scrambled_epochs_blocks(evidence):
    h=evidence[2].copy();h.iloc[[0,1]]=h.iloc[[1,0]].to_numpy()
    with pytest.raises(a.Stop):a.tables_audit(evidence[0],evidence[1],h)

@pytest.mark.parametrize('field,value',[('seed',17),('patience',30),('dataset_hash','changed'),('score_mean',.9),('best_epochs',[3,99,93])])
def test_incompatible_summary_blocks(evidence,field,value):
    evidence[0][field]=value
    with pytest.raises(a.Stop):a.tables_audit(*evidence)

def test_contradictory_alias_blocks(evidence):
    evidence[2].loc[0,'map50_95']=.9
    with pytest.raises(a.Stop):a.tables_audit(*evidence)

def test_malformed_csv_not_accepted(tmp_path,evidence):
    p=tmp_path/'broken.csv';p.write_text('fold,epoch,train_loss\n0,1,3\n0,2,"unterminated')
    with pytest.raises(pd.errors.ParserError):pd.read_csv(p)

def test_plot_preserves_originals(tmp_path,evidence):
    before={p.name:a.digest(p) for p in (ROOT/'fixtures').iterdir()}
    a.plot_by_fold(evidence[2],tmp_path)
    assert (tmp_path/'curvas_por_fold.png').stat().st_size>10000
    assert (tmp_path/'curvas_por_fold.svg').is_file()
    assert before=={p.name:a.digest(p) for p in (ROOT/'fixtures').iterdir()}

def queue_fixture(tmp_path):
    db=tmp_path/'queue.sqlite'
    with sqlite3.connect(db) as c:
        c.execute('CREATE TABLE jobs(job_id TEXT, kind TEXT, architecture TEXT, payload TEXT, depends_on TEXT,status TEXT)')
        c.execute('CREATE TABLE context(key TEXT,value TEXT)')
        c.execute('INSERT INTO context VALUES (?,?)',('fingerprint',a.EXPECTED_FP))
        payload=json.dumps(dict(fingerprint=a.EXPECTED_FP,round_id='r50vd_01'))
        for kind,state,dep in [('baseline','PENDING_REVIEW',[]),('hpo','PENDING',['baseline'])]:
            c.execute('INSERT INTO jobs VALUES (?,?,?,?,?,?)',(kind,kind,a.MODEL,payload,json.dumps(dep),state))
    return db

def test_queue_readonly_no_approval(tmp_path,monkeypatch):
    db=queue_fixture(tmp_path)
    import utils.config_loader
    monkeypatch.setattr(utils.config_loader,'cfg_path',lambda cfg,key:tmp_path)
    before=db.read_bytes();jobs=a.read_jobs({})
    assert jobs['baseline']['status']=='PENDING_REVIEW' and jobs['hpo']['status']=='PENDING'
    assert db.read_bytes()==before
    with a.read_only_sqlite(db) as c:
        with pytest.raises(sqlite3.OperationalError):c.execute("UPDATE jobs SET status='COMPLETE'")

def test_missing_queue_not_created(tmp_path):
    p=tmp_path/'absent.sqlite'
    with pytest.raises(a.MissingEvidence):a.read_only_sqlite(p)
    assert not p.exists()

def test_wrong_queue_identity_blocks(tmp_path,monkeypatch):
    db=queue_fixture(tmp_path)
    with sqlite3.connect(db) as c:c.execute("UPDATE context SET value='foreign'")
    import utils.config_loader
    monkeypatch.setattr(utils.config_loader,'cfg_path',lambda cfg,key:tmp_path)
    before=db.read_bytes()
    with pytest.raises(a.Stop):a.read_jobs({})
    assert db.read_bytes()==before

def test_native_prepared_diagnostics_redirected_only(tmp_path,monkeypatch):
    import utils.config_loader,utils.prepared,utils.isolation
    original=tmp_path/'original';original.mkdir();(original/'split_manifest.csv').write_text('sealed');(original/'dataset_hash.json').write_text('{}')
    cfg=dict(paths=dict(artifacts_dir=str(original)))
    monkeypatch.setattr(utils.config_loader,'cfg_path',lambda cfg,key:Path(cfg['paths']['artifacts_dir']))
    def native(cfg,with_folds):
        dest=Path(cfg['paths']['artifacts_dir']);assert dest!=original
        (dest/'pairing_audit.csv').write_text('new diagnostics');return 'data'
    monkeypatch.setattr(utils.prepared,'load_prepared_data',native)
    monkeypatch.setattr(utils.isolation,'require_near_duplicate_review',lambda cfg,data:None)
    before={p.name:a.digest(p) for p in original.iterdir()}
    local,data=a.readonly_prepared(cfg,tmp_path/'scratch')
    assert data=='data' and cfg['paths']['artifacts_dir']==str(original)
    assert (tmp_path/'scratch/pairing_audit.csv').is_file()
    assert before=={p.name:a.digest(p) for p in original.iterdir()}

def test_no_hpo_or_approval_or_training_entrypoint():
    tree=ast.parse((ROOT/'botrytis_r50_auditoria.py').read_text())
    calls=[n for n in ast.walk(tree) if isinstance(n,ast.Call)]
    attrs=[n.func.attr for n in calls if isinstance(n.func,ast.Attribute)]
    assert not set(attrs)&{'approve','computed','claim_next','plan','train_detector','optimize','backward','step','selftest'}
    assert 'audit' in (ROOT/'botrytis_r50_auditoria.py').read_text()

def test_exact_adapter_pinned():
    assert a.digest(ROOT/'botrytis_r50_oficial_pinned.py')==a.ADAPTER_SHA
    adapter=a.load_adapter()
    assert adapter.REV=='df2a96c8c3a098567a1149bf0248da6efe6d495d'
    assert adapter.WEIGHT_SHA=='dfaafa5a3d29958a360c35f1fad553a6f287bfae9967e8ee007e65cff21ddf45'

def test_official_model_strict_load_without_forward_or_training():
    import torch
    source=os.environ.get('BT_R50_SOURCE');weights=os.environ.get('BT_R50_WEIGHTS')
    if not source or not weights:pytest.skip('Set BT_R50_SOURCE/BT_R50_WEIGHTS to verified official resources')
    adapter=a.load_adapter();threads=torch.get_num_threads();torch.set_num_threads(2)
    try:
        model=adapter.build_model(Path(source),Path(weights))
        assert sum(p.numel() for p in model.parameters())==42729027
        assert model.decoder.num_classes==1 and model.decoder.enc_score_head.out_features==1
        assert not [n for n,p in model.named_parameters() if n.startswith('backbone.') and n.endswith('.bias') and p.requires_grad]
        assert all(p.grad is None for p in model.parameters())
    finally:torch.set_num_threads(threads)


def test_optuna_status_readonly(tmp_path,monkeypatch):
    import optuna
    import utils.optuna_storage,utils.config_loader,utils.experiments
    from utils.architectures import get_arch
    cfg=dict(optuna=dict(study_prefix='botrytis'),experiment=dict(experiment_id='botrytis_v4',round_id='r50vd_01'))
    db=tmp_path/(get_arch(a.MODEL).study_name('botrytis')+'.db')
    study=optuna.create_study(storage='sqlite:///'+str(db),study_name=utils.optuna_storage.study_name(get_arch(a.MODEL),cfg),direction='maximize')
    study.set_user_attr('experiment_identity',{'local':'fixture'})
    study.add_trial(optuna.trial.create_trial(value=.5,user_attrs=dict(config_id='cfg_1',fold_scores=[.4,.5,.6])))
    study.ask()
    monkeypatch.setattr(utils.config_loader,'cfg_path',lambda cfg,key:tmp_path)
    monkeypatch.setattr(utils.experiments,'identity',lambda cfg:{'local':'fixture'})
    before=db.read_bytes();result=a.study_status(cfg)
    assert result['used']==1 and result['running']==[1]
    assert result['trials'][0]['fold_scores']==[.4,.5,.6]
    assert before==db.read_bytes()


def test_own_fold_receipts_and_checkpoint_validation_readonly(tmp_path,evidence,monkeypatch):
    import torch
    import utils.config_loader,utils.hashing,utils.search_spaces
    m,f,h=evidence;model=torch.nn.Linear(2,1);adapter=a.load_adapter();cfg={};code=dict(code_hash='code',git_commit='commit');hp=m['hp']
    monkeypatch.setattr(utils.config_loader,'load_config',lambda p:cfg)
    monkeypatch.setattr(utils.hashing,'scientific_config_hash',lambda cfg:'science')
    monkeypatch.setattr(utils.search_spaces,'baseline_hyperparameters',lambda spec,cfg:hp)
    def receipt_check(path,model,cfg,**kwargs):
        meta=json.loads(Path(str(path)+'.resume.json').read_text())
        if meta.get('sha256')!=a.digest(path):raise a.Stop('Bad hash')
        return path,meta
    monkeypatch.setattr(a,'validate_checkpoint',receipt_check)
    a.atomic_json(tmp_path/'metrics.json',m);a.atomic_json(tmp_path/'summary.json',m)
    a.atomic_json(tmp_path/'metadata.json',dict(status='COMPLETE',pc_id=a.PC,**code));(tmp_path/'config.yaml').write_text('original')
    f.to_csv(tmp_path/'fold_metrics.csv',index=False);h.to_csv(tmp_path/'history.csv',index=False)
    checkpoints=[]
    for fold in range(3):
        out=tmp_path/f'fold{fold}';out.mkdir();a.atomic_json(out/'adapter.json',dict(format=adapter.FORMAT,source_revision=adapter.REV,pretrained_sha256=adapter.WEIGHT_SHA,adapter_sha256=a.ADAPTER_SHA));history=h[h.fold==fold].reset_index(drop=True);history.to_csv(out/'completed_history.csv',index=False)
        history.drop(columns=['fold']).to_csv(out/'results.csv',index=False)
        row=f[f.fold==fold].iloc[0].to_dict();row['score']=m['fold_scores'][fold]
        sig=a.fold_signature(a.EXPECTED_FP,a.MODEL,hp,42,fold,100,20)
        a.atomic_json(out/'completed_fold.json',dict(signature=sig,history_sha256=a.digest(out/'completed_history.csv'),row=row))
        internal=adapter.canonical(dict(experiment=a.EXPECTED_FP,hp=hp,seed=42,fold=fold,epochs=100,patience=20,fixed=False))
        best=out/'best.pt';payload=dict(format=adapter.FORMAT,signature=internal,experiment_fingerprint=a.EXPECTED_FP,official_revision=adapter.REV,pretrained_sha256=adapter.WEIGHT_SHA,best_epoch=row['best_epoch'],model=model.state_dict(),metrics={'metrics/mAP50-95(B)':row['score']})
        torch.save(payload,best);a.atomic_json(Path(str(best)+'.resume.json'),dict(sha256=a.digest(best)))
        checkpoints.append(dict(role=f'fold{fold}_best',sha256=a.digest(best),weights_discarded=False))
        state=out/'cp'/('e%04d.pt'%row['epochs_run']);state.parent.mkdir()
        torch.save(dict(format=adapter.FORMAT,signature=internal,epoch=int(row['epochs_run']),best_epoch=int(row['best_epoch']),best_score=row['score'],history=history.to_dict('records'),model=model.state_dict(),ema=model.state_dict(),best_state=model.state_dict(),gradients={},optimizer=dict(param_groups=[dict(backbone=True,initial_lr=.0001,lr=.000001,param_group='weight',weight_decay=.0001),dict(backbone=False,initial_lr=.001,lr=.00001,param_group='bias',weight_decay=0)]),scheduler={},scaler={},rng={},stopped=bool(row['stopped_early'])),state)
        a.atomic_json(Path(str(state)+'.resume.json'),dict(sha256=a.digest(state),extra=dict(adapter_sha256=a.ADAPTER_SHA)))
    a.atomic_json(tmp_path/'checkpoint_metadata.json',dict(checkpoints=checkpoints,**code))
    before=a.frozen_inventory(tmp_path)
    result,_,_,verified=a.baseline_local(cfg,tmp_path,adapter,model,code)
    assert result['numeric_integrity'] and len(verified)==3
    assert before==a.frozen_inventory(tmp_path)
    state=tmp_path/'fold1/cp/e0100.pt';state.write_bytes(state.read_bytes()[:-25])
    with pytest.raises(a.Stop,match='Bad hash'):a.baseline_local(cfg,tmp_path,adapter,model,code)


def test_missing_evidence_results_in_D_without_queue_changes(tmp_path,monkeypatch):
    import utils.config_loader
    project=tmp_path/'project';store=tmp_path/'r50';baseline=store/'r/rtdetr_r50/baseline';baseline.mkdir(parents=True)
    (baseline/'original.csv').write_text('preserve')
    art=tmp_path/'art';art.mkdir();config=tmp_path/'config.yaml';config.write_text('original')
    monkeypatch.setattr(utils.config_loader,'cfg_path',lambda cfg,key:art)
    monkeypatch.setattr(a,'processes',lambda:[])
    def missing(project):raise a.MissingEvidence('No metadata original')
    monkeypatch.setattr(a,'readonly_code',missing)
    out=store/'audit/check';before=a.frozen_inventory(baseline)
    assert a.perform_audit(project,store,config,{},out)==2
    result=a.read(out/'audit.json')
    assert result['decision']=='D_EVIDENCIA_INSUFICIENTE'
    assert result['queue_modified'] is False and result['human_approval_registered'] is False
    assert result['missing']==['No metadata original'] and before==a.frozen_inventory(baseline)


@pytest.mark.parametrize('bad',[.001,float('nan'),-1])
def test_bad_backbone_group_lr_blocks(bad):
    groups=[dict(backbone=True,initial_lr=bad,lr=.00001,param_group='weight',weight_decay=.0001),dict(backbone=False,initial_lr=.001,lr=.00001,param_group='bias',weight_decay=0)]
    with pytest.raises(a.Stop):a.check_optimizer_state(dict(optimizer=dict(param_groups=groups),scaler={}),dict(lr0=.001,backbone_lr_ratio=.1,weight_decay=.0001))


def test_original_input_lists_match_saved_fold_membership(tmp_path,monkeypatch):
    import yaml
    import utils.config_loader
    cfg=dict(detection_dataset=dict(split_dir_names=dict(train='train')))
    monkeypatch.setattr(utils.config_loader,'cfg_path',lambda cfg,key:tmp_path/'derived')
    table=pd.DataFrame(dict(image_id=['a','b','c'],outer_split=['train']*3,inner_fold_det=[0,1,2],image_file=['a.png','b.png','c.png']))
    for fold in range(3):
        folder=tmp_path/f'fold{fold}';folder.mkdir();data=dict(nc=1,names={0:'fruit'})
        for role in ('train','val'):
            selected=table[table.inner_fold_det==fold] if role=='val' else table[table.inner_fold_det!=fold]
            path=folder/(role+'.txt');path.write_text(''.join(str((tmp_path/'derived/images/train'/f).resolve())+'\n' for f in selected.image_file))
            data[role]=str(path)
        (folder/'data.yaml').write_text(yaml.safe_dump(data))
    result=a.verify_original_lists(cfg,tmp_path,table)
    assert all(r['train']['images']==2 and r['val']['images']==1 for r in result)
    (tmp_path/'fold0/val.txt').write_text(str(tmp_path/'unrelated.png'))
    with pytest.raises(a.Stop,match='Lista original'):a.verify_original_lists(cfg,tmp_path,table)
