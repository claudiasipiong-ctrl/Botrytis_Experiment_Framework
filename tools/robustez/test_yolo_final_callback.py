"""Regression: genuine final_eval callback versus truncated/altered evidence.

All data are synthetic. The trainer callback test uses actual 8.4.170 code on
fake trainer attributes, with no network, checkpoint deserialization or GPU.
"""
import contextlib
import hashlib
import importlib.util
import json
from pathlib import Path
from types import SimpleNamespace
import sys

import numpy as np
import pandas as pd
import pytest

HERE = Path(__file__).parent
spec_ = importlib.util.spec_from_file_location('final_callback_controller', HERE/'botrytis_robustez_reanudar.py')
controller = importlib.util.module_from_spec(spec_); spec_.loader.exec_module(controller)
bt = controller.bt
from utils.config_loader import load_config, cfg_path, PROJECT_ROOT
from utils.experiments import fingerprint
from utils.io import write_yaml
from utils.architectures import get_arch


def pair(tmp, *, final_wins=False, last_before_cap=False):
    folder = tmp/'yolov8n/hpo/trial_001/fold0'; folder.mkdir(parents=True)
    cfg = load_config(); cfg['experiment'].update(architecture='yolov8n', round_id='fixture', experiment_id='botrytis_v4')
    n = 59 if last_before_cap else 23
    peak_epoch = n - 20
    values = [.5 + i*.001 for i in range(n)]
    values[peak_epoch-1] = .900123456789
    raw = pd.DataFrame({'epoch':range(1,n+1), 'train/box_loss':np.linspace(1.,.2,n),
        'train/cls_loss':np.linspace(2.,.3,n), 'train/dfl_loss':np.linspace(1.2,.7,n),
        'metrics/mAP50-95(B)':values, 'metrics/mAP50(B)':[min(v+.04,1.) for v in values],
        'metrics/precision(B)':[.9]*n,'metrics/recall(B)':[.85]*n,
        'val/box_loss':[.5]*n, 'val/cls_loss':[.6]*n, 'val/dfl_loss':[.8]*n})
    # Native raw precision is %.6g, while the signed callback is %.12g.
    raw.to_csv(folder/'results.csv',index=False,float_format='%.6g')
    signed = raw.copy(); extra = dict(signed.iloc[-1]); extra['epoch'] = n+1
    extra['metrics/mAP50-95(B)'] = .901 if final_wins else .899
    extra['metrics/mAP50(B)'] = .95
    for c in ['val/box_loss','val/cls_loss','val/dfl_loss']: extra[c] = np.nan
    signed.loc[len(signed)] = extra
    signed['fold']=0; signed['map50_95']=signed['metrics/mAP50-95(B)']; signed['map50']=signed['metrics/mAP50(B)']
    signed['train_loss']=np.nan; signed['val_loss']=np.nan
    signed.to_csv(folder/'completed_history.csv',index=False,float_format='%.12g')
    selected_epoch = n+1 if final_wins else peak_epoch
    hp = dict(optimizer='AdamW',lr0=.001)
    row=dict(fold=0,score=float(signed['map50_95'].max()),best_epoch=selected_epoch,
             epochs_run=n+1,stopped_early=n+1<60)
    bt.atomic_json(folder/'completed_fold.json',dict(signature=bt.fold_signature(fingerprint(cfg),'yolov8n',hp,42,0,60,20),
        row=row,history_sha256=bt.digest(folder/'completed_history.csv')))
    bt.atomic_json(folder.parent/'metadata.json',dict(status='COMPLETE',git_commit=controller.COMMIT,code_hash=controller.CODE,
        dataset_hash=controller.DATA,split_hash=controller.SPLIT,extra=dict(hp=hp),seed=42))
    bt.atomic_json(folder.parent/'environment.json',dict(ultralytics='8.4.170',dataset_hash=controller.DATA,split_hash=controller.SPLIT))
    write_yaml(folder.parent/'config.yaml',cfg)
    return folder, cfg, row


@pytest.mark.parametrize('final_wins',[False,True])
@pytest.mark.parametrize('last_before_cap',[False,True])
def test_verified_extra_callback_preserves_all_originals(tmp_path, final_wins, last_before_cap):
    folder,cfg,row = pair(tmp_path,final_wins=final_wins,last_before_cap=last_before_cap)
    original = {p:p.read_bytes() for p in tmp_path.rglob('*') if p.is_file()}
    result = bt.curve_evidence(folder/'completed_history.csv',True,row['best_epoch'])
    assert result['final_callback']['extra_event_is_training_epoch'] is False
    assert result['best_checkpoint_training_epoch'] == len(pd.read_csv(folder/'results.csv'))-20
    assert result['epochs'] == len(pd.read_csv(folder/'results.csv'))
    assert result['best_value'] == pytest.approx(row['score'])
    assert result['signed_history_sha256'] == bt.digest(folder/'completed_history.csv')
    assert all(p.read_bytes()==data for p,data in original.items())


@pytest.mark.parametrize('damage', ['two_missing','missing_middle','changed_metric','changed_train',
    'changed_extra_train','nan_objective','wrong_version','wrong_hash','wrong_signature',
    'missing_producer','missing_receipt','wrong_patience','wrong_cap','mixed_data','receipt_epochs'])
def test_corrupt_or_unexplained_discrepancies_still_stop(tmp_path,damage):
    folder,cfg,row = pair(tmp_path)
    raw=folder/'results.csv'; signed=folder/'completed_history.csv'
    frame=pd.read_csv(raw)
    if damage=='two_missing':frame=frame.iloc[:-1]
    if damage=='missing_middle':frame=frame.drop(index=4)
    if damage=='changed_metric':frame.loc[0,'metrics/mAP50-95(B)']=.1
    if damage=='changed_train':frame.loc[0,'train/box_loss']=10
    if damage=='changed_extra_train':
        s=pd.read_csv(signed);s.loc[len(s)-1,'train/box_loss']=20;s.to_csv(signed,index=False)
        r=bt.read(folder/'completed_fold.json');r['history_sha256']=bt.digest(signed);bt.atomic_json(folder/'completed_fold.json',r)
    if damage=='nan_objective':frame.loc[0,'metrics/mAP50-95(B)']=np.nan
    if damage=='wrong_version':
        r=bt.read(folder.parent/'environment.json');r['ultralytics']='different';bt.atomic_json(folder.parent/'environment.json',r)
    if damage=='wrong_hash':
        r=bt.read(folder/'completed_fold.json');r['history_sha256']='bad';bt.atomic_json(folder/'completed_fold.json',r)
    if damage=='wrong_signature':
        r=bt.read(folder/'completed_fold.json');r['signature']='bad';bt.atomic_json(folder/'completed_fold.json',r)
    if damage=='missing_producer':(folder.parent/'metadata.json').unlink()
    if damage=='missing_receipt':(folder/'completed_fold.json').unlink()
    if damage=='wrong_patience':
        cfg['detector']['early_stopping_patience']=30
        write_yaml(folder.parent/'config.yaml',cfg)
    if damage=='wrong_cap':
        cfg['detector']['max_epochs']=23;write_yaml(folder.parent/'config.yaml',cfg)
    if damage=='mixed_data':
        r=bt.read(folder.parent/'environment.json');r['dataset_hash']='other';bt.atomic_json(folder.parent/'environment.json',r)
    if damage=='receipt_epochs':
        r=bt.read(folder/'completed_fold.json');r['row']['epochs_run']=999;bt.atomic_json(folder/'completed_fold.json',r)
    frame.to_csv(raw,index=False,float_format='%.6g')
    original = {p:p.read_bytes() for p in tmp_path.rglob('*') if p.is_file()}
    with pytest.raises(bt.Stop):bt.curve_evidence(signed,True,row['best_epoch'])
    assert all(p.read_bytes()==data for p,data in original.items())


@pytest.mark.parametrize('early',[False,True])
def test_actual_pinned_ultralytics_final_eval_explains_extra_event(tmp_path,monkeypatch,early):
    import utils.detector_train as dt
    import utils.logging_utils as logging
    import ultralytics.engine.trainer as trainer
    monkeypatch.setattr(logging,'log_epoch',lambda *a,**k:None)
    monkeypatch.setattr(trainer,'torch_distributed_zero_first',lambda *a:contextlib.nullcontext())
    monkeypatch.setattr(trainer,'strip_optimizer',lambda *a,**k:{})
    best=tmp_path/'best.pt';best.write_bytes(b'SYNTHETIC_NOT_DESERIALIZED')
    metrics={'metrics/mAP50-95(B)':.9,'metrics/mAP50(B)':.95,'metrics/precision(B)':.9,'metrics/recall(B)':.8}
    class Validator:
        args=SimpleNamespace(plots=False,compile=False)
        metrics=SimpleNamespace(box=SimpleNamespace(all_ap=np.array([[.9]])))
        def __call__(self,**kw):return dict(metrics)
    fake=SimpleNamespace(best=best,last=tmp_path/'absent.pt',validator=Validator(),args=SimpleNamespace(plots=False),
        epoch=3,epochs=60 if early else 4,metrics=dict(metrics),fitness=.9,tloss=None,
        label_loss_items=lambda _:{'train/box_loss':.5},save_dir=tmp_path)
    history=[];final={}
    fake.run_callbacks=lambda event:dt._collect(fake,history,final,None,0,lambda *a:None,False)
    trainer.BaseTrainer.final_eval(fake)
    assert fake.epoch==3
    if early:
        assert len(history)==1 and history[0]['epoch']==5 and final=={}
    else:
        assert history==[] and final['metrics/mAP50-95(B)']==.9


def test_no_false_positive_for_a_genuinely_missing_csv_record_without_producer(tmp_path):
    folder,cfg,row=pair(tmp_path)
    # A bare row-count difference is insufficient to declare final_eval.
    (folder.parent/'metadata.json').unlink()
    with pytest.raises(bt.Stop,match='ausente'):
        bt.curve_evidence(folder/'completed_history.csv',True,row['best_epoch'])


def test_real_native_aggregation_uses_derived_training_epochs_without_rewriting_sources(tmp_path,monkeypatch):
    folder,cfg,row=pair(tmp_path,final_wins=True)
    for key,value in dict(runs_dir='r',results_dir='e',best_configs_dir='best').items():cfg['paths'][key]=str(tmp_path/value)
    config_path=tmp_path/'config.yaml';write_yaml(config_path,cfg)
    spec=get_arch('yolov8n');rob=cfg_path(cfg,'paths.runs_dir')/'yolov8n/robustness'
    cands=[dict(config_id='cfg_'+str(i),hp=dict(lr0=.001*(i+1)),trial_number=i,value=.7+i*.01) for i in range(3)]
    for c in cands:
        for seed in (17,42,73):
            out=rob/c['config_id']/f'seed_{seed}';out.mkdir(parents=True)
            bt.atomic_json(out/'result.json',dict(config_id=c['config_id'],hp=c['hp'],seed=seed,
                score=c['value'],fold_scores=[c['value']]*3,best_epochs=[24]*3))
    originals={p:p.read_bytes() for p in rob.rglob('result.json')}
    native=bt.load_native(PROJECT_ROOT/'src/08_repeat_top_configs.py','_actual_08_epoch_fixture')
    native.plot_robustness=lambda *a,**k:None
    class Context:
        pc_id='PC-01';hashes={}
        runs_base=cfg_path(cfg,'paths.runs_dir')
        logger=SimpleNamespace(info=lambda *a:None)
        def save_figure(self,*a,**k):pass
        def register_artifact(self,*a):pass
        def log_metrics(self,*a):pass
    captured=[]
    def cli(*a,**kw):
        captured.append(native.aggregate(Context(),spec,cfg,cands,rob))
        return 0
    native.run_cli=cli
    monkeypatch.setattr(bt,'load_native',lambda *a:native)
    monkeypatch.setattr(bt.PostHooks,'install_post',lambda *a:None)
    monkeypatch.setattr(bt,'curve_evidence',lambda *a,**k:dict(best_checkpoint_training_epoch=3))
    monkeypatch.setattr(sys,'argv',list(sys.argv))
    assert bt.child_entry('aggregate',PROJECT_ROOT,['--config',str(config_path),'--architecture','yolov8n','--aggregate-only'],tmp_path/'state')==0
    best=captured[0]
    assert best['final_epochs']==3 and best['score_mean']==pytest.approx(.72)
    assert len(best['post_training_callback_epoch_interpretations'])==9
    assert all(p.read_bytes()==data for p,data in originals.items())
