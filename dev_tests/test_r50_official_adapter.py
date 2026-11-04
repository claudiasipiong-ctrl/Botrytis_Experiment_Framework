import importlib.util
import importlib
import contextlib
import logging
import os
from pathlib import Path
import shutil
from types import SimpleNamespace

import numpy as np
from PIL import Image
import pytest
import torch

source=Path(__file__).with_name('botrytis_r50_oficial.py')
if not source.is_file():source=Path(__file__).resolve().parents[1]/'tools/botrytis_r50_oficial.py'
spec=importlib.util.spec_from_file_location('r50_adapter',source);bt=importlib.util.module_from_spec(spec);spec.loader.exec_module(bt)
from utils.config_loader import load_config,PROJECT_ROOT
from utils.architectures import get_arch
from utils.io import write_json

torch.set_num_threads(4)
OFFICIAL=Path(os.environ.get('R50_TEST_SOURCE',str(Path.home()/'r50/official')))
WEIGHT=Path(os.environ.get('R50_TEST_WEIGHT',str(Path.home()/'r50/weights/rtdetr_r50vd_6x_coco_from_paddle.pth')))
ARCHIVE=Path(os.environ.get('R50_TEST_ARCHIVE',str(Path.home()/'r50/official.zip')))
pytestmark=pytest.mark.skipif(not OFFICIAL.is_dir() or not WEIGHT.is_file() or not ARCHIVE.is_file(),reason='Optional official-backend integration assets are required; no implicit downloads in tests')


def cfg(tmp_path):
    c=load_config();c['project']['profile']='smoke';c['compute']['pc_id']='PC-03'
    for key,path in [('artifacts_dir','a'),('queue_dir','q'),('runs_dir','r'),('results_dir','e')]:c['paths'][key]=str(tmp_path/path)
    c['experiment'].update(architecture='rtdetr_r50',round_id='r50_test',experiment_id='botrytis_v4')
    c['weights']['r50_source']=str(OFFICIAL)
    c['detector'].update(workers=0,batch_size=2,nominal_batch_size=4,image_size=640)
    c['detector']['baseline'].update(warmup_epochs=0)
    return c


def test_download_existing_hash_refuses_mutation(tmp_path):
    p=tmp_path/'existing';p.write_bytes(b'KEEP')
    with pytest.raises(RuntimeError,match='incompatible'):bt.download('https://example.com/',p,'0'*64,False)
    assert p.read_bytes()==b'KEEP'


def test_no_network_permission_never_fetches(tmp_path,monkeypatch):
    monkeypatch.setattr(bt.urllib.request,'urlopen',lambda *a,**k:pytest.fail('Unexpected network'))
    with pytest.raises(RuntimeError,match='allow-network'):bt.download(bt.WEIGHT_URL,tmp_path/'missing','0'*64,False)


def test_official_source_extraction_and_tamper_check_against_archive(tmp_path):
    shutil.copyfile(ARCHIVE,tmp_path/'official.zip')
    root=bt.acquire_source(tmp_path,False)
    assert bt.acquire_source(tmp_path,False)==root
    f=root/'src/nn/backbone/presnet.py';f.write_text(f.read_text()+'\n# changed\n')
    marker=bt.read(root/'source.json');marker['files']['src/nn/backbone/presnet.py']=bt.sha(f)
    write_json(root/'source.json',marker)
    with pytest.raises(RuntimeError,match='modificada'):bt.acquire_source(tmp_path,False)


def test_real_checkpoint_size_and_digest():
    assert WEIGHT.stat().st_size==bt.WEIGHT_SIZE and bt.sha(WEIGHT)==bt.WEIGHT_SHA


def test_full_official_coco_load_then_only_fruit_heads_change():
    original=torch.load(WEIGHT,map_location='cpu',weights_only=False)['ema']['module']
    model=bt.build_model(OFFICIAL,WEIGHT)
    current=model.state_dict()
    allowed=lambda n:n.startswith(('decoder.enc_score_head.','decoder.dec_score_head.','decoder.denoising_class_embed.'))
    changed=[]
    for name,value in current.items():
        if allowed(name):changed.append(name)
        else:assert torch.equal(value,original[name]),name
    assert len(changed)==15 and model.decoder.num_classes==1
    assert model.decoder.denoising_class_embed.weight.shape==(2,256)
    assert model.multi_scale is None and all(p.requires_grad for p in model.backbone.parameters())
    assert bt.sha(WEIGHT)==bt.WEIGHT_SHA


def test_real_backend_forward_backward_and_finite_gradient():
    result=bt.selftest(OFFICIAL,WEIGHT,torch.device('cpu'))
    assert result['status']=='PASS' and result['backward_size']==160 and result['strict_coco_load']
    assert result['optimizer_steps']==0 and result['synthetic']


@pytest.mark.parametrize('n',[0,1,3])
def test_real_official_criterion_handles_background_and_fruit(n):
    loss_fn=bt.criterion(OFFICIAL)
    logits=torch.zeros(2,300,1,requires_grad=True)
    boxes=torch.full((2,300,4),.5,requires_grad=True)
    targets=[{'labels':torch.zeros(n,dtype=torch.long),'boxes':torch.full((n,4),.5)} for _ in range(2)]
    losses=loss_fn({'pred_logits':logits,'pred_boxes':boxes},targets)
    assert set(losses)=={'loss_vfl','loss_bbox','loss_giou'}
    total=sum(losses.values());assert torch.isfinite(total);total.backward()
    assert torch.isfinite(logits.grad).all() and torch.isfinite(boxes.grad).all()


def dataset_fixture(tmp_path):
    images=tmp_path/'images/train';labels=tmp_path/'labels/train';images.mkdir(parents=True);labels.mkdir(parents=True)
    files=[]
    for i in range(4):
        p=images/f'im{i}.png';Image.new('RGB',(80,40),(230,20,10)).save(p)
        (labels/(p.stem+'.txt')).write_text('0 0.5 0.5 0.5 0.5\n' if i%2 else '')
        files.append(p)
    for role,ids in [('train',files[:2]),('val',files[2:])]:
        (tmp_path/(role+'.txt')).write_text('\n'.join(str(p) for p in ids)+'\n')
    import yaml
    path=tmp_path/'data.yaml';path.write_text(yaml.safe_dump(dict(train=str(tmp_path/'train.txt'),val=str(tmp_path/'val.txt'),nc=1,names={0:'fruit'})))
    return path,files


def test_exact_dataset_preserves_files_no_cache_and_normalized_box_geometry(tmp_path):
    data,files=dataset_fixture(tmp_path);c=cfg(tmp_path)
    c['detector']['augmentation'].update(mosaic=0,mixup=0,scale=0,translate=0,fliplr=0)
    before={p:p.read_bytes() for p in tmp_path.rglob('*') if p.is_file()}
    loaders,args,generator=bt.make_loaders(c,data,2,42)
    batch=next(iter(loaders['val']));x,targets=bt.batch_inputs(batch,torch.device('cpu'))
    assert x.shape==(2,3,640,640) and float(x[0,0].mean())>float(x[0,2].mean())
    assert torch.allclose(targets[1]['boxes'],torch.tensor([[.5,.5,.5,.5]]))
    assert all(before[p]==p.read_bytes() for p in before)
    assert not list(tmp_path.rglob('*.cache'))
    assert len(loaders['train'].dataset)==2 and len(loaders['val'].dataset)==2
    assert args.mosaic==0 and args.scale==0


def test_training_rejects_test_dataset_key(tmp_path):
    data,files=dataset_fixture(tmp_path);data.write_text(data.read_text()+'test: forbidden\n')
    with pytest.raises(RuntimeError,match='test'):bt.make_loaders(cfg(tmp_path),data,2,42)


def test_loader_rejects_same_image_in_training_and_validation(tmp_path):
    data,files=dataset_fixture(tmp_path)
    (tmp_path/'val.txt').write_text((tmp_path/'train.txt').read_text())
    with pytest.raises(RuntimeError,match='Fuga'):bt.make_loaders(cfg(tmp_path),data,2,42)


def test_loader_rejects_corrupt_image_instead_of_training_on_smaller_subset(tmp_path):
    data,files=dataset_fixture(tmp_path);files[0].write_bytes(b'broken PNG, preserve bytes')
    with pytest.raises(RuntimeError,match='incompleta|No valid images'):bt.make_loaders(cfg(tmp_path),data,2,42)
    assert files[0].read_bytes()==b'broken PNG, preserve bytes'


def test_native_jpeg_repair_is_blocked_without_changing_the_input(tmp_path):
    data,files=dataset_fixture(tmp_path)
    jpeg=files[0].with_suffix('.jpg');Image.new('RGB',(80,40)).save(jpeg)
    truncated=jpeg.read_bytes()[:-2];jpeg.write_bytes(truncated)
    text=(tmp_path/'train.txt').read_text().replace(str(files[0]),str(jpeg));(tmp_path/'train.txt').write_text(text)
    (tmp_path/'labels/train'/jpeg.with_suffix('.txt').name).write_text('0 .5 .5 .5 .5\n')
    with pytest.raises(RuntimeError,match='incompleta|No valid images'):bt.make_loaders(cfg(tmp_path),data,2,42)
    assert jpeg.read_bytes()==truncated


@pytest.mark.parametrize('kind',['perfect','duplicate','missing','none'])
def test_internal_metrics_use_exact_ultralytics_definitions(kind):
    gt=np.array([[.1,.1,.5,.5]])
    if kind=='perfect':pred=(gt.copy(),np.array([.9]))
    elif kind=='duplicate':pred=(np.concatenate([gt,gt]),np.array([.9,.8]))
    else:pred=(np.zeros((0,4)),np.zeros(0))
    metrics,ap=bt.detection_metrics([pred],[gt if kind!='none' else np.zeros((0,4))])
    assert 0<=metrics['metrics/mAP50-95(B)']<=1
    if kind=='perfect':assert metrics['metrics/mAP50-95(B)']==pytest.approx(.995)
    if kind=='missing':assert metrics['metrics/recall(B)']==0


def test_float_output_cast_preserves_integer_metadata_and_gradients():
    x=torch.ones(2,dtype=torch.float64,requires_grad=True)
    value={'pred_logits':x,'aux_outputs':[{'boxes':x}],'dn_meta':{'indices':torch.tensor([1])}}
    out=bt.float_outputs(value)
    assert out['pred_logits'].dtype==torch.float32 and out['dn_meta']['indices'].dtype==torch.int64
    out['pred_logits'].sum().backward();assert torch.equal(x.grad,torch.ones(2,dtype=torch.float64))


def test_optimizer_respects_backbone_lr_ratio_and_does_not_drop_parameters():
    model=torch.nn.Sequential();model.add_module('backbone',torch.nn.Linear(2,2));model.add_module('head',torch.nn.Linear(2,1))
    from ultralytics.cfg import get_cfg
    args=get_cfg();hp=dict(optimizer='AdamW',lr0=.001,backbone_lr_ratio=.1,weight_decay=.0001)
    opt=bt.build_optimizer(model,args,hp,16)
    assert {id(p) for g in opt.param_groups for p in g['params']}=={id(p) for p in model.parameters()}
    assert all(g['lr']==pytest.approx(.0001 if g['backbone'] else .001) for g in opt.param_groups)


def test_rng_round_trip_with_loader_generator():
    import random
    generator=torch.Generator().manual_seed(42);state=bt.rng_state(generator)
    a=(random.random(),np.random.rand(),torch.rand(1),torch.rand(1,generator=generator))
    bt.restore_rng(state,generator)
    b=(random.random(),np.random.rand(),torch.rand(1),torch.rand(1,generator=generator))
    assert a[0]==b[0] and a[1]==b[1] and torch.equal(a[2],b[2]) and torch.equal(a[3],b[3])


def test_signed_resume_atomic_generation_and_corrupt_latest_blocker(tmp_path):
    c=cfg(tmp_path);arch=get_arch('rtdetr_r50');out=tmp_path/'fold0';out.mkdir()
    state=dict(format=bt.FORMAT,signature='same',seed=42,hp={'lr0':.001},fold=0)
    for epoch in (1,2,3):
        state['epoch']=epoch;bt.save_resume(out,state,arch,c)
    assert sorted(p.name for p in (out/'cp').glob('*.pt'))==['e0002.pt','e0003.pt']
    assert bt.load_resume(out,'same',arch,c)['epoch']==3
    with pytest.raises(RuntimeError,match='otra configuracion'):bt.load_resume(out,'other',arch,c)
    newest=out/'cp/e0003.pt';newest.write_bytes(b'corrupt retained')
    with pytest.raises(RuntimeError,match='incompatible'):bt.load_resume(out,'same',arch,c)
    assert newest.read_bytes()==b'corrupt retained' and (out/'cp/e0002.pt').exists()


def test_native_round_new_catalog_and_old_baselines_remain_untouched(tmp_path,monkeypatch):
    c=cfg(tmp_path);c['project']['profile']='official';store=tmp_path/'r50';(store/'weights').mkdir(parents=True)
    c['detector'].update(batch_size=16,nominal_batch_size=64)
    (store/'weights/rtdetr_r50vd_6x_coco_from_paddle.pth').write_bytes(b'fixture transfer already verified by acquisition stage')
    home=tmp_path/'home';home.mkdir();monkeypatch.setattr(Path,'home',classmethod(lambda cls:home));monkeypatch.setenv('USERNAME','I21313')
    index=home/'bt-hoy-20261103/rondas.json';index.parent.mkdir()
    original={'vit_b16':'preserve-vit','yolov8n':'preserve-yolo'};write_json(index,original)
    source_catalog=bt.sha(PROJECT_ROOT/'configs/weights_registry.json')
    config,effective,overrides=bt.prepare_round(c,store,True)
    from utils.experiments import verify_round
    verify_round(effective,architecture='rtdetr_r50')
    assert bt.read(index)['vit_b16']=='preserve-vit' and bt.read(index)['yolov8n']=='preserve-yolo'
    row=next(r for r in bt.read(store/'catalog.json')['models'] if r['slug']=='rtdetr_r50')
    assert row['compatible_pretrained'] is True and row['implementation']=='RT-DETR official PyTorch'
    assert row['official_sha256'] is None and row['adapter_sha256']==bt.sha(source)
    assert bt.sha(PROJECT_ROOT/'configs/weights_registry.json')==source_catalog
    again=bt.prepare_round(c,store,True)
    assert again[0]==config and effective['detector']['baseline']['epochs']==100
    assert effective['detector']['trials_per_architecture']==15 and effective['inner_hpo']['folds']==3
    assert effective['detector']['augmentation']==c['detector']['augmentation']


def test_no_automatic_approval_or_completed_old_model_launch():
    s=source.read_text()
    assert '.approve(' not in s and '"--force"' not in s
    assert "import_module('06_run_baseline')" in s
    assert "'--architecture','rtdetr_r50'" in s
    assert "import_module('07_run_hpo_study')" not in s


def test_decoder_predictions_scale_to_original_image_without_nms():
    class Model(torch.nn.Module):
        def __init__(self):super().__init__();self.p=torch.nn.Parameter(torch.zeros(1));self._botrytis_official_r50=True
        def forward(self,x):return {'pred_boxes':torch.tensor([[[.5,.5,.5,.5]]]),'pred_logits':torch.tensor([[[2.]]])}
    c=load_config();model=Model()
    boxes,scores=bt.predict(model,[Image.new('RGB',(200,100))],conf=.1,iou=.7,cfg=c)[0]
    assert np.allclose(boxes,[[50,25,150,75]]) and scores[0]==pytest.approx(torch.sigmoid(torch.tensor(2.)).item())


def test_validation_matches_in_pixel_coordinates_like_native_validator(monkeypatch):
    class Model(torch.nn.Module):
        def forward(self,x):return {'pred_boxes':torch.tensor([[[.5,.5,.5,.5]]]),'pred_logits':torch.tensor([[[2.]]])}
    batch=dict(img=torch.zeros(1,3,640,640,dtype=torch.uint8),batch_idx=torch.tensor([0]),cls=torch.tensor([[0.]]),bboxes=torch.tensor([[.5,.5,.5,.5]]))
    def metrics(pred,gt):
        assert np.array_equal(pred[0][0],np.array([[160,160,480,480]]))
        assert np.array_equal(gt[0],np.array([[160,160,480,480]]))
        return {'metrics/mAP50-95(B)':.995},[[.995]*10]
    class Loss(torch.nn.Module):
        def forward(self,*a):return dict(loss_vfl=torch.tensor(0.),loss_bbox=torch.tensor(0.),loss_giou=torch.tensor(0.))
    monkeypatch.setattr(bt,'detection_metrics',metrics)
    assert bt.validate(Model(),Loss(),[batch],torch.device('cpu'))['metrics/mAP50-95(B)']==.995


class TinyDetector(torch.nn.Module):
    def __init__(self):
        super().__init__();self.backbone=torch.nn.Linear(1,1);self.head=torch.nn.Linear(1,1);self.box=torch.nn.Parameter(torch.tensor([0.,0.,-1.,-1.]))
    def forward(self,x,targets=None):
        logits=self.head(self.backbone(x.mean((1,2,3)).reshape(-1,1)))
        return {'pred_logits':logits[:,None,:].expand(-1,300,-1),'pred_boxes':self.box.sigmoid()[None,None,:].expand(len(x),300,4)}


def test_training_loop_resume_preserves_accumulated_gradients_and_complete_epochs(tmp_path,monkeypatch):
    data,files=dataset_fixture(tmp_path);c=cfg(tmp_path)
    c['detector']['augmentation'].update(mosaic=0,mixup=0,scale=0,translate=0,fliplr=0)
    monkeypatch.setattr(bt,'build_model',lambda *a,**k:TinyDetector())
    monkeypatch.setattr('utils.weights.verified_local_weight',lambda *a,**k:WEIGHT)
    monkeypatch.setattr('utils.model_stats.count_gflops',lambda *a,**k:0.001)
    hp=dict(optimizer='AdamW',lr0=.001,backbone_lr_ratio=.1,weight_decay=.0001)
    full=bt.train_detector(get_arch('rtdetr_r50'),hp,c,data_yaml=data,project_dir=tmp_path/'full',name='fold0',seed=42,epochs=2,patience=20)
    observed=[]
    def interrupt(step,row):
        observed.append(row['epoch'])
        if row['epoch']==1:raise RuntimeError('SIMULATED_INTERRUPTION')
    with pytest.raises(RuntimeError,match='SIMULATED_INTERRUPTION'):
        bt.train_detector(get_arch('rtdetr_r50'),hp,c,data_yaml=data,project_dir=tmp_path/'partial',name='fold0',seed=42,epochs=2,patience=20,epoch_callback=interrupt)
    state=torch.load(tmp_path/'partial/fold0/cp/e0001.pt',map_location='cpu',weights_only=False)
    assert state['gradients'] and state['epoch']==1 and state['last_opt_step']==-1
    assert state['scheduler']['last_epoch']==0
    assert all(g['lr']==pytest.approx(.0001 if g['backbone'] else .001) for g in state['optimizer']['param_groups'])
    resumed=bt.train_detector(get_arch('rtdetr_r50'),hp,c,data_yaml=data,project_dir=tmp_path/'partial',name='fold0',seed=42,epochs=2,patience=20,
        epoch_callback=lambda step,row:observed.append(row['epoch']))
    assert observed==[1,2] and resumed.history.epoch.tolist()==[1,2]
    assert full.history.equals(resumed.history)
    a=torch.load(tmp_path/'full/fold0/cp/e0002.pt',map_location='cpu',weights_only=False)
    b=torch.load(tmp_path/'partial/fold0/cp/e0002.pt',map_location='cpu',weights_only=False)
    assert all(torch.equal(a['model'][k],b['model'][k]) for k in a['model'])
    count=[]
    again=bt.train_detector(get_arch('rtdetr_r50'),hp,c,data_yaml=data,project_dir=tmp_path/'partial',name='fold0',seed=42,epochs=2,patience=20,
        epoch_callback=lambda *a:count.append(1))
    assert not count and again.history.equals(resumed.history)


def test_native_06_three_folds_summary_and_second_invocation_never_trains(tmp_path,monkeypatch):
    data,files=dataset_fixture(tmp_path);c=cfg(tmp_path)
    c['detector']['augmentation'].update(mosaic=0,mixup=0,scale=0,translate=0,fliplr=0)
    c['detector']['baseline']['epochs']=1
    c['paths']['detection_dir']=str(tmp_path)
    import pandas as pd
    table=pd.DataFrame([dict(image_id=p.stem,outer_split='train',inner_fold_det=i%3,n_fruits=i%2,image_file=p.name) for i,p in enumerate(files)])
    native=importlib.import_module('06_run_baseline')
    import utils.hpo_runner as hpo
    from utils.logging_utils import RunContext
    monkeypatch.setattr(bt,'build_model',lambda *a,**k:TinyDetector())
    monkeypatch.setattr('utils.weights.verified_local_weight',lambda *a,**k:WEIGHT)
    monkeypatch.setattr('utils.model_stats.count_gflops',lambda *a,**k:.001)
    monkeypatch.setattr(hpo,'detection_images_table',lambda c:table)
    monkeypatch.setattr(native,'preflight_training',lambda *a:SimpleNamespace(dataset_hash='synthetic',split_hash='synthetic'))
    monkeypatch.setattr(native,'GpuJobLock',lambda *a,**k:contextlib.nullcontext())
    monkeypatch.setattr(native,'save_training_figures',lambda *a,**k:None)
    ctx=SimpleNamespace(cfg=c,pc_id='PC-03',run_id='CPU_NATIVE_BASELINE',run_dir=tmp_path/'baseline',logger=logging.getLogger('r50-test'),checkpoints=[])
    ctx.run_dir.mkdir()
    ctx.set_seed=lambda seed:None;ctx.add_metadata=lambda **k:None;ctx.note_split_access=lambda *a:None
    ctx.log_metrics=lambda *a:None;ctx.register_artifact=lambda *a:None;ctx._write_checkpoint_metadata=lambda:None
    original_dict=bt.ORIGINALS.copy();bt.ORIGINALS.clear()
    old=(RunContext.record_checkpoint, __import__('utils.detector_train',fromlist=['train_detector']).train_detector)
    import utils.inference as infer
    old_inference=(infer.load_detector,infer.predict_detector)
    old_runner=(hpo.evaluate_config,hpo.train_detector)
    try:
        bt.install_hooks(c)
        ctx.record_checkpoint=lambda *a,**k:RunContext.record_checkpoint(ctx,*a,**k)
        calls=[]
        def train(*a,**k):calls.append(k['name']);return bt.train_detector(*a,**k)
        monkeypatch.setattr(hpo,'train_detector',train)
        args=SimpleNamespace(architecture='rtdetr_r50',folds=None,force=False,skip_env_check=False)
        native.main(ctx,args)
        summary=bt.read(ctx.run_dir/'summary.json');before=(ctx.run_dir/'summary.json').read_bytes()
        assert calls==['fold0','fold1','fold2'] and len(summary['fold_scores'])==3
        assert summary['architecture']=='RT-DETR-R50' and summary['objective']=='map50_95'
        assert len(ctx.checkpoints)==3 and all(r['weights_discarded'] is False for r in ctx.checkpoints)
        assert all(Path(r['retained_path']).is_file() for r in ctx.checkpoints)
        native.main(ctx,args)
        assert len(calls)==3 and before==(ctx.run_dir/'summary.json').read_bytes()
        (ctx.run_dir/'summary.json').unlink();ctx.checkpoints.clear()
        native.main(ctx,args)
        assert len(calls)==3 and len(ctx.checkpoints)==3  # native cache export, without training any valid fold
        assert all(r['weights_discarded'] is False for r in ctx.checkpoints)
    finally:
        RunContext.record_checkpoint=old[0]
        __import__('utils.detector_train',fromlist=['train_detector']).train_detector=old[1]
        infer.load_detector,infer.predict_detector=old_inference
        hpo.evaluate_config,hpo.train_detector=old_runner
        bt.ORIGINALS.clear();bt.ORIGINALS.update(original_dict)


def test_partial_training_without_optimizer_is_not_restarted(tmp_path):
    c=cfg(tmp_path);out=tmp_path/'prior/fold0';out.mkdir(parents=True);(out/'results.csv').write_text('epoch,loss\n1,1\n')
    from unittest.mock import patch
    with patch('utils.weights.verified_local_weight',return_value=WEIGHT):
        with pytest.raises(RuntimeError,match='sin estado de optimizador'):
            bt.train_detector(get_arch('rtdetr_r50'),dict(lr0=.001),c,data_yaml=tmp_path/'unused.yaml',project_dir=out.parent,name='fold0',seed=42,epochs=2)
    assert (out/'results.csv').read_text()=='epoch,loss\n1,1\n'


def test_public_pc03_setup_and_native_baseline_handoff_are_idempotent(tmp_path,monkeypatch):
    import sys
    from orchestration.study_queue import StudyQueue
    home=tmp_path/'home';project=home/'Documents/s2_no_borrar/Entrenamiento/Botrytis_Experiment_Framework_git'
    py=project/'.venv/Scripts/python.exe';py.parent.mkdir(parents=True);py.touch()
    c=cfg(tmp_path);c['project']['profile']='official';c['detector'].update(batch_size=16,nominal_batch_size=64)
    monkeypatch.chdir(project);monkeypatch.setattr(Path,'home',classmethod(lambda cls:home));monkeypatch.setenv('USERNAME','i21313')
    monkeypatch.setattr(bt,'os',SimpleNamespace(name='nt',chdir=os.chdir,environ=os.environ,replace=os.replace,fsync=os.fsync))
    monkeypatch.setattr(sys,'executable',str(py));monkeypatch.setattr(sys,'argv',['r50','--allow-network','--accept-license'])
    original=load_config
    monkeypatch.setattr('utils.config_loader.load_config',lambda p=None,s=None:c if p==project/'artifacts/v4/local.yaml' else original(p,s))
    monkeypatch.setattr('orchestration.machine.read_binding',lambda cfg:dict(pc_id='PC-03',machine_id='i21313'))
    gates=[];monkeypatch.setattr('utils.manual_gates.require_gate',lambda c,pc,g:gates.append((pc,g)))
    monkeypatch.setattr('psutil.process_iter',lambda *a:[])
    monkeypatch.setattr(torch.cuda,'is_available',lambda:True);monkeypatch.setattr(torch.cuda,'get_device_name',lambda *a:'GPU_TEST')
    monkeypatch.setattr(torch.cuda,'device_count',lambda:1)
    monkeypatch.setattr('utils.guards.GpuJobLock',lambda *a,**k:contextlib.nullcontext())
    monkeypatch.setattr(bt,'acquire_source',lambda *a:OFFICIAL)
    def downloaded(url,path,expected,network):
        path=Path(path);path.parent.mkdir(parents=True,exist_ok=True)
        if not path.exists():os.link(WEIGHT,path)
        return path
    monkeypatch.setattr(bt,'download',downloaded)
    tests=[]
    def selftest(*a):
        tests.append(1);return dict(status='PASS',adapter_sha256=bt.sha(source),torch=torch.__version__,cuda_runtime=torch.version.cuda,gpu='GPU_TEST')
    monkeypatch.setattr(bt,'selftest',selftest)
    hooks=[];monkeypatch.setattr(bt,'install_hooks',lambda c:hooks.append(1))
    native=importlib.import_module('06_run_baseline');launched=[]
    def fake_run(script,main,**kw):
        args=kw['argv'];launched.append(args)
        assert args[args.index('--architecture')+1]=='rtdetr_r50'
        path=args[args.index('--config')+1];overrides=[args[i+1] for i,x in enumerate(args) if x=='--set']
        effective=original(path,overrides);q=StudyQueue.from_cfg(effective);job=next(j for j in q.jobs() if j.kind=='baseline')
        q.finish(job.job_id,'PENDING_REVIEW');return 0
    monkeypatch.setattr(native,'run_cli',fake_run)
    assert bt.main()==0 and bt.main()==0
    assert len(launched)==1 and len(tests)==1 and len(hooks)==1
    assert bt.read(home/'r50/summary.json')['queue']['PENDING_REVIEW']==1
    assert all(pc=='PC-03' and g=='READY_FOR_EXPERIMENT' for pc,g in gates)


@pytest.mark.parametrize('state',['RUNNING','FAILED'])
def test_queue_recovery_requires_dead_matching_owner_not_heartbeat_age(tmp_path,monkeypatch,state):
    import psutil
    from orchestration.study_queue import StudyQueue
    c=cfg(tmp_path);q=StudyQueue.from_cfg(c);q.plan(c,[get_arch('rtdetr_r50')]);job=next(j for j in q.jobs() if j.kind=='baseline')
    q.claim_next('PC-03',kinds=['baseline'],slugs=['rtdetr_r50'])
    if state=='FAILED':q.finish(job.job_id,state)
    job=q.get(job.job_id)
    owner=tmp_path/'owner.json'
    with pytest.raises(RuntimeError,match='propietario'):bt.recover_owned_baseline(q,job,owner)
    write_json(owner,dict(job_id=job.job_id,attempt=job.attempt,pid=12345678,created=1.,adapter_sha256=bt.sha(source)))
    monkeypatch.setattr(psutil,'Process',lambda pid:SimpleNamespace(create_time=lambda:1.))
    with pytest.raises(RuntimeError,match='sigue activo'):bt.recover_owned_baseline(q,job,owner)
    def absent(pid):raise psutil.NoSuchProcess(pid)
    monkeypatch.setattr(psutil,'Process',absent)
    bt.recover_owned_baseline(q,job,owner)
    assert q.get(job.job_id).status=='PENDING' and q.get(job.job_id).attempt==job.attempt


def test_acquisition_download_before_bootstrap_then_training_still_blocks_network(tmp_path):
    """Real HTTP regression in a fresh process; importing a trainer must not leak permissions."""
    import subprocess,sys,textwrap
    code=textwrap.dedent('''
        import hashlib,importlib.util,os,sys,threading,socket
        from pathlib import Path
        from http.server import BaseHTTPRequestHandler,ThreadingHTTPServer
        module_path,output,project=sys.argv[1:]
        spec=importlib.util.spec_from_file_location('adapter',module_path)
        bt=importlib.util.module_from_spec(spec);spec.loader.exec_module(bt)
        class Handler(BaseHTTPRequestHandler):
            def do_GET(self):
                self.send_response(200);self.end_headers();self.wfile.write(b'OFFICIAL_FIXTURE')
            def log_message(self,*a):pass
        server=ThreadingHTTPServer(('127.0.0.1',0),Handler)
        worker=threading.Thread(target=server.serve_forever,daemon=True);worker.start()
        try:
            assert 'utils.bootstrap' not in sys.modules
            bt.download('http://127.0.0.1:'+str(server.server_port),Path(output)/'verified.bin',hashlib.sha256(b'OFFICIAL_FIXTURE').hexdigest(),True)
            assert 'utils.bootstrap' not in sys.modules
            assert (Path(output)/'verified.bin').read_bytes()==b'OFFICIAL_FIXTURE'
            sys.path.insert(0,str(Path(project)/'src'))
            os.environ['BOTRYTIS_BOOTSTRAP_DIR']=str(Path(output)/'logs')
            from utils import bootstrap
            bootstrap.install('r50_training.py',Path(project))
            with socket.socket() as sock:
                try:sock.connect(('127.0.0.1',server.server_port))
                except PermissionError as error:assert 'bloqueada' in str(error)
                else:raise AssertionError('Training network guard was disabled')
            print('ACQUISITION_OK_TRAINING_STILL_BLOCKED')
        finally:
            server.shutdown();server.server_close()
    ''')
    result=subprocess.run([sys.executable,'-c',code,str(source),str(tmp_path),str(PROJECT_ROOT)],capture_output=True,text=True,timeout=30)
    assert result.returncode==0,result.stdout+result.stderr
    assert 'ACQUISITION_OK_TRAINING_STILL_BLOCKED' in result.stdout


def test_explicit_acquisition_requires_license(tmp_path,monkeypatch):
    monkeypatch.setattr(bt,'acquire_source',lambda *a:pytest.fail('No authorized acquisition'))
    with pytest.raises(RuntimeError,match='accept-license'):bt.acquire_before_framework(tmp_path,True,False)


def test_main_acquires_before_any_framework_import():
    import ast
    tree=ast.parse(source.read_text());main=next(n for n in tree.body if isinstance(n,ast.FunctionDef) and n.name=='main')
    acquisition=next(n.lineno for n in ast.walk(main) if isinstance(n,ast.Call) and isinstance(n.func,ast.Name) and n.func.id=='acquire_before_framework')
    first=min(n.lineno for n in ast.walk(main) if isinstance(n,ast.ImportFrom) and (n.module or '').startswith(('utils','orchestration')))
    assert acquisition<first
