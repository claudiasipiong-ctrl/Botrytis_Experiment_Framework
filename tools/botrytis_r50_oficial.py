"""RT-DETR-R50-vd oficial: adaptador externo, rondas nuevas, no modifica otras arquitecturas."""
from pathlib import Path
import argparse
import copy
import gc
import hashlib
import importlib
import json
import math
import os
import random
import re
import shutil
import subprocess
import sys
import time
import types
import urllib.request
import uuid
import warnings
import zipfile

REV='df2a96c8c3a098567a1149bf0248da6efe6d495d'
SOURCE_URL='https://codeload.github.com/lyuwenyu/RT-DETR/zip/'+REV
SOURCE_SHA='e6014ae32d2d384799e353a7a6a511aaef1dad77763492738dfef70e714d6564'
WEIGHT_URL='https://github.com/lyuwenyu/storage/releases/download/v0.1/rtdetr_r50vd_6x_coco_from_paddle.pth'
WEIGHT_SHA='dfaafa5a3d29958a360c35f1fad553a6f287bfae9967e8ee007e65cff21ddf45'
WEIGHT_SIZE=172373587
FORMAT='botrytis-r50vd-official-v1'
CLASSES=None
ORIGINALS={}


def sha(p):
    h=hashlib.sha256()
    with Path(p).open('rb') as f:
        for part in iter(lambda:f.read(1<<20),b''):h.update(part)
    return h.hexdigest()


def emit(event,**fields):
    print(event,json.dumps(fields,ensure_ascii=False,default=str),flush=True)


def read(p):
    return json.loads(Path(p).read_text(encoding='utf-8-sig'))


def canonical(value):
    return hashlib.sha256(json.dumps(value,sort_keys=True,separators=(',',':'),default=str).encode()).hexdigest()


def download(url,path,expected,allow_network):
    from utils.durable import atomic_json
    path=Path(path)
    if path.is_file():
        if sha(path)!=expected:raise RuntimeError('Archivo existente incompatible, se conserva: '+str(path))
        return path
    if not allow_network:raise RuntimeError('Falta archivo oficial y --allow-network: '+str(path))
    path.parent.mkdir(parents=True,exist_ok=True)
    temp=path.with_name(path.name+'.download-'+uuid.uuid4().hex[:8])
    emit('DESCARGANDO_FUENTE_OFICIAL',url=url,path=path)
    with urllib.request.urlopen(url,timeout=60) as source,temp.open('xb') as out:
        received=0;reported=0
        for chunk in iter(lambda:source.read(1<<20),b''):
            out.write(chunk);received+=len(chunk)
            if received-reported>=16<<20:emit('DESCARGA_EN_CURSO',megabytes=round(received/1024**2));reported=received
        out.flush();os.fsync(out.fileno())
    if sha(temp)!=expected:raise RuntimeError('SHA256 incorrecto, descarga conservada: '+str(temp))
    os.replace(temp,path)
    atomic_json(path.with_suffix(path.suffix+'.source.json'),dict(url=url,sha256=expected,
        checksum_basis='Hash calculado durante la verificacion de esta integracion; no se afirma publicado por upstream'))
    return path


def acquire_source(base,allow_network=False):
    from utils.durable import atomic_json
    base=Path(base);archive=download(SOURCE_URL,base/'official.zip',SOURCE_SHA,allow_network)
    root=base/'official';marker=root/'source.json'
    if not root.exists():
        root.mkdir(parents=True)
        with zipfile.ZipFile(archive) as z:
            prefix='RT-DETR-'+REV+'/rtdetr_pytorch/'
            for item in z.infolist():
                if not item.filename.startswith(prefix) or item.is_dir():continue
                rel=Path(item.filename[len(prefix):])
                if rel.is_absolute() or '..' in rel.parts:raise RuntimeError('Ruta insegura en archivo oficial')
                dest=root/rel;dest.parent.mkdir(parents=True,exist_ok=True);dest.write_bytes(z.read(item))
        manifest={p.relative_to(root).as_posix():sha(p) for p in root.rglob('*') if p.is_file()}
        atomic_json(marker,dict(revision=REV,archive_sha256=SOURCE_SHA,files=manifest))
    if not marker.is_file():raise RuntimeError('Extraccion previa incompleta: '+str(root))
    m=read(marker)
    if m['revision']!=REV or m['archive_sha256']!=SOURCE_SHA:raise RuntimeError('Otra fuente oficial; se conserva')
    with zipfile.ZipFile(archive) as z:
        prefix='RT-DETR-'+REV+'/rtdetr_pytorch/'
        for item in z.infolist():
            if item.is_dir() or not item.filename.startswith(prefix):continue
            rel=item.filename[len(prefix):]
            expected=hashlib.sha256(z.read(item)).hexdigest()
            if not (root/rel).is_file() or sha(root/rel)!=expected:raise RuntimeError('Fuente oficial modificada: '+rel)
    return root


def official_classes(root):
    """Import only untouched model modules; exclude COCO/ONNX/trainer optional dependencies."""
    global CLASSES
    root=Path(root).resolve()
    if CLASSES is not None:
        if CLASSES['root']!=str(root):raise RuntimeError('Dos fuentes oficiales en un proceso')
        return CLASSES
    for name,rel in [('src','.'),('src.nn','nn'),('src.nn.backbone','nn/backbone'),
                     ('src.zoo','zoo'),('src.zoo.rtdetr','zoo/rtdetr'),('src.misc','misc')]:
        path=root/'src'/rel
        if name in sys.modules and str(path) not in getattr(sys.modules[name],'__path__',[]):
            raise RuntimeError('Conflicto de namespace oficial: '+name)
        module=types.ModuleType(name);module.__path__=[str(path)];sys.modules[name]=module
        module.__package__=name
        module.__spec__=importlib.machinery.ModuleSpec(name,loader=None,is_package=True)
        module.__spec__.submodule_search_locations=module.__path__
    for name in ['src.nn.backbone.presnet','src.zoo.rtdetr.rtdetr','src.zoo.rtdetr.hybrid_encoder',
                 'src.zoo.rtdetr.rtdetr_decoder','src.zoo.rtdetr.rtdetr_criterion',
                 'src.zoo.rtdetr.rtdetr_postprocessor','src.zoo.rtdetr.matcher']:
        importlib.import_module(name)
    from src.core import YAMLConfig
    from src.zoo.rtdetr.rtdetr_criterion import SetCriterion
    from src.zoo.rtdetr.matcher import HungarianMatcher
    from src.zoo.rtdetr.rtdetr_postprocessor import RTDETRPostProcessor
    CLASSES=dict(root=str(root),YAMLConfig=YAMLConfig,Criterion=SetCriterion,Matcher=HungarianMatcher,Post=RTDETRPostProcessor)
    return CLASSES


def build_model(root,weights=None,classes=1):
    """Load the COMPLETE 80-class COCO graph strictly, then replace only classification heads."""
    import torch
    api=official_classes(root)
    config=api['YAMLConfig'](str(Path(root)/'configs/rtdetr/rtdetr_r50vd_6x_coco.yml'))
    config.yaml_cfg['PResNet'].update(pretrained=False,freeze_at=-1)
    config.yaml_cfg['RTDETR']['multi_scale']=None  # framework fixes 640, not upstream multi-scale COCO training
    config.yaml_cfg['remap_mscoco_category']=False
    config.yaml_cfg['num_classes']=80
    model=config.model
    if weights is not None:
        if sha(weights)!=WEIGHT_SHA:raise RuntimeError('Checkpoint preentrenado no verificado')
        state=torch.load(weights,map_location='cpu',weights_only=False)
        if set(state)!={'ema'} or 'module' not in state['ema']:raise RuntimeError('Formato COCO oficial inesperado')
        model.load_state_dict(state['ema']['module'],strict=True)
    if classes not in (1,80):raise RuntimeError('Solo COCO80 o fruit1')
    if classes==1:
        decoder=model.decoder
        decoder.num_classes=1
        decoder.enc_score_head=torch.nn.Linear(decoder.hidden_dim,1)
        decoder.dec_score_head=torch.nn.ModuleList([torch.nn.Linear(decoder.hidden_dim,1) for _ in decoder.dec_score_head])
        decoder.denoising_class_embed=torch.nn.Embedding(2,decoder.hidden_dim,padding_idx=1)
        for head in [decoder.enc_score_head,*decoder.dec_score_head]:
            torch.nn.init.constant_(head.bias,-math.log(99.))
    return model


def criterion(root):
    a=official_classes(root)
    return a['Criterion'](matcher=a['Matcher']({'cost_class':2,'cost_bbox':5,'cost_giou':2},use_focal_loss=True,alpha=.25,gamma=2),
                          weight_dict={'loss_vfl':1,'loss_bbox':5,'loss_giou':2},losses=['vfl','boxes'],alpha=.75,gamma=2,num_classes=1)


def float_outputs(value):
    import torch
    if isinstance(value,torch.Tensor):return value.float() if value.is_floating_point() else value
    if isinstance(value,dict):return {k:float_outputs(v) for k,v in value.items()}
    if isinstance(value,list):return [float_outputs(v) for v in value]
    if isinstance(value,tuple):return tuple(float_outputs(v) for v in value)
    return value


def make_loaders(cfg,data_yaml,batch,seed):
    import torch
    import yaml
    from PIL import Image
    from ultralytics.cfg import get_cfg
    from ultralytics.models.rtdetr.val import RTDETRDataset
    import ultralytics.data.dataset as dataset_module
    from utils.reproducibility import seed_worker
    data=yaml.safe_load(Path(data_yaml).read_text(encoding='utf-8'))
    if 'test' in data:raise RuntimeError('No se admite test en entrenamiento/HPO')
    args=get_cfg(overrides=dict(cfg['detector']['augmentation'],imgsz=cfg['detector']['image_size'],cache=False,
                              batch=batch,workers=cfg['detector']['workers'],single_cls=False))
    generator=torch.Generator().manual_seed(seed)
    loaders={}
    for role in ('train','val'):
        if role not in data:continue
        original=dataset_module.save_dataset_cache_file
        dataset_module.save_dataset_cache_file=lambda prefix,path,x,version:x.update(version=version)  # memory schema, no input cache writes
        original_save=Image.Image.save
        def refuse_input_repair(*a,**k):raise RuntimeError('No se permite reparar/escribir imagenes de entrada')
        Image.Image.save=refuse_input_repair
        try:
            ds=RTDETRDataset(img_path=data[role],imgsz=args.imgsz,batch_size=batch,augment=role=='train',hyp=copy.deepcopy(args),
                             rect=False,cache=False,single_cls=False,classes=None,data={'nc':1,'names':{0:'fruit'},'channels':3},fraction=1.)
        finally:
            dataset_module.save_dataset_cache_file=original
            Image.Image.save=original_save
        if Path(data[role]).suffix.lower()=='.txt':
            declared=[Path(p).resolve() for p in Path(data[role]).read_text(encoding='utf-8-sig').splitlines() if p.strip()]
            actual=[Path(p).resolve() for p in ds.im_files]
            if len(set(declared))!=len(declared) or set(declared)!=set(actual):raise RuntimeError('Lista de imagenes incompleta/duplicada/corrupta: '+role)
        loaders[role]=torch.utils.data.DataLoader(ds,batch_size=batch,shuffle=role=='train',num_workers=int(args.workers),
              collate_fn=ds.collate_fn,worker_init_fn=seed_worker,generator=generator if role=='train' else torch.Generator().manual_seed(seed),
              pin_memory=torch.cuda.is_available(),drop_last=False,persistent_workers=False)
    if 'val' in loaders and Path(data['train']).suffix.lower()=='.txt' and Path(data['val']).suffix.lower()=='.txt':
        if set(loaders['train'].dataset.im_files)&set(loaders['val'].dataset.im_files):raise RuntimeError('Fuga train/validation en listas de imagenes')
    return loaders,args,generator


def batch_inputs(batch,device):
    import torch
    images=batch['img'].to(device,non_blocking=True).float()/255
    targets=[]
    for i in range(len(images)):
        mask=batch['batch_idx']==i
        labels=batch['cls'][mask].reshape(-1).to(device,dtype=torch.int64)
        if labels.numel() and labels.max().item()!=0:raise RuntimeError('Solo clase fruit=0')
        boxes=batch['bboxes'][mask].to(device,dtype=torch.float32)
        if not torch.isfinite(boxes).all() or ((boxes<0)|(boxes>1)).any():raise RuntimeError('Cajas fuera de cxcywh normalizado')
        targets.append(dict(labels=labels,boxes=boxes))
    return images,targets


def detection_metrics(predictions,ground_truths):
    import numpy as np
    import torch
    from ultralytics.engine.validator import BaseValidator
    from ultralytics.utils.metrics import DetMetrics,box_iou
    result=DetMetrics(names={0:'fruit'})
    validator=types.SimpleNamespace(iouv=torch.linspace(.5,.95,10))
    for i,(pred,gt) in enumerate(zip(predictions,ground_truths)):
        boxes,scores=pred;boxes=torch.as_tensor(boxes,dtype=torch.float32).reshape(-1,4)
        gt=torch.as_tensor(gt,dtype=torch.float32).reshape(-1,4)
        pclass=torch.zeros(len(boxes));gclass=torch.zeros(len(gt))
        correct=BaseValidator.match_predictions(validator,pclass,gclass,box_iou(gt,boxes)).numpy()
        result.update_stats(dict(tp=correct,conf=np.asarray(scores),pred_cls=pclass.numpy(),target_cls=gclass.numpy(),
                                 target_img=np.unique(gclass.numpy()),im_name=str(i)))
    result.process(plot=False)
    return {k:float(v) for k,v in result.results_dict.items()},result.box.all_ap.tolist()


def validate(model,loss_fn,loader,device,conf=.001,amp=False):
    import torch
    from torchvision.ops import box_convert
    model.eval();loss_fn.eval();predictions=[];gts=[];loss_sums={};batches=0
    for batch in loader:
        images,targets=batch_inputs(batch,device)
        with torch.no_grad(),torch.autocast(device_type=device.type,dtype=torch.float16,enabled=amp):
            output=model(images)
        output=float_outputs(output)
        with torch.no_grad(),torch.autocast(device_type=device.type,enabled=False):
            losses=loss_fn(output,targets)
        if any(not torch.isfinite(v).all() for v in losses.values()):raise RuntimeError('Loss de validacion no finita; no se publica fold')
        for k in ('loss_vfl','loss_bbox','loss_giou'):loss_sums[k]=loss_sums.get(k,0)+float(losses[k])
        boxes=box_convert(output['pred_boxes'],'cxcywh','xyxy')
        scores=output['pred_logits'].sigmoid().squeeze(-1)
        if not torch.isfinite(boxes).all() or not torch.isfinite(scores).all():raise RuntimeError('Predicciones no finitas')
        scale=torch.tensor([images.shape[3],images.shape[2],images.shape[3],images.shape[2]],device=device)
        for i,target in enumerate(targets):
            order=torch.argsort(scores[i],descending=True)[:300];keep=order[scores[i,order]>conf]
            predictions.append(((boxes[i,keep]*scale).cpu().numpy(),scores[i,keep].cpu().numpy()))
            gts.append((box_convert(target['boxes'],'cxcywh','xyxy')*scale).cpu().numpy())
        batches+=1
    metrics,ap=detection_metrics(predictions,gts)
    metrics['ap_per_iou']=ap[0] if ap else None
    for k,v in loss_sums.items():metrics['val/'+k[5:]+'_loss']=v/max(batches,1)
    metrics['val_loss']=sum(loss_sums.values())/max(batches,1)
    return metrics


def build_optimizer(model,args,hp,batch):
    from ultralytics.engine.trainer import BaseTrainer
    ratio=float(hp['backbone_lr_ratio'])
    if not 0<ratio<=1:raise RuntimeError('backbone_lr_ratio invalido')
    optimizer=BaseTrainer.build_optimizer(types.SimpleNamespace(args=args),model,name=hp['optimizer'],lr=hp['lr0'],
                                         momentum=args.momentum,decay=hp['weight_decay'])
    backbone={id(p) for n,p in model.named_parameters() if n.startswith('backbone.')}
    groups=list(optimizer.param_groups);optimizer.param_groups.clear()
    for group in groups:
        for is_backbone in (False,True):
            params=[p for p in group['params'] if (id(p) in backbone)==is_backbone]
            if not params:continue
            new={k:v for k,v in group.items() if k!='params'}
            new.update(params=params,lr=float(group['lr'])*(ratio if is_backbone else 1),backbone=is_backbone)
            optimizer.add_param_group(new)
    return optimizer


def rng_state(generator):
    import numpy as np
    import torch
    return dict(python=random.getstate(),numpy=np.random.get_state(),torch=torch.get_rng_state(),
                cuda=torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None,generator=generator.get_state())


def restore_rng(value,generator):
    import numpy as np
    import torch
    random.setstate(value['python']);np.random.set_state(value['numpy']);torch.set_rng_state(value['torch'])
    if value['cuda'] is not None:torch.cuda.set_rng_state_all(value['cuda'])
    generator.set_state(value['generator'])


def save_resume(out,state,spec,cfg):
    """Publish signed immutable epoch before pointer; retain two complete optimizer generations."""
    from utils.durable import atomic_json,atomic_torch_save
    from utils.resume import receipt
    out=Path(out);folder=out/'cp';folder.mkdir(parents=True,exist_ok=True)
    path=folder/('e%04d.pt'%state['epoch'])
    if path.exists():raise RuntimeError('Checkpoint de epoca ya existe; se conserva: '+str(path))
    atomic_torch_save(state,path)
    receipt(path,spec,cfg,dict(seed=state['seed'],hp=state['hp'],context={'fold':state['fold']},epoch=state['epoch'],adapter_sha256=sha(__file__)))
    atomic_json(out/'resume.json',dict(path=str(path),sha256=sha(path),epoch=state['epoch'],signature=state['signature']))
    # Only this new adapter's signed, superseded rolling states are retired.
    states=sorted(folder.glob('e*.pt'))
    for previous in states[:-2]:
        side=previous.with_suffix('.pt.resume.json')
        if side.is_file() and read(side).get('extra',{}).get('adapter_sha256')==sha(__file__):
            previous.unlink();side.unlink()
    return path


def load_resume(out,signature,spec,cfg):
    import torch
    from utils.resume import require_checkpoint
    candidates=sorted((Path(out)/'cp').glob('e*.pt'),reverse=True)
    if not candidates:return None
    # A corrupt newest state is a blocker, never a silent return to pretrained.
    path=require_checkpoint(candidates[0],spec,cfg)
    state=torch.load(path,map_location='cpu',weights_only=False)
    if state.get('format')!=FORMAT or state.get('signature')!=signature:raise RuntimeError('Estado de otra configuracion/fold')
    return state


def train_detector(spec,hp,cfg,*,data_yaml,project_dir,name,seed,epochs=None,patience=None,fixed_epochs=False,
                   epoch_callback=None,step_offset=0,log=print,keep_weights=False):
    import numpy as np
    import pandas as pd
    import torch
    from utils.detector_train import DetectorRun
    from utils.experiments import fingerprint
    from utils.weights import verified_local_weight
    from utils.reproducibility import set_global_seed
    from utils.durable import atomic_json,atomic_torch_save
    from utils.resume import receipt
    from ultralytics.utils.torch_utils import ModelEMA,EarlyStopping
    if spec.slug!='rtdetr_r50':return ORIGINALS['train'](spec,hp,cfg,data_yaml=data_yaml,project_dir=project_dir,name=name,seed=seed,
        epochs=epochs,patience=patience,fixed_epochs=fixed_epochs,epoch_callback=epoch_callback,step_offset=step_offset,log=log,keep_weights=keep_weights)
    if cfg.get('initialization',{}).get('mode')!='pretrained' or not cfg['detector']['pretrained']:
        raise RuntimeError('Este adaptador solo admite el protocolo COCO pretrained; no sustituye un experimento scratch')
    if not torch.cuda.is_available() and cfg['project']['profile']=='official':raise RuntimeError('No se permite entrenamiento operativo en CPU')
    device=torch.device('cuda:0' if torch.cuda.is_available() else 'cpu');amp=device.type=='cuda' and cfg['detector']['precision']=='fp16'
    root=Path(cfg['weights']['r50_source']);weight=verified_local_weight(spec,cfg)
    epochs=int(epochs or cfg['detector']['max_epochs']);patience=int(patience if patience is not None else cfg['detector']['early_stopping_patience'])
    fold=int(re.fullmatch(r'fold(\d+)',name)[1]) if re.fullmatch(r'fold(\d+)',name) else None
    out=Path(project_dir)/name;out.mkdir(parents=True,exist_ok=True)
    signature=canonical(dict(experiment=fingerprint(cfg),hp=hp,seed=seed,fold=fold,epochs=epochs,patience=patience,fixed=fixed_epochs))
    previous=load_resume(out,signature,spec,cfg)
    if previous is None and ((out/'results.csv').exists() or list(out.rglob('*.pt'))):
        raise RuntimeError('Trabajo previo sin estado de optimizador verificable; se conserva sin reiniciar desde COCO: '+str(out))
    set_global_seed(seed,deterministic=bool(cfg['reproducibility']['deterministic']))
    model=build_model(root,weight).to(device);loss_fn=criterion(root).to(device)
    emit('FOLD_RTDETR_OFICIAL',name=name,seed=seed,epochs=epochs,checkpoint='RESUME' if previous else 'COCO_COMPLETO_VERIFICADO',output=out)
    loaders,args,generator=make_loaders(cfg,data_yaml,int(cfg['detector']['batch_size']),seed)
    args.nbs=int(cfg['detector']['nominal_batch_size']);args.warmup_epochs=float(hp.get('warmup_epochs',cfg['detector']['baseline']['warmup_epochs']))
    args.lrf=float(hp.get('lrf',cfg['detector']['baseline']['lrf']));args.cos_lr=bool(cfg['detector']['baseline']['cos_lr'])
    optimizer=build_optimizer(model,args,hp,int(args.batch));ema=ModelEMA(model);stopper=EarlyStopping(0 if fixed_epochs else patience)
    lf=(lambda e:((1-math.cos(math.pi*e/epochs))/2)*(args.lrf-1)+1) if args.cos_lr else (lambda e:max(1-e/epochs,0)*(1-args.lrf)+args.lrf)
    scheduler=torch.optim.lr_scheduler.LambdaLR(optimizer,lr_lambda=lf)
    scheduler.last_epoch=-1  # same epoch-zero scheduling as the pinned native trainer
    scaler=torch.amp.GradScaler('cuda',enabled=amp)
    history=[];best_score=-math.inf;best_epoch=0;best_state=None;best_metrics={};last_opt_step=-1;start=0;elapsed=0.
    if previous:
        model.load_state_dict(previous['model'],strict=True);ema.ema.load_state_dict(previous['ema'],strict=True);ema.updates=previous['ema_updates']
        optimizer.load_state_dict(previous['optimizer']);scheduler.load_state_dict(previous['scheduler']);scaler.load_state_dict(previous['scaler'])
        history=previous['history'];best_score=previous['best_score'];best_epoch=previous['best_epoch'];best_state=previous['best_state'];best_metrics=previous['best_metrics']
        last_opt_step=previous['last_opt_step'];start=previous['epoch'];elapsed=previous['elapsed']
        stopper.best_epoch=previous['stopper_epoch'];stopper.best_fitness=previous['stopper_fitness']
        for n,p in model.named_parameters():
            if n in previous['gradients']:p.grad=previous['gradients'][n].to(device)
        restore_rng(previous['rng'],generator)
        emit('R50_REANUDADO',fold=fold,epoch=start,checkpoint='Ultima epoca completa, optimizador/EMA/scaler/RNG/gradientes conservados')
    else:
        optimizer.zero_grad(set_to_none=True)
    if device.type=='cuda':torch.cuda.reset_peak_memory_stats()
    nb=len(loaders['train']);nw=round(min(args.warmup_epochs,max(epochs-1,0))*nb)
    t0=time.perf_counter()
    if previous and previous.get('stopped'):start=epochs
    for epoch in range(start,epochs):
        model.train();loss_fn.train()
        with warnings.catch_warnings():
            warnings.filterwarnings('ignore',message='Detected call of .*lr_scheduler.step')
            scheduler.step()  # native epoch-start schedule; last_epoch=-1 was set explicitly
        sums={};total_full=0.
        if args.close_mosaic and (epoch==epochs-int(args.close_mosaic) or (previous and epoch==start and epoch>epochs-int(args.close_mosaic))) and hasattr(loaders['train'].dataset,'close_mosaic'):
            loaders['train'].dataset.close_mosaic(hyp=copy.deepcopy(args))
        for i,batch in enumerate(loaders['train']):
            ni=epoch*nb+i;accumulate=max(round(args.nbs/int(args.batch)),1)
            if ni<nw:
                accumulate=max(1,int(np.interp(ni,[0,nw],[1,args.nbs/int(args.batch)]).round()))
                for group in optimizer.param_groups:
                    group['lr']=float(np.interp(ni,[0,nw],[args.warmup_bias_lr if group.get('param_group')=='bias' else 0.,group['initial_lr']*lf(epoch)]))
            images,targets=batch_inputs(batch,device)
            with torch.autocast(device_type=device.type,dtype=torch.float16,enabled=amp):output=model(images,targets)
            with torch.autocast(device_type=device.type,enabled=False):losses=loss_fn(float_outputs(output),targets)
            loss=sum(losses.values())
            if not torch.isfinite(loss):raise RuntimeError('R50 loss no finita; ultimo checkpoint valido conservado')
            scaler.scale(loss).backward()
            if ni-last_opt_step>=accumulate:
                scaler.unscale_(optimizer);torch.nn.utils.clip_grad_norm_(model.parameters(),10.)
                scaler.step(optimizer);scaler.update();optimizer.zero_grad(set_to_none=True);ema.update(model);last_opt_step=ni
            total_full+=float(loss.detach())
            for k in ('loss_vfl','loss_bbox','loss_giou'):sums[k]=sums.get(k,0)+float(losses[k].detach())
        metrics={} if fixed_epochs else validate(ema.ema,loss_fn,loaders['val'],device,float(cfg['detector']['val_conf']),amp)
        row=dict(epoch=epoch+1,optimization_loss_total=total_full/nb,train_loss=sum(sums.values())/nb)
        row.update({'train/'+k[5:]+'_loss':v/nb for k,v in sums.items()});row.update(metrics)
        for k,alias in [('metrics/mAP50-95(B)','map50_95'),('metrics/mAP50(B)','map50'),('metrics/precision(B)','precision'),('metrics/recall(B)','recall')]:
            if k in metrics:row[alias]=metrics[k]
        row['f1']=2*row['precision']*row['recall']/(row['precision']+row['recall']) if row.get('precision',0)+row.get('recall',0)>0 else 0.
        history.append(row)
        score=float(metrics.get('metrics/mAP50-95(B)',0.))
        if fixed_epochs or score>best_score:
            best_score=score;best_epoch=epoch+1;best_metrics=copy.deepcopy(metrics)
            best_state={k:v.detach().cpu().clone() for k,v in ema.ema.state_dict().items()}
        stopped=False if fixed_epochs else stopper(epoch+1,score)
        state=dict(format=FORMAT,signature=signature,seed=seed,hp=dict(hp),fold=fold,epoch=epoch+1,
                   model={k:v.detach().cpu() for k,v in model.state_dict().items()},ema={k:v.detach().cpu() for k,v in ema.ema.state_dict().items()},
                   ema_updates=ema.updates,optimizer=optimizer.state_dict(),scheduler=scheduler.state_dict(),scaler=scaler.state_dict(),
                   gradients={n:p.grad.detach().cpu() for n,p in model.named_parameters() if p.grad is not None},rng=rng_state(generator),
                   history=history,best_score=best_score,best_epoch=best_epoch,best_state=best_state,best_metrics=best_metrics,
                   stopper_epoch=stopper.best_epoch,stopper_fitness=stopper.best_fitness,last_opt_step=last_opt_step,
                   elapsed=elapsed+time.perf_counter()-t0,stopped=stopped)
        save_resume(out,state,spec,cfg)
        pd.DataFrame(history).to_csv(out/'results.csv',index=False,lineterminator='\n')
        log('RT-DETR-R50-vd | fold=%s epoca=%d/%d seed=%d mAP50-95=%s mAP50=%s P=%s R=%s loss=%.6g'%(fold,epoch+1,epochs,seed,row.get('map50_95'),row.get('map50'),row.get('precision'),row.get('recall'),row['train_loss']))
        if epoch_callback is not None and not fixed_epochs:epoch_callback(step_offset+epoch,row)
        if stopped:break
    seconds=elapsed+time.perf_counter()-t0
    if best_state is None:raise RuntimeError('No hay checkpoint calculado, no se publican resultados')
    pd.DataFrame(history).to_csv(out/'results.csv',index=False,lineterminator='\n')  # recover export after last-epoch power loss without retraining
    chosen=out/'best.pt'
    payload=dict(format=FORMAT,model=best_state,pretrained_sha256=WEIGHT_SHA,official_revision=REV,signature=signature,
                 experiment_fingerprint=fingerprint(cfg),best_epoch=best_epoch,metrics=best_metrics)
    if not chosen.is_file():atomic_torch_save(payload,chosen);receipt(chosen,spec,cfg,dict(seed=seed,hp=hp,context={'fold':fold},epoch=best_epoch))
    else:
        from utils.resume import require_checkpoint
        require_checkpoint(chosen,spec,cfg)
        old=torch.load(chosen,map_location='cpu',weights_only=False)
        if old.get('signature')!=signature or old.get('best_epoch')!=best_epoch:raise RuntimeError('Best previo distinto; se conserva')
    stats=dict(params_m=sum(p.numel() for p in model.parameters())/1e6,gflops=None)
    from utils.model_stats import count_gflops
    stats['gflops']=count_gflops(ema.ema,(1,3,int(cfg['detector']['image_size']),int(cfg['detector']['image_size'])))
    final=dict(best_metrics)
    for k,alias in [('metrics/mAP50-95(B)','map50_95'),('metrics/mAP50(B)','map50'),('metrics/precision(B)','precision'),('metrics/recall(B)','recall')]:
        if k in final:final[alias]=final[k]
    p,r=final.get('precision',0),final.get('recall',0);final['f1']=2*p*r/(p+r) if p+r else 0.
    atomic_json(out/'adapter.json',dict(format=FORMAT,source_revision=REV,pretrained_sha256=WEIGHT_SHA,adapter_sha256=sha(__file__),
          augmentation='RTDETRDataset Ultralytics 8.4.170, sin cambiar parametros',loss_comparison='train/val componentes primarias; total optimizado incluye auxiliares y denoising',
          resume_policy='Dos generaciones de optimizador, EMA, scaler, RNG y gradientes pendientes; best conservado',head_change='COCO80 -> fruit1 solamente'))
    vram=torch.cuda.max_memory_allocated()/1024**2 if device.type=='cuda' else None
    return DetectorRun(score=None if fixed_epochs else best_score,best_epoch=best_epoch,epochs_run=len(history),stopped_early=len(history)<epochs,
          history=pd.DataFrame(history),final_metrics=final,train_seconds=seconds,vram_peak_mb=vram,micro_batch=int(args.batch),
          weights_path=str(chosen),weights_sha256=sha(chosen),pretrained_sha256=WEIGHT_SHA,extra={'official_revision':REV},**stats)


def predict(model,images,*,conf,iou,cfg,max_det=300):
    import cv2
    import numpy as np
    import torch
    from torchvision.ops import box_convert
    if not getattr(model,'_botrytis_official_r50',False):
        return ORIGINALS['predict'](model,images,conf=conf,iou=iou,cfg=cfg,max_det=max_det)
    model.eval();device=next(model.parameters()).device;size=int(cfg['detector']['image_size']);results=[]
    for image in images:
        arr=np.asarray(image);height,width=arr.shape[:2]
        x=cv2.resize(arr,(size,size),interpolation=cv2.INTER_LINEAR)
        x=torch.from_numpy(np.ascontiguousarray(x.transpose(2,0,1))).unsqueeze(0).to(device).float()/255
        with torch.no_grad(),torch.autocast(device_type=device.type,dtype=torch.float16,enabled=device.type=='cuda' and cfg['detector']['precision']=='fp16'):
            output=model(x)
        boxes=box_convert(output['pred_boxes'][0].float(),'cxcywh','xyxy')
        scores=output['pred_logits'][0].float().sigmoid().squeeze(-1)
        if not torch.isfinite(boxes).all() or not torch.isfinite(scores).all():raise RuntimeError('Predicciones oficiales no finitas')
        keep=torch.argsort(scores,descending=True)[:max_det];keep=keep[scores[keep]>float(conf)]
        boxes=boxes[keep]*torch.tensor([width,height,width,height],device=device)
        results.append((boxes.cpu().numpy().astype(float),scores[keep].cpu().numpy().astype(float)))
    return results


def install_hooks(cfg):
    import torch
    import utils.detector_train as dt
    import utils.inference as infer
    from utils.logging_utils import RunContext
    if ORIGINALS:return
    ORIGINALS.update(train=dt.train_detector,load=infer.load_detector,predict=infer.predict_detector,record=RunContext.record_checkpoint)
    dt.train_detector=train_detector
    def load(path,spec):
        if spec.slug!='rtdetr_r50':return ORIGINALS['load'](path,spec)
        value=torch.load(path,map_location='cpu',weights_only=False)
        if value.get('format')!=FORMAT:raise RuntimeError('No es checkpoint generado por el backend oficial')
        from utils.experiments import fingerprint
        if value.get('experiment_fingerprint')!=fingerprint(cfg):raise RuntimeError('Checkpoint de otra ronda')
        model=build_model(cfg['weights']['r50_source'],None)
        model.load_state_dict(value['model'],strict=True);model._botrytis_official_r50=True
        return model.to('cuda:0' if torch.cuda.is_available() else 'cpu').eval()
    def record(ctx,role,*,sha256,path=None,**extra):
        if ctx.cfg.get('experiment',{}).get('architecture')=='rtdetr_r50' and extra.get('fold') is not None:
            retained=Path(ctx.run_dir)/('fold'+str(extra['fold']))/'best.pt'
            if retained.is_file() and sha(retained)==sha256:
                path=retained;extra.update(weights_discarded=False,retained_path=str(retained),official_revision=REV)
        return ORIGINALS['record'](ctx,role,sha256=sha256,path=path,**extra)
    RunContext.record_checkpoint=record
    infer.load_detector=load;infer.predict_detector=predict
    import utils.hpo_runner as runner
    ORIGINALS.update(evaluate=runner.evaluate_config,hpo_train=runner.train_detector)
    runner.train_detector=train_detector
    def evaluate(spec,hp,c,ctx,folder,**kw):
        result=ORIGINALS['evaluate'](spec,hp,c,ctx,folder,**kw)
        if spec.slug=='rtdetr_r50':
            from utils.resume import require_checkpoint
            for row in result.folds:
                role='fold'+str(row['fold'])+'_best'
                if any(r['role']==role for r in ctx.checkpoints):continue
                retained=Path(folder)/('fold'+str(row['fold']))/'best.pt'
                require_checkpoint(retained,spec,c)
                payload=torch.load(retained,map_location='cpu',weights_only=False)
                if payload.get('best_epoch')!=row['best_epoch'] or not math.isclose(payload['metrics']['metrics/mAP50-95(B)'],row['score'],abs_tol=1e-9):
                    raise RuntimeError('Checkpoint retenido discrepa del fold cacheado')
                ctx.record_checkpoint(role,sha256=sha(retained),path=retained,fold=row['fold'],weights_discarded=False,retained_path=str(retained))
        return result
    runner.evaluate_config=evaluate


def selftest(source,weight,device):
    import torch
    with torch.random.fork_rng(devices=[0] if device.type=='cuda' else []):
        torch.manual_seed(42)
        model=build_model(source,weight).to(device)
        loss_fn=criterion(source).to(device)
        model.eval()
        with torch.no_grad():output=model(torch.zeros(1,3,640,640,device=device))
        if output['pred_logits'].shape!=(1,300,1) or output['pred_boxes'].shape!=(1,300,4):raise RuntimeError('Geometria del modelo no coincide')
        if not all(torch.isfinite(x).all() for x in output.values()):raise RuntimeError('Forward oficial no finito')
        model.train()
        target=[dict(labels=torch.tensor([0],device=device),boxes=torch.tensor([[.5,.5,.2,.2]],device=device)) for _ in range(2)]
        # Actual 640x640 on the user's GPU; smaller CPU input only for developer verification.
        size=640 if device.type=='cuda' else 160  # >=300 encoder tokens for the 300 official queries
        with torch.autocast(device_type=device.type,dtype=torch.float16,enabled=device.type=='cuda'):
            out=model(torch.rand(2,3,size,size,device=device),target)
        with torch.autocast(device_type=device.type,enabled=False):losses=loss_fn(float_outputs(out),target)
        loss=sum(losses.values())
        if not torch.isfinite(loss):raise RuntimeError('Loss de autoprueba no finita')
        loss.backward()
        if any(p.grad is not None and not torch.isfinite(p.grad).all() for p in model.parameters()):raise RuntimeError('Gradiente de autoprueba no finito')
        result=dict(status='PASS',device=str(device),strict_coco_load=True,forward_shape=[1,300,1],backward_size=size,
                    synthetic=True,optimizer_steps=0,loss=float(loss.detach()),checkpoint_sha256=WEIGHT_SHA,source_revision=REV,
                    torch=torch.__version__,adapter_sha256=sha(__file__))
        result.update(cuda_runtime=torch.version.cuda,gpu=torch.cuda.get_device_name(0) if device.type=='cuda' else None)
        del model,loss_fn,output,out,losses,loss
    gc.collect()
    if device.type=='cuda':torch.cuda.empty_cache()
    emit('BACKEND_OFICIAL_VERIFICADO',**result)
    return result


def prepare_round(base,store,accept_license):
    from utils.config_loader import load_config,cfg_path
    from utils.weights import registry
    from utils.durable import atomic_json,atomic_copy
    from utils.experiments import create_round,verify_round
    store=Path(store)
    work=Path.home()/'bt-hoy-20261103';index_file=work/'rondas.json';index=read(index_file) if index_file.exists() else {}
    destination=store/'x'/base['experiment']['experiment_id']/'r50vd_01/config.yaml'
    old=index.get('rtdetr_r50')
    if old and Path(old).resolve()!=destination.resolve():
        prior=load_config(old)
        key=hashlib.sha256(str(Path(old).resolve()).casefold().encode()).hexdigest()[:8]
        for runs in (cfg_path(prior,'paths.runs_dir'),Path.home()/'bf'/key/'r'):
            model_dir=runs/'rtdetr_r50'
            if any(list(model_dir.rglob(pattern)) for pattern in ('summary.json','*.pt','study_summary.json','completed_fold.json','history.csv')):
                raise RuntimeError('Otra ronda R50 tiene trabajo previo; no se sustituye ni se reentrena: '+str(model_dir))
    catalog=copy.deepcopy(registry(base));expected=store/'catalog.json'
    row=next(r for r in catalog['models'] if r['slug']=='rtdetr_r50')
    row.update(filename='rtdetr_r50vd_6x_coco_from_paddle.pth',implementation='RT-DETR official PyTorch',
        implementation_model='rtdetr_r50vd_6x_coco.yml',source_version=REV,url=WEIGHT_URL,source='https://github.com/lyuwenyu/RT-DETR',
        license='Apache-2.0 upstream; Ultralytics data/metrics retain their AGPL-3.0 terms',size_bytes=WEIGHT_SIZE,
        official_sha256=None,checksum_prefix=None,expected_sha256=WEIGHT_SHA,checksum_basis='Computed and strict-load verified for this adapter, not upstream-published',
        compatible_pretrained=True,pretraining_dataset='COCO, complete official R50-vd EMA',availability='DESCARGABLE',block_reason='',
        comparability='Official R50-vd backend; prior catalog was an unexecutable placeholder. Cohort compatibility requires explicit review.',
        adapter_sha256=sha(__file__))
    if expected.exists() and read(expected)!=catalog:raise RuntimeError('Catalogo de otra version; no se sobrescribe: '+str(expected))
    if not expected.exists():atomic_json(expected,catalog)
    weight=store/'weights'/row['filename'];side=weight.with_suffix(weight.suffix+'.provenance.json')
    if side.exists():
        metadata=read(side)
        if metadata.get('sha256')!=WEIGHT_SHA or metadata.get('source_url')!=WEIGHT_URL or not metadata.get('license_accepted'):
            raise RuntimeError('Procedencia anterior incompatible; no se sobrescribe')
    else:
        if not accept_license:raise RuntimeError('Adquisicion requiere --accept-license, como el gestor original')
        atomic_json(side,dict(slug='rtdetr_r50',state='DESCARGADO_VERIFICADO',source_url=WEIGHT_URL,source_version=REV,
            sha256=WEIGHT_SHA,size_bytes=weight.stat().st_size,license=row['license'],license_accepted=True,
            verification_basis=row['checksum_basis']))
    source=copy.deepcopy(base)
    source['experiment']['registry_root']=str(store/'x')
    source['weights'].update(registry=str(expected),storage_dir=str(store/'weights'),r50_source=str(store/'official'))
    destination=store/'x'/source['experiment']['experiment_id']/'r50vd_01/config.yaml'
    if destination.exists():
        verify_round(load_config(destination),architecture='rtdetr_r50')
    else:
        destination=create_round(source,round_id='r50vd_01',architecture='rtdetr_r50',operator=os.environ['USERNAME'],mode='supervised',
            reason='Desbloqueo solicitado: grafo R50-vd y checkpoint COCO oficiales, cabeza fruit1, sin modificar rondas anteriores')
    if index.get('rtdetr_r50')!=str(destination):
        if index_file.exists():atomic_copy(index_file,store/('rondas-before-'+canonical(index)[:10]+'.json'))
        index['rtdetr_r50']=str(destination);atomic_json(index_file,index)
    overrides=['paths.runs_dir='+str(store/'r'),'paths.results_dir='+str(store/'e')]
    effective=load_config(destination,overrides);verify_round(effective,architecture='rtdetr_r50')
    atomic_json(store/'methodology.json',dict(official_revision=REV,adapter_sha256=sha(__file__),weight_sha256=WEIGHT_SHA,
        strict_loading='All COCO80 keys, then only class/objectness/denoising-class heads replaced for fruit1',
        changed_backend=True,changed_previous_rounds=False,changed_dataset_or_splits=False,
        training='Framework AdamW/LR/backbone ratio/epochs/batch/augmentation/metrics retained; official model and criterion',
        batchnorm='Official FrozenBatchNorm backbone retained; freeze_at=-1 honors no frozen trainable backbone layers',
        multi_scale=False,image_size=640,comparison_requires_review=True,prior_hpos_reexecuted=False,
        note='Do not claim catalog/cohort identity with older rounds; no automatic global selection or test release'))
    return destination,effective,overrides


def recover_owned_baseline(queue,job,owner_file):
    import psutil
    if job.kind!='baseline' or job.architecture!='rtdetr_r50' or job.status not in ('FAILED','RUNNING'):
        raise RuntimeError('No se recupera ni cambia este estado: '+job.status)
    if not Path(owner_file).is_file():raise RuntimeError('Falta propietario verificable de este intento; se conserva sin reiniciar')
    owner=read(owner_file)
    if owner.get('job_id')!=job.job_id or owner.get('attempt')!=job.attempt or owner.get('adapter_sha256')!=sha(__file__):
        raise RuntimeError('Propietario de otra configuracion/intento; no se recupera')
    try:
        process=psutil.Process(owner['pid'])
        if abs(process.create_time()-owner['created'])<.01:raise RuntimeError('El propietario sigue activo; no se duplica')
    except psutil.NoSuchProcess:pass
    queue.finish(job.job_id,'PENDING',error='Recuperacion de intento propio terminado; conserva caches y estados de optimizador')
    emit('BASELINE_R50_RECUPERADO',job=job.job_id,previous=job.status,training='Solo epocas/folds no completados y verificables')


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--allow-network',action='store_true');parser.add_argument('--accept-license',action='store_true')
    parser.add_argument('--check-only',action='store_true',help='Adquiere/verifica backend sin iniciar baseline')
    args=parser.parse_args()
    if os.name!='nt':raise RuntimeError('Operacion de PC-03 requiere Windows; pruebas CPU se ejecutan por separado')
    project=Path.home()/'Documents/s2_no_borrar/Entrenamiento/Botrytis_Experiment_Framework_git'
    py=project/'.venv/Scripts/python.exe'
    if Path(sys.executable).resolve()!=py.resolve():return subprocess.call([str(py),'-u',str(Path(__file__).resolve()),*sys.argv[1:]])
    os.chdir(project);sys.path.insert(0,str(project/'src'))
    os.environ.update(PYTHONUTF8='1',PYTHONIOENCODING='utf-8',MPLBACKEND='Agg',BOTRYTIS_BOOTSTRAP_DIR=str(Path.home()/'btlog'))
    os.environ.setdefault('CUBLAS_WORKSPACE_CONFIG',':4096:8')
    from utils.config_loader import load_config
    from utils.durable import FileLock,atomic_json
    from utils.guards import gpu_lock_path,GpuJobLock
    from utils.manual_gates import require_gate
    from orchestration.machine import read_binding
    from orchestration.pc_registry import PCRegistry
    from orchestration.study_queue import StudyQueue
    import psutil
    import torch
    import ultralytics
    if ultralytics.__version__!='8.4.170':raise RuntimeError('Se requiere la version de datos/metricas aprobada: Ultralytics 8.4.170')
    base=load_config(project/'artifacts/v4/local.yaml');binding=read_binding(base)
    if os.environ.get('USERNAME','').casefold()!='i21313' or binding['pc_id']!='PC-03' or base['compute']['pc_id']!='PC-03':
        raise RuntimeError('Este ejecutor es exclusivo de PC-03 / I21313')
    if PCRegistry(base).owner('rtdetr_r50')!='PC-03':raise RuntimeError('RT-DETR no pertenece a esta PC')
    if base['initialization']['mode']!='pretrained' or not base['detector']['pretrained']:raise RuntimeError('Se requiere protocolo pretrained original')
    require_gate(base,'PC-03','READY_FOR_EXPERIMENT')
    store=Path.home()/'r50';store.mkdir(exist_ok=True);lock=gpu_lock_path(base,'PC-03')
    own=psutil.Process();excluded={own.pid,*(p.pid for p in own.parents())}
    with FileLock(lock.with_suffix('.r50.guard')):
        try:
            with FileLock(lock.with_suffix('.priority.guard')):pass
        except RuntimeError:
            raise RuntimeError('El controlador de prioridad sigue activo; dejarlo terminar antes de iniciar R50') from None
        source=acquire_source(store,args.allow_network)
        weight=download(WEIGHT_URL,store/'weights/rtdetr_r50vd_6x_coco_from_paddle.pth',WEIGHT_SHA,args.allow_network)
        if weight.stat().st_size!=WEIGHT_SIZE:raise RuntimeError('Tamano del checkpoint oficial incorrecto')
        previous=None
        while True:
            running=[p.pid for p in psutil.process_iter(['pid','name']) if p.pid not in excluded and (p.info['name'] or '').lower() in ('python.exe','pythonw.exe')]
            if not running:break
            if running!=previous:emit('ESPERANDO_GPU_LIBRE_SIN_INTERRUMPIR',processes=running);previous=running
            time.sleep(3)
        with FileLock(lock.with_suffix('.final.guard')):
            if not torch.cuda.is_available() or torch.cuda.device_count()!=1:raise RuntimeError('Se requiere una GPU CUDA; no se permite caer a CPU')
            verified=read(store/'backend_verified.json') if (store/'backend_verified.json').is_file() else {}
            if any(verified.get(k)!=v for k,v in dict(adapter_sha256=sha(__file__),torch=torch.__version__,cuda_runtime=torch.version.cuda,gpu=torch.cuda.get_device_name(0)).items()):
                with GpuJobLock(base,'PC-03',job='selftest:rtdetr_r50_official'):
                    test=selftest(source,weight,torch.device('cuda:0'));atomic_json(store/'backend_verified.json',test)
            config,cfg,overrides=prepare_round(base,store,args.accept_license)
            require_gate(cfg,'PC-03','READY_FOR_EXPERIMENT')
            q=StudyQueue.from_cfg(cfg)
            from utils.architectures import get_arch
            q.plan(cfg,[get_arch('rtdetr_r50')])
            baseline=next(j for j in q.jobs() if j.kind=='baseline')
            if args.check_only:
                emit('RTDETR_HABILITADO_BASELINE_NO_INICIADO',config=config,report=store/'methodology.json');return 0
            if baseline.status in ('COMPLETE','PENDING_REVIEW','PUBLICATION_PENDING'):
                emit('BASELINE_R50_YA_CALCULADO_NO_SE_REPITE',status=baseline.status,results=store/'r/rtdetr_r50/baseline');return 0
            if baseline.status not in ('PENDING','FAILED','RUNNING'):
                raise RuntimeError('Estado previo requiere recuperacion explicita, no se relanza: '+baseline.status)
            owner_file=store/'owner.json'
            if baseline.status in ('FAILED','RUNNING'):
                if baseline.attempt>=int(cfg.get('operations',{}).get('max_job_attempts',3)):raise RuntimeError('Limite de intentos; revisar error anterior')
                atomic_json(store/'retry.json',dict(job=baseline.as_json(),authorization='Usuario solicita continuar; sin procesos Python activos y locks locales disponibles',operator=os.environ['USERNAME']))
                recover_owned_baseline(q,baseline,owner_file)
            install_hooks(cfg)
            module=importlib.import_module('06_run_baseline')
            native=['--architecture','rtdetr_r50','--pc-id','PC-03','--config',str(config)]
            for value in overrides:native.extend(['--set',value])
            emit('INICIANDO_BASELINE_RTDETR_OFICIAL',config=config,results=store/'r/rtdetr_r50/baseline',folds=3,
                 epochs=cfg['detector']['baseline']['epochs'],hpo='No se inicia; requiere revision posterior del baseline')
            previous_run=store/'r/rtdetr_r50/baseline'
            attempt=store/'attempts'/uuid.uuid4().hex[:8]
            for filename in ('metadata.json','checkpoint_metadata.json','execution.log'):
                old=previous_run/filename
                if old.is_file():
                    attempt.mkdir(parents=True,exist_ok=True);shutil.copy2(old,attempt/filename)
            attempt.mkdir(parents=True,exist_ok=True)
            atomic_json(attempt/'queue-before.json',q.transfer_snapshot('rtdetr_r50'))
            atomic_json(owner_file,dict(job_id=baseline.job_id,attempt=baseline.attempt+1,pid=own.pid,created=own.create_time(),adapter_sha256=sha(__file__)))
            rc=module.run_cli(module.SCRIPT,module.main,argv=native,description=module.__doc__,add_arguments=module.add_arguments,run_layout=module.layout_for('baseline'))
            atomic_json(store/'summary.json',dict(exit_code=rc,queue=q.counts(),jobs=[j.as_json() for j in q.jobs()],
                 results=str(store/'r/rtdetr_r50/baseline'),methodology=str(store/'methodology.json'),adapter_sha256=sha(__file__),
                 final_validation=False,previous_three_hpos_untouched=True))
            emit('FIN_BASELINE_R50',code=rc,report=store/'summary.json',status=q.counts())
            return rc


if __name__=='__main__':
    try:sys.exit(main())
    except KeyboardInterrupt:
        emit('INTERRUMPIDO',message='Checkpoint valido conservado; no borrar locks ni reiniciar desde COCO');sys.exit(130)
    except Exception as error:
        import traceback
        emit('DETENIDO_SIN_ALTERAR_VIT',error=str(error));traceback.print_exc();sys.exit(2)
