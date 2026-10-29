"""Weight inventory and strictly offline initialization. Network is a separate manual command."""
from __future__ import annotations
import hashlib,json,os
from pathlib import Path

STATES=('NO_DISPONIBLE','DESCARGABLE','DESCARGADO_VERIFICADO','GENERADO_POR_ENTRENAMIENTO','NO_EXISTE_FUENTE_FIABLE')
ROOT=Path(__file__).resolve().parents[2]
def digest(p):
    h=hashlib.sha256()
    with Path(p).open('rb') as f:
        for b in iter(lambda:f.read(1<<20),b''):h.update(b)
    return h.hexdigest()
def registry(cfg=None):
    n=(cfg or {}).get('weights',{}).get('registry','configs/weights_registry.json')
    p=Path(n);p=p if p.is_absolute() else ROOT/p
    return json.loads(p.read_text(encoding='utf-8'))
def entry_for(spec,cfg=None):
    return next(r for r in registry(cfg)['models'] if r['slug']==spec.slug)
def weight_path(row,cfg=None):
    storage=Path((cfg or {}).get('weights',{}).get('storage_dir','weights'))
    return (storage if storage.is_absolute() else ROOT/storage)/row['filename']
def verified_local_weight(spec,cfg=None):
    r=entry_for(spec,cfg)
    if not r['compatible_pretrained']:
        raise RuntimeError('BLOQUEADO: '+r['block_reason']+'; decision manual en docs/PESOS.md')
    p=weight_path(r,cfg);side=p.with_suffix(p.suffix+'.provenance.json')
    if not p.is_file() or not side.is_file():
        raise FileNotFoundError(f'NO_DISPONIBLE: {spec.name}. Adquiera y verifique manualmente con tools/manage_weights.py. Esperado {p}')
    m=json.loads(side.read_text(encoding='utf-8'))
    if m.get('state')!='DESCARGADO_VERIFICADO' or m.get('slug')!=spec.slug or m.get('source_url')!=r['url']:
        raise RuntimeError('Proveniencia incompatible de pesos: '+str(side))
    if m.get('sha256')!=digest(p) or m.get('size_bytes')!=p.stat().st_size:
        raise RuntimeError('Pesos alterados; repetir verificacion manual: '+str(p))
    from utils.bootstrap import event
    event("verified_weight_loaded",architecture=spec.slug,path=str(p.resolve()),sha256=m["sha256"],size_bytes=m["size_bytes"],source_url=r["url"],initialization="pretrained")
    return p.resolve()
def initialization_source(spec,cfg):
    mode=cfg.get('initialization',{}).get('mode')
    if mode=='pretrained':
        p=verified_local_weight(spec,cfg);return p,digest(p)
    if mode!='scratch' or not cfg['initialization'].get('scratch_decision'):
        raise RuntimeError('BLOQUEADO: modalidad no predeclarada')
    if spec.family=='rtdetr' and not cfg['initialization'].get('rtdetr_variant_decision'):
        raise RuntimeError('BLOQUEADO: Ultralytics ResNet50 es variante de R50-vd original; requiere decision manual explicita')
    return ('rtdetr-resnet50.yaml' if spec.family=='rtdetr' else spec.slug+'.yaml'),None
def inventory(cfg=None):
    rows=[]
    for r in registry(cfg)['models']:
        from types import SimpleNamespace
        p=weight_path(r,cfg);state='NO_DISPONIBLE';reason='Estado inicial: cero pesos'
        if p.is_file():
            try:verified_local_weight(SimpleNamespace(slug=r['slug'],name=r['name']),cfg);state='DESCARGADO_VERIFICADO';reason='Archivo local y procedencia coinciden'
            except (OSError,RuntimeError,ValueError) as exc:reason=str(exc)
        rows.append({**r,'state':state,'path':str(p),'reason':reason})
    return rows
