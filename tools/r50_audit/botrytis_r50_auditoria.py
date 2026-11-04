"""Auditoria local y consulta de RT-DETR-R50-vd; no entrena ni aprueba.
Modos audit/status. HPO queda preparado y pendiente de APROBAR explicito.
"""
from __future__ import annotations
import argparse,copy,hashlib,html,importlib.util,json,math,os,re,shutil,sqlite3,subprocess,sys,tempfile,traceback,uuid
from datetime import datetime,timezone
from pathlib import Path
MODEL='rtdetr_r50';PC='PC-03';USER='i21313';BUDGET=15
EXPECTED_FP='8aa24ae8da9a178a3d864f3e447667f52be6bbda55691abc585dff3d8c654352'
ADAPTER_SHA='e62e47fdc4cdbe6648d0830a8efac9701ab04ae82b77757d4a0880ec73381df7'
EXPECTED_DATA='f9cc08785271eecb6566669e1eff7fc1dcb821b8a8a407592a6153fe248018ed'
EXPECTED_SPLIT='617a3fc380835763c6f43f84e770df586d96d06aab7de9101a10c44b072ee64b'
EXECUTOR_SHA256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest()
class Stop(RuntimeError):pass
class MissingEvidence(Stop):pass


def utc(): return datetime.now(timezone.utc).isoformat()

def read(p):
    try: return json.loads(Path(p).read_text(encoding='utf-8-sig'))
    except (OSError, ValueError) as e: raise Stop(f'Evidencia ilegible/ausente: {p}: {e}') from e

def digest(p):
    h = hashlib.sha256()
    with Path(p).open('rb') as f:
        for b in iter(lambda: f.read(1 << 20), b''): h.update(b)
    return h.hexdigest()

def canonical(x): return hashlib.sha256(json.dumps(x, sort_keys=True, separators=(',', ':'), ensure_ascii=False, default=str).encode()).hexdigest()

def atomic_bytes(p, data):
    p = Path(p); p.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp = tempfile.mkstemp(dir=str(p.parent), prefix='.bt-', suffix='.tmp')
    try:
        with os.fdopen(fd, 'wb') as f: f.write(data); f.flush(); os.fsync(f.fileno())
        os.replace(tmp, p)
    finally:
        if os.path.exists(tmp): os.unlink(tmp)

def atomic_json(p, data): atomic_bytes(p, (json.dumps(data, ensure_ascii=False, indent=2, allow_nan=False, default=str) + '\n').encode())

def emit(message, **fields):
    print(message + (' ' + json.dumps(fields, ensure_ascii=False, default=str) if fields else ''), flush=True)




def verified_copy(src, dst):
    src, dst = Path(src), Path(dst)
    if src.is_symlink(): raise Stop('No se copia un enlace como evidencia: ' + str(src))
    if dst.exists():
        if digest(src) != digest(dst): raise Stop('Conflicto; se preservan ambos archivos: ' + str(dst))
        return
    dst.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp = tempfile.mkstemp(dir=str(dst.parent), prefix='.copy-', suffix='.tmp'); os.close(fd)
    try:
        shutil.copy2(src, tmp)
        if digest(src) != digest(tmp): raise Stop('La fuente cambio mientras se copiaba: ' + str(src))
        os.replace(tmp, dst)
    finally:
        if os.path.exists(tmp): os.unlink(tmp)


def git(root, *args):
    return subprocess.check_output(['git', '-C', str(root), *args], text=True, stderr=subprocess.PIPE).strip()


def fold_signature(experiment, model, hp, seed, fold, epochs, patience):
    from utils.hashing import sha256_json
    return sha256_json(dict(experiment=experiment, architecture=model, hp=dict(hp), seed=seed, fold=fold, epochs=epochs, patience=patience))

def verify_fold(folder, signature, fold):
    import pandas as pd
    folder = Path(folder); receipt = read(folder / 'completed_fold.json'); hist = folder / 'completed_history.csv'
    if receipt.get('signature') != signature: raise Stop(f'Fold {fold}: firma de otra ronda/configuracion')
    if not hist.is_file() or digest(hist) != receipt.get('history_sha256'): raise Stop(f'Fold {fold}: historial ausente/alterado: {hist}')
    row = receipt.get('row', {})
    if row.get('fold') != fold: raise Stop('Recibo de fold incorrecto: ' + str(folder))
    h = pd.read_csv(hist)
    if h.empty or len(h) != int(row.get('epochs_run', -1)): raise Stop(f'Fold {fold}: numero de epocas inconsistente')
    if 'fold' not in h or set(h['fold'].astype(int)) != {fold} or h['epoch'].duplicated().any(): raise Stop(f'Fold {fold}: filas duplicadas o mezcladas')
    epochs_actual = h['epoch'].astype(int).tolist()
    if epochs_actual != list(range(1, len(h) + 1)): raise Stop(f'Fold {fold}: epocas incompletas o desordenadas')
    if not math.isfinite(float(row.get('score', float('nan')))) or not 0 <= float(row['score']) <= 1: raise Stop(f'Fold {fold}: metrica invalida')
    return receipt, h

def validate_checkpoint(path, model, cfg, seed=None, hp=None, fold=None):
    """Use the actual repository receipt validator; never load an unsigned pickle."""
    from utils.architectures import get_arch
    from utils.resume import require_checkpoint
    p = require_checkpoint(Path(path), get_arch(model), cfg)
    meta = read(p.with_suffix(p.suffix + '.resume.json')); extra = meta.get('extra', {})
    if seed is not None and extra.get('seed') != seed: raise Stop('Checkpoint de otra semilla: ' + str(p))
    if hp is not None and extra.get('hp') is not None and extra['hp'] != hp: raise Stop('Checkpoint de otros hiperparametros: ' + str(p))
    context = extra.get('context', {})
    if fold is not None and context and context.get('fold') != fold: raise Stop('Checkpoint de otro fold: ' + str(p))
    return p, meta







def partition_evidence(cfg, spec):
    """Audit outer AND inner partitions. Metadata describes physical groups only when explicit."""
    import pandas as pd
    from utils.config_loader import cfg_path
    from utils.splits import load_split_manifest
    from utils.folds import load_inner_folds, DET_FOLD_COL, CLS_FOLD_COL
    art = cfg_path(cfg, 'paths.artifacts_dir'); m = load_split_manifest(art, cfg['outer_split']['manifest_name'])
    det, cls = load_inner_folds(art); inv_path = art/'dataset_inventory.csv'; inv = pd.read_csv(inv_path, dtype={'dhash':str})
    if inv.image_id.duplicated().any() or set(inv.image_id)!=set(m.image_id): raise Stop('Inventario/manifiesto no corresponden uno a uno')
    outer = m.set_index('image_id').outer_split.to_dict(); di = det.set_index('image_id')[DET_FOLD_COL].to_dict(); ci = cls.set_index('image_id')[CLS_FOLD_COL].to_dict()
    train = set(m.loc[m.outer_split=='train','image_id'])
    for table,col in ((det,DET_FOLD_COL),(cls,CLS_FOLD_COL)):
        if table.image_id.duplicated().any() or set(table.image_id)!=train or set(table[col])!={0,1,2}: raise Stop('Folds ausentes/duplicados/incompatibles')
    def crosses(a,b):
        return dict(outer=outer[a]!=outer[b], detection_inner=a in di and b in di and di[a]!=di[b],
                    classification_inner=a in ci and b in ci and ci[a]!=ci[b])
    exact = []; audit_file = art/'split_content_audit.csv'
    if not audit_file.is_file(): raise Stop('Falta auditoria original de contenido: ' + str(audit_file))
    audit = pd.read_csv(audit_file)
    if audit.image_id.duplicated().any() or set(audit.image_id)!=set(m.image_id): raise Stop('Auditoria de contenido incompleta')
    for key in ('file_sha256','pixel_sha256'):
        if key not in audit or audit[key].isna().any(): raise Stop('Falta hash de contenido: '+key)
        for _,group in audit.groupby(key):
            ids = list(group.image_id)
            for i,a in enumerate(ids):
                for b in ids[i+1:]:
                    cross = crosses(a,b)
                    if any(cross.values()): exact.append(dict(a=a,b=b,hash_type=key,crosses=cross))
    near = []; threshold = int(cfg['dataset']['validation']['near_duplicate_hamming'])
    if 'dhash' not in inv or inv.dhash.isna().any(): raise Stop('Faltan hashes perceptuales; no se declara revisada la similitud')
    hashes = [(str(r.image_id),int(str(r.dhash),16)) for r in inv.itertuples()]
    native_review = read(art/'near_duplicate_review.json') if (art/'near_duplicate_review.json').is_file() else {}
    from utils.dataset import recorded_dataset_hash
    decisions = native_review.get('pairs',[]) if native_review.get('dataset_hash')==recorded_dataset_hash(art) else []
    for i,(a,ha) in enumerate(hashes):
        for b,hb in hashes[i+1:]:
            distance = (ha^hb).bit_count(); cross = crosses(a,b)
            if distance<=threshold and any(cross.values()):
                decision = next((r for r in decisions if {r.get('image_a'),r.get('image_b')}=={a,b}),{})
                reviewed = decision.get('decision')=='distinct_sources' and bool(decision.get('evidence'))
                near.append(dict(a=a,b=b,dhash_hamming=distance,crosses=cross,outer=[outer[a],outer[b]],
                                 detection_folds=[di.get(a),di.get(b)],classification_folds=[ci.get(a),ci.get(b)],
                                 disposition='REVIEWED_DISTINCT_SOURCES' if reviewed else 'SUSPECTED_NOT_VISUALLY_CONFIRMED',
                                 existing_review=decision if reviewed else None))
    groups = []; conflicts = []; metadata_sources = [inv]
    image_manifest = art/'image_manifest.csv'
    if image_manifest.is_file(): metadata_sources.append(pd.read_csv(image_manifest))
    # Optional documented evidence, never inferred from sequential filenames.
    meta_path = Path.home()/'bt-hoy-20261103'/'capture_groups.csv'
    if meta_path.is_file(): metadata_sources.append(pd.read_csv(meta_path))
    for frame in metadata_sources:
        for col in ('plant_id','leaf_id','physical_fruit_id','session_id','capture_session_id','specimen_id'):
            if col not in frame or 'image_id' not in frame: continue
            t = frame[['image_id',col]].dropna().drop_duplicates()
            if (t.groupby('image_id')[col].nunique()>1).any(): raise Stop('Metadatos de origen contradictorios: '+col)
            t = t[t.image_id.isin(outer)];groups.append(dict(column=col,images_covered=int(t.image_id.nunique()),total_images=len(m)))
            for value,group in t.groupby(col):
                ids=list(group.image_id)
                for i,a in enumerate(ids):
                    for b in ids[i+1:]:
                        cross=crosses(a,b)
                        if any(cross.values()): conflicts.append(dict(group_column=col,group_value=str(value),a=a,b=b,crosses=cross))
    from utils.classifier_data import load_crop_manifest
    crops = load_crop_manifest(cfg)
    if not crops.image_id.map(outer).eq(crops.outer_split).all(): raise Stop('Un crop no hereda el split de su imagen')
    if crops.loc[crops.outer_split=='train','image_id'].map(ci).ne(crops.loc[crops.outer_split=='train','inner_fold']).any(): raise Stop('Un crop no hereda el fold de su imagen')
    relevant = 'detection_inner' if spec.is_detector else 'classification_inner'
    confirmed = [r for r in exact+conflicts if r['crosses']['outer'] or r['crosses'][relevant]]
    suspects = [r for r in near if (r['crosses']['outer'] or r['crosses'][relevant]) and r['disposition']=='SUSPECTED_NOT_VISUALLY_CONFIRMED']
    return dict(status='CONFIRMED_CONFLICT' if confirmed else ('SUSPECTED_SOURCE_OVERLAP' if suspects else 'NO_DETECTED_OVERLAP'),
                outer_counts=m.outer_split.value_counts().to_dict(),inner_detection_counts=det[DET_FOLD_COL].value_counts().to_dict(),
                inner_classification_counts=cls[CLS_FOLD_COL].value_counts().to_dict(),exact_content_conflicts=exact,
                perceptual_pairs=near,group_metadata=groups,physical_group_conflicts=conflicts,
                confirmed_relevant_conflicts=confirmed,suspected_relevant_pairs=suspects,
                physical_independence='NOT_VERIFIABLE_WITH_AVAILABLE_METADATA' if not groups else 'CHECKED_ONLY_FOR_EXPLICIT_METADATA',
                limitation='dHash no acredita identidad ni detecta todas las vistas del mismo ejemplar; ausencia de coincidencia no prueba independencia biologica.',
                crop_split_and_fold_inheritance=True,hashes_from_original_inventory=True)


def read_only_sqlite(path):
    path=Path(path)
    if not path.is_file():raise MissingEvidence('Falta SQLite original: '+str(path))
    c=sqlite3.connect(path.resolve().as_uri()+'?mode=ro',uri=True,timeout=10);c.row_factory=sqlite3.Row
    c.execute('PRAGMA query_only=ON');return c


def read_jobs(cfg):
    from utils.config_loader import cfg_path
    with read_only_sqlite(cfg_path(cfg,'paths.queue_dir')/'queue.sqlite') as c:
        if c.execute('PRAGMA integrity_check').fetchone()[0]!='ok':raise Stop('Cola SQLite corrupta')
        rows=[dict(r) for r in c.execute('SELECT * FROM jobs')]
        fingerprint_row=c.execute("SELECT value FROM context WHERE key='fingerprint'").fetchone()
    if not fingerprint_row or fingerprint_row[0]!=EXPECTED_FP:raise Stop('Cola de otra identidad')
    if any(r['architecture']!=MODEL for r in rows):raise Stop('Otra arquitectura en la cola; no se modifica')
    selected={}
    for kind in ('baseline','hpo'):
        found=[r for r in rows if r['kind']==kind]
        if len(found)!=1:raise Stop('Job ausente o ambiguo: '+kind)
        selected[kind]=found[0];payload=json.loads(found[0]['payload'])
        if payload.get('fingerprint')!=EXPECTED_FP or payload.get('round_id')!='r50vd_01':raise Stop('Job de otra ronda')
    if json.loads(selected['hpo']['depends_on'])!=[selected['baseline']['job_id']]:raise Stop('Dependencia HPO-baseline alterada')
    return selected


def readonly_code(project):
    from utils.hashing import compute_code_hash
    # Account for LF/CRLF without refreshing the index during the audit.
    raw=subprocess.check_output(['git','--no-optional-locks','-C',str(project),'status','--porcelain','-z','--untracked-files=no'])
    equivalent=[]
    for item in raw.split(b'\0'):
        if not item:continue
        if item[:3]!=b' M ':raise Stop('Cambios Git reales/preparados; se conservan')
        rel=os.fsdecode(item[3:])
        if git(project,'rev-parse','HEAD:'+rel)!=git(project,'hash-object','--path='+rel,str(Path(project)/rel)):
            raise Stop('Codigo modificado: '+rel)
        equivalent.append(rel)
    return dict(git_commit=git(project,'rev-parse','HEAD'),code_hash=compute_code_hash(project),eol_equivalent=equivalent,index_refreshed=False)


def load_adapter():
    path=Path(__file__).with_name('botrytis_r50_oficial_pinned.py')
    if not path.is_file():raise MissingEvidence('Falta adaptador oficial fijado: '+str(path))
    if digest(path)!=ADAPTER_SHA:raise Stop('Adaptador diferente del que entreno el baseline')
    spec=importlib.util.spec_from_file_location('botrytis_r50_pinned',path)
    module=importlib.util.module_from_spec(spec);sys.modules[spec.name]=module;spec.loader.exec_module(module)
    return module


def tables_audit(metrics,fold_metrics,history):
    import numpy as np
    import pandas as pd
    h=history.copy();f=fold_metrics.copy();m=metrics
    if not {'fold','epoch'}.issubset(h) or 'fold' not in f:raise MissingEvidence('Falta identificador fold/epoch')
    if h.empty or f.empty:raise MissingEvidence('CSV vacio')
    if m.get('slug')!=MODEL or m.get('objective')!='map50_95' or m.get('folds')!=[0,1,2]:raise Stop('Resumen de otra tarea/arquitectura')
    if m.get('dataset_hash')!=EXPECTED_DATA or m.get('split_hash')!=EXPECTED_SPLIT or m.get('experiment_fingerprint')!=EXPECTED_FP:raise Stop('Hashes/identidad del resumen incompatibles')
    if m.get('seed')!=42 or m.get('max_epochs')!=100 or m.get('patience')!=20:raise Stop('Baseline de otro protocolo')
    if any(len(m.get(k,[]))!=3 for k in ('fold_scores','best_epochs','epochs_run')):raise Stop('Resumen de folds incompleto')
    if h.duplicated(['fold','epoch']).any() or set(h.fold)!={0,1,2} or f.fold.duplicated().any() or set(f.fold)!={0,1,2}:
        raise Stop('Folds/epocas ausentes o duplicados')
    mandatory=['train_loss','val_loss','optimization_loss_total','train/vfl_loss','train/bbox_loss','train/giou_loss','val/vfl_loss','val/bbox_loss','val/giou_loss',
        'map50_95','map50','precision','recall','f1','metrics/mAP50-95(B)','metrics/mAP50(B)','metrics/precision(B)','metrics/recall(B)']
    for col in mandatory:
        if col not in h:raise MissingEvidence('Falta columna: '+col)
        values=pd.to_numeric(h[col],errors='coerce')
        if not np.isfinite(values).all() or (values<0).any():raise Stop('Valor obligatorio no finito/negativo: '+col)
        if not ('loss' in col) and (values>1).any():raise Stop('Metrica fuera de rango: '+col)
    for split in ('train','val'):
        components=sum(h[split+'/'+c+'_loss'] for c in ('vfl','bbox','giou'))
        if not np.allclose(components,h[split+'_loss'],rtol=2e-6,atol=1e-7):raise Stop('Agregado loss no concuerda con componentes: '+split)
    for alias,col in [('map50_95','metrics/mAP50-95(B)'),('map50','metrics/mAP50(B)'),('precision','metrics/precision(B)'),('recall','metrics/recall(B)')]:
        if not np.allclose(h[alias],h[col],rtol=0,atol=1e-7):raise Stop('Aliases de metricas contradictorios: '+alias)
    if not h.map50_95.le(h.map50+1e-7).all():raise Stop('mAP50-95 mayor que mAP50')
    if 'seed' in h and set(h.seed)!={42}:raise Stop('Semillas mezcladas')
    if 'outer_split' in h and set(h.outer_split)!={'train'}:raise Stop('HPO/baseline accedio otro split')
    rows=[];scores=[]
    for fold in range(3):
        g=h[h.fold==fold];row=f[f.fold==fold].iloc[0];best=g.loc[g.map50_95.idxmax()]
        if g.epoch.tolist()!=list(range(1,len(g)+1)):raise Stop('Epocas incompletas/desordenadas')
        score=float(best.map50_95);scores.append(score);be=int(m['best_epochs'][fold])
        selected=g[g.epoch==be]
        if len(selected)!=1 or not math.isclose(float(selected.map50_95.iloc[0]),score,abs_tol=1e-7):raise Stop('Best epoch incorrecto')
        if not math.isclose(score,float(m['fold_scores'][fold]),abs_tol=1e-7) or not math.isclose(score,float(row.score),abs_tol=1e-7):raise Stop('Scores resumen/CSV distintos')
        if int(row.best_epoch)!=be or int(row.epochs_run)!=len(g) or m['epochs_run'][fold]!=len(g):raise Stop('Epocas resumen/CSV no concuerdan')
        stopped=len(g)<m['max_epochs']
        if stopped and len(g)-be!=m['patience']:raise Stop('Parada no explicada por paciencia ni limite')
        if row.stopped_early not in (True,False) or bool(row.stopped_early)!=stopped:raise Stop('Indicador de parada inconsistente')
        at60=g[g.epoch==60];early=g[g.epoch<=60]
        detail=dict(fold=fold,epochs=len(g),best_epoch=be,best_mAP50_95=float(m['fold_scores'][fold]),last_mAP50_95=float(g.map50_95.iloc[-1]),
            best_before60_observed=float(early.map50_95.max()),at60_observed=float(at60.map50_95.iloc[0]) if len(at60) else None,
            early_stopping_consistent=True,n_train=int(row.n_train),n_val=int(row.n_val),
            metrics_at_best={k:float(selected[k].iloc[0]) for k in ('map50','map50_95','precision','recall','f1')},
            losses={col:dict(first=float(g[col].iloc[0]),at_best=float(selected[col].iloc[0]),last=float(g[col].iloc[-1]),min=float(g[col].min()),max=float(g[col].max())) for col in mandatory if 'loss' in col},
            zero_detection_metric_epochs=g.loc[(g.map50==0)&(g.precision==0)&(g.recall==0),'epoch'].astype(int).tolist())
        rows.append(detail)
    if not math.isclose(float(np.mean(scores)),float(m['score_mean']),abs_tol=1e-7) or not math.isclose(float(np.std(scores,ddof=1)),float(m['score_sd']),abs_tol=1e-7):raise Stop('Media/SD no concuerdan')
    counts=h.groupby('epoch').fold.nunique()
    return dict(folds=rows,score_mean=float(m['score_mean']),score_sd=float(m['score_sd']),numeric_integrity=True,
        variable_fold_count=dict(at_epoch1=int(counts.iloc[0]),after23=int(counts.loc[24]) if 24 in counts else None,
            last_available_fold_mean=float(h[h.epoch==h.epoch.max()].map50_95.mean()),warning='La figura original cambia de 3 a 2 folds; no equivale al promedio CV de los tres mejores checkpoints.'),
        truncated100_curve_mean_before60=float(np.mean([r['best_before60_observed'] for r in rows])),
        truncation_is_not_60_epoch_training=True,learning_rates_logged=False,gradient_norms_logged=False,
        overfit='No demostrado: las perdidas totales train y validation bajan; VFL usa targets de calidad dependientes de IoU y no exige monotonia.')


def plot_by_fold(history,out):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    out=Path(out);out.mkdir(parents=True,exist_ok=True)
    fig,axes=plt.subplots(2,3,figsize=(13,7),sharex='col')
    for fold in range(3):
        g=history[history.fold==fold]
        axes[0,fold].plot(g.epoch,g.map50_95,label='mAP50-95');axes[0,fold].plot(g.epoch,g.map50,label='mAP50',alpha=.65)
        best=g.loc[g.map50_95.idxmax()];axes[0,fold].scatter([best.epoch],[best.map50_95],color='black',s=20)
        axes[0,fold].set_title('Fold '+str(fold)+' | '+str(len(g))+' epocas');axes[0,fold].set_ylim(0,1)
        axes[1,fold].plot(g.epoch,g.train_loss,label='train primaria');axes[1,fold].plot(g.epoch,g.val_loss,label='val primaria')
        for row in (0,1):axes[row,fold].grid(alpha=.2);axes[row,fold].legend(fontsize=8)
        axes[1,fold].set_xlabel('Epoca')
    fig.suptitle('RT-DETR-R50-vd: curvas individuales, sin imputar folds detenidos');fig.tight_layout()
    for suffix in ('png','svg'):fig.savefig(out/('curvas_por_fold.'+suffix),dpi=150)
    plt.close(fig)


def frozen_inventory(folder):
    folder=Path(folder)
    return {str(p.relative_to(folder)):dict(sha256=digest(p),bytes=p.stat().st_size) for p in sorted(folder.rglob('*'))
        if p.is_file() and 'previous_attempts' not in p.relative_to(folder).parts and p.suffix in ('.json','.csv','.yaml','.pt')}


def readonly_prepared(cfg,destination):
    from utils.config_loader import cfg_path
    from utils.prepared import load_prepared_data
    from utils.isolation import require_near_duplicate_review
    root=cfg_path(cfg,'paths.artifacts_dir');dest=Path(destination);dest.mkdir(parents=True,exist_ok=True)
    # Preparation helpers regenerate diagnostic CSVs: redirect those writes to
    # a separate audit workspace; do not rewrite sealed input manifests.
    for p in root.iterdir():
        if p.is_file() and p.suffix in ('.csv','.json','.sha256'):verified_copy(p,dest/p.name)
    local=copy.deepcopy(cfg);local['paths']['artifacts_dir']=str(dest)
    data=load_prepared_data(local,with_folds=True);require_near_duplicate_review(local,data)
    return local,data


def fold_distribution(data,fold_metrics):
    import numpy as np
    from utils.folds import DET_FOLD_COL
    t=data.det_folds;train=set(data.manifest.loc[data.manifest.outer_split=='train','image_id'])
    if t.image_id.duplicated().any() or set(t.image_id)!=train or set(t[DET_FOLD_COL])!={0,1,2}:raise Stop('Particiones internas invalidas')
    if len(data.images)!=324:raise Stop('Conteo activo distinto de 324')
    fruits=data.fruits;result=[]
    for fold in range(3):
        va=set(t.loc[t[DET_FOLD_COL]==fold,'image_id']);tr=train-va;expected=fold_metrics[fold_metrics.fold==fold].iloc[0]
        if va&tr or len(va)!=int(expected.n_val) or len(tr)!=int(expected.n_train):raise Stop('TRAIN/VAL no concuerda con los resultados')
        roles={}
        for name,ids in (('train',tr),('val',va)):
            boxes=fruits[fruits.image_id.isin(ids)].copy();areas=(boxes.x2-boxes.x1)*(boxes.y2-boxes.y1)/(boxes.image_w*boxes.image_h)
            if areas.empty or not np.isfinite(areas).all() or (areas<=0).any():raise Stop('Cajas de fruto invalidas')
            counts=boxes.groupby('image_id').size().reindex(sorted(ids),fill_value=0)
            scaled=areas*640**2
            roles[name]=dict(images=len(ids),boxes=len(boxes),background_images=int((counts==0).sum()),boxes_per_image_mean=float(counts.mean()),
                relative_area_quantiles={str(q):float(areas.quantile(q)) for q in (.05,.25,.5,.75,.95)},
                area_categories_after_resize640=dict(small=int((scaled<32**2).sum()),medium=int(((scaled>=32**2)&(scaled<96**2)).sum()),large=int((scaled>=96**2).sum())))
        result.append(dict(fold=fold,**roles))
    return result


def write_report(out,report):
    out=Path(out);out.mkdir(parents=True,exist_ok=True);atomic_json(out/'audit.json',report)
    data=report.get('tables',{});rows=''.join('<tr><td>'+str(r['fold'])+'</td><td>'+f"{r['best_mAP50_95']:.6f}"+'</td><td>'+str(r['best_epoch'])+'</td><td>'+str(r['epochs'])+'</td></tr>' for r in data.get('folds',[]))
    page='<!doctype html><html lang="es"><meta charset="utf-8"><title>Auditoria RT-DETR PC-03</title><style>body{max-width:1200px;margin:25px auto;padding:15px;font:16px/1.5 system-ui}pre{white-space:pre-wrap;overflow-wrap:anywhere}td,th{padding:8px;border:1px solid #aaa}img{max-width:100%}</style><h1>Auditoria RT-DETR-R50-vd</h1><p>Decision: <b>'+html.escape(report.get('decision','D_EVIDENCIA_INSUFICIENTE'))+'</b></p><table><tr><th>Fold</th><th>Mejor mAP50-95</th><th>Mejor epoca</th><th>Epocas</th></tr>'+rows+'</table><p>Las curvas individuales evitan cambiar la poblacion de folds al finalizar el fold 0. Los originales se conservan.</p>'
    if (out/'curvas_por_fold.png').is_file():page+='<img src="curvas_por_fold.png">'
    page+='<p>Sin telemetria de gradientes/LR por paso ni predicciones crudas no se demuestra la causa de la inestabilidad. APROBAR autoriza solo HPO original, no un cambio metodologico ni independencia biologica.</p><pre>'+html.escape(json.dumps(report,ensure_ascii=False,indent=2,default=str))+'</pre></html>'
    atomic_bytes(out/'AUDITORIA.html',page.encode())


def require_file(path):
    path=Path(path)
    if not path.is_file():raise MissingEvidence('Falta evidencia local: '+str(path))
    return path


def study_status(cfg):
    """Read Optuna SQLite directly: no migrations, study creation or RNG changes."""
    from utils.architectures import get_arch
    from utils.config_loader import cfg_path
    from utils.optuna_storage import study_name
    spec=get_arch(MODEL);db=cfg_path(cfg,'paths.optuna_dir')/(spec.study_name(str(cfg['optuna']['study_prefix']))+'.db')
    result=dict(database=str(db),exists=db.is_file(),trials=[],budget=15,folds=3,used=0)
    if not db.is_file():return result
    with read_only_sqlite(db) as c:
        if c.execute('PRAGMA integrity_check').fetchone()[0]!='ok':raise Stop('Estudio SQLite corrupto')
        found=c.execute('SELECT study_id FROM studies WHERE study_name=?',(study_name(spec,cfg),)).fetchall()
        if len(found)!=1:raise Stop('Estudio local de otra ronda o ambiguo')
        identity=c.execute("SELECT value_json FROM study_user_attributes WHERE study_id=? AND key='experiment_identity'",(found[0][0],)).fetchone()
        from utils.experiments import identity as current_identity
        if not identity or json.loads(identity[0])!=current_identity(cfg):raise Stop('Identidad Optuna incompatible')
        trials=[dict(r) for r in c.execute('SELECT trial_id,number,state,datetime_start,datetime_complete FROM trials WHERE study_id=? ORDER BY number',(found[0][0],))]
        for trial in trials:
            attrs={r['key']:json.loads(r['value_json']) for r in c.execute('SELECT key,value_json FROM trial_user_attributes WHERE trial_id=?',(trial['trial_id'],))}
            trial['config_id']=attrs.get('config_id');trial['run_dir']=attrs.get('run_dir');trial['fold_scores']=attrs.get('fold_scores')
            value=c.execute('SELECT value FROM trial_values WHERE trial_id=? AND objective=0',(trial['trial_id'],)).fetchone()
            trial['value']=value[0] if value else None
            intermediate=c.execute('SELECT step,intermediate_value FROM trial_intermediate_values WHERE trial_id=? ORDER BY step DESC LIMIT 1',(trial['trial_id'],)).fetchone()
            if intermediate:trial['latest_reported']=dict(step=intermediate[0],map50_95=intermediate[1])
    result.update(trials=trials,used=sum(t['state'] in ('COMPLETE','PRUNED') for t in trials))
    if result['used']>BUDGET:raise Stop('Optuna excede presupuesto; no se altera')
    result['running']=[t['number'] for t in trials if t['state']=='RUNNING']
    return result


def validate_protocol(cfg):
    from utils.architectures import get_arch,trials_for
    from utils.guards import assert_equal_hpo_budget,assert_same_pretrained_policy,assert_image_size_fixed,assert_not_frozen
    from utils.search_spaces import baseline_hyperparameters,space_for,search_space_hash
    assert_equal_hpo_budget(cfg);assert_same_pretrained_policy(cfg);assert_image_size_fixed(cfg);assert_not_frozen(cfg,'HPO')
    spec=get_arch(MODEL);d=cfg['detector'];o=cfg['optuna'];b=d['baseline']
    if cfg['project']['profile']!='official' or cfg['experiment']['mode']!='supervised':raise Stop('Se requiere ronda oficial supervisada')
    expected=dict(max_epochs=60,early_stopping_patience=20,batch_size=16,image_size=640,precision='fp16')
    if any(d.get(k)!=v for k,v in expected.items()) or d.get('oom_fallback'):raise Stop('Protocolo detector distinto del auditado')
    if trials_for(spec,cfg)!=15 or int(cfg['inner_hpo']['folds'])!=3 or cfg['reproducibility']['seed']!=42:raise Stop('Presupuesto/semilla distinto')
    if int(b['epochs'])!=100 or int(b['patience'])!=20:raise Stop('Protocolo baseline distinto')
    if o['sampler']['name']!='TPE' or o['pruner']['name']!='MedianPruner' or o['sampler']['startup_trials']!=5 or o['pruner']['startup_trials']!=5 or o['pruner']['warmup_epochs']!=8 or o['pruner']['interval_steps']!=1:raise Stop('Sampler/pruner diferente')
    return dict(trials=15,folds=3,max_epochs=60,patience=20,seed=42,precision='fp16',batch_size=16,objective='map50_95',
        space=space_for(spec,cfg),search_space_sha256=search_space_hash(),baseline_hp=baseline_hyperparameters(spec,cfg),
        sampler=o['sampler'],pruner=o['pruner'],pruning_step='fold_index * max_epochs + epoch_zero',
        warning='Los 8 pasos de warmup son globales; no se reinician al pasar al fold siguiente. Los trials lentos pueden ser descartados.',
        unchanged=True,independent_test=False,robustness=False)


def verify_backend(store,cfg,adapter):
    import torch
    from utils.weights import verified_local_weight
    from utils.architectures import get_arch
    # Pre-existing source only: never extract/download from the audit.
    require_file(store/'official.zip');require_file(store/'official/source.json')
    root=adapter.acquire_source(store,False)
    if root.resolve()!=Path(cfg['weights']['r50_source']).resolve():raise Stop('Fuente oficial de otra carpeta')
    weight=verified_local_weight(get_arch(MODEL),cfg)
    if digest(weight)!=adapter.WEIGHT_SHA or weight.stat().st_size!=adapter.WEIGHT_SIZE:raise Stop('Pesos COCO incompatibles')
    methodology=read(require_file(store/'methodology.json'));verified=read(require_file(store/'backend_verified.json'))
    for k,v in dict(adapter_sha256=ADAPTER_SHA,official_revision=adapter.REV,weight_sha256=adapter.WEIGHT_SHA).items():
        if methodology.get(k)!=v:raise Stop('Metodologia incompatible: '+k)
    for k,v in dict(adapter_sha256=ADAPTER_SHA,source_revision=adapter.REV,checkpoint_sha256=adapter.WEIGHT_SHA,status='PASS',strict_coco_load=True,optimizer_steps=0).items():
        if verified.get(k)!=v:raise Stop('Autoprueba previa incompatible: '+k)
    # Reuse the real prior GPU self-test; do not run backward or training now.
    if verified.get('device')!='cuda:0':raise MissingEvidence('Falta autoprueba GPU original compatible')
    if str(verified.get('torch'))!=str(torch.__version__) or verified.get('cuda_runtime')!=torch.version.cuda:raise Stop('Runtime diferente de autoprueba existente')
    model=adapter.build_model(root,weight)
    count=sum(p.numel() for p in model.parameters())
    if count!=42729027:raise Stop('Conteo de parametros inesperado para R50-vd fruit1')
    backbone_biases=[n for n,p in model.named_parameters() if n.startswith('backbone.') and n.endswith('.bias') and p.requires_grad]
    return model,dict(strict_coco80_load_then_fruit1=True,params=count,source_revision=adapter.REV,adapter_sha256=ADAPTER_SHA,
        pretrained_sha256=adapter.WEIGHT_SHA,backbone_trainable_biases=backbone_biases,methodology=methodology,
        prior_gpu_selftest=verified,training_executed_by_audit=False,stale_config_label=cfg['initialization'].get('rtdetr_backend'),
        config_label_note='La etiqueta heredada ultralytics_resnet50 no describe el grafo ejecutado; se verifica el adaptador y payload oficial sin cambiar config/fingerprint.')


def compare_histories(a,b,columns):
    import numpy as np
    if len(a)!=len(b):raise Stop('Historiales de distinta longitud')
    for col in columns:
        if col not in a or col not in b:raise MissingEvidence('Columna original ausente: '+col)
        if not np.allclose(a[col],b[col],rtol=2e-6,atol=1e-7,equal_nan=False):raise Stop('Historiales discrepan en '+col)


def check_optimizer_state(state,hp):
    import torch
    groups=state['optimizer']['param_groups']
    if not groups or not any(g.get('backbone') is True for g in groups) or not any(g.get('backbone') is False for g in groups):raise Stop('Faltan grupos backbone/head del optimizador')
    for group in groups:
        expected=float(hp['lr0'])*(float(hp['backbone_lr_ratio']) if group['backbone'] else 1)
        if not math.isclose(float(group.get('initial_lr',-1)),expected,abs_tol=1e-12):raise Stop('LR inicial de grupo incompatible con protocolo')
        if not math.isfinite(float(group['lr'])) or float(group['lr'])<0:raise Stop('LR final no finita/negativa')
        expected_decay=float(hp['weight_decay']) if group.get('param_group')=='weight' else 0.
        if not math.isclose(float(group['weight_decay']),expected_decay,abs_tol=1e-12):raise Stop('Weight decay del grupo incompatible')
    def finite(value):
        if isinstance(value,torch.Tensor):return not value.is_floating_point() or bool(torch.isfinite(value).all())
        if isinstance(value,float):return math.isfinite(value)
        if isinstance(value,dict):return all(finite(v) for v in value.values())
        if isinstance(value,(list,tuple)):return all(finite(v) for v in value)
        return True
    if not finite(state['optimizer']) or not finite(state['scaler']):raise Stop('Estado de optimizador/scaler no finito')


def baseline_local(cfg,folder,adapter,model,code):
    import numpy as np
    import pandas as pd
    import torch
    from utils.architectures import get_arch
    from utils.config_loader import load_config
    from utils.hashing import scientific_config_hash
    from utils.search_spaces import baseline_hyperparameters
    folder=Path(folder);metrics=read(require_file(folder/'metrics.json'));summary=read(require_file(folder/'summary.json'))
    meta=read(require_file(folder/'metadata.json'));checkpoints=read(require_file(folder/'checkpoint_metadata.json'))
    if meta.get('status')!='COMPLETE' or meta.get('pc_id')!=PC:raise Stop('Baseline sin cierre COMPLETE de PC-03')
    for item in (meta,checkpoints):
        for k in ('code_hash','git_commit'):
            if item.get(k)!=code[k]:raise Stop('Baseline/codigo vigente incompatible: '+k)
    if scientific_config_hash(load_config(require_file(folder/'config.yaml')))!=scientific_config_hash(cfg):raise Stop('Config baseline y ronda incompatibles')
    for key in ('fold_scores','best_epochs','epochs_run','score_mean','score_sd','dataset_hash','split_hash','experiment_fingerprint'):
        if summary.get(key)!=metrics.get(key):raise Stop('Resumen/metricas diferentes: '+key)
    f=pd.read_csv(require_file(folder/'fold_metrics.csv'));h=pd.read_csv(require_file(folder/'history.csv'));tables=tables_audit(metrics,f,h)
    if {p.name for p in folder.glob('fold*') if p.is_dir() and re.fullmatch(r'fold\d+',p.name)}!={'fold0','fold1','fold2'}:raise Stop('Folds adicionales/ausentes')
    hp=baseline_hyperparameters(get_arch(MODEL),cfg)
    if metrics.get('hp')!=hp:raise Stop('HP baseline distintos del protocolo')
    verified=[];cols=[c for c in h if c in ('epoch','train_loss','val_loss','optimization_loss_total','map50_95','map50','precision','recall','f1') or c.startswith(('train/','val/','metrics/'))]
    for fold in range(3):
        emit('AUDITANDO_FOLD_SIN_ENTRENAR',fold=fold,checkpoints='hash, recibos y carga CPU estricta')
        out=folder/f'fold{fold}'
        adapter_record=read(require_file(out/'adapter.json'))
        if any(adapter_record.get(k)!=v for k,v in dict(format=adapter.FORMAT,source_revision=adapter.REV,pretrained_sha256=adapter.WEIGHT_SHA,adapter_sha256=ADAPTER_SHA).items()):raise Stop('Fold con otro adaptador/backend')
        sig=fold_signature(EXPECTED_FP,MODEL,hp,42,fold,100,20)
        receipt,signed=verify_fold(out,sig,fold)
        aggregate=h[h.fold==fold].reset_index(drop=True)
        compare_histories(signed,aggregate,cols)
        raw=pd.read_csv(require_file(out/'results.csv'));compare_histories(raw,aggregate,cols)
        row=receipt['row'];fm=f[f.fold==fold].iloc[0]
        for col in ('score','best_epoch','epochs_run','n_train','n_val'):
            if not math.isclose(float(row[col]),float(fm[col]),rel_tol=2e-6,abs_tol=1e-7):raise Stop('Recibo/fold_metrics discrepan: '+col)
        signature=adapter.canonical(dict(experiment=EXPECTED_FP,hp=hp,seed=42,fold=fold,epochs=100,patience=20,fixed=False))
        best=out/'best.pt';require_file(best);require_file(best.with_suffix('.pt.resume.json'));validate_checkpoint(best,MODEL,cfg,seed=42,hp=hp,fold=fold)
        cp=[r for r in checkpoints.get('checkpoints',[]) if r.get('role')==f'fold{fold}_best']
        if len(cp)!=1 or cp[0].get('sha256')!=digest(best) or cp[0].get('weights_discarded') is not False:raise Stop('Metadata del mejor checkpoint ausente/incompatible')
        payload=torch.load(best,map_location='cpu',weights_only=False,mmap=True)
        expected=dict(format=adapter.FORMAT,signature=signature,experiment_fingerprint=EXPECTED_FP,official_revision=adapter.REV,pretrained_sha256=adapter.WEIGHT_SHA,best_epoch=int(row['best_epoch']))
        if any(payload.get(k)!=v for k,v in expected.items()):raise Stop('Best checkpoint de otra ejecucion')
        if not math.isclose(float(payload['metrics']['metrics/mAP50-95(B)']),float(row['score']),abs_tol=1e-9):raise Stop('Best checkpoint/score discrepan')
        model.load_state_dict(payload['model'],strict=True)
        if any(t.is_floating_point() and not torch.isfinite(t).all().item() for t in payload['model'].values()):raise Stop('Pesos no finitos')
        del payload
        states=sorted((out/'cp').glob('e*.pt'))
        if not states:raise MissingEvidence('Falta estado original de optimizador: '+str(out/'cp'))
        for state in states:
            require_file(state.with_suffix('.pt.resume.json'));_,signed_state=validate_checkpoint(state,MODEL,cfg,seed=42,hp=hp,fold=fold)
            if signed_state.get('extra',{}).get('adapter_sha256')!=ADAPTER_SHA:raise Stop('Estado de optimizador de otro adaptador')
        state=torch.load(states[-1],map_location='cpu',weights_only=False,mmap=True)
        if state.get('format')!=adapter.FORMAT or state.get('signature')!=signature or state.get('epoch')!=int(row['epochs_run']) or state.get('best_epoch')!=int(row['best_epoch']):raise Stop('Estado de optimizador no corresponde al cierre')
        if not math.isclose(float(state['best_score']),float(row['score']),abs_tol=1e-9):raise Stop('Estado final/best score discrepan')
        compare_histories(pd.DataFrame(state['history']),aggregate,cols)
        for key in ('optimizer','scheduler','scaler','rng','gradients','ema','best_state'):
            if key not in state:raise MissingEvidence('Estado de recuperacion sin '+key)
        for key in ('model','ema','best_state','gradients'):
            if any(t.is_floating_point() and not torch.isfinite(t).all().item() for t in state[key].values()):raise Stop('Estado/gradientes finales no finitos: '+key)
        best_payload=torch.load(best,map_location='cpu',weights_only=False,mmap=True)
        if set(state['best_state'])!=set(best_payload['model']) or any(not torch.equal(state['best_state'][k],best_payload['model'][k]) for k in state['best_state']):raise Stop('Best exportado no coincide con el mejor EMA guardado')
        del best_payload
        check_optimizer_state(state,hp)
        if bool(state['stopped'])!=(int(row['epochs_run'])<100):raise Stop('Early stop/estado discrepan')
        verified.append(dict(fold=fold,best_sha256=digest(best),best_epoch=int(row['best_epoch']),last_epoch=int(state['epoch']),
            original_optimizer_states=[str(p) for p in states],final_optimizer_groups=[{k:g.get(k) for k in ('initial_lr','lr','param_group','backbone','weight_decay','betas')} for g in state['optimizer']['param_groups']],
            strict_model_load=True,final_tensors_finite=True))
        del state
    return tables,h,f,verified


def verify_original_lists(cfg,baseline,derived):
    import yaml
    from utils.config_loader import cfg_path
    from utils.folds import DET_FOLD_COL
    root=cfg_path(cfg,'paths.detection_dir')/'images'/cfg['detection_dataset']['split_dir_names']['train']
    train=derived[derived.outer_split=='train'];result=[]
    for fold in range(3):
        folder=Path(baseline)/f'fold{fold}';yml=require_file(folder/'data.yaml');data=yaml.safe_load(yml.read_text(encoding='utf-8-sig'))
        if 'test' in data or data.get('nc')!=1 or data.get('names')!={0:'fruit'}:raise Stop('YAML original no corresponde a fruit1 interno')
        record=dict(fold=fold,yaml_sha256=digest(yml))
        for role in ('train','val'):
            declared=require_file(folder/(role+'.txt'))
            if Path(data[role]).resolve()!=declared.resolve():raise Stop('YAML original remite a otra lista')
            rows=train[train[DET_FOLD_COL]==fold] if role=='val' else train[train[DET_FOLD_COL]!=fold]
            expected={str((root/filename).resolve()) for filename in rows.image_file}
            paths=[str(Path(line.strip()).resolve()) for line in declared.read_text(encoding='utf-8-sig').splitlines() if line.strip()]
            if len(paths)!=len(set(paths)) or set(paths)!=expected or len(paths)!=len(rows):raise Stop('Lista original train/val no corresponde a imagenes/folds sellados')
            record[role]=dict(images=len(paths),list_sha256=digest(declared))
        result.append(record)
    return result


def processes():
    import psutil
    own=psutil.Process();excluded={own.pid,*(p.pid for p in own.parents())}
    return [dict(pid=p.pid,name=p.info['name']) for p in psutil.process_iter(['pid','name']) if p.pid not in excluded and (p.info['name'] or '').casefold() in ('python.exe','pythonw.exe')]


def resolve(project,home):
    from utils.config_loader import load_config
    from orchestration.machine import read_binding
    from orchestration.pc_registry import PCRegistry
    base=load_config(project/'artifacts/v4/local.yaml');binding=read_binding(base)
    if os.environ.get('USERNAME','').casefold()!=USER or base['compute']['pc_id']!=PC or binding['pc_id']!=PC or binding['machine_id'].casefold()!=USER:raise Stop('No corresponde a PC-03/I21313')
    store=home/'r50';config=require_file(store/'x/botrytis_v4/r50vd_01/config.yaml')
    overrides=['paths.runs_dir='+str(store/'r'),'paths.results_dir='+str(store/'e')]
    cfg=load_config(config,overrides)
    if PCRegistry(cfg).owner(MODEL)!=PC:raise Stop('PC-03 no es propietaria de RT-DETR')
    return store,config,cfg


def perform_audit(project,store,config,cfg,out):
    import logging
    import torch
    from types import SimpleNamespace
    from utils.architectures import get_arch
    from utils.config_loader import cfg_path
    from utils.experiments import verify_round,fingerprint
    from utils.manual_gates import require_gate,ledger_path
    from utils.reproducibility import collect_environment
    from utils.run_common import check_environment_matches_reference
    from utils.detector_data import detection_images_table
    baseline=store/'r/rtdetr_r50/baseline';before=frozen_inventory(baseline)
    art=cfg_path(cfg,'paths.artifacts_dir');prepared_before=frozen_inventory(art)
    report=dict(created_utc=utc(),pc_id=PC,architecture=MODEL,decision='D_EVIDENCIA_INSUFICIENTE',executor_sha256=EXECUTOR_SHA256,
        original_results=str(baseline),training_executed=False,queue_modified=False,human_approval_registered=False,previous_architectures_modified=False,
        independent_test_evaluated=False,robustness_executed=False,errors=[],missing=[])
    threads=torch.get_num_threads()
    try:
        emit('FASE_IDENTIFICACION_Y_COLA_SOLO_LECTURA',pc=PC,training=False)
        active=processes();report['other_python_processes']=active
        if active:raise Stop('Hay procesos Python activos; se conservan sin interrumpir. Reejecutar auditoria cuando terminen: '+str(active))
        report['code']=readonly_code(project)
        require_file(config.parent/'manifest.json')
        require_file(cfg_path(cfg,'compute.reference_environment'));require_file(ledger_path(cfg,PC))
        verify_round(cfg,architecture=MODEL)
        if fingerprint(cfg)!=EXPECTED_FP:raise Stop('Fingerprint diferente de r50vd_01')
        manifest=read(config.parent/'manifest.json')
        if manifest.get('config_sha256')!=digest(config):raise Stop('Config inmutable alterada')
        report['config_sha256']=digest(config);report['fingerprint']=EXPECTED_FP;report['hpo_protocol']=validate_protocol(cfg)
        report['jobs']=read_jobs(cfg);report['study']=study_status(cfg)
        if report['jobs']['baseline']['status'] not in ('PENDING_REVIEW','COMPLETE'):raise Stop('Baseline no esta listo para revision')
        if report['study'].get('running') or report['jobs']['hpo']['status']=='RUNNING':raise Stop('HPO activo; no se duplica ni modifica')
        require_gate(cfg,PC,'READY_FOR_EXPERIMENT')
        report['qa_reused']=dict(ledger=str(ledger_path(cfg,PC)),sha256=digest(ledger_path(cfg,PC)),gate='READY_FOR_EXPERIMENT',full_qa_repeated=False)
        env=collect_environment(cfg=cfg,project_root=project,pc_id=PC)
        check_environment_matches_reference(SimpleNamespace(cfg=cfg,env=env,logger=logging.getLogger('r50_audit')),strict=True)
        if not torch.cuda.is_available() or torch.cuda.device_count()!=1:raise Stop('No hay una unica GPU CUDA disponible para HPO; auditoria sin entrenar')
        emit('FASE_BACKEND_OFICIAL_CPU',training=False,selftest_reused=True)
        adapter=load_adapter();torch.set_num_threads(2)
        model,report['backend']=verify_backend(store,cfg,adapter)
        tables,h,f,verified=baseline_local(cfg,baseline,adapter,model,report['code']);del model
        report['tables']=tables;report['checkpoints']=verified;plot_by_fold(h,out)
        emit('FASE_DATOS_Y_PARTICIONES',splits_regenerated=False,ignored_off_unchanged=True)
        scratch,data=readonly_prepared(cfg,Path(out)/'prep')
        if data.dataset_hash!=EXPECTED_DATA or data.split_hash!=EXPECTED_SPLIT:raise Stop('Datos/splits no corresponden al baseline')
        derived=detection_images_table(scratch)
        report['original_input_lists']=verify_original_lists(cfg,baseline,derived)
        report['fold_distributions']=fold_distribution(data,f);report['isolation']=partition_evidence(scratch,get_arch(MODEL))
        if report['isolation']['confirmed_relevant_conflicts']:raise Stop('Conflicto de contenido/origen demostrado entre particiones; no se aprueba')
        report['limitations']=[
            'Variabilidad alta; fold0 funciona mal pero no hay fallo invalidante demostrado solo por su mAP.',
            'No hay LR por paso ni norma de gradientes historica; solo se verifican tensores/gradientes guardados finales.',
            'No se evalua train como conjunto de inferencia; no hay brecha mAP train-val calculable sin inferencia adicional.',
            'dHash es sospecha; independencia de hoja/planta/sesion solo comprobable si existen metadatos explicitos.',
            'No hay predicciones crudas adjuntas: no se confirma colapso, conteo de falsos positivos ni causalidad del fold0.',
            'El presupuesto HPO60 puede favorecer aprendizaje rapido; se conserva para comparabilidad, no se cambia metodo.',
            'Las metricas son validacion interna de fruit, no rendimiento del pipeline Botrytis completo ni test independiente.',
            'El backend oficial conserva su politica de dos estados rolling de optimizador y best; cualquier cambio de retencion debe acordarse antes de ejecutar HPO.'
        ]
        report['decision']='B_APTO_CON_OBSERVACIONES'
        report['continuation']=dict(status='PREPARADO_PENDIENTE_APROBAR',entrypoint='src/07_run_hpo_study.py',
            adapter_required_sha256=ADAPTER_SHA,approval_api='StudyQueue.approve(baseline.job_id, operator=..., evidence=...)',
            review_semantics='computed() termina en PENDING_REVIEW; approve() registra reviews y pasa a COMPLETE para habilitar hpo dependiente.',
            no_baseline_repetition=True,no_hpo_start=True,no_new_queue=True,no_method_change=True,
            note='El comando operativo de HPO se entrega despues de APROBAR explicito, como solicita la instruccion 7. No ejecutar 07 sin instalar el adaptador oficial.')
    except MissingEvidence as e:report['missing'].append(str(e))
    except (FileNotFoundError,KeyError) as e:report['missing'].append(str(e))
    except Exception as e:report['decision']='C_BLOQUEADO_POR_ERROR_CRITICO';report['errors'].append(str(e));report['traceback']=traceback.format_exc()
    finally:
        torch.set_num_threads(threads)
        report['baseline_unchanged']=frozen_inventory(baseline)==before
        report['prepared_artifacts_unchanged']=frozen_inventory(art)==prepared_before
        if not report['baseline_unchanged'] or not report['prepared_artifacts_unchanged']:
            report['decision']='C_BLOQUEADO_POR_ERROR_CRITICO';report['errors'].append('Archivos originales cambiaron durante auditoria; comprobar procesos externos')
        report['baseline_inventory']=before;write_report(out,report)
        atomic_json(store/'audit/latest.json',dict(report=str(Path(out)/'AUDITORIA.html'),json=str(Path(out)/'audit.json'),sha256=digest(Path(out)/'audit.json'),decision=report['decision']))
    emit('AUDITORIA_FINALIZADA',decision=report['decision'],report=Path(out)/'AUDITORIA.html',errors=report['errors'],missing=report['missing'],hpo_started=False,queue_modified=False)
    if report['decision'].startswith(('A_','B_')):emit('APROBACION_PENDIENTE',instruction='Revise AUDITORIA.html y escriba APROBAR en la conversacion para recibir la continuacion HPO. Esta auditoria no registra una aprobacion.')
    return 0 if report['decision'].startswith(('A_','B_')) else 2


def main():
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('mode',choices=['audit','status']);args=parser.parse_args()
    if os.name!='nt':raise Stop('Modo operativo exclusivo de Windows PC-03; pruebas CPU separadas')
    project=Path.home()/'Documents/s2_no_borrar/Entrenamiento/Botrytis_Experiment_Framework_git';py=project/'.venv/Scripts/python.exe'
    require_file(py)
    if Path(sys.executable).resolve()!=py.resolve():return subprocess.call([str(py),'-u',str(Path(__file__).resolve()),*sys.argv[1:]])
    os.chdir(project);sys.path.insert(0,str(project/'src'))
    os.environ.update(PYTHONUTF8='1',PYTHONIOENCODING='utf-8',MPLBACKEND='Agg',BOTRYTIS_BOOTSTRAP_DIR=str(Path.home()/'r50/log'),BOTRYTIS_PC_ID=PC,GIT_OPTIONAL_LOCKS='0',PYTHONDONTWRITEBYTECODE='1');sys.dont_write_bytecode=True
    from utils.config_loader import load_config
    store,config,cfg=resolve(project,Path.home())
    if args.mode=='status':
        emit('ESTADO_RTDETR_SOLO_LECTURA',jobs=read_jobs(cfg),study=study_status(cfg),processes=processes());return 0
    out=store/'audit'/uuid.uuid4().hex[:8];out.mkdir(parents=True)
    return perform_audit(project,store,config,cfg,out)


if __name__=='__main__':
    try:sys.exit(main())
    except KeyboardInterrupt:emit('INTERRUMPIDO_SIN_ENTRENAR_NI_APROBAR');sys.exit(130)
    except Exception as e:emit('ERROR_CRITICO',error=str(e),hpo_started=False,queue_modified=False);sys.exit(2)
