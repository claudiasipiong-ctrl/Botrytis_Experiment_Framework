"""PC-01: cerrar baseline YOLOv8l tras float(None), usando exclusivamente folds firmados.
No entrena, no inicia HPO, no aprueba revisiones, no modifica el checkout.
"""
from __future__ import annotations
import ast, hashlib, importlib, inspect, json, math, os
from pathlib import Path
import re, socket, sqlite3, subprocess, sys, tempfile, textwrap, traceback, uuid
MODEL='yolov8l';PC='PC-01';USER='i21315'
EXECUTOR_SHA256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest()
class Stop(RuntimeError):pass


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

def owner_alive(owner):
    import psutil
    if owner.get('hostname') != socket.gethostname() or not owner.get('pid') or not owner.get('created'):
        raise Stop('Propietario de proceso ausente/ajeno; no se declara muerto automaticamente: ' + str(owner))
    try: return abs(psutil.Process(int(owner['pid'])).create_time() - float(owner['created'])) < 1
    except psutil.NoSuchProcess: return False
    except psutil.AccessDenied as e: raise Stop('No se puede comprobar el proceso propietario') from e

def own_process():
    import psutil
    return dict(hostname=socket.gethostname(), pid=os.getpid(), created=psutil.Process().create_time())

def git(root, *args):
    return subprocess.check_output(['git', '-C', str(root), *args], text=True, stderr=subprocess.PIPE).strip()

def check_git(root, audit):
    """Refresh only index entries already proven identical to HEAD after Git filters."""
    if subprocess.run(['git', '-C', str(root), 'diff', '--cached', '--quiet']).returncode:
        raise Stop('Hay cambios preparados para commit; se conservan')
    if subprocess.run(['git', '-C', str(root), 'diff', '--quiet']).returncode:
        raise Stop('Hay modificaciones reales de codigo. No se resetean ni se reemplazan')
    raw = subprocess.check_output(['git', '-C', str(root), 'status', '--porcelain', '-z', '--untracked-files=no'])
    paths = []
    for entry in raw.split(b'\0'):
        if not entry: continue
        if entry[:3] != b' M ': raise Stop('Estado Git requiere revision; se conserva: ' + repr(entry))
        rel = os.fsdecode(entry[3:]); expected = git(root, 'rev-parse', 'HEAD:' + rel)
        actual = git(root, 'hash-object', '--path=' + rel, str(Path(root) / rel))
        if actual != expected: raise Stop('Git detecta contenido distinto en ' + rel)
        paths.append(rel)
    if paths:
        subprocess.run(['git', '-C', str(root), 'add', '--', *paths], check=True)
        if git(root, 'diff', '--cached', '--name-only') or git(root, 'status', '--porcelain', '--untracked-files=no'):
            raise Stop('El indice no quedo limpio; no se toca el codigo')
        audit['index_refresh_only'] = paths
    from utils.hashing import compute_code_hash
    return git(root, 'rev-parse', 'HEAD'), compute_code_hash(root)

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

def snapshot_sqlite(db, destination):
    if not Path(db).is_file(): return
    Path(destination).parent.mkdir(parents=True,exist_ok=True)
    with sqlite3.connect(Path(db).resolve().as_uri() + '?mode=ro', uri=True, timeout=15) as src:
        with sqlite3.connect(destination, timeout=15) as dst: src.backup(dst)
    with sqlite3.connect(destination) as test:
        if test.execute('PRAGMA integrity_check').fetchone()[0] != 'ok': raise Stop('Snapshot SQLite no integro')


def component_loss_checks(history):
    """Check authentic components; an empty aggregate is ND, never a zero."""
    import pandas as pd
    flags=[];details=[]
    if 'fold' not in history or set(history.fold)!={0,1,2}:raise Stop('Historial sin los tres folds')
    for fold,g in history.groupby('fold',sort=True):
        if len(g)<2:raise Stop('Fold sin curva suficiente: '+str(fold))
        for name in ('train/box_loss','train/cls_loss','train/dfl_loss'):
            if name not in g:raise Stop('Falta componente obligatoria: '+name)
            values=pd.to_numeric(g[name],errors='coerce')
            if values.isna().any() or not values.map(math.isfinite).all() or (values<0).any():raise Stop('Componente no finita/incompleta: '+name)
            decreased=float(values.iloc[0])>float(values.iloc[-1]);flags.append(decreased)
            details.append(dict(fold=int(fold),component=name,first=float(values.iloc[0]),last=float(values.iloc[-1]),decreased=decreased))
        if 'train_loss' in g and g.train_loss.notna().any():
            values=pd.to_numeric(g.train_loss,errors='coerce')
            if values.isna().any() or not values.map(math.isfinite).all():raise Stop('Agregado parcialmente informado/no finito')
    return flags,details


def verify_saved_folds(cfg,folder):
    import pandas as pd
    from utils.architectures import get_arch,section_for
    from utils.experiments import fingerprint
    from utils.search_spaces import baseline_hyperparameters
    folder=Path(folder);spec=get_arch(MODEL);section=section_for(spec,cfg);base=section['baseline']
    hp=baseline_hyperparameters(spec,cfg);exp=fingerprint(cfg)
    actual={p.name for p in folder.glob('fold*') if p.is_dir() and re.fullmatch(r'fold\d+',p.name)}
    if actual!={'fold0','fold1','fold2'}:raise Stop('Folds ausentes/adicionales: '+str(actual))
    receipts=[];histories=[];hashes={}
    metric='metrics/mAP50-95(B)'
    for fold in range(3):
        f=folder/f'fold{fold}'
        sig=fold_signature(exp,MODEL,hp,int(base['seed']),fold,int(base['epochs']),int(base['patience']))
        receipt,h=verify_fold(f,sig,fold);row=receipt['row']
        raw=pd.read_csv(f/'results.csv');raw.columns=raw.columns.str.strip()
        for frame in (raw,h):
            if frame.empty or frame.epoch.tolist()!=list(range(1,int(row['epochs_run'])+1)):raise Stop('Epocas originales incompletas/desordenadas')
            for name in (metric,'metrics/mAP50(B)','metrics/precision(B)','metrics/recall(B)'):
                if name not in frame:raise Stop('Metrica obligatoria ausente: '+name)
                values=pd.to_numeric(frame[name],errors='coerce')
                if values.isna().any() or not values.map(math.isfinite).all() or not values.between(0,1).all():raise Stop('Metrica no finita/fuera de rango: '+name)
            if not math.isclose(float(frame[metric].max()),float(row['score']),abs_tol=1e-5):raise Stop('Score de recibo/CSV incompatible')
            selected=frame.loc[frame.epoch==int(row['best_epoch']),metric]
            if len(selected)!=1 or not math.isclose(float(selected.iloc[0]),float(row['score']),abs_tol=1e-5):raise Stop('Best epoch no corresponde al score')
        for name in ('train/box_loss','train/cls_loss','train/dfl_loss',metric,'metrics/mAP50(B)','metrics/precision(B)','metrics/recall(B)'):
            if name not in raw or name not in h:raise Stop('CSV e historial sin columna obligatoria: '+name)
            a=pd.to_numeric(raw[name],errors='coerce');b=pd.to_numeric(h[name],errors='coerce')
            if len(a)!=len(b) or any(not math.isclose(float(x),float(y),rel_tol=1e-5,abs_tol=1e-5) for x,y in zip(a,b)):
                raise Stop('El CSV original discrepa del historial firmado: '+name)
        for name in ('completed_fold.json','completed_history.csv','results.csv'):
            hashes[str(f/name)]=digest(f/name)
        receipts.append(receipt);histories.append(h)
    history=pd.concat(histories,ignore_index=True);flags,details=component_loss_checks(history)
    return dict(receipts=receipts,hashes=hashes,components=details,all_components_decreased=all(flags),
        training_seconds=sum(float(r['row']['train_seconds']) for r in receipts)),history


def original_error_is_supported(metadata):
    error=metadata.get('error') or {}
    return metadata.get('status')=='FAILED' and error.get('type')=='TypeError' and 'NoneType' in error.get('message','') and 'train_loss' in error.get('traceback','') and '06_run_baseline.py' in error.get('traceback','')


def install_recovery(native,cfg,folder,checkpoint_metadata,evidence):
    """Patch just summary convergence in RAM. The native training functions are forbidden."""
    import utils.hpo_runner as runner
    expected=Path(folder).resolve();original_evaluate=runner.evaluate_config
    def forbidden(*args,**kwargs):raise Stop('PROHIBIDO_ENTRENAR: recuperacion solo de folds completos')
    runner.train_detector=forbidden;runner.train_classifier=forbidden
    def cached(spec,hp,effective,ctx,run_dir,**options):
        if spec.slug!=MODEL or Path(run_dir).resolve()!=expected or options.get('trial') is not None or options.get('folds')!=[0,1,2]:
            raise Stop('Solo baseline YOLOv8l con los tres folds originales')
        checked,_=verify_saved_folds(effective,run_dir)
        if checked['hashes']!=evidence['hashes']:raise Stop('Los folds cambiaron antes de recuperarlos')
        for entry in checkpoint_metadata['checkpoints']:
            ctx.record_checkpoint(entry['role'],**{k:v for k,v in entry.items() if k not in ('role','size_bytes')})
        ctx.add_metadata(recovery_executor_sha256=EXECUTOR_SHA256,recovery_native06_sha256=digest(native.__file__),recovery_original_training_seconds=evidence['training_seconds'],
            recovery_loss_components=evidence['components'],recovery_patch='Summary only: authentic component losses replace unavailable aggregate; source checkout unchanged')
        result=original_evaluate(spec,hp,effective,ctx,run_dir,**options)
        result.train_seconds=evidence['training_seconds']
        return result
    runner.evaluate_config=cached
    source=textwrap.dedent(inspect.getsource(native.main));tree=ast.parse(source);fn=tree.body[0]
    position=next(i for i,n in enumerate(fn.body) if isinstance(n,ast.Assign) and any(isinstance(t,ast.Name) and t.id=='first_last' for t in n.targets))
    if not isinstance(fn.body[position+1],ast.For) or 'train_loss' not in ast.unparse(fn.body[position+1]):raise Stop('Interfaz 06 cambio; no parchear a ciegas')
    fn.body[position:position+2]=ast.parse('first_last = _recovery_component_loss_checks(ev.history)[0]').body
    native.__dict__['_recovery_component_loss_checks']=component_loss_checks
    exec(compile(ast.fix_missing_locations(tree),str(native.__file__)+' [summary recovery only]','exec'),native.__dict__)


def verify_checkpoint_metadata(meta,folder):
    entries=meta.get('checkpoints',[])
    if {e.get('role') for e in entries}!={f'fold{i}_best' for i in range(3)} or len(entries)!=3:
        raise Stop('Faltan los hashes originales de los tres checkpoints')
    for e in entries:
        if not re.fullmatch(r'[a-f0-9]{64}',str(e.get('sha256',''))):raise Stop('Hash de checkpoint original invalido')
        if e.get('weights_discarded') is not True:
            path=e.get('retained_path') or e.get('path')
            if not path or not Path(path).is_file() or digest(path)!=e['sha256']:raise Stop('Peso declarado conservado ausente/alterado')


def main():
    if os.name!='nt':raise Stop('Ejecutor operativo exclusivo de Windows PC-01')
    home=Path.home();project=home/'Documents/s2_no_borrar/Entrenamiento/Botrytis_Experiment_Framework_git'
    py=project/'.venv/Scripts/python.exe'
    if Path(sys.executable).resolve()!=py.resolve():return subprocess.call([str(py),'-u',str(Path(__file__).resolve())])
    os.chdir(project);sys.path.insert(0,str(project/'src'))
    os.environ.update(PYTHONUTF8='1',PYTHONIOENCODING='utf-8',MPLBACKEND='Agg',BOTRYTIS_BOOTSTRAP_DIR=str(home/'btlog'))
    from utils.config_loader import load_config,cfg_path
    from utils.experiments import verify_round,fingerprint
    from utils.hashing import scientific_config_hash
    from utils.durable import FileLock
    from utils.guards import gpu_lock_path
    from utils.manual_gates import require_gate
    from orchestration.machine import read_binding
    from orchestration.pc_registry import PCRegistry
    from orchestration.study_queue import StudyQueue
    import psutil
    base=load_config(project/'artifacts/v4/local.yaml');binding=read_binding(base)
    if os.environ.get('USERNAME','').casefold()!=USER or base['compute']['pc_id']!=PC or binding['pc_id']!=PC or binding['machine_id'].casefold()!=USER:
        raise Stop('PC, binding y usuario deben ser PC-01/I21315')
    config=Path(read(home/'bt-hoy-20261103/rondas.json')[MODEL]).resolve()
    key=hashlib.sha256(str(config).casefold().encode()).hexdigest()[:8];short=home/'bf'/key
    overrides=['paths.runs_dir='+str(short/'r'),'paths.results_dir='+str(short/'e')]
    cfg=load_config(config,overrides)
    if PCRegistry(cfg).owner(MODEL)!=PC or cfg['compute']['pc_id']!=PC:raise Stop('Ronda de otra PC')
    lock=gpu_lock_path(cfg,PC)
    with FileLock(lock.with_suffix('.recovery.guard')),FileLock(lock.with_suffix('.final.guard')),FileLock(lock.with_suffix('.priority.guard')):
        own=psutil.Process();exclude={own.pid,*(p.pid for p in own.parents())}
        active=[p.pid for p in psutil.process_iter(['pid','name']) if p.pid not in exclude and (p.info['name'] or '').casefold() in ('python.exe','pythonw.exe')]
        if active:raise Stop('Otro Python activo; no se interrumpe: '+str(active))
        audit={};commit,code=check_git(project,audit);manifest=verify_round(cfg,architecture=MODEL)
        if digest(config)!=manifest['config_sha256']:raise Stop('Configuracion inmutable alterada')
        require_gate(cfg,PC,'READY_FOR_EXPERIMENT')
        db=cfg_path(cfg,'paths.queue_dir')/'queue.sqlite'
        if not db.is_file():raise Stop('Cola original ausente; no se crea una nueva')
        folder=short/'r'/MODEL/'baseline'
        emit('VERIFICANDO_FOLDS_SIN_ENTRENAR',path=folder)
        evidence,history=verify_saved_folds(cfg,folder)
        metadata=read(folder/'metadata.json')
        if metadata.get('code_hash')!=code or metadata.get('git_commit')!=commit or metadata.get('pc_id')!=PC:
            raise Stop('Codigo o PC distintos de la ejecucion original')
        if scientific_config_hash(load_config(folder/'config.yaml'))!=scientific_config_hash(cfg):raise Stop('Configuracion distinta del baseline original')
        q=StudyQueue.from_cfg(cfg);jobs=[j for j in q.jobs() if j.kind=='baseline' and j.architecture==MODEL]
        if len(jobs)!=1:raise Stop('Job baseline ausente/ambiguo')
        job=jobs[0]
        if job.payload.get('fingerprint')!=fingerprint(cfg):raise Stop('Job de otra ronda')
        if job.status in ('PENDING_REVIEW','COMPLETE') and (folder/'summary.json').is_file():
            summary=read(folder/'summary.json')
            scores=[r['row']['score'] for r in evidence['receipts']]
            if metadata.get('status')!='COMPLETE' or summary.get('fold_scores')!=scores:raise Stop('Cierre previo inconsistente')
            emit('BASELINE_YA_RECUPERADO_NO_SE_REPITE',status=job.status,summary=folder/'summary.json');return 0
        checkpoint_meta=read(folder/'checkpoint_metadata.json');verify_checkpoint_metadata(checkpoint_meta,folder)
        for k in ('git_commit','code_hash','dataset_hash','split_hash'):
            if checkpoint_meta.get(k)!=metadata.get(k):raise Stop('Metadata de checkpoints incompatible: '+k)
        ledger=short/'recover_loss';ledger.mkdir(parents=True,exist_ok=True);owner_file=ledger/'owner.json'
        first_error=ledger/'original_error.json'
        if not first_error.exists():
            if job.status!='FAILED' or not original_error_is_supported(metadata):raise Stop('Fallo diferente: no se recupera a ciegas')
            atomic_json(first_error,dict(metadata=metadata,job=job.as_json(),config=str(config),fingerprint=fingerprint(cfg),executor_sha256=EXECUTOR_SHA256))
        else:
            prior=read(first_error)
            if prior.get('fingerprint')!=fingerprint(cfg) or prior.get('executor_sha256')!=EXECUTOR_SHA256:raise Stop('Recuperacion de otra identidad/version')
            if job.status not in ('FAILED','PENDING','RUNNING','PUBLICATION_PENDING'):raise Stop('Estado no recuperable: '+job.status)
            if job.status=='RUNNING':
                owner=read(owner_file)
                if owner.get('job_id')!=job.job_id or owner.get('attempt')!=job.attempt or owner_alive(owner['process']):raise Stop('No se demostro proceso anterior muerto')
        if job.attempt>=3:raise Stop('Limite de intentos; conservar evidencias')
        attempt=ledger/uuid.uuid4().hex[:8];attempt.mkdir()
        snapshot_sqlite(db,attempt/'queue-before.sqlite')
        atomic_json(attempt/'verification.json',dict(**evidence,executor_sha256=EXECUTOR_SHA256,training_forbidden=True))
        # Native preparation/export rechecks sealed dataset/splits and gates. It may
        # regenerate reports, but trainers are forbidden even if a fold disappears.
        native=importlib.import_module('06_run_baseline')
        install_recovery(native,cfg,folder,checkpoint_meta,evidence)
        if job.status=='PUBLICATION_PENDING':
            from orchestration.exchange import publish_architecture
            publish_architecture(cfg,PC,MODEL,cfg_path(cfg,'paths.runs_dir'),stage='baseline',complete=True)
            q.computed(job.job_id,published=True)
        else:
            atomic_json(owner_file,dict(job_id=job.job_id,attempt=job.attempt+1,process=own_process(),executor_sha256=EXECUTOR_SHA256))
            q.finish(job.job_id,'PENDING',error='Recuperar SOLO exportacion del baseline; entrenamiento prohibido; '+str(attempt/'verification.json'))
            args=['--config',str(config),'--pc-id',PC,'--architecture',MODEL]
            for value in overrides:args.extend(['--set',value])
            rc=native.run_cli(native.SCRIPT,native.main,argv=args,description=native.__doc__,add_arguments=native.add_arguments,run_layout=native.layout_for('baseline'))
            if rc:raise Stop('Cierre aun pendiente; entrenamiento no ejecutado. Codigo '+str(rc))
        after,_=verify_saved_folds(cfg,folder)
        if after['hashes']!=evidence['hashes']:raise Stop('Los CSV/recibos originales cambiaron durante el cierre')
        summary=read(folder/'summary.json');meta=read(folder/'metadata.json')
        if meta.get('status')!='COMPLETE' or q.get(job.job_id).status!='PENDING_REVIEW':raise Stop('No se completo el cierre supervisado')
        if summary.get('fold_scores')!=[r['row']['score'] for r in evidence['receipts']]:raise Stop('Scores de recuperacion diferentes')
        emit('BASELINE_RECUPERADO',status=q.get(job.job_id).status,mean=summary['score_mean'],sd=summary['score_sd'],
            summary=folder/'summary.json',folds_retrained=0,hpo_started=False,approval='Pendiente de revision real antes de HPO')
        atomic_json(attempt/'result.json',dict(status=q.get(job.job_id).status,summary=summary,folds_retrained=0,hpo_started=False))
        return 0


if __name__=='__main__':
    try:sys.exit(main())
    except KeyboardInterrupt:emit('INTERRUMPIDO',training='No se ejecutaron entrenamientos');sys.exit(130)
    except Exception as error:emit('DETENIDO_SIN_REENTRENAR',error=str(error));traceback.print_exc();sys.exit(2)
