"""Posponer post-HPO sin senales y lanzar SOLO un baseline pendiente del propietario."""
from pathlib import Path
import copy
import hashlib
import json
import os
import sqlite3
import subprocess
import sys
import time
import uuid
from datetime import datetime, timezone

FINISHED = {'yolov8n', 'yolo26l', 'vit_b16'}
USERS = {'i21315': 'PC-01', 'i21314': 'PC-02', 'i21313': 'PC-03'}
POST = ('repeat', 'aggregate', 'outer_train', 'outer_eval', 'thresholds')


def emit(event, **values):
    print(event, json.dumps(values, ensure_ascii=False, default=str), flush=True)


def read(path):
    return json.loads(Path(path).read_text(encoding='utf-8-sig'))


def pause_queue(db, model, destination, operator):
    """Atomic PENDING -> BLOCKED only. Never mutate RUNNING or dependencies."""
    from utils.durable import atomic_json
    db, destination = Path(db), Path(destination)
    if not db.is_file():
        return dict(database=str(db), state='NO_QUEUE', postponed=[], running=[])
    destination.mkdir(parents=True, exist_ok=True)
    snapshot = destination/'queue-before.sqlite'
    with sqlite3.connect(db.resolve().as_uri()+'?mode=ro', uri=True, timeout=30) as source:
        with sqlite3.connect(snapshot) as backup:
            source.backup(backup)
    with sqlite3.connect(snapshot) as backup:
        if backup.execute('PRAGMA integrity_check').fetchone()[0] != 'ok':
            raise RuntimeError('Respaldo SQLite no integro; no se pospone nada')
    ledger = destination/'pause.json'
    marker = 'POSPUESTO_POR_PRIORIDAD '+str(ledger)
    with sqlite3.connect(db, timeout=30) as c:
        c.row_factory = sqlite3.Row
        c.execute('BEGIN IMMEDIATE')
        before = [dict(r) for r in c.execute('SELECT * FROM jobs WHERE architecture=?', (model,))]
        pending = [r for r in before if r['status']=='PENDING' and r['kind'] in POST]
        record = dict(database=str(db), architecture=model, operator=operator,
                      reason='Priorizar arquitecturas pendientes; robustez diferida, no validacion definitiva',
                      created_utc=datetime.now(timezone.utc).isoformat(), marker=marker, postponed=pending,
                      running=[r for r in before if r['status']=='RUNNING'], backup=str(snapshot), state='INTENT')
        atomic_json(ledger, record)  # durable write-ahead intent, also safe if process dies before COMMIT
        for r in pending:
            c.execute("UPDATE jobs SET status='BLOCKED', error=?, updated_at=? WHERE job_id=? AND status='PENDING'",
                      (marker, record['created_utc'], r['job_id']))
        c.commit()
    record['state']='APPLIED'
    atomic_json(ledger, record)
    emit('POST_HPO_POSPUESTO', architecture=model, jobs=[r['job_id'] for r in pending],
         running_preserved=[r['job_id'] for r in record['running']], evidence=ledger)
    return record


def restore_pause(ledger):
    """Explicit restore of ONLY this pause. No approval, retry or training."""
    row=read(ledger)
    with sqlite3.connect(row['database'], timeout=30) as c:
        c.execute('BEGIN IMMEDIATE')
        for original in row['postponed']:
            c.execute("UPDATE jobs SET status='PENDING',error=?,updated_at=? WHERE job_id=? AND status='BLOCKED' AND error=?",
                      (original.get('error'), datetime.now(timezone.utc).isoformat(), original['job_id'], row['marker']))
        c.commit()
    emit('PENDIENTES_RESTAURADOS_SIN_EJECUTAR', evidence=ledger)


def other_python():
    import psutil
    own=psutil.Process()
    excluded={own.pid, *(p.pid for p in own.parents())}
    found=[]
    for p in psutil.process_iter(['pid','name']):
        if p.pid not in excluded and (p.info['name'] or '').lower() in ('python.exe','pythonw.exe'):
            found.append(p.pid)
    return sorted(found)


def wait_idle(lock):
    """Keep caller alive, preserve current worker and acquire its parent's machine guard."""
    from utils.durable import FileLock
    previous=None
    while True:
        processes=other_python()
        guard=None
        if not processes:
            try:
                guard=FileLock(lock.with_suffix('.final.guard'))
                guard.__enter__()
                # No new worker/GPU job may have appeared during inspection.
                with FileLock(lock.with_suffix('.worker.guard')), FileLock(lock.with_suffix('.guard')):
                    if not other_python():
                        return guard
            except RuntimeError:
                pass
            if guard is not None and guard.handle is not None:
                guard.__exit__(None,None,None)
        if processes != previous:
            emit('ESPERANDO_FIN_DEL_TRABAJO_ACTUAL', processes=processes, action='No se interrumpe ni mata ningun proceso; se espera fin del job completo')
            previous=processes
        time.sleep(3)


def locate_rounds(base, work):
    from utils.config_loader import cfg_path, load_config
    from utils.experiments import verify_round
    index=read(work/'rondas.json') if (work/'rondas.json').is_file() else {}
    roots={work/'registro', Path.home()/'bte', Path.home()/'bn', cfg_path(base,'paths.artifacts_dir')/'experiments'}
    if base.get('experiment',{}).get('registry_root'):
        roots.add(Path(base['experiment']['registry_root']))
    found={}
    for model, value in index.items():
        path=Path(value).resolve()
        cfg=load_config(path)
        verify_round(cfg, architecture=model)
        found[model]=(path,cfg)
    for root in roots:
        if not root.is_dir():
            continue
        for path in root.rglob('config.yaml'):
            if 'previous_attempts' in path.parts or not path.with_name('manifest.json').is_file():
                continue
            cfg=load_config(path)
            model=cfg.get('experiment',{}).get('architecture')
            if not model or model in index:
                continue
            try:
                verify_round(cfg, architecture=model)
            except (RuntimeError,ValueError):
                continue
            if model in found and found[model][0]!=path.resolve():
                raise RuntimeError('Rondas compatibles ambiguas; no se elige ni crea otra: '+model)
            found[model]=(path.resolve(),cfg)
    return found


def effective_round(path):
    from utils.config_loader import load_config
    from utils.experiments import verify_round
    key=hashlib.sha256(str(Path(path).resolve()).casefold().encode()).hexdigest()[:8]
    short=Path.home()/'bf'/key
    overrides=['paths.runs_dir='+str(short/'r'), 'paths.results_dir='+str(short/'e')]
    cfg=load_config(path, overrides)
    verify_round(cfg)
    return cfg, overrides, short


def baseline_present(cfg, effective, model):
    """Any completed/attempted stage is preserved rather than silently restarted."""
    from utils.config_loader import cfg_path
    database=cfg_path(effective,'paths.queue_dir')/'queue.sqlite'
    job=None
    if database.is_file():
        with sqlite3.connect(database.resolve().as_uri()+'?mode=ro',uri=True) as c:
            c.row_factory=sqlite3.Row
            rows=[dict(r) for r in c.execute("SELECT * FROM jobs WHERE architecture=? AND kind='baseline'",(model,))]
        if len(rows)>1:
            raise RuntimeError('Baseline duplicado; conservar y revisar: '+model)
        job=rows[0] if rows else None
    bases={cfg_path(c,'paths.runs_dir')/model/'baseline' for c in (cfg,effective)}
    for folder in bases:
        if (folder/'summary.json').is_file():
            emit('BASELINE_EXISTENTE_NO_REPETIR',architecture=model,path=folder,queue_status=job['status'] if job else 'NO_JOB',approval='No se aprueba ni se valida definitivamente')
            return True
    if job and job['status']!='PENDING':
        raise RuntimeError('Baseline existente sin resumen o requiere revision/recuperacion explicita: '+str(job))
    # No change of run root when the original location contains any partial work.
    native=cfg_path(cfg,'paths.runs_dir')/model/'baseline'
    short=cfg_path(effective,'paths.runs_dir')/model/'baseline'
    if native!=short and native.is_dir() and any(native.iterdir()):
        raise RuntimeError('Trabajo previo en ruta original; se conserva sin migrar/reiniciar: '+str(native))
    if short.is_dir():
        for fold in range(3):
            f=short/f'fold{fold}'
            if f.is_dir() and any(f.iterdir()) and not (f/'completed_fold.json').is_file():
                raise RuntimeError('Fold parcial conservado; requiere reanudacion desde su checkpoint: '+str(f))
    return False


def choose_next(base, rounds, pc):
    from orchestration.pc_registry import PCRegistry
    from utils.architectures import get_arch
    from utils.weights import entry_for, verified_local_weight
    registry=PCRegistry(base)
    rejected=[]
    owned=registry.owned_slugs(pc)
    order=[get_arch(name).slug for name in registry.first_batch.get(pc,[]) if get_arch(name).slug in owned]
    order.extend(model for model in owned if model not in order)
    for model in order:
        if model in FINISHED:
            continue
        spec=get_arch(model)
        source=rounds[model][1] if model in rounds else base
        if model in rounds:
            effective,_,_=effective_round(rounds[model][0])
            if baseline_present(source,effective,model):
                rejected.append(dict(model=model,reason='Baseline existente: se conserva; revision/HPO de esa arquitectura quedan pendientes'))
                continue
        row=entry_for(spec,source)
        if not row['compatible_pretrained']:
            rejected.append(dict(model=model,reason=row['block_reason'],kind='BACKEND_BLOCKED'))
            continue
        try:
            weight=verified_local_weight(spec,source)
        except (OSError,RuntimeError,ValueError) as e:
            rejected.append(dict(model=model,reason=str(e),kind='WEIGHT_UNAVAILABLE',source=row['url']))
            continue
        return model,rejected,str(weight)
    return None,rejected,None


def main():
    if os.name!='nt':
        raise RuntimeError('Operacion exclusivamente en Windows; pruebas de funciones disponibles en CPU')
    root=Path.home()/'Documents/s2_no_borrar/Entrenamiento/Botrytis_Experiment_Framework_git'
    os.chdir(root)
    py=root/'.venv/Scripts/python.exe'
    if Path(sys.executable).resolve()!=py.resolve():
        return subprocess.call([str(py),'-u',str(Path(__file__).resolve()),*sys.argv[1:]])
    sys.path.insert(0,str(root/'src'))
    os.environ.update(PYTHONUTF8='1',PYTHONIOENCODING='utf-8',MPLBACKEND='Agg',BOTRYTIS_BOOTSTRAP_DIR=str(Path.home()/'btlog'))
    from utils.config_loader import load_config,cfg_path
    from utils.durable import FileLock,atomic_json
    from utils.guards import gpu_lock_path
    from orchestration.machine import read_binding
    from orchestration.pc_registry import PCRegistry
    from utils.experiments import create_round,verify_round
    from utils.manual_gates import require_gate
    from orchestration.study_queue import StudyQueue
    operator=os.environ.get('USERNAME','')
    pc=USERS.get(operator.casefold())
    base=load_config(root/'artifacts/v4/local.yaml')
    binding=read_binding(base)
    if not pc or base['compute']['pc_id']!=pc or binding['pc_id']!=pc or binding['machine_id'].casefold()!=operator.casefold():
        raise RuntimeError('Usuario, configuracion y binding no corresponden')
    lock=gpu_lock_path(base,pc)
    work=Path.home()/'bt-hoy-20261103'
    auditdir=Path.home()/'bp'/uuid.uuid4().hex[:8]
    auditdir.mkdir(parents=True)
    with FileLock(lock.with_suffix('.priority.guard')):
        if len(sys.argv)==3 and sys.argv[1]=='--restore':
            if read(sys.argv[2])['operator'].casefold()!=operator.casefold():
                raise RuntimeError('Restauracion de otro operador/equipo no autorizada')
            guard=wait_idle(lock)
            try:
                restore_pause(sys.argv[2]);return 0
            finally:
                guard.__exit__(None,None,None)
        if len(sys.argv)!=1:
            raise RuntimeError('Argumentos desconocidos')
        rounds=locate_rounds(base,work)
        paused=[]
        for model in FINISHED:
            if model in rounds and PCRegistry(rounds[model][1]).owner(model)==pc:
                cfg=rounds[model][1]
                paused.append(pause_queue(cfg_path(cfg,'paths.queue_dir')/'queue.sqlite',model,auditdir/model,operator))
        if not paused:
            raise RuntimeError('No se localizo una ronda propia de los modelos ya terminados; no se lanza otra GPU a ciegas')
        emit('PLAN_ORIGINAL_LOCAL',pc=pc,owned=PCRegistry(base).owned_slugs(pc),paused=paused)
        guard=wait_idle(lock)
        try:
            model,rejected,weight=choose_next(base,rounds,pc)
            report=dict(pc=pc,paused=paused,selected=model,deferred=rejected,weight=weight,
                        robustness='POSPUESTA, NO COMPLETE',final_validation=False,hpo_reexecution=False,
                        executor_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest())
            atomic_json(auditdir/'summary.json',report)
            if not model:
                emit('PAUSA_APLICADA_SIN_OTRO_MODELO_HABILITADO',report=auditdir/'summary.json',reasons=rejected)
                return 0
            require_gate(base,pc,'READY_FOR_EXPERIMENT')
            if model in rounds:
                path=rounds[model][0]
            else:
                source=copy.deepcopy(base)
                source['experiment']['registry_root']=str(Path.home()/'bn')
                path=create_round(source,round_id='n_'+model,architecture=model,operator=operator,
                                  reason='Prioridad solicitada: siguiente arquitectura; robustez anterior pospuesta',mode='supervised')
                index=read(work/'rondas.json') if (work/'rondas.json').is_file() else {}
                index[model]=str(path);atomic_json(work/'rondas.json',index)
            effective,overrides,short=effective_round(path)
            verify_round(effective,architecture=model)
            if PCRegistry(effective).owner(model)!=pc:
                raise RuntimeError('Otra PC es propietaria; no se transfiere implicitamente')
            require_gate(effective,pc,'READY_FOR_EXPERIMENT')
            q=StudyQueue.from_cfg(effective)
            from utils.architectures import get_arch
            q.plan(effective,[get_arch(model)])
            if baseline_present(load_config(path),effective,model):
                return 0
            command=[str(py),'-u',str(root/'src/06_run_baseline.py'),'--config',str(path),'--pc-id',pc,'--architecture',model]
            for value in overrides:
                command.extend(['--set',value])
            atomic_json(auditdir/'launch.json',dict(command=command,model=model,config=str(path),short=str(short),old_models_never_launched=sorted(FINISHED)))
            emit('INICIANDO_BASELINE_SIGUIENTE',model=model,command=command,report=auditdir/'summary.json')
            rc=subprocess.call(command,cwd=root)
            report.update(baseline_exit_code=rc,queue_counts=q.counts(),jobs=[j.as_json() for j in q.jobs()],config=str(path),short=str(short))
            atomic_json(auditdir/'summary.json',report)
            baseline_jobs=[j for j in q.jobs() if j.kind=='baseline' and j.architecture==model]
            if rc==0 and (len(baseline_jobs)!=1 or baseline_jobs[0].status not in ('COMPLETE','PENDING_REVIEW','PUBLICATION_PENDING')):
                raise RuntimeError('El proceso termino sin cierre del baseline; se conserva y no se reinicia: '+str(report['queue_counts']))
            emit('FIN_BASELINE_SIGUIENTE',code=rc,jobs=[j.as_json() for j in q.jobs()],report=auditdir/'summary.json',
                 review='PENDING_REVIEW requiere evidencia y aprobacion; no se inicia HPO automaticamente')
            return rc
        finally:
            guard.__exit__(None,None,None)


if __name__=='__main__':
    try:
        sys.exit(main())
    except KeyboardInterrupt:
        emit('CONTROL_INTERRUMPIDO',message='Pausas y resultados conservados; no borrar locks');sys.exit(130)
    except Exception as e:
        emit('DETENIDO_SIN_REINICIAR_EXPERIMENTOS',error=str(e));sys.exit(2)
