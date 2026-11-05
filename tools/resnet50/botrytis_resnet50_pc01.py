"""PC-01 exclusivamente ResNet50: baseline -> APROBAR -> HPO -> APROBAR -> robustez.
No pausa ni ejecuta YOLO; VAL20 solo tras aprobacion; TEST cerrado. Fuentes originales intactas.
"""
from pathlib import Path
import argparse,contextlib,copy,hashlib,importlib.util,json,os,subprocess,sys,uuid
HERE=Path(__file__).resolve().parent
MODEL='resnet50';PC='PC-01';USER='i21315'


def load(name):
    manifest=json.loads((HERE/'SHA256SUMS.json').read_text())
    path=HERE/name
    if hashlib.sha256(path.read_bytes()).hexdigest()!=manifest[name]:raise RuntimeError('Dependencia alterada: '+name)
    spec=importlib.util.spec_from_file_location('_resnet_'+path.stem,path)
    mod=importlib.util.module_from_spec(spec);sys.modules[spec.name]=mod;spec.loader.exec_module(mod);return mod


def check_identity(base,binding,operator):
    if operator.casefold()!=USER or base['compute']['pc_id']!=PC or binding['pc_id']!=PC or binding['machine_id'].casefold()!=USER:
        raise RuntimeError('Ejecutor exclusivo PC-01 / i21315')


def acquire_weight(project,base,spec):
    from utils.weights import verified_local_weight,weight_path,entry_for
    try:return verified_local_weight(spec,base)
    except FileNotFoundError:
        row=entry_for(spec,base);path=weight_path(row,base)
        if path.exists():raise RuntimeError('Peso existente sin procedencia valida; se conserva: '+str(path))
        print('PESOS_RESNET50: '+row['url']+' | '+row['license'],flush=True)
        if not sys.stdin.isatty() or input('Si acepta las condiciones indicadas y la descarga, escriba ACEPTO: ').strip()!='ACEPTO':
            raise RuntimeError('Descarga/licencia no autorizada; no se entrena')
        subprocess.run([sys.executable,'-u',str(project/'tools/manage_weights.py'),'acquire','--architecture',MODEL,'--allow-network','--accept-license'],cwd=project,check=True)
        return verified_local_weight(spec,base)


def prepare_round(project,base,home,operator,pr):
    from orchestration.pc_registry import PCRegistry
    from utils.architectures import get_arch
    from utils.manual_gates import require_gate
    from utils.experiments import create_round,verify_round
    from utils.durable import atomic_json
    if PCRegistry(base).owner(MODEL)!=PC:raise RuntimeError('ResNet50 no pertenece a PC-01')
    require_gate(base,PC,'READY_FOR_EXPERIMENT')
    acquire_weight(project,base,get_arch(MODEL))
    work=home/'bt-hoy-20261103';rounds=pr.locate_rounds(base,work)
    if MODEL in rounds:path=rounds[MODEL][0]
    else:
        source=copy.deepcopy(base);source['experiment']['registry_root']=str(home/'bn')
        path=create_round(source,round_id='n_resnet50',architecture=MODEL,operator=operator,
                          reason='Completar ResNet50 del plan original de PC-01; YOLO intactos',mode='supervised')
    index=pr.read(work/'rondas.json') if (work/'rondas.json').is_file() else {}
    if MODEL in index and Path(index[MODEL]).resolve()!=Path(path).resolve():raise RuntimeError('Indice ResNet50 ambiguo; no se sobrescribe')
    if MODEL not in index:index[MODEL]=str(path);atomic_json(work/'rondas.json',index)
    cfg,sets,short=pr.effective_round(path);verify_round(cfg,architecture=MODEL)
    require_gate(cfg,PC,'READY_FOR_EXPERIMENT')
    return path,cfg,sets,short


@contextlib.contextmanager
def held_idle(pr,lock):
    guard=pr.wait_idle(lock)  # wait_idle already acquired this guard.
    try:yield
    finally:guard.__exit__(None,None,None)


def baseline(project,base,home,pr):
    from utils.config_loader import load_config
    from utils.guards import gpu_lock_path
    from utils.durable import FileLock,atomic_json
    from utils.architectures import get_arch
    from orchestration.study_queue import StudyQueue
    lock=gpu_lock_path(base,PC)
    with FileLock(lock.with_suffix('.resnet-controller.guard')),held_idle(pr,lock):
        path,cfg,sets,short=prepare_round(project,base,home,os.environ['USERNAME'],pr)
        q=StudyQueue.from_cfg(cfg);q.plan(cfg,[get_arch(MODEL)])
        if pr.baseline_present(load_config(path),cfg,MODEL):
            return path,short
        cmd=[sys.executable,'-u',str(project/'src/06_run_baseline.py'),'--config',str(path),'--pc-id',PC,'--architecture',MODEL]
        for val in sets:cmd+=['--set',val]
        audit=short/'resnet_control'/uuid.uuid4().hex[:8];audit.mkdir(parents=True)
        atomic_json(audit/'launch.json',dict(command=cmd,architecture=MODEL,other_models_started=False))
        pr.emit('INICIANDO_BASELINE_RESNET50',folds=3,max_epochs=cfg['classifier']['baseline']['epochs'],results=str(short/'r/resnet50/baseline'))
        rc=subprocess.call(cmd,cwd=project)
        atomic_json(audit/'resultado.json',dict(code=rc,jobs=[j.as_json() for j in q.jobs()]))
        if rc:raise RuntimeError('Baseline detenido; resultados conservados. Codigo '+str(rc))
        jobs=[j for j in q.jobs() if j.kind=='baseline' and j.architecture==MODEL]
        if len(jobs)!=1 or jobs[0].status not in ('COMPLETE','PENDING_REVIEW','PUBLICATION_PENDING'):
            raise RuntimeError('Baseline sin cierre verificable; no se inicia HPO')
        return path,short


def robust():
    r=load('botrytis_robustez_reanudar.py')
    r.MODELS[PC]=[MODEL]
    sys.argv=[str(r.SELF),'--expected-pc',PC]
    return r.main()


def main():
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--robust-only',action='store_true');args=parser.parse_args()
    if os.name!='nt':raise RuntimeError('Operacion exclusiva Windows PC-01; pruebas CPU separadas')
    project=Path.home()/'Documents/s2_no_borrar/Entrenamiento/Botrytis_Experiment_Framework_git'
    py=project/'.venv/Scripts/python.exe'
    if Path(sys.executable).resolve()!=py.resolve():return subprocess.call([str(py),'-u',str(Path(__file__).resolve()),*sys.argv[1:]])
    sys.path.insert(0,str(project/'src'));os.chdir(project)
    os.environ.update(PYTHONUTF8='1',PYTHONIOENCODING='utf-8',MPLBACKEND='Agg',BOTRYTIS_BOOTSTRAP_DIR=str(Path.home()/'btlog'))
    from utils.config_loader import load_config
    from orchestration.machine import read_binding
    base=load_config(project/'artifacts/v4/local.yaml');check_identity(base,read_binding(base),os.environ.get('USERNAME',''))
    if args.robust_only:return robust()
    from utils.guards import gpu_lock_path
    from utils.durable import FileLock
    with FileLock(gpu_lock_path(base,PC).with_suffix('.resnet-pipeline.guard')):
        pr=load('botrytis_prioridad.py')
        # Read-only code check; do not update or checkout source files.
        r=load('botrytis_robustez_reanudar.py');code=r.audit.readonly_code(project)
        if code['git_commit']!=r.COMMIT or code['code_hash']!=r.CODE:raise RuntimeError('Codigo incompatible; no se modifica')
        path,short=baseline(project,base,Path.home(),pr)
        rc=subprocess.call([sys.executable,'-u',str(HERE/'botrytis_resnet_hpo_pc01.py'),'--expected-pc',PC],cwd=project)
        if rc:return rc
        rc=subprocess.call([sys.executable,'-u',str(Path(__file__).resolve()),'--robust-only'],cwd=project)
        if rc:return rc
        return subprocess.call([sys.executable,'-u',str(HERE/'botrytis_train_val_continuar.py'),'--expected-pc',PC,'--only-model',MODEL],cwd=project)



if __name__=='__main__':
    try:sys.exit(main())
    except KeyboardInterrupt:print('INTERRUMPIDO: resultados conservados; no borrar locks',flush=True);sys.exit(130)
    except Exception as exc:print('RESNET50_DETENIDO: '+str(exc),flush=True);sys.exit(2)
