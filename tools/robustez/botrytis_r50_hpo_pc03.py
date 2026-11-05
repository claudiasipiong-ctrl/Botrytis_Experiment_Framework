"""PC-03: verifica evidencias, registra autorizacion explicita y ejecuta HPO R50-vd.
No llama a 06 ni a 08, no cambia protocolo ni aprueba automaticamente el HPO.
Requiere --aprobar-original-hpo, solicitado expresamente por el usuario.
"""
from __future__ import annotations
import argparse,copy,hashlib,importlib,importlib.util,json,math,os,pickle,re,shutil,socket,sqlite3,subprocess,sys,tempfile,traceback,uuid
from pathlib import Path
from datetime import datetime,timezone
AUDIT_SHA='432881891c91bc6e20becc7fec97d245a657ce5c7e6f1677fb99c224d344149e'
p=Path(__file__).with_name('botrytis_r50_auditoria.py')
if not p.is_file() or hashlib.sha256(p.read_bytes()).hexdigest()!=AUDIT_SHA:raise RuntimeError('Falta auditoria fijada o su SHA256 no coincide')
spec=importlib.util.spec_from_file_location('r50_audit_pinned',p)
audit=importlib.util.module_from_spec(spec);sys.modules[spec.name]=audit;spec.loader.exec_module(audit)
MODEL= audit.MODEL;PC=audit.PC;USER=audit.USER;BUDGET=15
EXECUTOR_SHA256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest()
# Same pinned trainer and scientific protocol. Accept a known predecessor
# solely with matching job/attempt and
# a demonstrably dead local process (never just an old heartbeat).
LEGACY_EXECUTOR_SHAS=('160ce95cb02e3ce2a4decf4a2698e5646032b05bbfcbb6b0887b0a8bfef806e7',
                      'a6967e3073f5a9ce14e6fa389a59c5b9d8c410a6b2f99a3e49f83bf9d33d14e1')
Stop=audit.Stop
class RecoveryStop(BaseException):pass
utc=audit.utc;read=audit.read;digest=audit.digest;canonical=audit.canonical
atomic_bytes=audit.atomic_bytes;atomic_json=audit.atomic_json;emit=audit.emit
git=audit.git;verified_copy=audit.verified_copy;fold_signature=audit.fold_signature;verify_fold=audit.verify_fold
validate_checkpoint=audit.validate_checkpoint


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

def inside(path, root):
    try: Path(path).resolve().relative_to(Path(root).resolve()); return True
    except ValueError: return False

def copy_tree_safe(src, dst, skip=()):
    src, dst = Path(src), Path(dst)
    for p in sorted(src.rglob('*')):
        rel = p.relative_to(src)
        if any(x in skip for x in rel.parts): continue
        if p.is_symlink(): raise Stop('Enlace inesperado en evidencias: ' + str(p))
        if p.is_file(): verified_copy(p, dst / rel)

def allowed_trial_folder(cfg, spec, trial):
    from utils.config_loader import cfg_path
    value = trial.user_attrs.get('run_dir')
    if not value: raise Stop(f'Trial {trial.number}: falta run_dir')
    folder = Path(value).resolve()
    roots = [cfg_path(cfg, 'paths.runs_dir'), cfg_path(cfg, 'paths.artifacts_dir').parent / 'runs']
    expected = [(Path(r) / spec.slug / 'hpo' / f'trial_{trial.number:03d}').resolve() for r in roots]
    if folder not in expected: raise Stop(f'Trial {trial.number}: carpeta fuera de las rutas aprobadas: {folder}')
    return folder

class OptimizerState:
    """Atomic, hash-checked local sampler/pruner state. Never read from Drive."""
    def __init__(self, root, identity):
        self.root = Path(root); self.identity = identity
    def save(self, study):
        import optuna
        data = pickle.dumps((study.sampler, study.pruner), protocol=pickle.HIGHEST_PROTOCOL)
        sha = hashlib.sha256(data).hexdigest(); self.root.mkdir(parents=True, exist_ok=True)
        # Content-addressed files make a power loss between data and pointer safe.
        target = self.root / (sha[:20] + '.pkl')
        if not target.exists(): atomic_bytes(target, data)
        atomic_json(self.root / 'state.json', dict(schema=1, identity=self.identity, sha256=sha, file=target.name,
                    optuna=optuna.__version__, python=list(sys.version_info[:2]), hostname=socket.gethostname(), draws=[dict(number=t.number,params=t.params) for t in study.trials if t.params], utc=utc()))
    def restore(self, study):
        import optuna
        pointer = self.root / 'state.json'
        if not pointer.exists(): return False
        meta = read(pointer); file = self.root / meta['file']
        if Path(meta['file']).name != meta['file'] or meta.get('identity') != self.identity or meta.get('hostname') != socket.gethostname(): raise Stop('Estado Optuna local de otra identidad/maquina')
        if meta.get('optuna') != optuna.__version__ or meta.get('python') != list(sys.version_info[:2]) or digest(file) != meta.get('sha256'): raise Stop('Estado sampler/pruner alterado o incompatible')
        draws=[dict(number=t.number,params=t.params) for t in study.trials if t.params]
        if meta.get('draws')!=draws:return False  # caller requires exact native replay
        sampler, pruner = pickle.loads(file.read_bytes())
        study.sampler, study.pruner = sampler, pruner
        return True

def checkpoint_candidates(folder):
    """Only original signed snapshots, not optimizer-free pretrained weights."""
    folder = Path(folder); found = []
    for side in folder.rglob('*.resume.json'):
        if any(p in ('previous_attempts', 'archive') for p in side.relative_to(folder).parts): continue
        path = Path(str(side)[:-len('.resume.json')])
        if path.is_file(): found.append((path, read(side)))
    return found


def recovery_family(study,trial,cfg,spec):
    """Follow only explicit identical-parameter retries, never arbitrary paths."""
    by_number={t.number:t for t in study.trials};family=[];seen=set();hp=trial.user_attrs.get('hp')
    while True:
        if trial.number in seen:raise Stop('Ciclo en linaje de recuperacion')
        seen.add(trial.number)
        folder=allowed_trial_folder(cfg,spec,trial)
        if not folder.is_dir():raise Stop('Carpeta de recuperacion ausente: '+str(folder))
        if trial.user_attrs.get('hp')!=hp:raise Stop('HP diferentes en linaje de recuperacion')
        family.append((trial,folder))
        parent=trial.user_attrs.get('bt_retry_of')
        if parent is None:break
        previous=by_number.get(parent)
        if (previous is None or previous.state.name!='FAIL' or previous.params!=trial.params
                or trial.user_attrs.get('bt_recovery_source')!=previous.user_attrs.get('run_dir')):
            raise Stop('Linaje de recuperacion no comprobable: trial '+str(trial.number))
        trial=previous
    return family


def select_partial_checkpoint(folder,cfg,spec,hp,seed,fold,epochs,patience):
    """Exclude only uncommitted unsigned files; signed corruption stays blocking."""
    folder=Path(folder);pointer=read(folder/'resume.json') if (folder/'resume.json').is_file() else None
    files=sorted((folder/'cp').glob('e*.pt'));valid=[];unpublished=[]
    if pointer:
        committed=folder/'cp'/Path(pointer.get('path','')).name
        if not committed.is_file() or not committed.with_suffix('.pt.resume.json').is_file():
            raise Stop('El puntero publicado carece de checkpoint/recibo: '+str(committed))
        if digest(committed)!=pointer.get('sha256'):raise Stop('Puntero de checkpoint alterado: '+str(committed))
    for cp in files:
        match=re.fullmatch(r'e(\d+)\.pt',cp.name)
        if not cp.is_file() or not match:raise Stop('Ruta de checkpoint inesperada: '+str(cp))
        epoch=int(match[1]);side=cp.with_suffix('.pt.resume.json')
        if not side.is_file():
            unpublished.append(dict(path=str(cp),sha256=digest(cp),epoch=epoch,receipt_missing=str(side),
                                    disposition='UNPUBLISHED_NOT_USED_OR_DELETED'))
            continue
        validate_checkpoint(cp,spec.slug,cfg,seed=seed,hp=hp,fold=fold)
        meta=read(side);extra=meta.get('extra',{})
        if extra.get('adapter_sha256')!=audit.ADAPTER_SHA or extra.get('epoch')!=epoch or not 1<=epoch<=epochs:
            raise Stop('Contrato de estado/epoca incompatible: '+str(cp))
        valid.append((epoch,cp))
    if not valid:
        if files or (folder/'results.csv').is_file() or any(folder.glob('*.pt')):
            return dict(unrecoverable=True,folder=str(folder),unpublished=unpublished)
        return None
    epoch,selected=max(valid,key=lambda item:item[0])
    # Hash/receipt checked before deserializing. Read only the selected state.
    import torch
    state=torch.load(selected,map_location='cpu',weights_only=False)
    from utils.experiments import fingerprint
    expected=canonical(dict(experiment=fingerprint(cfg),hp=dict(hp),seed=seed,fold=fold,epochs=epochs,patience=patience,fixed=False))
    adapter=audit.load_adapter()
    if (state.get('format')!=adapter.FORMAT or state.get('signature')!=expected or state.get('epoch')!=epoch
            or state.get('seed')!=seed or state.get('hp')!=dict(hp) or state.get('fold')!=fold):
        raise Stop('Payload del estado no corresponde a su contrato: '+str(selected))
    required=('model','ema','optimizer','scheduler','scaler','gradients','rng','best_state','best_metrics')
    if any(not isinstance(state.get(k),dict) for k in required):raise Stop('Estado incompleto de optimizador/modelo/RNG: '+str(selected))
    if any(k not in state for k in ('ema_updates','best_epoch','best_score','last_opt_step','elapsed','stopper_epoch','stopper_fitness','stopped')):
        raise Stop('Estado incompleto de reanudacion: '+str(selected))
    import pandas as pd
    h=pd.DataFrame(state.get('history',[]))
    if len(h)!=epoch:raise Stop('Historial del checkpoint incompleto: '+str(selected))
    inspect_history(h,str(selected))
    if not 1<=int(state['best_epoch'])<=epoch or not math.isfinite(float(state['best_score'])):
        raise Stop('Mejor epoca/score del checkpoint invalido: '+str(selected))
    if not math.isclose(float(h.map50_95.max()),float(state['best_score']),abs_tol=1e-7):
        raise Stop('Score del checkpoint discrepa de su historial: '+str(selected))
    return dict(path=str(selected),folder=str(folder),epoch=epoch,sha256=digest(selected),
                receipt_sha256=digest(selected.with_suffix('.pt.resume.json')),unpublished=unpublished)


def recovery_plan(study,trial,cfg,spec):
    """Merge verified per-fold progress across the explicit retry family."""
    from utils.experiments import fingerprint
    from utils.architectures import section_for
    family=recovery_family(study,trial,cfg,spec);sec=section_for(spec,cfg);hp=trial.user_attrs['hp']
    seed=int(cfg['reproducibility']['seed']);epochs=int(sec['max_epochs']);patience=int(sec['early_stopping_patience'])
    plan=dict(trial=trial.number,family=[t.number for t,_ in family],folds=[],has_work=False)
    for fold in range(3):
        completed=[];partials=[];unrecoverable=[]
        for source,folder in family:
            out=folder/f'fold{fold}'
            if not out.is_dir():continue
            if (out/'completed_fold.json').is_file():
                sig=fold_signature(fingerprint(cfg),spec.slug,hp,seed,fold,epochs,patience)
                receipt,h=verify_fold(out,sig,fold);inspect_curve(out/'completed_history.csv')
                validate_checkpoint(out/'best.pt',spec.slug,cfg,seed=seed,hp=hp,fold=fold)
                completed.append(dict(kind='COMPLETE',fold=fold,source=str(out),source_trial=source.number,
                    history_sha256=receipt['history_sha256'],best_sha256=digest(out/'best.pt')))
        if completed:
            if len({(r['history_sha256'],r['best_sha256']) for r in completed})!=1:
                raise Stop('Folds completos contradictorios en linaje: '+str(fold))
            plan['folds'].append(completed[0]);plan['has_work']=True;continue
        for source,folder in family:
            out=folder/f'fold{fold}'
            if not out.is_dir():continue
            selected=select_partial_checkpoint(out,cfg,spec,hp,seed,fold,epochs,patience)
            if selected and selected.get('unrecoverable'):unrecoverable.append(selected)
            elif selected:partials.append(dict(selected,source_trial=source.number,fold=fold,kind='PARTIAL'))
        if partials:
            latest=max(r['epoch'] for r in partials);chosen=[r for r in partials if r['epoch']==latest]
            if len({r['sha256'] for r in chosen})!=1:raise Stop('Checkpoints de igual epoca contradictorios: '+str(fold))
            plan['folds'].append(dict(chosen[0],unrecoverable_copies=unrecoverable));plan['has_work']=True
        elif unrecoverable:
            raise Stop('No hay estado de optimizador con recibo integro en el linaje del fold '+str(fold)+': '+str(unrecoverable))
        else:plan['folds'].append(dict(kind='NEW',fold=fold))
    return plan


def retry_leaves(study):
    trials=study.get_trials(deepcopy=False)
    parents={t.user_attrs.get('bt_retry_of') for t in trials if t.user_attrs.get('bt_retry_of') is not None}
    return [t for t in trials if t.state.name in ('FAIL','RUNNING') and t.number not in parents]


def preview_recovery(study,cfg,spec,directory):
    plans=[]
    for trial in retry_leaves(study):
        if trial.state.name=='RUNNING' and owner_alive(trial.user_attrs.get('process_owner',{})):
            raise Stop('Trial sigue activo; no se recupera')
        if not trial.user_attrs.get('hp') or not trial.user_attrs.get('run_dir') or not trial.params:continue
        plans.append(recovery_plan(study,trial,cfg,spec))
    by_number={t.number:t for t in study.trials}
    for pending in study.trials:
        if pending.state.name!='WAITING' or pending.user_attrs.get('bt_retry_of') is None:continue
        previous=by_number.get(pending.user_attrs['bt_retry_of'])
        if (previous is None or previous.state.name!='FAIL'
                or pending.user_attrs.get('bt_recovery_source')!=previous.user_attrs.get('run_dir')
                or pending.system_attrs.get('fixed_params')!=previous.params):
            raise Stop('Retry WAITING sin vinculo y parametros comprobables')
        plans.append(dict(recovery_plan(study,previous,cfg,spec),waiting_trial=pending.number))
    atomic_json(Path(directory)/'recovery-preflight.json',dict(plans=plans,queue_modified=False,training=False))
    emit('RECUPERACION_COMPROBADA_ANTES_DE_RECLAMAR',plans=plans)
    return plans


def recovered_failures(study,cfg,spec):
    """FAIL remains FAIL; only a verified closed identical retry covers it."""
    covered={}
    for trial in study.trials:
        if trial.state.name not in ('COMPLETE','PRUNED') or trial.user_attrs.get('bt_retry_of') is None:continue
        for ancestor,_ in recovery_family(study,trial,cfg,spec)[1:]:covered[ancestor.number]=trial.number
    failed={t.number for t in study.trials if t.state.name=='FAIL'}
    if failed-set(covered):raise Stop('Trials fallidos sin recuperacion cerrada verificable: '+str(sorted(failed-set(covered))))
    return [dict(failed_trial=n,recovered_by=covered[n],original_state_preserved='FAIL') for n in sorted(failed)]

def schedule_retries(study, cfg, spec, audit_dir):
    """Public Optuna APIs: retain old FAIL, enqueue the identical params once.

    Completed and PRUNED trials never get replayed. A retry links its original
    path and reuses only receipts matching the same experiment/hp/seed/fold.
    """
    import optuna
    from utils.experiments import fingerprint
    from utils.architectures import section_for
    sec = section_for(spec, cfg); exp = fingerprint(cfg); actions = []
    trials = study.get_trials(deepcopy=False)
    for t in trials:
        if t.state.name == 'RUNNING':
            if owner_alive(t.user_attrs.get('process_owner', {})): raise Stop(f'YA_ACTIVO: trial {t.number} conserva un proceso vivo')
            study.tell(t.number, state=optuna.trial.TrialState.FAIL)
            # Keep the project's trial mirror consistent with Optuna after a
            # power loss; updating the old trial is not approving/repeating it.
            from orchestration.study_queue import StudyQueue
            from utils.experiments import identity
            from utils.config_loader import cfg_path
            from utils.splits import current_split_hash
            ident = identity(cfg)
            StudyQueue.from_cfg(cfg).record_trial(study.study_name, t.number, 'FAILED', pc_id=cfg['compute']['pc_id'],
                run_dir=t.user_attrs.get('run_dir'), provenance=dict(git_commit=ident['git_commit'], dataset_hash=ident['dataset'],
                    split_hash=current_split_hash(cfg_path(cfg, 'paths.artifacts_dir'), cfg['outer_split']['manifest_name'])), finished=True)
            actions.append(dict(action='DEAD_PROCESS_TO_FAIL', trial=t.number))
    trials = study.get_trials(deepcopy=False)
    already = {t.user_attrs.get('bt_retry_of') for t in trials if t.user_attrs.get('bt_retry_of') is not None}
    for t in trials:
        if t.state.name != 'FAIL' or t.number in already: continue
        hp, directory = t.user_attrs.get('hp'), t.user_attrs.get('run_dir')
        if not hp or not directory or not t.params: continue  # no completed work exists to recover
        if t.user_attrs.get('pc_id') not in (None, cfg['compute']['pc_id']): raise Stop('Trial fallido de otra PC')
        folder = allowed_trial_folder(cfg, spec, t)
        plan=recovery_plan(study,t,cfg,spec)
        if plan['has_work']:
            study.enqueue_trial(dict(t.params), user_attrs=dict(bt_retry_of=t.number, bt_recovery_source=str(folder.resolve()),
                                bt_recovered_folds=[r['fold'] for r in plan['folds'] if r['kind']=='COMPLETE'],
                                bt_recovery_plan=plan,bt_executor_sha256=EXECUTOR_SHA256), skip_if_exists=False)
            actions.append(dict(action='RETRY_IDENTICAL_PARAMS', trial=t.number, plan=plan))
    # A callback failure may leave the queue's trial mirror RUNNING even after
    # Optuna durably recorded FAIL. Reconcile only existing nonfinal mirror
    # rows; do not change Optuna, historical values, timestamps or approvals.
    from utils.config_loader import cfg_path
    mirror=cfg_path(cfg,'paths.queue_dir')/'queue.sqlite'
    if mirror.is_file():
        with sqlite3.connect(mirror,timeout=30) as con:
            con.execute('BEGIN IMMEDIATE')
            for failed in study.get_trials(deepcopy=False):
                if failed.state.name!='FAIL':continue
                row=con.execute('SELECT status FROM trials WHERE study_name=? AND trial_number=?',(study.study_name,failed.number)).fetchone()
                if row and row[0] in ('COMPLETE','PRUNED'):raise Stop('Estado Optuna/cola contradictorio para trial '+str(failed.number))
                if row and row[0]!='FAILED':
                    con.execute("UPDATE trials SET status='FAILED',updated_at=? WHERE study_name=? AND trial_number=?",(utc(),study.study_name,failed.number))
    if actions: atomic_json(Path(audit_dir) / ('recovery-' + uuid.uuid4().hex[:8] + '.json'), dict(utc=utc(), study=study.study_name, actions=actions))
    return actions

def replay_optimizer_state(study, cfg, spec, state_dir):
    """Recover legacy sampler state only when a deterministic replay matches all draws.

    The project creates one study per SQLite (study_id=1). Use an isolated replay
    SQLite with the same native ID, so Optuna's cached search-space identity stays
    valid after restoring the sampler. The real study is read-only during replay.
    """
    import optuna
    from utils.optuna_storage import make_sampler, make_pruner, study_db_path
    from utils.search_spaces import load_search_spaces, suggest_hyperparameters
    db = study_db_path(cfg, spec)
    with sqlite3.connect(db.resolve().as_uri() + '?mode=ro', uri=True) as con:
        rows = con.execute('SELECT study_id,study_name FROM studies').fetchall()
    if rows != [(1, study.study_name)]: raise Stop('Sin estado sampler y SQLite multiexperimento; no se adivina el estado de optimizacion')
    destination = Path(state_dir) / ('replay-' + uuid.uuid4().hex[:8] + '.db')
    sampler = make_sampler(cfg, int(cfg['reproducibility']['seed'])); pruner = make_pruner(cfg)
    replay = optuna.create_study(storage='sqlite:///' + destination.as_posix(), study_name=study.study_name,
                                 direction=study.direction, sampler=sampler, pruner=pruner)
    spaces = load_search_spaces(); traced = []
    for old in study.get_trials(deepcopy=False):
        fixed = old.system_attrs.get('fixed_params')
        if old.state.name == 'WAITING':
            if fixed is None: raise Stop('Trial WAITING sin parametros fijos; no se puede reconstruir')
            replay.enqueue_trial(dict(fixed), user_attrs=dict(old.user_attrs)); continue
        if fixed is not None: replay.enqueue_trial(dict(fixed), user_attrs=dict(old.user_attrs))
        trial = replay.ask(); hp = suggest_hyperparameters(trial, spec, cfg, spaces)
        if trial.number != old.number or trial.params != old.params or (old.user_attrs.get('hp') and hp != old.user_attrs['hp']):
            raise Stop(f'No se puede reconstruir el sampler sin alterar las propuestas: trial {old.number}. Se conserva el estudio original.')
        for step,value in sorted(old.intermediate_values.items()): trial.report(value, step)
        if old.state.name == 'COMPLETE': replay.tell(trial, old.value)
        elif old.state.name == 'PRUNED': replay.tell(trial, state=optuna.trial.TrialState.PRUNED)
        elif old.state.name == 'FAIL': replay.tell(trial, state=optuna.trial.TrialState.FAIL)
        elif old.state.name != 'RUNNING': raise Stop('Estado de trial no soportado para replay')
        traced.append(dict(number=old.number, state=old.state.name, exact_params_match=True))
    study.sampler, study.pruner = replay.sampler, replay.pruner
    atomic_json(Path(state_dir) / 'sampler-replay.json', dict(status='EXACT_NATIVE_REPLAY', trials=traced, replay_sqlite=str(destination), utc=utc()))
    return traced

def snapshot_sqlite(db, destination):
    if not Path(db).is_file(): return
    Path(destination).parent.mkdir(parents=True,exist_ok=True)
    with sqlite3.connect(Path(db).resolve().as_uri() + '?mode=ro', uri=True, timeout=15) as src:
        with sqlite3.connect(destination, timeout=15) as dst: src.backup(dst)
    with sqlite3.connect(destination) as test:
        if test.execute('PRAGMA integrity_check').fetchone()[0] != 'ok': raise Stop('Snapshot SQLite no integro')

class R50Recovery:
    def __init__(self,cfg,spec,state_dir):
        from utils.experiments import fingerprint
        self.cfg,self.spec,self.state_dir=cfg,spec,Path(state_dir)
        self.state_dir.mkdir(parents=True,exist_ok=True)
        self.state=OptimizerState(self.state_dir/'sampler',fingerprint(cfg));self.study=None
    def event(self, name, **data):
        path = self.state_dir / 'events.jsonl'
        with path.open('a', encoding='utf-8') as f:
            f.write(json.dumps(dict(utc=utc(), event=name, executor_sha256=EXECUTOR_SHA256, **data), ensure_ascii=False, default=str) + '\n')
            f.flush()
        emit(name,**data)

    def create_study(self, original, spec, cfg, seed):
        study = original(spec, cfg, seed)
        self.study = study
        restored = self.state.restore(study)
        if not restored and study.trials:
            replay_optimizer_state(study, cfg, spec, self.state_dir)
            self.event('LEGACY_OPTIMIZER_STATE_RECONSTRUCTED', existing_trials=len(study.trials))
        self.event('OPTIMIZER_STATE_RESTORED' if restored else 'OPTIMIZER_STATE_INITIALIZED', existing_trials=len(study.trials))
        from utils.optuna_storage import budget_used
        from utils.architectures import trials_for
        if budget_used(study) < trials_for(spec, cfg):
            schedule_retries(study, cfg, spec, self.state_dir)
        self.state.save(study)
        optimize = study.optimize

        def robust_optimize(objective, *a, **kw):
            callbacks = list(kw.pop('callbacks', []) or [])

            def begin(trial):
                trial.set_user_attr('process_owner', own_process())
                trial.set_user_attr('bt_executor_sha256', EXECUTOR_SHA256)
                try:
                    return objective(trial)
                except (KeyboardInterrupt, SystemExit):
                    trial.set_user_attr('bt_interrupted', True)
                    raise

            def persist(st, trial):
                self.state.save(st)
                self.event('TRIAL_DURABLE', trial=trial.number, state=trial.state.name, value=trial.value)
                if trial.state.name == 'FAIL':
                    actions = schedule_retries(st, cfg, spec, self.state_dir)
                    if not any((x.get('action') == 'RETRY_IDENTICAL_PARAMS' and x.get('trial') == trial.number for x in actions)):
                        raise RecoveryStop(f'Trial {trial.number} fallo sin evidencia suficiente de recuperacion; archivos y estudio conservados')
            try:
                return optimize(begin, *a, callbacks=[persist, *callbacks], **kw)
            finally:
                self.state.save(study)
        study.optimize = robust_optimize
        return study

    def evaluate(self,original,spec,hp,cfg,ctx,run_dir,**kw):
        from utils.experiments import fingerprint
        trial=kw.get('trial')
        if spec.slug!=MODEL or trial is None:raise RecoveryStop('Hook exclusivo del HPO R50-vd')
        run_dir=Path(run_dir);trial.set_user_attr('bt_executor_sha256',EXECUTOR_SHA256)
        trial.set_user_attr('bt_adapter_sha256',audit.ADAPTER_SHA)
        source_name=trial.user_attrs.get('bt_recovery_source')
        if not source_name and any((p/'completed_fold.json').exists() or (p/'results.csv').exists() or any(p.rglob('*.pt')) for p in run_dir.glob('fold*') if p.is_dir()):raise RecoveryStop('Trial nuevo contiene trabajo previo sin vinculo de recuperacion; se conserva sin entrenar')
        if source_name:
            previous=next((t for t in self.study.trials if t.number==trial.user_attrs.get('bt_retry_of')),None)
            if previous is None or previous.state.name!='FAIL' or previous.user_attrs.get('hp')!=dict(hp) or previous.user_attrs.get('run_dir')!=source_name:raise RecoveryStop('Retry de otra fuente/HP/estado')
            plan=recovery_plan(self.study,previous,cfg,spec)
            atomic_json(self.state_dir/('plan-trial-'+str(trial.number)+'.json'),plan)
            trial.set_user_attr('bt_recovery_plan',plan)
            for entry in plan['folds']:
                fold=entry['fold'];dst=run_dir/f'fold{fold}'
                if entry['kind']=='COMPLETE':
                    src=Path(entry['source'])
                    copy_tree_safe(src,dst)
                    self.event('FOLD_CONSERVADO_SIN_ENTRENAR',trial=trial.number,source=str(src),fold=fold)
                elif entry['kind']=='PARTIAL':
                    cp=Path(entry['path']);src=Path(entry['folder'])
                    # Keep the original directory untouched. The new active view
                    # contains only the selected genuine file and its genuine
                    # receipt, so the pinned backend cannot select an orphan.
                    copy_tree_safe(src,dst,skip=('cp','data.yaml','train.txt','val.txt','best.pt','best.pt.resume.json',
                                                'adapter.json','resume.json','results.csv'))
                    verified_copy(cp,dst/'cp'/cp.name)
                    verified_copy(cp.with_suffix('.pt.resume.json'),dst/'cp'/(cp.name+'.resume.json'))
                    self.event('ESTADO_R50_RECUPERADO',trial=trial.number,fold=fold,checkpoint=str(cp),epoch=entry['epoch'],
                               unpublished_preserved=entry['unpublished'],original_preserved=True)
            intermediates={}
            for ancestor,_ in recovery_family(self.study,previous,cfg,spec):
                for step,value in ancestor.intermediate_values.items():
                    if step in intermediates and not math.isclose(intermediates[step],float(value),rel_tol=0,abs_tol=1e-12):
                        raise RecoveryStop('Intermedios contradictorios en linaje')
                    intermediates[step]=float(value)
            for step,value in sorted(intermediates.items()):trial.report(value,int(step))
        self.state.save(self.study)  # sampler durable after draw, before any GPU work
        self.event('TRIAL_RTDETR_EN_EJECUCION',trial=trial.number,configuration=trial.user_attrs.get('config_id'),hp=dict(hp),folds=kw['folds'])
        return original(spec,hp,cfg,ctx,run_dir,**kw)

    def install(self,native,adapter):
        import utils.hpo_runner as runner
        if runner.train_detector is not adapter.train_detector:raise Stop('Backend R50 oficial no instalado; se impide fallback Ultralytics')
        create=native.create_or_load_study;evaluate=runner.evaluate_config
        native.create_or_load_study=lambda spec,cfg,seed:self.create_study(create,spec,cfg,seed)
        runner.evaluate_config=lambda *args,**kw:self.evaluate(evaluate,*args,**kw)


def inspect_curve(path):
    import pandas as pd
    return inspect_history(pd.read_csv(path),str(path))


def inspect_history(h,source):
    import numpy as np
    import pandas as pd
    columns=['epoch','optimization_loss_total','train_loss','val_loss','train/vfl_loss','train/bbox_loss','train/giou_loss',
        'val/vfl_loss','val/bbox_loss','val/giou_loss','map50_95','map50','precision','recall']
    for column in columns:
        if column not in h:raise Stop('Falta columna garantizada por adaptador oficial: '+column+' en '+str(source))
        values=pd.to_numeric(h[column],errors='coerce')
        if h.empty or not np.isfinite(values).all() or (values<0).any():raise Stop('Historial vacio/no finito/negativo: '+str(source)+' '+column)
    if h.epoch.tolist()!=list(range(1,len(h)+1)):raise Stop('Historial de epocas incompleto/duplicado: '+str(source))
    if any((h[c]>1).any() for c in ('map50_95','map50','precision','recall')):raise Stop('Objetivos fuera de rango')
    if not (h.map50_95<=h.map50+1e-7).all():raise Stop('mAP50-95 > mAP50')
    for role in ('train','val'):
        if not np.allclose(h[role+'_loss'],sum(h[role+'/'+k+'_loss'] for k in ('vfl','bbox','giou')),rtol=2e-6,atol=1e-7):raise Stop('Loss primaria contradictoria')
    return h


def compare_training_histories(raw,history):
    """Compare scalars numerically and the adapter's AP vector structurally.

    pandas reads ap_per_iou as text because the producer writes a list inside
    one cell. Never subtract those strings, drop the column, or rewrite a CSV.
    """
    import ast
    import numpy as np
    import pandas as pd
    if len(raw)!=len(history):raise Stop('Historiales de distinta longitud')
    if raw.columns.duplicated().any() or history.columns.duplicated().any():raise Stop('Columnas duplicadas en historial R50')
    provenance={'architecture','config_id','run_id','outer_split','validation_role'}
    columns=[c for c in raw if c in history and c not in provenance]
    # A vector published on only one side is missing evidence, not a number.
    if ('ap_per_iou' in raw)!=('ap_per_iou' in history):raise Stop('ap_per_iou ausente en uno de los historiales')
    scalar=[c for c in columns if c!='ap_per_iou']
    left=raw[scalar].copy();right=history[scalar].copy()
    for col in scalar:
        try:
            left[col]=pd.to_numeric(left[col],errors='raise')
            right[col]=pd.to_numeric(right[col],errors='raise')
        except (TypeError,ValueError) as exc:
            raise Stop('Columna escalar R50 no numerica: '+col) from exc
        if not np.isfinite(left[col]).all() or not np.isfinite(right[col]).all():
            raise Stop('Columna escalar R50 no finita: '+col)
    audit.compare_histories(left,right,scalar)
    vectors=0;unavailable=[]
    def vector(value,epoch):
        if value is None or (isinstance(value,(float,np.floating)) and np.isnan(value)):
            return None
        if isinstance(value,str):
            try:value=ast.literal_eval(value)
            except (ValueError,SyntaxError,TypeError,RecursionError) as exc:
                raise Stop('ap_per_iou no es una lista valida en epoca '+str(epoch)) from exc
        if value is None:return None
        if not isinstance(value,(list,tuple,np.ndarray)) or len(value)!=10:
            raise Stop('ap_per_iou requiere diez valores (IoU 0.50:0.95), epoca '+str(epoch))
        if any(isinstance(x,(bool,np.bool_)) or not isinstance(x,(int,float,np.integer,np.floating)) for x in value):
            raise Stop('ap_per_iou contiene valores no numericos, epoca '+str(epoch))
        result=np.asarray(value,dtype=float)
        if result.shape!=(10,) or not np.isfinite(result).all() or ((result<0)|(result>1)).any():
            raise Stop('ap_per_iou no finito/fuera de rango, epoca '+str(epoch))
        return result
    if 'ap_per_iou' in columns:
        for index,(a,b) in enumerate(zip(raw['ap_per_iou'],history['ap_per_iou'])):
            epoch=raw['epoch'].iloc[index] if 'epoch' in raw else index+1
            a=vector(a,epoch);b=vector(b,epoch)
            if a is None and b is None:
                unavailable.append(int(epoch));continue
            if a is None or b is None:raise Stop('ap_per_iou ausente solo en un registro, epoca '+str(epoch))
            if not np.allclose(a,b,rtol=2e-6,atol=1e-7,equal_nan=False):
                raise Stop('Historiales discrepan en ap_per_iou, epoca '+str(epoch))
            vectors+=1
    return dict(scalar_columns=scalar,rows=len(raw),ap_vectors_compared=vectors,
                ap_vector_unavailable_epochs=unavailable,originals_modified=False)


def verify_trials(study,cfg,spec):
    from utils.experiments import fingerprint
    from utils.search_spaces import config_id
    seed=int(cfg['reproducibility']['seed']);sec=cfg['detector'];verified=[]
    for t in study.get_trials(deepcopy=False):
        if t.state.name not in ('COMPLETE','PRUNED'):continue
        emit('VERIFICANDO_TRIAL_EXISTENTE',trial=t.number,state=t.state.name,training=False)
        folder=allowed_trial_folder(cfg,spec,t);hp=t.user_attrs.get('hp')
        if not hp or t.user_attrs.get('config_id')!=config_id(hp):raise Stop('HP/configuracion ausente o incompatible')
        if any(not math.isfinite(float(v)) or not 0<=float(v)<=1 for v in t.intermediate_values.values()):raise Stop('Intermedios no finitos/fuera de rango')
        if t.state.name=='PRUNED':
            if not t.intermediate_values or t.user_attrs.get('pruned_at_step') not in t.intermediate_values:raise Stop('PRUNED sin recibo de pruning')
            checkpoints=checkpoint_candidates(folder)
            if not checkpoints:raise Stop('PRUNED sin checkpoint propio: '+str(folder))
            for cp,_ in checkpoints:validate_checkpoint(cp,MODEL,cfg,seed=seed,hp=hp)
            verified.append(dict(number=t.number,state='PRUNED'));continue
        if t.value is None or not math.isfinite(float(t.value)) or not 0<=float(t.value)<=1:raise Stop('Objetivo COMPLETE invalido')
        scores=[];history_checks=[];metadata=read(folder/'checkpoint_metadata.json')
        for fold in range(3):
            out=folder/f'fold{fold}';sig=fold_signature(fingerprint(cfg),MODEL,hp,seed,fold,int(sec['max_epochs']),int(sec['early_stopping_patience']))
            receipt,h=verify_fold(out,sig,fold);history=inspect_curve(out/'completed_history.csv');scores.append(float(receipt['row']['score']))
            raw=inspect_curve(out/'results.csv');history_checks.append(compare_training_histories(raw,history))
            emit('HISTORIAL_RTDETR_VERIFICADO',trial=t.number,fold=fold,rows=len(raw),ap_vectors_compared=history_checks[-1]['ap_vectors_compared'])
            be=int(receipt['row']['best_epoch'])
            n=int(receipt['row']['epochs_run'])
            if n>int(sec['max_epochs']) or (n<int(sec['max_epochs']) and n-be!=int(sec['early_stopping_patience'])):raise Stop('Parada de fold HPO no corresponde al protocolo')
            selected=history.loc[history.epoch==be,'map50_95']
            if len(selected)!=1 or not math.isclose(float(selected.iloc[0]),scores[-1],abs_tol=1e-7) or not math.isclose(float(history.map50_95.max()),scores[-1],abs_tol=1e-7):raise Stop('Mejor metrica/epoca incompatible con recibo')
            best=out/'best.pt';validate_checkpoint(best,MODEL,cfg,seed=seed,hp=hp,fold=fold)
            records=[r for r in metadata.get('checkpoints',[]) if r.get('role')==f'fold{fold}_best']
            if len(records)!=1 or records[0].get('sha256')!=digest(best) or records[0].get('weights_discarded') is not False:raise Stop('Pesos entrenados ausentes/discrepan')
        if t.user_attrs.get('n_folds_done')!=3 or len(t.user_attrs.get('fold_scores',[]))!=3:raise Stop('COMPLETE sin tres folds documentados')
        if any(not math.isclose(a,float(b),abs_tol=1e-9) for a,b in zip(scores,t.user_attrs['fold_scores'])) or not math.isclose(sum(scores)/3,float(t.value),abs_tol=1e-9):raise Stop('Objetivo HPO no concuerda con folds')
        for cp,_ in checkpoint_candidates(folder):validate_checkpoint(cp,MODEL,cfg,seed=seed,hp=hp)
        verified.append(dict(number=t.number,state='COMPLETE',value=t.value,scores=scores,history_checks=history_checks))
    return verified


def cached_audit_compatible(prior,fp,config_sha,code,inventory):
    return (prior.get('decision') in ('A_APTO_PARA_CONTINUAR','B_APTO_CON_OBSERVACIONES')
        and prior.get('executor_sha256')==AUDIT_SHA and prior.get('fingerprint')==fp
        and prior.get('config_sha256')==config_sha
        and all(prior.get('code',{}).get(k)==code[k] for k in ('git_commit','code_hash'))
        and prior.get('baseline_inventory')==inventory and prior.get('baseline_unchanged') is True
        and prior.get('prepared_artifacts_unchanged') is True)


def run_initial_audit(project,store,config,cfg,out):
    original=audit.emit
    def progress(name,**fields):
        if name=='APROBACION_PENDIENTE':emit('AUTORIZACION_EXPLICITA_RECIBIDA',source='--aprobar-original-hpo',no_additional_prompt=True)
        else:original(name,**fields)
    audit.emit=progress
    try:return audit.perform_audit(project,store,config,cfg,out)
    finally:audit.emit=original


def fresh_critical_inputs(cfg,baseline,attempt):
    from utils.architectures import get_arch
    from utils.detector_data import detection_images_table
    emit('VERIFICANDO_DATOS_ACTUALES',qa_full_suite=False,training=False)
    scratch,data=audit.readonly_prepared(cfg,attempt/'prep-current')
    if data.dataset_hash!=audit.EXPECTED_DATA or data.split_hash!=audit.EXPECTED_SPLIT:raise Stop('Datos/splits actuales diferentes del baseline autorizado')
    derived=detection_images_table(scratch)
    inputs=audit.verify_original_lists(cfg,baseline,derived)
    isolation=audit.partition_evidence(scratch,get_arch(MODEL))
    if isolation['confirmed_relevant_conflicts']:raise Stop('Conflicto de origen/contenido confirmado; no se registra aprobacion')
    result=dict(utc=utc(),dataset_hash=data.dataset_hash,split_hash=data.split_hash,original_input_lists=inputs,isolation=isolation,training=False)
    atomic_json(attempt/'critical-inputs.json',result)
    return result


def obtain_audit(project,store,config,cfg,attempt):
    from utils.experiments import fingerprint,verify_round
    from utils.manual_gates import require_gate
    from utils.reproducibility import collect_environment
    from utils.run_common import check_environment_matches_reference
    from types import SimpleNamespace
    import logging,torch
    code=audit.readonly_code(project);verify_round(cfg,architecture=MODEL)
    if fingerprint(cfg)!=audit.EXPECTED_FP:raise Stop('Fingerprint diferente del autorizado')
    audit.validate_protocol(cfg);require_gate(cfg,PC,'READY_FOR_EXPERIMENT')
    env=collect_environment(cfg=cfg,project_root=project,pc_id=PC)
    check_environment_matches_reference(SimpleNamespace(cfg=cfg,env=env,logger=logging.getLogger('r50_hpo')),strict=True)
    if not torch.cuda.is_available() or torch.cuda.device_count()!=1:raise Stop('Se requiere una GPU CUDA; no se cae a CPU')
    baseline=store/'r/rtdetr_r50/baseline';inventory=audit.frozen_inventory(baseline)
    pointer=store/'audit/latest.json';report=None
    if pointer.is_file():
        latest=read(pointer);path=Path(latest.get('json',''))
        if path.is_file() and inside(path,store/'audit') and digest(path)==latest.get('sha256'):
            prior=read(path)
            if cached_audit_compatible(prior,fingerprint(cfg),digest(config),code,inventory):
                report=prior;verified_copy(path,attempt/'baseline-audit.json')
                emit('AUDITORIA_VALIDA_REUTILIZADA',source=str(path),baseline_retrained=False)
    if report is None:
        jobs=audit.read_jobs(cfg)
        if jobs['hpo']['status']=='RUNNING':raise Stop('Falta auditoria local valida para recuperar HPO RUNNING; se conserva su estado')
        out=store/'audit'/uuid.uuid4().hex[:8]
        if run_initial_audit(project,store,config,cfg,out):raise Stop('La auditoria local no habilita HPO; revisar '+str(out/'AUDITORIA.html'))
        report=read(out/'audit.json');verified_copy(out/'audit.json',attempt/'baseline-audit.json')
    # Verify source bytes again; this performs no acquisition/extraction.
    adapter=audit.load_adapter();audit.require_file(store/'official.zip');audit.require_file(store/'official/source.json');adapter.acquire_source(store,False)
    from utils.architectures import get_arch
    from utils.weights import verified_local_weight
    if digest(verified_local_weight(get_arch(MODEL),cfg))!=adapter.WEIGHT_SHA:raise Stop('Pesos COCO incompatibles')
    fresh_critical_inputs(cfg,baseline,attempt)
    return report,adapter,inventory


def approve_baseline(queue,baseline,report,attempt,baseline_dir,authorized):
    if baseline.status=='COMPLETE':
        with queue._conn() as con:
            reviews=con.execute("SELECT operator,evidence FROM reviews WHERE job_id=? AND decision='APPROVED'",(baseline.job_id,)).fetchall()
        if not reviews or not all(r['operator'] and r['evidence'] for r in reviews):raise Stop('Baseline COMPLETE sin aprobacion humana documentada')
        emit('APROBACION_PREVIA_CONSERVADA',job=baseline.job_id);return
    if baseline.status!='PENDING_REVIEW':raise Stop('Baseline no esta finalizado para aprobar: '+baseline.status)
    if not authorized:raise Stop('Requiere autorizacion explicita --aprobar-original-hpo')
    if report.get('decision') not in ('A_APTO_PARA_CONTINUAR','B_APTO_CON_OBSERVACIONES'):raise Stop('Auditoria no aprobable')
    if audit.frozen_inventory(baseline_dir)!=report['baseline_inventory']:raise Stop('Baseline cambio desde auditoria')
    evidence=attempt/'aprobacion.json'
    atomic_json(evidence,dict(decision='APROBAR',operator=os.environ.get('USERNAME',USER),utc=utc(),job_id=baseline.job_id,
        fingerprint=audit.EXPECTED_FP,executor_sha256=EXECUTOR_SHA256,scope='Solo baseline R50-vd -> HPO original 15x3/60epocas/paciencia20',
        authorization_source='Usuario solicito iniciar y aprobar HPO en chat; invocacion explicita --aprobar-original-hpo',
        human_visual_inspection_claimed=False,absence_of_overfitting_certified=False,final_validation=False,
        report=str(attempt/'baseline-audit.json'),report_sha256=digest(attempt/'baseline-audit.json'),critical_inputs_sha256=digest(attempt/'critical-inputs.json'),limitations=report.get('limitations',[])))
    queue.approve(baseline.job_id,operator=os.environ.get('USERNAME',USER),evidence=str(evidence))
    emit('APROBACION_HPO_REGISTRADA',evidence=str(evidence),no_overfit_certified=False)


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


def recover_owned_job(queue,job,state_dir):
    if job.status not in ('RUNNING','FAILED'):return
    path=Path(state_dir)/'owner.json'
    if not path.is_file():raise Stop('Job requiere propietario verificable; falta '+str(path))
    owner=read(path)
    if owner.get('job_id')!=job.job_id or owner.get('attempt')!=job.attempt or owner.get('executor_sha256') not in (EXECUTOR_SHA256,*LEGACY_EXECUTOR_SHAS):raise Stop('Propietario de otra ejecucion; no se modifica la cola')
    if owner_alive(owner['process']):raise Stop('HPO sigue activo; no se duplica')
    if job.attempt>=3:raise Stop('Limite nativo de intentos; revisar error conservado')
    queue.finish(job.job_id,'PENDING',error='Reanudar intento propio muerto demostrado; conserva Optuna y checkpoints originales')
    emit('HPO_REANUDABLE',previous=job.status,attempt=job.attempt)


def run_hpo(project,store,config,cfg,queue,job,adapter,state_dir,attempt):
    from utils.architectures import get_arch
    from utils.config_loader import cfg_path
    from utils.optuna_storage import study_db_path,load_study,budget_used
    spec=get_arch(MODEL);db=study_db_path(cfg,spec)
    hpo=cfg_path(cfg,'paths.runs_dir')/MODEL/'hpo'
    if not db.is_file() and (any(hpo.glob('trial_*/fold*/completed_fold.json')) or any(hpo.glob('trial_*/fold*/*.pt')) or any(hpo.glob('trial_*/fold*/cp/*.pt'))):raise Stop('Falta SQLite Optuna pero hay trabajo HPO previo; no se crea otro estudio ni se reinicia')
    if db.is_file():
        snapshot_sqlite(db,attempt/'study-before.sqlite')
        existing=load_study(spec,cfg);verify_trials(existing,cfg,spec)
        if budget_used(existing)>BUDGET:raise Stop('Presupuesto excedido')
    else:existing=None
    if job.status in ('PENDING_REVIEW','COMPLETE') and not job.payload.get('partial_budget'):
        if existing is None or budget_used(existing)!=BUDGET:raise Stop('Cola cerrada sin estudio completo verificable')
        emit('HPO_YA_TERMINADO_NO_SE_REPITE',best_trial=existing.best_trial.number,value=existing.best_value,status=job.status);return existing
    if job.status=='PUBLICATION_PENDING':
        if existing is None or budget_used(existing)!=BUDGET:raise Stop('Publicacion no corresponde a HPO completo')
        from orchestration.exchange import publish_architecture
        from orchestration.worker import top_config_ids
        publish_architecture(cfg,PC,MODEL,cfg_path(cfg,'paths.runs_dir'),stage='hpo',complete=True)
        queue.computed(job.job_id,published=True);queue.expand_after_hpo(cfg,spec,top_config_ids(cfg,MODEL))
        emit('HPO_PUBLICADO_SIN_REENTRENAR');return existing
    if job.status=='PENDING_REVIEW':raise Stop('Bloque HPO parcial necesita su propia revision, se conserva')
    if job.status not in ('PENDING','FAILED','RUNNING'):raise Stop('Estado HPO bloqueado: '+job.status)
    if existing is not None:preview_recovery(existing,cfg,spec,attempt)
    recover_owned_job(queue,job,state_dir);job=queue.get(job.job_id)
    # Existing originals live outside every new retry folder; native rolling
    # retention applies only to the newly generated/copy optimizer generations.
    remaining=BUDGET-(budget_used(existing) if existing is not None else 0)
    states=list((store/'r/rtdetr_r50/baseline').glob('fold*/cp/e*.pt'));bests=list((store/'r/rtdetr_r50/baseline').glob('fold*/best.pt'))
    estimate=int(remaining*3*(2*max((p.stat().st_size for p in states),default=1_000_000_000)+max((p.stat().st_size for p in bests),default=180_000_000))*1.15+5*1024**3)
    free=shutil.disk_usage(store).free
    emit('ESPACIO_HPO',available_GB=round(free/1024**3,1),estimated_maximum_GB=round(estimate/1024**3,1),retention='best + dos estados rolling por fold; originales anteriores conservados')
    if free<estimate:raise Stop('Espacio insuficiente para preservar resultados y checkpoints; no se inicia GPU')
    adapter.install_hooks(cfg)
    native=importlib.import_module('07_run_hpo_study');recovery=R50Recovery(cfg,spec,state_dir/'recovery');recovery.install(native,adapter)
    overrides=['paths.runs_dir='+str(store/'r'),'paths.results_dir='+str(store/'e')]
    argv=['--config',str(config),'--pc-id',PC,'--architecture',MODEL,'--trial-block-size',str(BUDGET)]
    for value in overrides:argv+=['--set',value]
    atomic_json(state_dir/'owner.json',dict(job_id=job.job_id,attempt=job.attempt+1,executor_sha256=EXECUTOR_SHA256,process=own_process()))
    emit('INICIANDO_HPO_RTDETR',remaining=remaining,trials=15,folds=3,max_epochs=60,patience=20,results=str(store/'r/rtdetr_r50/hpo'),baseline_retrained=False)
    rc=native.run_cli(native.SCRIPT,native.main,argv=argv,description=native.__doc__,add_arguments=native.add_arguments,run_layout=lambda a:(MODEL,'hpo/study',True))
    if rc:raise Stop('HPO detenido/interrumpido; trabajo preservado, codigo '+str(rc))
    study=load_study(spec,cfg);verify_trials(study,cfg,spec)
    if budget_used(study)!=BUDGET or not any(t.state.name=='COMPLETE' for t in study.trials):raise Stop('Estudio aun incompleto; no se declara terminado')
    return study


def main():
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--aprobar-original-hpo',action='store_true');args=parser.parse_args()
    if not args.aprobar_original_hpo:raise Stop('Para autorizar la transicion solicitada ejecutar con --aprobar-original-hpo')
    if os.name!='nt':raise Stop('Ejecutor operativo exclusivo de Windows PC-03; tests separados sin GPU')
    project=Path.home()/'Documents/s2_no_borrar/Entrenamiento/Botrytis_Experiment_Framework_git';py=project/'.venv/Scripts/python.exe'
    if Path(sys.executable).resolve()!=py.resolve():return subprocess.call([str(py),'-u',str(Path(__file__).resolve()),*sys.argv[1:]])
    os.chdir(project);sys.path.insert(0,str(project/'src'))
    os.environ.update(PYTHONUTF8='1',PYTHONIOENCODING='utf-8',MPLBACKEND='Agg',BOTRYTIS_BOOTSTRAP_DIR=str(Path.home()/'r50/log'),BOTRYTIS_PC_ID=PC,GIT_OPTIONAL_LOCKS='0',PYTHONDONTWRITEBYTECODE='1');sys.dont_write_bytecode=True
    if os.environ.get('BOTRYTIS_WORKER_JOB_ID'):raise Stop('Ejecutar fuera de otro worker')
    store,config,cfg=audit.resolve(project,Path.home())
    from utils.durable import FileLock
    from utils.guards import gpu_lock_path
    from orchestration.study_queue import StudyQueue
    from contextlib import ExitStack
    lock=gpu_lock_path(cfg,PC)
    with ExitStack() as stack:
        for suffix in ('.r50_hpo.guard','.r50.guard','.final.guard','.priority.guard'):stack.enter_context(FileLock(lock.with_suffix(suffix)))
        active=audit.processes()
        if active:raise Stop('Otro Python sigue activo; se conserva sin interrumpir: '+str(active))
        os.environ.setdefault('CUBLAS_WORKSPACE_CONFIG',':4096:8')
        state_dir=store/'hpo_control';attempt=state_dir/'a'/uuid.uuid4().hex[:8];attempt.mkdir(parents=True)
        emit('FASE_IDENTIFICACION_OK',pc=PC,architecture=MODEL,round='r50vd_01')
        baseline=store/'r/rtdetr_r50/baseline';before=audit.frozen_inventory(baseline)
        try:
            report,adapter,inventory=obtain_audit(project,store,config,cfg,attempt)
            check_git(project,{})  # refresh only index entries already proven content-identical
            jobs=audit.read_jobs(cfg);queue=StudyQueue.from_cfg(cfg)
            snapshot_sqlite(queue.path,attempt/'queue-before.sqlite')
            approve_baseline(queue,queue.get(jobs['baseline']['job_id']),report,attempt,baseline,args.aprobar_original_hpo)
            study=run_hpo(project,store,config,cfg,queue,queue.get(jobs['hpo']['job_id']),adapter,state_dir,attempt)
            state=queue.get(jobs['hpo']['job_id']).status
            summary=dict(status='HPO_TERMINADO',queue_status=state,trials_complete=sum(t.state.name=='COMPLETE' for t in study.trials),
                trials_pruned=sum(t.state.name=='PRUNED' for t in study.trials),trials_failed=sum(t.state.name=='FAIL' for t in study.trials),
                best_trial=study.best_trial.number,best_config=study.best_trial.user_attrs.get('config_id'),best_value=study.best_value,
                baseline_mean=report['tables']['score_mean'],baseline_sd=report['tables']['score_sd'],baseline_max_epochs=100,hpo_max_epochs=60,
                winner='NO_DECLARADO',overfitting='NO_DESCARTADO',robustness_started=False,independent_test_used=False,final_validation=False,
                baseline_unchanged=audit.frozen_inventory(baseline)==before,results=str(store/'r/rtdetr_r50/hpo'))
            if not summary['baseline_unchanged']:raise Stop('Baseline cambio durante HPO; no se valida automaticamente')
            atomic_json(attempt/'summary.json',summary);atomic_json(state_dir/'latest.json',dict(summary=str(attempt/'summary.json')))
            emit('HPO_RTDETR_TERMINADO',**summary);return 0
        except BaseException as e:
            atomic_json(attempt/'error.json',dict(error=str(e),traceback=traceback.format_exc(),baseline_unchanged=audit.frozen_inventory(baseline)==before,results_preserved=True))
            raise

if __name__=='__main__':
    try:sys.exit(main())
    except KeyboardInterrupt:emit('INTERRUMPIDO_RESULTADOS_CONSERVADOS');sys.exit(130)
    except BaseException as e:
        if isinstance(e,SystemExit):raise
        emit('ERROR_CRITICO',error=str(e),results_preserved=True);sys.exit(2)
