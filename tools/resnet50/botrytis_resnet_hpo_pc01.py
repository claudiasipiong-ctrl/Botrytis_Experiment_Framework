"""PC-01: revisar baseline ResNet50, pedir APROBAR y ejecutar SOLO HPO20x3.
Ejecutar fuera del checkout con su .venv. No invoca baseline/robustez/test.
La evidencia local vigente se comprueba antes de cualquier aprobacion o GPU.
"""
from __future__ import annotations
import argparse, copy, hashlib, html, importlib.util, json, logging, math, os
from pathlib import Path
import pickle, re, shutil, socket, sqlite3, subprocess, sys, tempfile, traceback, uuid
from datetime import datetime, timezone
VERSION='1.0.0'
MODEL='resnet50'; PC='PC-01'; USER='i21315'
EXECUTOR_SHA256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest()
class Stop(RuntimeError): pass
class RecoveryStop(BaseException): pass


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

def copy_tree_safe(src, dst, skip=()):
    src, dst = Path(src), Path(dst)
    for p in sorted(src.rglob('*')):
        rel = p.relative_to(src)
        if any(x in skip for x in rel.parts): continue
        if p.is_symlink(): raise Stop('Enlace inesperado en evidencias: ' + str(p))
        if p.is_file(): verified_copy(p, dst / rel)

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

def inspect_baseline(cfg, spec, run_dir, code, commit, experiment, allow_missing_summary=False):
    import pandas as pd
    from utils.architectures import section_for
    from utils.search_spaces import baseline_hyperparameters
    sec = section_for(spec, cfg); b = sec['baseline']; hp = baseline_hyperparameters(spec, cfg)
    seed, epochs = int(b['seed']), int(b['epochs']); patience = int(b['patience']) if spec.is_detector else int(sec['early_stopping_patience'])
    count = int(cfg['inner_hpo']['folds'])
    if count != 3: raise Stop('El protocolo solicitado requiere exactamente tres folds')
    run_dir = Path(run_dir); details = []; scores = []; training_seconds = 0
    actual_folders = {p.name for p in run_dir.glob('fold*') if p.is_dir() and re.fullmatch(r'fold\d+', p.name)}
    if actual_folders != {'fold0', 'fold1', 'fold2'}: raise Stop('Folds baseline ausentes/adicionales: ' + str(actual_folders))
    for fold in range(count):
        signature = fold_signature(experiment, spec.slug, hp, seed, fold, epochs, patience)
        receipt, h = verify_fold(run_dir / f'fold{fold}', signature, fold); row = receipt['row']; scores.append(float(row['score']))
        metric = 'metrics/mAP50-95(B)' if spec.is_detector else 'val_ap'
        if metric not in h or not math.isclose(float(h[metric].max()), scores[-1], abs_tol=1e-5): raise Stop(f'Fold {fold}: metrica objetiva no concuerda con el historial ({metric})')
        if spec.is_detector:
            raw = pd.read_csv(run_dir / f'fold{fold}' / 'results.csv'); raw.columns = raw.columns.str.strip()
            if metric not in raw or not math.isclose(float(raw[metric].max()), scores[-1], abs_tol=1e-5): raise Stop(f'Fold {fold}: CSV original discrepa del score')
            loss_columns = [c for c in h if c.startswith('train/') and c.endswith('_loss')]
        else: loss_columns = ['train_loss']
        if not loss_columns: raise Stop(f'Fold {fold}: faltan perdidas originales de entrenamiento')
        trends = {}
        for col in loss_columns:
            values = pd.to_numeric(h[col], errors='coerce')
            if len(values) < 2 or values.isna().any() or not values.map(math.isfinite).all(): raise Stop(f'Fold {fold}: perdida incompleta/no finita: {col}')
            window = min(10, max(1, len(values) // 3)); first = float(values.head(window).mean()); last = float(values.tail(window).mean())
            trends[col] = dict(first_window_mean=first, last_window_mean=last, window=window)
            if last >= first: raise Stop(f'Fold {fold}: convergencia requiere revision en {col}: {first} -> {last}')
        training_seconds += float(row.get('train_seconds', 0))
        details.append(dict(fold=fold, score=scores[-1], history_sha256=receipt['history_sha256'], epochs=len(h), losses=trends))
    summary_path = run_dir / 'summary.json'
    if not summary_path.is_file():
        if allow_missing_summary: return dict(folds=details, scores=scores, summary_missing=True, hp=hp)
        raise Stop('Falta el resumen baseline: ' + str(summary_path))
    summary = read(summary_path); metadata = read(run_dir / 'metadata.json')
    if metadata.get('status') != 'COMPLETE': raise Stop('Baseline sin cierre COMPLETE: ' + str(run_dir))
    if metadata.get('git_commit') != commit or metadata.get('code_hash') != code: raise Stop('Baseline incompatible con el codigo actual')
    if metadata.get('pc_id') != cfg['compute']['pc_id'] or summary.get('pc_id') != cfg['compute']['pc_id']: raise Stop('Baseline registrado para otra maquina')
    from utils.config_loader import load_config
    from utils.hashing import scientific_config_hash
    recorded_cfg = load_config(run_dir / 'config.yaml')
    if scientific_config_hash(recorded_cfg) != scientific_config_hash(cfg): raise Stop('Configuracion efectiva del baseline no concuerda con la ronda actual')

    objective = 'map50_95' if spec.is_detector else 'average_precision_positive'
    if summary.get('slug') != spec.slug or summary.get('folds') != [0, 1, 2] or summary.get('objective') != objective: raise Stop('Resumen de otra arquitectura, folds u objetivo')
    if summary.get('hp') != hp or int(summary.get('seed', -1)) != seed or summary.get('max_epochs') != epochs or summary.get('patience') != patience: raise Stop('Configuracion baseline no corresponde al protocolo actual')
    from utils.dataset import recorded_dataset_hash
    from utils.config_loader import cfg_path
    from utils.splits import current_split_hash
    art = cfg_path(cfg, 'paths.artifacts_dir')
    if summary.get('dataset_hash') != recorded_dataset_hash(art) or summary.get('split_hash') != current_split_hash(art, cfg['outer_split']['manifest_name']): raise Stop('Baseline de otros datos/split')
    if any(not math.isclose(float(a), float(b), abs_tol=1e-9) for a, b in zip(scores, summary.get('fold_scores', []))) or len(summary.get('fold_scores', [])) != 3: raise Stop('Scores resumen/recibos diferentes')
    mean = sum(scores) / 3; sd = math.sqrt(sum((v - mean)**2 for v in scores) / 2)
    if not math.isclose(float(summary['score_mean']), mean, abs_tol=1e-9) or not math.isclose(float(summary['score_sd']), sd, abs_tol=1e-8): raise Stop('Media/desviacion del resumen inconsistente')
    # A baseline may intentionally discard trained weights. Validate their recorded
    # hashes and every physical signed resume checkpoint that remains, without
    # pretending a deleted best.pt is still available.
    checkpoint_sources = [run_dir / 'checkpoint_metadata.json', *sorted((run_dir / 'previous_attempts').glob('*/checkpoint_metadata.json'))]
    sources = []
    for path in checkpoint_sources:
        if not path.is_file(): continue
        item = read(path)
        if item.get('git_commit') != commit or item.get('code_hash') != code: continue
        if item.get('dataset_hash') != summary['dataset_hash'] or item.get('split_hash') != summary['split_hash']: continue
        if path.parent != run_dir:
            valid = all((path.parent / f'fold{i}' / 'completed_history.csv').is_file() and
                        digest(path.parent / f'fold{i}' / 'completed_history.csv') == details[i]['history_sha256'] for i in range(3))
            if not valid: continue
        sources.append((path, item))
    for fold in range(3):
        entries = [c for _, item in sources for c in item.get('checkpoints', []) if c.get('role') == f'fold{fold}_best']
        shas = {c.get('sha256') for c in entries}
        for entry in entries:
            if entry.get('weights_discarded') is False:
                retained = entry.get('retained_path') or entry.get('path')
                if not retained or not Path(retained).is_file() or digest(retained) != entry.get('sha256'):
                    raise Stop(f'Fold {fold}: checkpoint declarado conservado ausente/alterado')
        if len(shas) != 1 or any(not re.fullmatch('[a-f0-9]{64}', str(v)) for v in shas):
            raise Stop(f'Fold {fold}: falta o discrepa el hash del checkpoint original (incluidos intentos conservados)')
    resumes = []
    for side in run_dir.glob('**/*.resume.json'):
        if 'previous_attempts' in side.relative_to(run_dir).parts: continue
        cp = Path(str(side)[:-len('.resume.json')])
        if cp.is_file(): validate_checkpoint(cp, spec.slug, cfg, seed=seed, hp=hp); resumes.append(str(cp))
    return dict(status='REVISION_AUTOMATICA_OK', model=spec.slug, objective=objective, scores=scores, mean=mean, sd=sd,
                folds=details, original_training_seconds=training_seconds, physical_resume_checkpoints=resumes,
                trained_best_weights='Hashes comprobados; las ausencias declaradas por el backend no se inventan ni se recuperan de pretrained',
                source_summary_sha256=digest(summary_path), human_visual_review=False)

def allowed_trial_folder(cfg, spec, trial):
    from utils.config_loader import cfg_path
    value = trial.user_attrs.get('run_dir')
    if not value: raise Stop(f'Trial {trial.number}: falta run_dir')
    folder = Path(value).resolve()
    roots = [cfg_path(cfg, 'paths.runs_dir'), cfg_path(cfg, 'paths.artifacts_dir').parent / 'runs']
    expected = [(Path(r) / spec.slug / 'hpo' / f'trial_{trial.number:03d}').resolve() for r in roots]
    if folder not in expected: raise Stop(f'Trial {trial.number}: carpeta fuera de las rutas aprobadas: {folder}')
    return folder

def verify_trial_evidence(study, cfg, spec):
    """Validate existing COMPLETE trials before counting them toward the budget."""
    from utils.experiments import fingerprint
    from utils.architectures import section_for
    sec = section_for(spec, cfg); exp = fingerprint(cfg); verified = []
    for trial in study.get_trials(deepcopy=False):
        if trial.state.name not in ('COMPLETE', 'PRUNED'): continue
        emit('VERIFICANDO_TRIAL_EXISTENTE', trial=trial.number, state=trial.state.name, training=False)
        if trial.state.name == 'PRUNED':
            if trial.user_attrs.get('pruned_at_step') is None and not trial.intermediate_values: raise Stop(f'Trial {trial.number}: PRUNED sin evidencia intermedia')
            if any(not math.isfinite(float(v)) or not 0<=float(v)<=1 for v in trial.intermediate_values.values()):raise Stop('PRUNED con objetivo intermedio no finito/fuera de rango')
            folder=allowed_trial_folder(cfg,spec,trial)
            for cp,_ in checkpoint_candidates(folder):validate_checkpoint(cp,spec.slug,cfg,seed=int(cfg['reproducibility']['seed']),hp=trial.user_attrs.get('hp'))
            verified.append(dict(number=trial.number, state='PRUNED')); continue
        if trial.value is None or not math.isfinite(float(trial.value)): raise Stop(f'Trial {trial.number}: objetivo COMPLETE invalido')
        hp, folder = trial.user_attrs.get('hp'), trial.user_attrs.get('run_dir')
        if not hp or not folder: raise Stop(f'Trial {trial.number}: faltan hp/run_dir originales')
        folder = allowed_trial_folder(cfg, spec, trial)
        scores = []
        for fold in range(3):
            signature = fold_signature(exp, spec.slug, hp, int(cfg['reproducibility']['seed']), fold, int(sec['max_epochs']), int(sec['early_stopping_patience']))
            receipt, _ = verify_fold(Path(folder) / f'fold{fold}', signature, fold); scores.append(float(receipt['row']['score']))
            _classification_curve_evidence(Path(folder)/f'fold{fold}/completed_history.csv',False,receipt['row']['best_epoch'])
        if not math.isclose(sum(scores) / 3, float(trial.value), abs_tol=1e-8): raise Stop(f'Trial {trial.number}: objetivo no concuerda con los tres folds')
        meta = read(Path(folder) / 'checkpoint_metadata.json')
        for fold in range(3):
            entries = [e for e in meta.get('checkpoints', []) if e.get('role') == f'fold{fold}_best']
            if not entries: raise Stop(f'Trial {trial.number}, fold {fold}: falta metadata del checkpoint')
            for entry in entries:
                if not re.fullmatch('[a-f0-9]{64}', str(entry.get('sha256',''))): raise Stop('Hash de checkpoint de trial invalido')
                if entry.get('weights_discarded') is False:
                    physical = entry.get('retained_path') or entry.get('path')
                    if not physical or not Path(physical).is_file() or digest(physical) != entry['sha256']: raise Stop('Checkpoint retenido ausente o alterado: ' + str(physical))
        for cp, _ in checkpoint_candidates(folder):
            validate_checkpoint(cp, spec.slug, cfg, seed=int(cfg['reproducibility']['seed']), hp=hp)
        verified.append(dict(number=trial.number, state='COMPLETE', value=trial.value, scores=scores))
    return verified

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
        if not folder.is_dir(): raise Stop(f'Trial {t.number}: run_dir original ausente: {folder}')
        valid_folds = []; signed_partial = []
        for fold in range(3):
            fdir = folder / f'fold{fold}'
            if (fdir / 'completed_fold.json').exists():
                sig = fold_signature(exp, spec.slug, hp, int(cfg['reproducibility']['seed']), fold, int(sec['max_epochs']), int(sec['early_stopping_patience']))
                verify_fold(fdir, sig, fold); valid_folds.append(fold)
        for cp, meta in checkpoint_candidates(folder):
            context = meta.get('extra', {}).get('context', {}); fold = context.get('fold')
            if spec.is_detector:
                match = re.search(r'^fold(\d+)$', cp.parent.parent.name)
                fold = int(match.group(1)) if match else fold
            if fold is None or fold in valid_folds: continue
            validate_checkpoint(cp, spec.slug, cfg, seed=int(cfg['reproducibility']['seed']), hp=hp, fold=fold)
            signed_partial.append(str(cp))
        if valid_folds or signed_partial:
            study.enqueue_trial(dict(t.params), user_attrs=dict(bt_retry_of=t.number, bt_recovery_source=str(folder.resolve()),
                                bt_recovered_folds=valid_folds, bt_executor_sha256=EXECUTOR_SHA256), skip_if_exists=False)
            actions.append(dict(action='RETRY_IDENTICAL_PARAMS', trial=t.number, valid_folds=valid_folds, signed_partial=signed_partial))
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

def metric_number(row, keys):
    for k in keys:
        value = row.get(k)
        if value is not None and math.isfinite(float(value)): return float(value)
    return None

def _classification_curve_evidence(path, detector, selected_epoch=None):
    """Describe trends; a small late rise is a flag, never a fabricated failed QA."""
    import numpy as np
    import pandas as pd
    path=Path(path);signed_history_sha256=digest(path)
    # Ultralytics results.csv exposes validation component losses that may not
    # be present in the callback's signed completed_history.csv.
    if detector and path.name=='completed_history.csv' and path.with_name('results.csv').is_file():path=path.with_name('results.csv')
    h = pd.read_csv(path); h.columns = h.columns.str.strip()
    if h.empty or 'epoch' not in h: raise Stop('Historial vacio/sin epocas: ' + str(path))
    epochs = pd.to_numeric(h.epoch, errors='raise')
    if epochs.duplicated().any() or not epochs.is_monotonic_increasing: raise Stop('Epocas duplicadas/desordenadas: ' + str(path))
    objective = next((c for c in (['metrics/mAP50-95(B)', 'map50_95'] if detector else ['val_ap']) if c in h and h[c].notna().any()), None)
    result = dict(path=str(path), sha256=digest(path), signed_history_sha256=signed_history_sha256,epochs=len(h), losses={}, flags=[], train_validation_metric_gap=None,
                  gap_reason='No se registra mAP/AP sobre train con la misma evaluacion por epoca; no se infiere de las losses.')
    loss_cols = [c for c in h if (c.startswith(('train/', 'val/')) and c.endswith('_loss')) or c in ('train_loss','val_loss')]
    for col in loss_cols:
        v = pd.to_numeric(h[col], errors='coerce')
        if not v.notna().any(): continue  # empty aggregate is not a missing component loss
        if v.isna().any() or not np.isfinite(v).all(): raise Stop('Loss no finita/incompleta: ' + str(path) + ' ' + col)
        n = min(5, max(1, len(v)//3)); end = float(v.tail(n).mean()); previous = float(v.iloc[-2*n:-n].mean()) if len(v)>=2*n else end
        result['losses'][col] = dict(first_window=float(v.head(n).mean()), previous_window=previous, final_window=end,
                                    window=n, late_direction='up' if end>previous else 'down_or_equal')
    for train in [c for c in result['losses'] if c.startswith('train/') or c=='train_loss']:
        val = 'val/'+train.split('/',1)[1] if '/' in train else 'val_loss'
        if val in result['losses'] and result['losses'][train]['final_window'] < result['losses'][train]['previous_window'] and result['losses'][val]['final_window'] > result['losses'][val]['previous_window']:
            result['flags'].append(dict(kind='POSSIBLE_LATE_OVERFIT', train_component=train, validation_component=val,
                                        conclusion='Senal orientativa; augmentation/calendario y ruido tambien pueden influir.'))
    if objective:
        values = pd.to_numeric(h[objective], errors='coerce')
        if values.isna().any() or not np.isfinite(values).all(): raise Stop('Objetivo por epoca no finito: ' + str(path))
        maximum_index=values.idxmax();bi=maximum_index
        if selected_epoch is not None:
            candidates=h.index[h.epoch==selected_epoch]
            if len(candidates)!=1:raise Stop('Epoca del checkpoint ausente de la curva: '+str(path))
            bi=candidates[0]
            if not math.isclose(float(values.loc[bi]),float(values.max()),rel_tol=0,abs_tol=1e-5):raise Stop('Epoca seleccionada incompatible con maximo registrado')
        best = h.loc[bi]
        result.update(objective=objective, best_epoch=int(best.epoch), best_value=float(values.max()), last_value=float(values.iloc[-1]),
                      maximum_metric_epoch=int(h.loc[maximum_index,'epoch']),selected_checkpoint_epoch=selected_epoch,
                      drop_best_to_last=float(values.max()-values.iloc[-1]),
                      metrics_at_same_best_epoch=dict(map50=metric_number(best,['metrics/mAP50(B)','map50']),
                          map50_95=metric_number(best,['metrics/mAP50-95(B)','map50_95']),
                          precision=metric_number(best,['metrics/precision(B)','precision','val_precision']),
                          recall=metric_number(best,['metrics/recall(B)','recall','val_recall']),
                          f1=metric_number(best,['val_f1','f1']),roc_auc=metric_number(best,['val_roc_auc','roc_auc']),
                          accuracy=metric_number(best,['val_accuracy','accuracy']),
                          average_precision_positive=metric_number(best,['val_ap'])))
    return result

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

class RecoveryHooks:
    """Recovery-only instrumentation around the repository's native objective.

    Sampler, pruner, search spaces, budgets, seed, model functions and metrics stay
    native. We retain checkpoints, restore signed partials and reuse whole folds.
    """

    def __init__(self, cfg, spec, state_dir):
        self.cfg, self.spec, self.state_dir = (cfg, spec, Path(state_dir))
        self.state_dir.mkdir(parents=True, exist_ok=True)
        from utils.experiments import fingerprint
        self.exp = fingerprint(cfg)
        self.state = OptimizerState(self.state_dir / 'sampler', self.exp)
        self.study = None
        self.current = None
        self.retained = {}

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

    def evaluate(self,original,spec,hp,cfg,ctx,run_dir,**options):
        if spec.slug!=MODEL or options.get('trial') is None:raise RecoveryStop('Solo HPO ResNet50')
        from utils.experiments import fingerprint
        trial=options['trial'];run_dir=Path(run_dir);source=None;reused=[]
        source_name=trial.user_attrs.get('bt_recovery_source')
        if source_name:
            previous=next((t for t in self.study.trials if t.number==trial.user_attrs.get('bt_retry_of')),None)
            if previous is None or previous.user_attrs.get('hp')!=dict(hp) or previous.user_attrs.get('run_dir')!=source_name:
                raise RecoveryStop('Fuente de retry ajena al estudio/hp')
            source=allowed_trial_folder(cfg,spec,previous)
            for fold in options['folds']:
                f=source/f'fold{fold}'
                if not (f/'completed_fold.json').exists():continue
                signature=fold_signature(fingerprint(cfg),MODEL,hp,options['seed'],fold,options['max_epochs'],options['patience'])
                receipt,h=verify_fold(f,signature,fold)
                copy_tree_safe(f,run_dir/f'fold{fold}')
                meta=read(source/'checkpoint_metadata.json')
                entries=[e for e in meta.get('checkpoints',[]) if e.get('role')==f'fold{fold}_best']
                if len(entries)!=1:raise RecoveryStop('Metadata del fold previo ausente/ambigua')
                reused.append((fold,entries[0]))
                for _,row in h.iterrows():trial.report(float(row['val_ap']),fold*options['max_epochs']+int(row['epoch'])-1)
                self.event('FOLD_REUSED',trial=trial.number,source=str(f),fold=fold)
        self.event('TRIAL_EN_EJECUCION',trial=trial.number,config_id=trial.user_attrs.get('config_id'),hp=dict(hp),folds=options['folds'])
        self.context=dict(run_dir=run_dir,source=source,options=options)
        self.state.save(self.study)  # sampled TPE state, before GPU work
        record=ctx.record_checkpoint
        def checkpoint(role,**info):
            match=re.fullmatch(r'fold(\d+)_best',role)
            if match:
                path=run_dir/f'fold{match.group(1)}'/ 'best.pt'
                if path.is_file():
                    from utils.resume import require_checkpoint
                    require_checkpoint(path,spec,cfg)
                    import torch
                    saved=torch.load(path,map_location='cpu',weights_only=False)
                    expected_state=info.get('state_dict_sha256') or info.get('sha256')
                    if saved.get('state_dict_sha256')!=expected_state:raise RecoveryStop('El best conservado discrepa del estado original')
                    info.update(state_dict_sha256=expected_state,sha256=digest(path),weights_discarded=False,retained_path=str(path))
            return record(role,**info)
        ctx.record_checkpoint=checkpoint
        for fold,entry in reused:
            info={k:v for k,v in entry.items() if k!='role'}
            checkpoint(f'fold{fold}_best',**info)
        try:
            result=original(spec,hp,cfg,ctx,run_dir,**options)
            ctx.add_metadata(recovered_from=str(source) if source else None,executor_sha256=EXECUTOR_SHA256)
            return result
        finally:ctx.record_checkpoint=record;self.context=None

    def classifier(self, original, spec, hp, cfg, train_df, val_df, **kwargs):
        if spec.slug != MODEL or self.context is None:
            raise RecoveryStop('Hook limitado al HPO ResNet50')
        runtime=copy.deepcopy(cfg); fold=runtime['classifier']['_resume_context']['fold']
        destination=self.context['run_dir']/f'fold{fold}'
        source=self.context.get('source')
        checkpoints=[]
        if source:
            for cp,meta in checkpoint_candidates(source):
                if meta.get('extra',{}).get('context',{}).get('fold')==fold:
                    validate_checkpoint(cp,MODEL,runtime,seed=kwargs['seed'],hp=hp,fold=fold)
                    checkpoints.append((int(meta['extra'].get('epoch',0)),cp))
        if checkpoints:
            checkpoints.sort(key=lambda x:(x[0],str(x[1])))
            cp=checkpoints[-1][1]
            saved=load_saved_fold(cp,spec,hp,runtime,train_df,val_df,kwargs)
            if saved['epoch']>=kwargs['max_epochs'] or saved['bad_epochs']>=kwargs['patience']:
                callback=kwargs.get('epoch_callback')
                if callback:callback(kwargs.get('step_offset',0)+saved['epoch']-1,saved['history'][-1])
                run=finished_classifier(saved,spec,runtime,dict(kwargs,hp=dict(hp)))
                self.event('FOLD_FINISHED_CHECKPOINT_REUSED',fold=fold,checkpoint=str(cp))
                write_best(run,destination,spec,hp,runtime,kwargs['seed'],fold)
                return run
            runtime['classifier']['resume_checkpoint']=str(cp)
            self.event('CHECKPOINT_RESUME',fold=fold,epoch=saved['epoch']+1,checkpoint=str(cp))
        kwargs['keep_state']=True
        from botrytis_robustez_base import classifier_rng_restore
        with classifier_rng_restore(self.event):
            run=original(spec,hp,runtime,train_df,val_df,**kwargs)
        write_best(run,destination,spec,hp,runtime,kwargs['seed'],fold)
        return run

    def install(self,module):
        import utils.hpo_runner as runner
        original_evaluate=runner.evaluate_config;original_classifier=runner.train_classifier
        original_create=module.create_or_load_study
        module.create_or_load_study=lambda spec,cfg,seed:self.create_study(original_create,spec,cfg,seed)
        runner.evaluate_config=lambda *a,**kw:self.evaluate(original_evaluate,*a,**kw)
        runner.train_classifier=lambda *a,**kw:self.classifier(original_classifier,*a,**kw)


def load_saved_fold(cp,spec,hp,cfg,train_df,val_df,options):
    import torch
    from utils.hashing import sha256_json
    from utils.semantics import semantic_fingerprint
    validate_checkpoint(cp,spec.slug,cfg,seed=options['seed'],hp=hp,fold=cfg['classifier']['_resume_context']['fold'])
    saved=torch.load(cp,map_location='cpu',weights_only=False)
    # The approved native trainer has OOM fallback disabled; do not change batch.
    identity=sha256_json(dict(hp=dict(hp),seed=options['seed'],train=train_df.to_dict('records'),val=val_df.to_dict('records'),
        semantics=semantic_fingerprint(cfg),epochs=options['max_epochs'],micro_batch=int(cfg['classifier']['batch_size'])))
    if saved.get('identity')!=identity:raise RecoveryStop('Checkpoint de otro fold/hp/seed/batch/datos')
    needed={'model','optimizer','scaler','history','best','bad_epochs','step','epoch','python_rng','numpy_rng','torch_rng','cuda_rng','loader_rng'}
    if needed-set(saved):raise RecoveryStop('Checkpoint incompleto: '+str(needed-set(saved)))
    h=saved['history']
    if not h or [r['epoch'] for r in h]!=list(range(1,int(saved['epoch'])+1)):
        raise RecoveryStop('Checkpoint sin historial contiguo')
    for row in h:
        for key in ('train_loss','val_loss','val_ap'):
            if row.get(key) is None or not math.isfinite(float(row[key])):raise RecoveryStop('Checkpoint con metrica/loss invalida: '+key)
    if saved['best'].get('state') is None:raise RecoveryStop('Falta el estado del mejor modelo')
    return saved


def finished_classifier(saved,spec,cfg,options):
    import numpy as np
    import pandas as pd
    from utils.classifier_train import ClassifierRun
    from utils.metrics import binary_classification_report
    from utils.training_common import state_dict_sha256
    # Same constructor/metrics as native _train_once after the epoch loop.
    from utils.classifier_models import build_classifier
    from utils.model_stats import classifier_stats
    model,norm=build_classifier(spec,options['hp'],pretrained=False,cfg=cfg)
    model.load_state_dict(saved['best']['state'],strict=True)
    stats=classifier_stats(model,int(cfg['classifier']['image_size']))
    del model
    best=saved['best'];rep=binary_classification_report(best['true'],best['prob'],threshold=.5)
    if rep.get('average_precision') is None:raise RecoveryStop('AP no definido al recuperar fold finalizado')
    return ClassifierRun(score=float(rep['average_precision']),best_epoch=int(best['epoch']),epochs_run=len(saved['history']),
        stopped_early=int(saved['epoch'])<options['max_epochs'],history=pd.DataFrame(saved['history']),
        val_true=np.asarray(best['true']),val_prob=np.asarray(best['prob']),val_index=np.asarray(best['idx']),metrics=rep,
        train_seconds=sum(float(r.get('epoch_seconds',0)) for r in saved['history']),vram_peak_mb=None,
        micro_batch=int(cfg['classifier']['batch_size']),accum_steps=1,params_m=stats['params_m'],
        gflops=stats['gflops'],state_sha256=state_dict_sha256(best['state']),state=best['state'],norm={k:list(v) for k,v in norm.items()},ap_undefined=False)


def write_best(run,folder,spec,hp,cfg,seed,fold):
    from utils.durable import atomic_torch_save
    from utils.resume import receipt
    folder=Path(folder);folder.mkdir(parents=True,exist_ok=True);path=folder/'best.pt'
    if not run.state:raise RecoveryStop('No se conserva un checkpoint vacio')
    if path.exists():raise RecoveryStop('Se conserva best.pt existente; no sobrescribir: '+str(path))
    atomic_torch_save(dict(model=run.state,norm=run.norm,architecture=spec.slug,hp=dict(hp),seed=seed,fold=fold,
        epoch=run.best_epoch,state_dict_sha256=run.state_sha256),path)
    receipt(path,spec,cfg,dict(seed=seed,hp=dict(hp),epoch=run.best_epoch,context=dict(fold=fold),role='best_not_optimizer'))


def resolve_existing(project,home):
    from utils.config_loader import load_config
    from utils.experiments import verify_round,fingerprint
    index=read(home/'bt-hoy-20261103/rondas.json')
    if MODEL not in index:raise Stop('No existe ResNet50 en rondas.json; no se crea otra ronda')
    config=Path(index[MODEL]).resolve()
    if not config.is_file():raise Stop('Configuracion registrada ausente: '+str(config))
    original=load_config(config)
    if original.get('experiment',{}).get('architecture')!=MODEL:raise Stop('Ronda de otra arquitectura')
    key=hashlib.sha256(str(config).casefold().encode()).hexdigest()[:8]
    short=home/'bf'/key
    sets=['paths.runs_dir='+str(short/'r'),'paths.results_dir='+str(short/'e')]
    cfg=load_config(config,sets);manifest=verify_round(cfg,architecture=MODEL)
    if digest(config)!=manifest.get('config_sha256'):raise Stop('Config inmutable alterada')
    if fingerprint(original)!=fingerprint(cfg):raise Stop('Las rutas de salida cambiaron la identidad experimental')
    # The baseline must already exist at the very same root used by prioridad.
    if not (short/'r/resnet50/baseline/summary.json').is_file():
        raise Stop('Baseline ausente en su ruta original; no se inicia entrenamiento: '+str(short))
    return config,cfg,sets,short


def queue_read(cfg):
    from utils.config_loader import cfg_path
    db=cfg_path(cfg,'paths.queue_dir')/'queue.sqlite'
    if not db.is_file():raise Stop('Cola existente ausente; no se crea una nueva')
    with sqlite3.connect(db.resolve().as_uri()+'?mode=ro',uri=True) as c:
        c.row_factory=sqlite3.Row
        if c.execute('PRAGMA integrity_check').fetchone()[0]!='ok':raise Stop('SQLite de cola no integro')
        rows=[dict(r) for r in c.execute('SELECT * FROM jobs')]
    if not rows or any(r['architecture']!=MODEL for r in rows):raise Stop('Cola de otra arquitectura o compartida inesperadamente')
    selected={}
    for kind in ('baseline','hpo'):
        found=[r for r in rows if r['kind']==kind]
        if len(found)!=1:raise Stop('Job ausente/ambiguo: '+kind)
        selected[kind]=found[0]
    if json.loads(selected['hpo']['depends_on'])!=[selected['baseline']['job_id']]:raise Stop('Dependencia HPO-baseline distinta del plan')
    return db,selected


def preflight(project,cfg,audit):
    from utils.experiments import verify_round,fingerprint
    from utils.manual_gates import require_gate,ledger_path
    from utils.prepared import load_prepared_data
    from utils.isolation import require_near_duplicate_review
    from utils.weights import verified_local_weight
    from utils.classifier_data import load_crop_manifest
    from utils.classifier_train import load_fold_frames
    from utils.architectures import get_arch,trials_for
    from utils.guards import assert_equal_hpo_budget,assert_same_pretrained_policy,assert_image_size_fixed,assert_not_frozen
    from utils.reproducibility import collect_environment
    from utils.run_common import check_environment_matches_reference
    from types import SimpleNamespace
    import torch
    commit,code=check_git(project,audit);verify_round(cfg,architecture=MODEL)
    assert_equal_hpo_budget(cfg);assert_same_pretrained_policy(cfg);assert_image_size_fixed(cfg);assert_not_frozen(cfg,'HPO')
    if cfg['project']['profile']!='official' or cfg['experiment']['mode']!='supervised':raise Stop('Se requiere ronda oficial supervisada')
    spec=get_arch(MODEL)
    if int(cfg['inner_hpo']['folds'])!=3 or trials_for(spec,cfg)!=20:raise Stop('Presupuesto ResNet50 debe ser 20 trials x 3 folds')
    if cfg['classifier'].get('oom_fallback'):raise Stop('El protocolo aprobado desactiva cambios automaticos de batch')
    require_gate(cfg,PC,'READY_FOR_EXPERIMENT')
    data=load_prepared_data(cfg,with_folds=True);require_near_duplicate_review(cfg,data)
    crops=load_crop_manifest(cfg)
    for fold in range(3):
        tr,va=load_fold_frames(crops,cfg,fold)
        if tr.empty or va.empty or set(va.y.unique())!={0,1} or set(tr.image_id)&set(va.image_id):raise Stop('Fold sin ambas clases o con fuga de imagenes')
    weight=verified_local_weight(spec,cfg)
    if not torch.cuda.is_available() or torch.cuda.device_count()!=1:raise Stop('GPU CUDA unica no disponible; no se cae a CPU')
    environment=collect_environment(cfg=cfg,project_root=project,pc_id=PC)
    check_environment_matches_reference(SimpleNamespace(cfg=cfg,env=environment,logger=logging.getLogger('resnet')),strict=True)
    audit.update(git_commit=commit,code_hash=code,dataset_hash=data.dataset_hash,split_hash=data.split_hash,
        experiment_fingerprint=fingerprint(cfg),seed=cfg['reproducibility']['seed'],pretrained_sha256=digest(weight),
        qa_reused=dict(ledger=str(ledger_path(cfg,PC)),sha256=digest(ledger_path(cfg,PC)),gate='READY_FOR_EXPERIMENT',full_suite_reexecuted=False))
    return spec,commit,code,fingerprint(cfg)


def write_report(folder,audit):
    folder=Path(folder);atomic_json(folder/'revision.json',audit)
    base=audit.get('baseline',{});rows=''.join('<tr><td>'+str(f['fold'])+'</td><td>'+f"{f['score']:.8f}"+'</td><td>'+str(f['epochs'])+'</td></tr>' for f in base.get('folds',[]))
    page='<!doctype html><html lang="es"><meta charset="utf-8"><title>ResNet50 PC-01</title><style>body{max-width:1100px;margin:30px auto;font:16px/1.5 system-ui;padding:15px}td,th{padding:8px;border:1px solid #aaa}pre{white-space:pre-wrap;overflow-wrap:anywhere}</style><h1>Revision de baseline e HPO ResNet50</h1><p>'+html.escape(audit.get('status',''))+'</p><table><tr><th>Fold</th><th>AP positiva</th><th>Epocas</th></tr>'+rows+'</table><p>AP de clasificacion, no mAP de deteccion. Las curvas de train usan augmentation; las de validation no. La diferencia de losses no equivale a una brecha AP train-validacion.</p><p>Sin una evaluacion del mismo checkpoint sobre train no se inventa esa brecha. No se usa test para HPO. Senales de sobreajuste no prueban fuga; ausencia de coincidencias tampoco prueba independencia biologica.</p><pre>'+html.escape(json.dumps(audit,ensure_ascii=False,indent=2,default=str))+'</pre></html>'
    figures=''.join('<p><img style="max-width:100%" src="'+html.escape(Path(p).resolve().as_uri(),quote=True)+'"></p>' for p in audit.get('baseline_figures',[]))
    page=page.replace('<pre>',figures+'<pre>',1)
    atomic_bytes(folder/'REVISION.html',page.encode())


def human_approve(queue,baseline,folder,audit,input_fn=input):
    if baseline.status=='COMPLETE':
        with queue._conn() as c:reviews=c.execute("SELECT operator,evidence FROM reviews WHERE job_id=? AND decision='APPROVED'",(baseline.job_id,)).fetchall()
        if not reviews:raise Stop('Baseline COMPLETE sin registro de revision humana')
        emit('APROBACION_PREVIA_CONSERVADA',job=baseline.job_id);return
    if baseline.status!='PENDING_REVIEW':raise Stop('Baseline requiere recuperacion antes de revision: '+baseline.status)
    if not sys.stdin.isatty():raise Stop('La aprobacion debe escribirse en la consola interactiva de PC-01')
    report=Path(folder)/'REVISION.html'
    emit('REVISION_AUTOMATICA_OK',mean=audit['baseline']['mean'],sd=audit['baseline']['sd'],fold_range=audit['fold_range'],report=report,
        curves=audit['curves'],isolation=audit['isolation']['status'])
    print('Revise el informe y las limitaciones mostradas. APROBAR autoriza SOLO HPO20x3; no certifica independencia biologica ni valida test.',flush=True)
    answer=input_fn('Decision [APROBAR para iniciar HPO; otra respuesta conserva el estado]: ').strip()
    if answer!='APROBAR':raise Stop('Sin aprobacion. Baseline, HPO y resultados conservados')
    if baseline_evidence_manifest(Path(audit['baseline_dir']))!=audit['baseline_evidence']:
        raise Stop('Las evidencias del baseline cambiaron mientras se revisaba el informe; no se registra aprobacion')
    acceptance=Path(folder)/'aprobacion.json'
    atomic_json(acceptance,dict(decision='APROBAR',operator=os.environ.get('USERNAME',USER),utc=utc(),job_id=baseline.job_id,
        report=str(report),report_sha256=digest(report),evidence_sha256=digest(Path(folder)/'revision.json'),
        fingerprint=audit['experiment_fingerprint'],scope='Solo baseline ResNet50 -> HPO20x3',executor_sha256=EXECUTOR_SHA256))
    queue.approve(baseline.job_id,operator=os.environ.get('USERNAME',USER),evidence=str(acceptance))
    emit('APROBACION_MANUAL_REGISTRADA',evidence=acceptance)


def validate_jobs_identity(jobs,exp,cfg):
    for kind,row in jobs.items():
        payload=json.loads(row['payload'])
        if payload.get('fingerprint')!=exp or payload.get('round_id')!=cfg['experiment']['round_id']:
            raise Stop('Job de otra identidad experimental: '+kind)


def recover_hpo_job(queue,job,folder):
    if job.status not in ('RUNNING','FAILED'):return
    owner_file=Path(folder)/'owner.json'
    if not owner_file.is_file():raise Stop('Job requiere recuperacion con propietario verificable; falta '+str(owner_file))
    owner=read(owner_file)
    if owner.get('job_id')!=job.job_id or owner.get('attempt')!=job.attempt or owner.get('executor_sha256')!=EXECUTOR_SHA256:
        raise Stop('Propietario de otra ejecucion/version; no se modifica la cola')
    if owner_alive(owner['process']):raise Stop('HPO todavia activo; no se duplica')
    if job.attempt>=3:raise Stop('Limite de intentos alcanzado; revisar el error guardado')
    queue.finish(job.job_id,'PENDING',error='Reanudar intento propio: proceso muerto demostrado, evidencia y SQLite conservados')
    emit('HPO_RECUPERADO',job=job.job_id,attempt=job.attempt,baseline_reexecuted=False)


def run_hpo(project,config,cfg,sets,short,queue,job,folder):
    from utils.architectures import get_arch
    from utils.optuna_storage import load_study,budget_used,study_db_path
    from utils.config_loader import cfg_path
    spec=get_arch(MODEL)
    if job.status in ('COMPLETE','PENDING_REVIEW') and not job.payload.get('partial_budget'):
        study=load_study(spec,cfg);verified=verify_trial_evidence(study,cfg,spec)
        if budget_used(study)!=20:raise Stop('Cola cerrada pero presupuesto incompleto')
        atomic_json(Path(folder)/'hpo-final.json',dict(budget=20,trials=verified,best_number=study.best_trial.number,best_value=study.best_value,status=job.status,robustness_started=False))
        emit('HPO_YA_TERMINADO_NO_SE_REPITE',status=job.status,report=short/'r/resnet50/hpo/study_summary.json');return 0
    if job.status=='PUBLICATION_PENDING':
        study=load_study(spec,cfg);verify_trial_evidence(study,cfg,spec)
        if budget_used(study)!=20:raise Stop('Publicacion parcial requiere revisar el bloque, no se aprueba automaticamente')
        from orchestration.exchange import publish_architecture
        publish_architecture(cfg,PC,MODEL,cfg_path(cfg,'paths.runs_dir'),stage='hpo',complete=True)
        queue.computed(job.job_id,published=True)
        from orchestration.worker import top_config_ids
        queue.expand_after_hpo(cfg,spec,top_config_ids(cfg,MODEL))
        atomic_json(Path(folder)/'hpo-final.json',dict(budget=20,trials=verify_trial_evidence(study,cfg,spec),best_number=study.best_trial.number,best_value=study.best_value,status=queue.get(job.job_id).status,robustness_started=False))
        emit('HPO_PUBLICADO_SIN_REENTRENAR',status=queue.get(job.job_id).status);return 0
    if job.status=='PENDING_REVIEW':raise Stop('Bloque HPO parcial requiere revision propia; no se aprueba como baseline')
    if job.status not in ('PENDING','FAILED','RUNNING'):raise Stop('HPO bloqueado; no se elude '+job.status)
    recover_hpo_job(queue,job,folder);job=queue.get(job.job_id)
    db=study_db_path(cfg,spec)
    if db.is_file():
        study=load_study(spec,cfg);verify_trial_evidence(study,cfg,spec)
        if budget_used(study)>20:raise Stop('Estudio excede presupuesto; no se altera')
        snapshot_sqlite(db,Path(folder)/('study-before-'+uuid.uuid4().hex[:8]+'.sqlite'))
    snapshot_sqlite(queue.path,Path(folder)/('queue-before-'+uuid.uuid4().hex[:8]+'.sqlite'))
    hooks=RecoveryHooks(cfg,spec,Path(folder)/'recovery')
    native=importlib.import_module('07_run_hpo_study');hooks.install(native)
    native_args=['--config',str(config),'--pc-id',PC,'--architecture',MODEL,'--trial-block-size','20']
    for value in sets:native_args.extend(['--set',value])
    atomic_json(Path(folder)/'owner.json',dict(job_id=job.job_id,attempt=job.attempt+1,executor_sha256=EXECUTOR_SHA256,process=own_process()))
    emit('INICIANDO_HPO_RESNET50',trials=20,folds=3,max_epochs=cfg['classifier']['max_epochs'],results=short/'r/resnet50/hpo',baseline='Se conserva; no se invoca 06',robustness='No se ejecuta')
    rc=native.run_cli(native.SCRIPT,native.main,argv=native_args,description=native.__doc__,add_arguments=native.add_arguments,
        run_layout=lambda a:(MODEL,'hpo/study',True))
    if rc:raise Stop('HPO interrumpido/fallido; se conservan estudio y checkpoints. Codigo '+str(rc))
    study=load_study(spec,cfg);verified=verify_trial_evidence(study,cfg,spec)
    if budget_used(study)!=20:raise Stop('HPO incompleto: '+str(budget_used(study))+'/20')
    atomic_json(Path(folder)/'hpo-final.json',dict(budget=20,trials=verified,best_number=study.best_trial.number,best_value=study.best_value,
        status=queue.get(job.job_id).status,robustness_started=False))
    emit('HPO_RESNET50_TERMINADO',used=budget_used(study),best_trial=study.best_trial.number,best_AP=study.best_value,
        status=queue.get(job.job_id).status,report=Path(folder)/'hpo-final.json',robustness='Pendiente; no iniciada')
    return 0


def main():
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--expected-pc',default=PC,choices=[PC]);parser.parse_args()
    if os.name!='nt':raise Stop('Ejecutor operativo para Windows PC-01; pruebas separadas en CPU')
    project=Path.home()/'Documents/s2_no_borrar/Entrenamiento/Botrytis_Experiment_Framework_git';py=project/'.venv/Scripts/python.exe'
    if Path(sys.executable).resolve()!=py.resolve():return subprocess.call([str(py),'-u',str(Path(__file__).resolve()),*sys.argv[1:]])
    os.chdir(project);sys.path.insert(0,str(project/'src'))
    os.environ.update(PYTHONUTF8='1',PYTHONIOENCODING='utf-8',MPLBACKEND='Agg',BOTRYTIS_BOOTSTRAP_DIR=str(Path.home()/'btlog'))
    os.environ.setdefault('CUBLAS_WORKSPACE_CONFIG',':4096:8')
    if os.environ.get('BOTRYTIS_WORKER_JOB_ID'):raise Stop('Ejecutar directamente, fuera de otro worker')
    from utils.config_loader import load_config
    from orchestration.machine import read_binding
    from orchestration.pc_registry import PCRegistry
    from orchestration.study_queue import StudyQueue
    from utils.durable import FileLock
    from utils.guards import gpu_lock_path
    import psutil
    base=load_config(project/'artifacts/v4/local.yaml');binding=read_binding(base)
    if os.environ.get('USERNAME','').casefold()!=USER or base['compute']['pc_id']!=PC or binding['pc_id']!=PC or binding['machine_id'].casefold()!=USER:
        raise Stop('Usuario, PC y binding no corresponden a PC-01/I21315')
    config,cfg,sets,short=resolve_existing(project,Path.home())
    if PCRegistry(cfg).owner(MODEL)!=PC:raise Stop('ResNet50 no pertenece a PC-01')
    lock=gpu_lock_path(cfg,PC)
    with FileLock(lock.with_suffix('.resnet.guard')),FileLock(lock.with_suffix('.final.guard')),FileLock(lock.with_suffix('.priority.guard')):
        own=psutil.Process();excluded={own.pid,*(p.pid for p in own.parents())}
        active=[p.pid for p in psutil.process_iter(['pid','name']) if p.pid not in excluded and (p.info['name'] or '').casefold() in ('python.exe','pythonw.exe')]
        if active:raise Stop('Otro Python sigue activo; se conserva sin interrumpir ni duplicar: '+str(active))
        emit('FASE_IDENTIFICACION_OK',pc=PC,architecture=MODEL,config=str(config),baseline=str(short/'r/resnet50/baseline'))
        folder=short/'resnet_review';folder.mkdir(parents=True,exist_ok=True)
        # All reports/approvals are versioned attempts; state for resume stays fixed.
        auditdir=folder/'a'/uuid.uuid4().hex[:8];auditdir.mkdir(parents=True)
        audit=dict(pc_id=PC,architecture=MODEL,config=str(config),executor_sha256=EXECUTOR_SHA256,status='COMPROBANDO',
            baseline_reexecuted=False,robustness_started=False,independent_test_used=False)
        try:
            emit('FASE_COMPROBACIONES_CRITICAS',baseline_training=False)
            db,jobs=queue_read(cfg);spec,commit,code,exp=preflight(project,cfg,audit);validate_jobs_identity(jobs,exp,cfg)
            if jobs['baseline']['status'] not in ('PENDING_REVIEW','COMPLETE'):raise Stop('Baseline no finalizado: '+jobs['baseline']['status'])
            audit['baseline_dir']=str(short/'r/resnet50/baseline')
            audit['baseline_evidence']=baseline_evidence_manifest(Path(audit['baseline_dir']))
            audit['baseline_figures']=[str(p) for p in (Path(audit['baseline_dir'])/'figures').glob('*.png')]
            emit('FASE_REVISION_BASELINE_EXISTENTE',path=audit['baseline_dir'],training=False)
            audit['baseline']=inspect_baseline(cfg,spec,short/'r/resnet50/baseline',code,commit,exp)
            audit['curves']=[_classification_curve_evidence(short/f'r/resnet50/baseline/fold{i}/completed_history.csv',False,
                read(short/f'r/resnet50/baseline/fold{i}/completed_fold.json')['row']['best_epoch']) for i in range(3)]
            emit('FASE_AUDITORIA_PARTICIONES',changes_to_splits=False)
            audit['isolation']=partition_evidence(cfg,spec)
            audit['fold_range']=max(audit['baseline']['scores'])-min(audit['baseline']['scores'])
            audit['plan']=dict(baseline=jobs['baseline'],hpo=jobs['hpo'],budget=20,folds=3,following_repeats='Se planifican nativamente al finalizar; no se ejecutan')
            if audit['isolation']['confirmed_relevant_conflicts']:raise Stop('Fuga confirmada; no se aprueba ni se cambian splits')
            if audit['isolation']['suspected_relevant_pairs']:raise Stop('Pares similares entre particiones sin evidencia de fuentes distintas; no se simula revision visual')
            audit['status']='REVISION_AUTOMATICA_OK';write_report(auditdir,audit)
            emit('INFORME_BASELINE_DISPONIBLE',report=auditdir/'REVISION.html')
            if os.name=='nt':
                try:os.startfile(str(auditdir/'REVISION.html'))
                except OSError:pass
            q=StudyQueue.from_cfg(cfg)
            snapshot_sqlite(db,auditdir/'queue-before-approval.sqlite')
            human_approve(q,q.get(jobs['baseline']['job_id']),auditdir,audit)
            rc=run_hpo(project,config,cfg,sets,short,q,q.get(jobs['hpo']['job_id']),folder)
            audit['status']='HPO_TERMINADO';audit['queue']=[j.as_json() for j in q.jobs()];audit['results']=str(short/'r/resnet50/hpo')
            final=folder/'hpo-final.json'
            if final.exists():
                audit['hpo']=read(final)
                audit['comparison']=dict(baseline_AP=audit['baseline']['mean'],best_HPO_AP=audit['hpo']['best_value'],
                    difference_AP=audit['hpo']['best_value']-audit['baseline']['mean'],baseline_max_epochs=cfg['classifier']['baseline']['epochs'],
                    hpo_max_epochs=cfg['classifier']['max_epochs'],winner='NO_DECLARADO',
                    limitation='Validacion interna usada para seleccion, presupuestos de epocas distintos; no es test independiente ni prueba de superioridad causal')
            write_report(auditdir,audit);atomic_json(folder/'latest.json',dict(report=str(auditdir/'REVISION.html'),status=audit['status']))
            return rc
        except BaseException as error:
            audit.update(status='DETENIDO_RESULTADOS_CONSERVADOS',error=str(error));write_report(auditdir,audit)
            emit('ERROR_CRITICO',error=str(error),report=auditdir/'REVISION.html');raise



def baseline_evidence_manifest(folder):
    folder=Path(folder)
    return {str(p.relative_to(folder)):digest(p) for p in sorted(folder.rglob('*'))
        if p.is_file() and 'previous_attempts' not in p.relative_to(folder).parts and p.suffix in ('.json','.csv','.pt')}

if __name__=='__main__':
    try:sys.exit(main())
    except KeyboardInterrupt:emit('INTERRUMPIDO',message='Conservar archivos y reejecutar la misma orden');sys.exit(130)
    except (Exception,RecoveryStop) as error:
        emit('DETENIDO_SIN_REPETIR_BASELINE',error=str(error));traceback.print_exc();sys.exit(2)
