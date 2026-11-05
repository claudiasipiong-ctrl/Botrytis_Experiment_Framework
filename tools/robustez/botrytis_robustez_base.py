"""Botrytis: continuacion posterior al HPO, correccion de CSV para PC-02. Ejecutar fuera del checkout.

No requiere Drive. No modifica fuentes, datos ni configuraciones originales.
No reentrena baselines; conserva sus folds y los intentos anteriores de exportacion.
Los hooks de recuperacion se registran con SHA256 del ejecutor y reutilizan las
funciones y el worker reales del proyecto. No simula inspeccion visual humana.
"""
from __future__ import annotations
import argparse
import copy
from datetime import datetime, timezone
import hashlib
import importlib.util
import json
import math
import os
from pathlib import Path
import pickle
import re
import shutil
import socket
import sqlite3
import subprocess
import sys
import tempfile
import time
import traceback
import uuid

VERSION = '2.0.0'
EXECUTOR_SHA256 = hashlib.sha256(Path(__file__).read_bytes()).hexdigest()
MACHINES = {'PC-01': ('i21315', 'yolov8n', 15), 'PC-02': ('i21314', 'yolo26l', 15), 'PC-03': ('i21313', 'vit_b16', 20)}
QA_REQUIRED = {'static', 'dependencies', 'configuration', 'hardware', 'dataset_zip', 'prepared_data',
               'environment', 'safety', 'clock', 'exchange_state', 'storage_io', 'pip_check',
               'framework_safe_suite', 'cuda_selftest', 'training_smoke', 'dataset_validation'}

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

class OSLock:
    """Nonblocking OS lock; a crash releases it without deleting someone else's lock."""
    def __init__(self, p): self.path = Path(p); self.f = None
    def __enter__(self):
        self.path.parent.mkdir(parents=True, exist_ok=True); self.f = self.path.open('a+b')
        if self.f.seek(0, 2) == 0: self.f.write(b'0'); self.f.flush()
        self.f.seek(0)
        try:
            if os.name == 'nt':
                import msvcrt
                msvcrt.locking(self.f.fileno(), msvcrt.LK_NBLCK, 1)
            else:
                import fcntl
                fcntl.flock(self.f.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        except OSError:
            self.f.close(); self.f = None; raise Stop('YA_ACTIVO: bloqueo ocupado: ' + str(self.path))
        return self
    def __exit__(self, *a):
        if self.f:
            self.f.seek(0)
            if os.name == 'nt':
                import msvcrt
                msvcrt.locking(self.f.fileno(), msvcrt.LK_UNLCK, 1)
            else:
                import fcntl
                fcntl.flock(self.f.fileno(), fcntl.LOCK_UN)
            self.f.close(); self.f = None

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

def python_processes():
    import psutil
    own = psutil.Process(); excluded = {own.pid} | {p.pid for p in own.parents()}
    found = []
    for p in psutil.process_iter(['pid', 'name', 'cmdline', 'create_time']):
        if p.pid in excluded: continue
        try:
            if (p.info['name'] or '').casefold() in ('python.exe', 'pythonw.exe', 'python', 'python3'):
                found.append(dict(pid=p.pid, command=(p.info['cmdline'] or [])[:2], created=p.info['create_time']))
        except (psutil.NoSuchProcess, psutil.AccessDenied):
            raise Stop('No se puede verificar ausencia de otro proceso Python')
    return found

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


def find_project(explicit=None):
    candidates = [Path(explicit)] if explicit else [Path.cwd(), Path.home() / 'Documents/s2_no_borrar/Entrenamiento/Botrytis_Experiment_Framework_git']
    if os.name == 'nt' and not explicit:
        import winreg
        try:
            with winreg.OpenKey(winreg.HKEY_CURRENT_USER, r'Software\Microsoft\Windows\CurrentVersion\Explorer\User Shell Folders') as k:
                documents = os.path.expandvars(winreg.QueryValueEx(k, 'Personal')[0])
                candidates.append(Path(documents) / 's2_no_borrar/Entrenamiento/Botrytis_Experiment_Framework_git')
        except OSError: pass
    for p in candidates:
        if (p / 'src/07_run_hpo_study.py').is_file() and (p / 'artifacts/v4/local.yaml').is_file(): return p.resolve()
    raise Stop('No se encontro el proyecto con artifacts/v4/local.yaml y src/07_run_hpo_study.py')


def resolve_round(base_cfg, model, work):
    from utils.config_loader import load_config, cfg_path
    index = work / 'rondas.json'
    if index.exists() and model in read(index):
        p = Path(read(index)[model])
        if not p.is_file(): raise Stop('rondas.json apunta a un archivo ausente: ' + str(p))
        c = load_config(p)
        if c.get('experiment', {}).get('architecture') != model: raise Stop('rondas.json apunta a otra arquitectura')
        return p.resolve(), c
    roots = [work / 'registro', Path.home() / 'bte', cfg_path(base_cfg, 'paths.artifacts_dir') / 'experiments']
    if base_cfg.get('experiment', {}).get('registry_root'): roots.append(Path(base_cfg['experiment']['registry_root']))
    found = {}
    for r in roots:
        if not r.is_dir(): continue
        for p in r.rglob('config.yaml'):
            if 'previous_attempts' in p.parts or not (p.parent / 'manifest.json').is_file(): continue
            c = load_config(p)
            if c.get('experiment', {}).get('architecture') == model:
                from utils.experiments import verify_round
                try: verify_round(c, architecture=model)
                except (RuntimeError, ValueError): continue
                found[str(p.resolve())] = c
    if len(found) != 1: raise Stop('Ronda compatible ausente o ambigua: ' + str(list(found)))
    name = next(iter(found)); return Path(name), found[name]


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


def qa_evidence(cfg, pc, code, dataset, split):
    from utils.manual_gates import require_gate, ledger_path
    require_gate(cfg, pc, 'READY_FOR_EXPERIMENT')
    ledger = read(ledger_path(cfg, pc))
    if ledger.get('pc_id') != pc: raise Stop('Ledger QA de otra PC')
    row = ledger['gates']['READY_FOR_EXPERIMENT']
    path = Path(row['evidence'])
    if not path.is_file(): raise Stop('Falta la evidencia QA aprobada: ' + str(path))
    authorization = read(path)
    if authorization.get('status') != 'TECHNICAL_QA_PASS':
        # Retain genuinely personal approvals; their evidence need not mimic the
        # automatic runner schema, but the mandatory technical report is checked.
        candidate = Path.cwd() / 'logs_qa/latest.json'
        technical = read(candidate); report_path = Path(technical['evidence_dir']) / 'report.json'
    else:
        for key, expected in [('pc_id', pc), ('code_hash', code), ('dataset_hash', dataset), ('split_hash', split)]:
            if authorization.get(key) != expected: raise Stop('QA incompatible en ' + key)
        report_path = Path(authorization['technical_report'])
        if digest(report_path) != authorization.get('technical_report_sha256'): raise Stop('El informe QA cambio desde la autorizacion')
        xml = Path(authorization['logging_recovery_xml'])
        import xml.etree.ElementTree as ET
        suites = list(ET.parse(xml).getroot().iter('testsuite'))
        if sum(int(t.get('tests', 0)) for t in suites) != int(authorization.get('logging_recovery_tests', -1)) or sum(int(t.get('tests', 0)) for t in suites) < 1 or any(int(t.get(k, 0)) for t in suites for k in ('errors', 'failures', 'skipped')):
            raise Stop('Evidencia de logging/recuperacion incompleta o fallida')
    report = read(report_path); rows = report.get('tests', []); checks = {r['check']: r for r in rows}
    if len(checks) != len(rows) or QA_REQUIRED - set(checks): raise Stop('QA: faltan comprobaciones tecnicas: ' + str(QA_REQUIRED - set(checks)))
    bad = [n for n in QA_REQUIRED if checks[n].get('status') != 'PASS' and not (n == 'dataset_validation' and checks[n].get('status') == 'WARN')]
    if checks['dataset_validation'].get('status') == 'WARN' and cfg.get('dataset', {}).get('validation', {}).get('fail_on_warnings') is True: bad.append('dataset_validation: fail_on_warnings vigente')
    if bad: raise Stop('QA tecnica sin aprobar: ' + str(bad))
    if report.get('machine_id') != MACHINES[pc][0]: raise Stop('Informe QA de otra estacion')
    # Compare actual hashes in the original technical report, including the HPO
    # search/protocol seals where available; no stale remote report is required.
    hashes = report.get('hashes') or {}
    for key, expected in [('code_hash', code), ('dataset_hash', dataset), ('split_hash', split)]:
        if hashes.get(key) != expected: raise Stop('Hash de QA distinto en ' + key)
    return dict(authorization=str(path), technical_report=str(report_path), report_sha256=digest(report_path), reused=True,
                optional_development_full_suite=checks.get('framework_full_suite', {}).get('status', 'NOT_RUN'))


def jobs_read(cfg, model):
    from utils.config_loader import cfg_path
    db = cfg_path(cfg, 'paths.queue_dir') / 'queue.sqlite'
    if not db.is_file(): raise Stop('Cola original ausente: ' + str(db))
    with sqlite3.connect(db.resolve().as_uri() + '?mode=ro', uri=True, timeout=10) as con:
        con.row_factory = sqlite3.Row
        return [dict(r) for r in con.execute('SELECT * FROM jobs WHERE architecture=?', (model,))]


def one_job(jobs, kind):
    found = [j for j in jobs if j['kind'] == kind]
    if len(found) != 1: raise Stop('Job ausente/ambiguo: ' + kind)
    return found[0]


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
            verified.append(dict(number=trial.number, state='PRUNED')); continue
        if trial.value is None or not math.isfinite(float(trial.value)): raise Stop(f'Trial {trial.number}: objetivo COMPLETE invalido')
        hp, folder = trial.user_attrs.get('hp'), trial.user_attrs.get('run_dir')
        if not hp or not folder: raise Stop(f'Trial {trial.number}: faltan hp/run_dir originales')
        folder = allowed_trial_folder(cfg, spec, trial)
        scores = []
        for fold in range(3):
            signature = fold_signature(exp, spec.slug, hp, int(cfg['reproducibility']['seed']), fold, int(sec['max_epochs']), int(sec['early_stopping_patience']))
            receipt, _ = verify_fold(Path(folder) / f'fold{fold}', signature, fold); scores.append(float(receipt['row']['score']))
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
                    optuna=optuna.__version__, python=list(sys.version_info[:2]), hostname=socket.gethostname(), utc=utc()))
    def restore(self, study):
        import optuna
        pointer = self.root / 'state.json'
        if not pointer.exists(): return False
        meta = read(pointer); file = self.root / meta['file']
        if Path(meta['file']).name != meta['file'] or meta.get('identity') != self.identity or meta.get('hostname') != socket.gethostname(): raise Stop('Estado Optuna local de otra identidad/maquina')
        if meta.get('optuna') != optuna.__version__ or meta.get('python') != list(sys.version_info[:2]) or digest(file) != meta.get('sha256'): raise Stop('Estado sampler/pruner alterado o incompatible')
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


class RuntimeHooks:
    """Recovery-only instrumentation around the repository's native objective.

    Sampler, pruner, search spaces, budgets, seed, model functions and metrics stay
    native. We retain checkpoints, restore signed partials and reuse whole folds.
    """
    def __init__(self, cfg, spec, state_dir):
        self.cfg, self.spec, self.state_dir = cfg, spec, Path(state_dir)
        self.state_dir.mkdir(parents=True, exist_ok=True)
        from utils.experiments import fingerprint
        self.exp = fingerprint(cfg); self.state = OptimizerState(self.state_dir / 'sampler', self.exp)
        self.study = None; self.current = None; self.retained = {}

    def event(self, name, **data):
        path = self.state_dir / 'events.jsonl'
        with path.open('a', encoding='utf-8') as f: f.write(json.dumps(dict(utc=utc(), event=name, executor_sha256=EXECUTOR_SHA256, **data), ensure_ascii=False, default=str) + '\n'); f.flush()

    def create_study(self, original, spec, cfg, seed):
        study = original(spec, cfg, seed); self.study = study
        restored = self.state.restore(study)
        if not restored and study.trials:
            replay_optimizer_state(study, cfg, spec, self.state_dir)
            self.event('LEGACY_OPTIMIZER_STATE_RECONSTRUCTED', existing_trials=len(study.trials))
        self.event('OPTIMIZER_STATE_RESTORED' if restored else 'OPTIMIZER_STATE_INITIALIZED', existing_trials=len(study.trials))
        from utils.optuna_storage import budget_used
        from utils.architectures import trials_for
        if budget_used(study) < trials_for(spec, cfg): schedule_retries(study, cfg, spec, self.state_dir)
        self.state.save(study)
        optimize = study.optimize
        def robust_optimize(objective, *a, **kw):
            callbacks = list(kw.pop('callbacks', []) or [])
            def begin(trial):
                trial.set_user_attr('process_owner', own_process())
                trial.set_user_attr('bt_executor_sha256', EXECUTOR_SHA256)
                try: return objective(trial)
                except (KeyboardInterrupt, SystemExit):
                    trial.set_user_attr('bt_interrupted', True); raise
            def persist(st, trial):
                self.state.save(st)
                self.event('TRIAL_DURABLE', trial=trial.number, state=trial.state.name, value=trial.value)
                if trial.state.name == 'FAIL':
                    actions = schedule_retries(st, cfg, spec, self.state_dir)
                    if not any(x.get('action') == 'RETRY_IDENTICAL_PARAMS' and x.get('trial') == trial.number for x in actions):
                        raise RecoveryStop(f'Trial {trial.number} fallo sin evidencia suficiente de recuperacion; archivos y estudio conservados')
            try: return optimize(begin, *a, callbacks=[persist, *callbacks], **kw)
            finally: self.state.save(study)
        study.optimize = robust_optimize
        return study

    def evaluate(self, original, spec, hp, cfg, ctx, run_dir, **kwargs):
        from utils.experiments import fingerprint
        trial = kwargs.get('trial'); run_dir = Path(run_dir); source = None; reused = []; self.retained = {}
        if trial is not None:
            self.current = trial
            trial.set_user_attr('bt_executor_sha256', EXECUTOR_SHA256)
            source_name = trial.user_attrs.get('bt_recovery_source')
            if source_name:
                source = Path(source_name)
                # The source was written by this study. Match it to the original
                # frozen trial, rather than accepting an arbitrary folder.
                previous = next((t for t in self.study.trials if t.number == trial.user_attrs.get('bt_retry_of')), None)
                if previous is None or previous.user_attrs.get('run_dir') != source_name or previous.user_attrs.get('hp') != dict(hp): raise RecoveryStop('Fuente de retry no coincide con el trial original')
                for fold in kwargs['folds']:
                    original_fold = source / f'fold{fold}'
                    if (original_fold / 'completed_fold.json').is_file():
                        sig = fold_signature(fingerprint(cfg), spec.slug, hp, kwargs['seed'], fold, kwargs['max_epochs'], kwargs['patience'])
                        verify_fold(original_fold, sig, fold)
                        copy_tree_safe(original_fold, run_dir / f'fold{fold}')
                        meta = read(source / 'checkpoint_metadata.json')
                        known = [x for x in meta.get('checkpoints', []) if x.get('role') == f'fold{fold}_best']
                        if not known: raise RecoveryStop(f'Fold {fold}: falta metadata original del checkpoint')
                        reused.append((fold, known[-1]))
                        self.event('FOLD_REUSED', retry=trial.number, source=str(original_fold), fold=fold)
                        # Preserve the intermediate pruning evidence for the retry.
                        _, history = verify_fold(original_fold, sig, fold)
                        key = 'map50_95' if spec.is_detector else 'val_ap'
                        for _, r in history.iterrows():
                            if key in r and r[key] is not None and math.isfinite(float(r[key])):
                                trial.report(float(r[key]), int(fold * kwargs['max_epochs'] + int(r['epoch']) - 1))
                # A SHA-checked checkpoint for each incomplete fold is selected by
                # its signed context. The model wrappers below use the native APIs.
            self.state.save(self.study)  # after hyperparameter sampling, before GPU work
        current_context = dict(spec=spec, hp=dict(hp), cfg=cfg, ctx=ctx, run_dir=run_dir, source=source, options=kwargs)
        self.context = current_context
        record = ctx.record_checkpoint
        def checkpoint_record(role, **info):
            m = re.fullmatch(r'fold(\d+)_best', role)
            if m and spec.is_detector:
                p = self.retained.get(int(m.group(1)), run_dir / f'fold{int(m.group(1))}' / 'weights' / 'best.pt')
                if not p.is_file(): p = p.with_name('last.pt')
                if p.is_file():
                    if info.get('sha256') != digest(p): raise RecoveryStop('Checkpoint conservado discrepa de su hash original: ' + str(p))
                    info.update(weights_discarded=False, retained_path=str(p), retained_sha256=digest(p))
            return record(role, **info)
        ctx.record_checkpoint = checkpoint_record
        for fold, info in reused:
            if info.get('retained_path'): self.retained[fold] = Path(info['retained_path'])
            checkpoint_record(f'fold{fold}_best', sha256=info['sha256'], weights_discarded=info.get('weights_discarded', True),
                   recovered_from=str(source), original_trial=trial.user_attrs.get('bt_retry_of'))
        try:
            result = original(spec, hp, cfg, ctx, run_dir, **kwargs)
            original_training = sum(float(f.get('train_seconds', 0)) for f in result.folds)
            if trial is not None: trial.set_user_attr('bt_original_fold_training_seconds', original_training)
            ctx.add_metadata(original_fold_training_seconds=original_training, current_evaluation_seconds=result.train_seconds, recovered_from=str(source) if source else None)
            return result
        finally: ctx.record_checkpoint = record; self.context = None

    def partial_for(self, fold, spec, hp, cfg):
        source = (self.context or {}).get('source')
        if source is None: return None
        matches = []
        for cp, meta in checkpoint_candidates(source):
            extra = meta.get('extra', {}); signed_fold = extra.get('context', {}).get('fold')
            if spec.is_detector:
                m = re.fullmatch(r'fold(\d+)', cp.parent.parent.name)
                signed_fold = int(m.group(1)) if m else signed_fold
            if signed_fold != fold: continue
            validate_checkpoint(cp, spec.slug, cfg, seed=self.context['options']['seed'], hp=hp, fold=fold)
            matches.append((int(extra.get('epoch', 0)), cp))
        if not matches: return None
        matches.sort(key=lambda x: (x[0], str(x[1])))
        if matches[-1][0] >= self.context['options']['max_epochs']:
            raise RecoveryStop('Checkpoint al final de las epocas pero sin recibo de fold; se conserva, no se inventa una metrica')
        return matches[-1][1]

    def classifier(self, original, spec, hp, cfg, train_df, val_df, **kwargs):
        runtime = copy.deepcopy(cfg); fold = runtime.get('classifier', {}).get('_resume_context', {}).get('fold')
        cp = self.partial_for(fold, spec, hp, runtime) if fold is not None else None
        if cp:
            runtime['classifier']['resume_checkpoint'] = str(cp)
            self.event('CHECKPOINT_RESUME', model=spec.slug, fold=fold, checkpoint=str(cp), sha256=digest(cp))
        return original(spec, hp, runtime, train_df, val_df, **kwargs)

    def detector(self, original, spec, hp, cfg, **kwargs):
        runtime = copy.deepcopy(cfg); m = re.fullmatch(r'fold(\d+)', kwargs['name']); fold = int(m.group(1)) if m else None
        destination = Path(kwargs['project_dir']) / kwargs['name']; cp = self.partial_for(fold, spec, hp, runtime) if fold is not None else None
        actual_folder = destination
        if cp:
            # Ultralytics resumes into its original save_dir. Preserve the entire
            # incomplete folder before continuing there, then copy the completed
            # artifacts to the new trial. No completed fold is retrained.
            actual_folder = cp.parent.parent
            if actual_folder.name != f'fold{fold}': raise RecoveryStop('Checkpoint detector sin carpeta fold verificable')
            archive = self.state_dir / 'archive' / uuid.uuid4().hex[:10]
            copy_tree_safe(actual_folder, archive)
            self.event('CHECKPOINT_ARCHIVED', source=str(actual_folder), archive=str(archive))
            runtime['detector']['resume_checkpoint'] = str(cp)
            kwargs.update(project_dir=actual_folder.parent, name=actual_folder.name)
            self.event('CHECKPOINT_RESUME', model=spec.slug, fold=fold, checkpoint=str(cp), sha256=digest(cp))
        kwargs['keep_weights'] = True
        import utils.detector_train as detector_module
        native_new_model = detector_module._new_model
        def snapshot_factory(*a, **kw):
            model, pre_sha = native_new_model(*a, **kw)
            def preserve_optimizer(tr):
                source_cp = Path(tr.last)
                if not source_cp.is_file(): return
                snap = source_cp.parent.parent / 'recovery' / 'epoch.pt'
                from utils.durable import atomic_copy
                from utils.resume import receipt
                atomic_copy(source_cp, snap)
                receipt(snap, spec, runtime, dict(seed=kwargs['seed'], hp=dict(hp), epoch=int(tr.epoch)+1, context=dict(fold=fold)))
                self.event('OPTIMIZER_CHECKPOINT_SAVED', fold=fold, checkpoint=str(snap), sha256=digest(snap))
            model.add_callback('on_model_save', preserve_optimizer)
            return model, pre_sha
        detector_module._new_model = snapshot_factory
        try: run = original(spec, hp, runtime, **kwargs)
        finally: detector_module._new_model = native_new_model
        if run.weights_path:
            self.retained[fold] = Path(run.weights_path)
            actual_folder = Path(run.weights_path).parent.parent
        # Ultralytics strips optimizer fields during final_eval. Keep its original
        # receipt under a clearly non-resumable name only after proving the
        # optimizer-bearing bytes survived in our signed snapshot.
        last = actual_folder / 'weights' / 'last.pt'; side = last.with_suffix('.pt.resume.json')
        if last.is_file() and side.is_file() and digest(last) != read(side).get('sha256'):
            snapshot = actual_folder / 'recovery' / 'epoch.pt'
            if not snapshot.is_file() or digest(snapshot) != read(side).get('sha256'):
                raise RecoveryStop('Backend modifico last.pt sin snapshot de optimizador verificable')
            archived = last.with_suffix('.pt.pre_final_receipt.json')
            verified_copy(side, archived); side.unlink()
            atomic_json(actual_folder / 'recovery' / 'final_weights.json', dict(final_last_sha256=digest(last), optimizer_snapshot=str(snapshot), snapshot_sha256=digest(snapshot), original_receipt=str(archived)))

        # The native in-memory callback history contains only new epochs on
        # resume; the backend's original CSV has all actual epochs and metrics.
        if cp:
            import pandas as pd
            raw = pd.read_csv(actual_folder / 'results.csv'); raw.columns = raw.columns.str.strip()
            metric = 'metrics/mAP50-95(B)'
            if raw.empty or metric not in raw or not raw[metric].notna().any(): raise RecoveryStop('Resume detector sin historial completo original')
            raw['train_loss'] = None; raw['val_loss'] = None
            raw['map50_95'] = raw[metric]; raw['map50'] = raw.get('metrics/mAP50(B)')
            p, r = raw.get('metrics/precision(B)'), raw.get('metrics/recall(B)')
            if p is not None and r is not None: raw['f1'] = (2 * p * r / (p + r)).where((p + r) > 0, 0)
            best = raw[metric].astype(float).idxmax()
            run.history = raw; run.score = float(raw.loc[best, metric]); run.best_epoch = int(raw.loc[best, 'epoch'])
            run.epochs_run = len(raw); run.stopped_early = len(raw) < int(kwargs.get('epochs', len(raw)))
            # The generated fold YAML in the new trial remains its own verified
            # local dataset description; other backend outputs are byte copies.
            copy_tree_safe(actual_folder, destination, skip=('data.yaml', 'dataset.yaml'))
        return run

    def install(self, module):
        import utils.hpo_runner as runner
        original_evaluate = runner.evaluate_config
        original_detector, original_classifier = runner.train_detector, runner.train_classifier
        original_create = module.create_or_load_study
        module.create_or_load_study = lambda spec, cfg, seed: self.create_study(original_create, spec, cfg, seed)
        runner.evaluate_config = lambda *a, **kw: self.evaluate(original_evaluate, *a, **kw)
        runner.train_detector = lambda *a, **kw: self.detector(original_detector, *a, **kw)
        runner.train_classifier = lambda *a, **kw: self.classifier(original_classifier, *a, **kw)
        # Only the hpo_runner reference is replaced; shutil itself is unchanged.
        # Preserve the native completed detector weights instead of deleting them.
        native_shutil = runner.shutil
        class RetainWeights:
            def __getattr__(self, name): return getattr(native_shutil, name)
            def rmtree(inner, path, *a, **kw):
                if Path(path).name == 'weights' and self.context and inside(path, self.context['run_dir']):
                    self.event('CHECKPOINTS_RETAINED', path=str(path)); return
                return native_shutil.rmtree(path, *a, **kw)
        runner.shutil = RetainWeights()


def load_native(path, name):
    spec = importlib.util.spec_from_file_location(name, path); module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module; spec.loader.exec_module(module); return module


def stamp_review(review, audit, run_dir):
    review.update({k:audit[k] for k in ('pc_id','config','git_commit','code_hash','dataset_hash','split_hash','seed','experiment_fingerprint','executor_sha256') if k in audit})
    review.update(utc=utc(), original_run_dir=str(run_dir), operator=os.environ.get('USERNAME',''),
                  authorization='Revision tecnica automatica y continuacion HPO solicitadas por el usuario; sin firma de inspeccion visual humana.')
    return review


def make_short_paths(config, work):
    key = hashlib.sha256(str(Path(config).resolve()).casefold().encode()).hexdigest()[:8]
    root = Path.home() / 'bf' / key
    root.mkdir(parents=True, exist_ok=True)
    return root, root / 'r', root / 'e', root / 's'


def path_preflight(cfg, spec, trial_folder, out):
    """Check actual atomic CSV and PNG/PDF/SVG writes at HPO evidence depth."""
    from utils.evidence import token
    from utils.io import write_csv, write_json
    import pandas as pd
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    probe = Path(out) / 'probe' / spec.task / spec.slug / 'train/internal' / token(str(trial_folder)) / 'inner_validation'
    metrics = probe / 'metrics'; plots = probe / 'plots'; plots.mkdir(parents=True, exist_ok=True)
    path = plots / (token('fold0_metrics/mAP50-95(B)') + '.png')
    if os.name == 'nt' and len(str(path)) >= 248: raise Stop('Ruta propuesta sigue siendo demasiado larga: ' + str(path))
    write_csv(pd.DataFrame({'epoch':[1,2], 'value':[.2,.3]}), metrics / 'history.csv')
    write_json(metrics / 'provenance.json', dict(status='PATH_PROBE_ONLY', synthetic=True))
    fig, ax = plt.subplots(); ax.plot([1,2], [.2,.3])
    try:
        for ext in ('png','pdf','svg'): fig.savefig(path.with_suffix('.' + ext))
    finally: plt.close(fig)
    return dict(directory=str(probe), checked=['atomic_csv','json','png','pdf','svg'], sample_path_length=len(str(path)))


def snapshot_sqlite(db, destination):
    if not Path(db).is_file(): return
    with sqlite3.connect(Path(db).resolve().as_uri() + '?mode=ro', uri=True, timeout=15) as src:
        with sqlite3.connect(destination, timeout=15) as dst: src.backup(dst)
    with sqlite3.connect(destination) as test:
        if test.execute('PRAGMA integrity_check').fetchone()[0] != 'ok': raise Stop('Snapshot SQLite no integro')


def protocol_preflight(root, base, cfg, pc, spec, evidence, audit):
    from utils.config_loader import cfg_path
    from utils.experiments import verify_round, fingerprint
    from utils.hashing import sha256_file
    from utils.prepared import load_prepared_data
    from utils.isolation import require_near_duplicate_review
    from utils.weights import verified_local_weight
    from utils.detector_data import detection_images_table
    from utils.classifier_data import load_crop_manifest
    from utils.safety import SafetyMonitor
    from utils.reproducibility import collect_environment
    from utils.run_common import check_environment_matches_reference
    from utils.guards import assert_equal_hpo_budget, assert_same_pretrained_policy, assert_image_size_fixed, assert_not_frozen
    from types import SimpleNamespace
    commit, code = check_git(root, audit); manifest = verify_round(cfg, architecture=spec.slug)
    if manifest.get('config_sha256') != sha256_file(audit['config']): raise Stop('Configuracion inmutable modificada: ' + audit['config'])
    if int(cfg['inner_hpo']['folds']) != 3: raise Stop('Se requieren 3 folds')
    assert_equal_hpo_budget(cfg); assert_same_pretrained_policy(cfg); assert_image_size_fixed(cfg); assert_not_frozen(cfg, 'HPO')
    from utils.architectures import trials_for
    if trials_for(spec, cfg) != MACHINES[pc][2]: raise Stop('Presupuesto aprobado diferente de 15/20')
    from orchestration.pc_registry import PCRegistry
    if PCRegistry(cfg).owner(spec.slug) != pc: raise Stop('Propiedad de arquitectura distinta/transferencia pendiente')
    data = load_prepared_data(cfg, with_folds=True); require_near_duplicate_review(cfg, data)
    from utils.dataset import discover_images
    active_images = discover_images(cfg_path(cfg,'paths.raw_images_dir'), cfg['dataset']['image_extensions'])
    if cfg['dataset'].get('enforce_expected_count') and len(active_images) != int(cfg['dataset']['n_images_expected']): raise Stop('El inventario activo no coincide con el conteo aprobado')
    detection_images_table(cfg); load_crop_manifest(cfg)
    # PreparedData checks exact-content leakage, signed splits/folds, GT crops and
    # inherited image membership. Also explicitly inspect inner-fold ID sets.
    from utils.folds import DET_FOLD_COL, CLS_FOLD_COL
    for table, column in ((data.det_folds, DET_FOLD_COL), (data.cls_folds, CLS_FOLD_COL)):
        if table is None or table['image_id'].duplicated().any(): raise Stop('IDs de folds duplicados o ausentes')
        train_ids = set(data.manifest.loc[data.manifest.outer_split == 'train', 'image_id'].astype(str))
        if set(table['image_id'].astype(str)) != train_ids: raise Stop('Folds no corresponden exactamente al outer train')
        for fold in range(3):
            va = set(table.loc[table[column] == fold, 'image_id'].astype(str)); tr = set(table.loc[table[column] != fold, 'image_id'].astype(str))
            if not va or not tr or va & tr: raise Stop('Fold vacio o fuga entre entrenamiento/validacion')
    weight = verified_local_weight(spec, cfg)
    qa = qa_evidence(cfg, pc, code, data.dataset_hash, data.split_hash)
    import torch
    if not torch.cuda.is_available() or torch.cuda.device_count() != 1: raise Stop('GPU CUDA unica no disponible; no se permite caer a CPU')
    environment = collect_environment(cfg=cfg, project_root=root, pc_id=pc)
    import logging
    check_environment_matches_reference(SimpleNamespace(cfg=cfg, env=environment, logger=logging.getLogger('bt-final')), strict=True)
    SafetyMonitor(cfg, evidence / 'telemetry.jsonl').check()
    audit.update(git_commit=commit, code_hash=code, dataset_hash=data.dataset_hash, split_hash=data.split_hash,
                 seed=cfg['reproducibility']['seed'], experiment_fingerprint=fingerprint(cfg),
                 pretrained_weight=str(weight), pretrained_sha256=digest(weight), qa_reused=qa,
                 independent_local_verification=True, remote_pc3_required=False)
    return commit, code, fingerprint(cfg)


def study_counts(study):
    counts = {s:0 for s in ('COMPLETE','PRUNED','FAIL','RUNNING','WAITING')}
    for t in study.trials: counts[t.state.name] = counts.get(t.state.name,0) + 1
    return counts


def verify_study_summary(study, cfg, spec, roots):
    from utils.search_spaces import search_space_hash
    candidates = [Path(r)/spec.slug/'hpo/study_summary.json' for r in roots]
    for p in candidates:
        if not p.is_file(): continue
        s = read(p); counts = study_counts(study)
        if s.get('study_name') != study.study_name or s.get('search_space_sha256') != search_space_hash(): continue
        if any(s.get(field) != counts[state] for field,state in [('n_complete','COMPLETE'),('n_pruned','PRUNED'),('n_failed','FAIL')]): continue
        best = s.get('best') or {}
        if best.get('number') == study.best_trial.number and math.isclose(float(best.get('value',-1)),study.best_value,abs_tol=1e-8): return str(p)
    raise Stop('Falta un study_summary.json integro y concordante con Optuna: ' + str(candidates))


def finish_report(path, audit, status, **more):
    audit.update(status=status, finished_utc=utc(), **more); atomic_json(path, audit)
    emit(status, summary=str(path), **more)




"""Se incorpora al ejecutor unico al empaquetar; no es una dependencia en Windows."""

POST_VERSION = '3.1.1-PC02-CSV'
POST_HPO_REFERENCES = {
    'PC-02':dict(model='yolo26l',total=15,complete=10,pruned=5,best_trial=2,best_value=0.9014278550610199,
        baseline=0.8838733333333333,baseline_sd=0.006704254867868095,
        configs=['cfg_0b6b618a','cfg_eeeab50e','cfg_dc44aac5']),
    'PC-03':dict(model='vit_b16',total=20,complete=11,pruned=9,best_trial=6,best_value=0.9290032595921819,
        baseline=0.9320969025099615,baseline_sd=0.022811500189412432,
        configs=['cfg_134de332','cfg_16fcf576','cfg_29f010e3']),
}
TARGET_JOB_ID = None
POST_KINDS = ('repeat', 'aggregate', 'outer_train', 'outer_eval', 'thresholds')
POST_FILES = {'repeat':'08_repeat_top_configs.py', 'aggregate':'08_repeat_top_configs.py',
              'outer_train':'09_train_outer_candidate.py', 'outer_eval':'10_evaluate_outer_validation.py',
              'thresholds':'11_tune_inference_thresholds.py'}



class ReviewRequired(Stop): pass


def check_completed_hpo(pc,study,baseline,job):
    """Read-only comparison to the confirmed handoff; no recovery/optimization."""
    from utils.hpo_results import top_configs
    if pc not in POST_HPO_REFERENCES:raise Stop('Esta entrega es solo para PC-02/PC-03')
    ref=POST_HPO_REFERENCES[pc];counts=study_counts(study)
    expected=dict(COMPLETE=ref['complete'],PRUNED=ref['pruned'],FAIL=0,RUNNING=0,WAITING=0)
    if counts!=expected or len(study.trials)!=ref['total']:raise Stop('El estudio local no coincide con el cierre confirmado: '+str(counts)+'. No se vuelve a ejecutar HPO.')
    if study.best_trial.number!=ref['best_trial'] or not math.isclose(study.best_value,ref['best_value'],rel_tol=0,abs_tol=1e-10):raise Stop('Mejor trial/valor distinto del cierre confirmado; resultados conservados')
    if [c['config_id'] for c in top_configs(study,3)]!=ref['configs']:raise Stop('Top-3 distinto del cierre confirmado; no se modifica el estudio')
    if not math.isclose(baseline['mean'],ref['baseline'],rel_tol=0,abs_tol=1e-10) or not math.isclose(baseline['sd'],ref['baseline_sd'],rel_tol=0,abs_tol=1e-10):raise Stop('Baseline de otra ronda/version; no se sustituye ni se reentrena')
    if job.get('payload',{}).get('partial_budget') or job['status'] not in ('PENDING_REVIEW','COMPLETE','PUBLICATION_PENDING'):raise Stop('Estado de cola HPO incompatible con cierre confirmado: '+job['status'])
    return dict(status='CONFIRMED_COMPLETED_HPO_READ_ONLY',reference=ref,states=counts,optimizer_untouched=True)


def classification_prediction_audit(path,threshold=.5):
    """Metrics from immutable saved probabilities; no inference or threshold search."""
    import numpy as np
    import pandas as pd
    from utils.metrics import binary_classification_report,precision_recall_curve_np
    if not math.isfinite(float(threshold)) or not 0<=threshold<=1:raise Stop('Umbral de clasificacion no valido')
    frame=pd.read_csv(path)
    if not {'y_true','p_con_signos'}<=set(frame):raise Stop('Faltan etiquetas/probabilidades guardadas: '+str(path))
    if frame.empty:return dict(status='NOT_CALCULABLE',reason='No hay predicciones',path=str(path),sha256=digest(path))
    y=pd.to_numeric(frame.y_true,errors='raise').to_numpy();p=pd.to_numeric(frame.p_con_signos,errors='raise').to_numpy()
    if not np.isfinite(y).all() or not set(y)<={0,1} or not np.isfinite(p).all() or ((p<0)|(p>1)).any():raise Stop('Etiquetas/probabilidades invalidas: '+str(path))
    if {'image_id','fruit_id'}<=set(frame) and frame.duplicated(['image_id','fruit_id']).any():raise Stop('Predicciones duplicadas para un fruto: '+str(path))
    report=binary_classification_report(y.astype(int),p,threshold=threshold)
    area=None
    if y.sum()>0:
        precision,recall,_=precision_recall_curve_np(y,p)
        if np.any(np.diff(recall)>0):raise Stop('Orden inesperado de la curva PR nativa')
        integrate=np.trapezoid if hasattr(np,'trapezoid') else np.trapz
        area=float(integrate(precision[::-1],recall[::-1]))
    return dict(status='CALCULATED_FROM_SAVED_PROBABILITIES',path=str(path),sha256=digest(path),threshold=threshold,
        positive_class='con_signos',average_precision_positive=report.get('average_precision_positive'),
        pr_auc_trapezoidal=area,roc_auc=report.get('roc_auc'),precision=report['precision'],recall=report['recall'],
        f1=report['f1'],accuracy=report['accuracy'],confusion_matrix=report['confusion_matrix'],
        confusion_matrix_order='[[TN,FP],[FN,TP]]',n=report['n'],n_positive=report['n_positive'],
        definitions='AP escalonada nativa y area PR trapezoidal se informan por separado; el objetivo HPO sigue siendo AP positiva.',
        precision_note='Recalculo con probabilidades CSV redondeadas; el resultado oficial original se conserva, aun si hay empates de redondeo.')


def pure_job(job):
    if isinstance(job, dict):
        out = dict(job)
        for k in ('payload', 'depends_on'):
            if isinstance(out.get(k), str): out[k] = json.loads(out[k])
        return out
    return dict(job_id=job.job_id, kind=job.kind, architecture=job.architecture, status=job.status,
                payload=dict(job.payload), depends_on=list(job.depends_on), attempt=job.attempt,
                log_path=job.log_path, error=job.error, pc_id=job.pc_id)


def expected_post_jobs(cfg, spec, cands):
    from orchestration.study_queue import job_id_for
    exp = cfg['experiment']; prefix = exp['experiment_id'] + '/' + exp['round_id'] + '/'
    jid = lambda kind, **kw: prefix + job_id_for(kind, spec.slug, **kw)
    hpo = jid('hpo'); rows = []
    for c in cands:
        for seed in cfg['robustness']['seeds']:
            rows.append(dict(job_id=jid('repeat', config_id=c['config_id'], seed=int(seed)), kind='repeat',
                             config_id=c['config_id'], seed=int(seed), depends_on=[hpo]))
    deps = [r['job_id'] for r in rows]
    for kind in ('aggregate', 'outer_train', 'outer_eval', 'thresholds'):
        ident = jid(kind); rows.append(dict(job_id=ident, kind=kind, depends_on=deps)); deps = [ident]
    return rows


def validate_queue_plan(cfg, spec, cands, jobs, allow_missing=False):
    expected = expected_post_jobs(cfg, spec, cands); found = {j['job_id']:j for j in map(pure_job, jobs)}
    if len(cands) != int(cfg['robustness']['top_k_configs']): raise Stop('No existen suficientes configuraciones COMPLETE distintas')
    if len(found) != len(jobs): raise Stop('Jobs duplicados')
    for r in expected:
        j = found.get(r['job_id'])
        if j is None and allow_missing: continue
        if not j or j['kind'] != r['kind'] or sorted(j['depends_on']) != sorted(r['depends_on']):
            raise Stop('Plan local ausente/incompatible: ' + str(r))
        if r['kind'] == 'repeat' and any(j['payload'].get(k) != r[k] for k in ('config_id', 'seed')):
            raise Stop('Payload de repeticion incompatible: ' + j['job_id'])
    extras = [j['job_id'] for j in found.values() if j['kind'] in POST_KINDS and j['job_id'] not in {r['job_id'] for r in expected}]
    if extras: raise Stop('Jobs adicionales/otra ronda; no se ejecutan: ' + str(extras))
    return [dict(r, status=found.get(r['job_id'],{}).get('status','MISSING')) for r in expected]


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



YOLO_METRIC_COLUMNS = {
    'map50_95': ('metrics/mAP50-95(B)', 'map50_95'),
    'map50': ('metrics/mAP50(B)', 'map50'),
    'precision': ('metrics/precision(B)', 'precision'),
    'recall': ('metrics/recall(B)', 'recall'),
}


def yolo_csv_inspection(path, complete=True):
    """Read-only structural/numeric audit; validation losses are diagnostic only.

    This is not permission to ignore invalid training or objective values.
    Missing optional data and explicit nonfinite data have different diagnoses.
    """
    import csv
    import numpy as np
    import pandas as pd
    path = Path(path)
    try:
        with path.open('r', encoding='utf-8-sig', newline='') as f:
            reader = csv.reader(f, strict=True)
            header = [c.strip() for c in next(reader, [])]
            if not header or any(not c for c in header) or len(set(header)) != len(header):
                raise Stop('CSV corrupto: cabecera vacia/duplicada: ' + str(path))
            rows = []
            for row in reader:
                if len(row) != len(header):
                    raise Stop(f'CSV corrupto/truncado: {path}, linea {reader.line_num}: {len(row)} campos; esperados {len(header)}')
                rows.append([c.strip() for c in row])
        if not rows or 'epoch' not in header:
            raise Stop('CSV vacio/sin epoch: ' + str(path))
    except (OSError, UnicodeError, csv.Error) as e:
        raise Stop(f'CSV ilegible/corrupto: {path}: {e}') from e
    h = pd.DataFrame(rows, columns=header)
    # Known NA tokens remain absent; unexpected text is corruption, not an NA.
    missing_tokens = {'', 'nan', '+nan', '-nan', 'na', 'n/a', 'null', 'none', 'nd'}
    def numbers(col):
        raw = h[col]
        missing = raw.str.casefold().isin(missing_tokens)
        value = pd.to_numeric(raw.mask(missing), errors='coerce')
        invalid = ~missing & value.isna()
        if invalid.any():
            raise Stop(f'CSV no numerico: {path} {col}, filas ' + str((np.flatnonzero(invalid) + 2).tolist()))
        return value.astype(float), missing
    epochs, absent = numbers('epoch')
    if absent.any() or not np.isfinite(epochs).all() or (epochs % 1 != 0).any() or epochs.tolist() != list(range(1, len(h) + 1)):
        raise Stop('Epocas incompletas/duplicadas/desordenadas: ' + str(path))
    h['epoch'] = epochs.astype(int)
    loss_columns = [c for c in header if (c.startswith(('train/', 'val/')) and c.endswith('_loss')) or c in ('train_loss', 'val_loss')]
    training = [c for c in loss_columns if c.startswith('train/')]
    if not training and 'train_loss' in h and any(x.casefold() not in missing_tokens for x in h['train_loss']):
        training = ['train_loss']
    if complete and not training:
        raise Stop('Faltan perdidas originales de entrenamiento: ' + str(path))
    availability = {}
    warnings = []
    numeric = {}
    for col in loss_columns:
        value, missing = numbers(col)
        finite = np.isfinite(value)
        nan_token = h[col].str.casefold().isin({'nan', '+nan', '-nan'})
        absent = missing & ~nan_token
        explicit_nonfinite = (~missing & ~finite) | nan_token
        required = col in training
        if required and (not finite.all() or (value < 0).any()):
            raise Stop('Perdida obligatoria de entrenamiento no finita/incompleta/invalida: ' + str(path) + ' ' + col)
        if not required and (value[finite] < 0).any():
            raise Stop('Perdida negativa invalida: ' + str(path) + ' ' + col)
        status = 'AVAILABLE' if finite.all() else ('PARTIAL' if finite.any() else 'ND')
        entry = dict(status=status, role='TRAINING_REQUIRED' if required else 'OPTIONAL_DIAGNOSTIC',
                     n_finite=int(finite.sum()), n_missing=int(absent.sum()), n_nonfinite=int(explicit_nonfinite.sum()), n_nan=int(nan_token.sum()),
                     missing_epochs=h.loc[absent, 'epoch'].tolist(), nonfinite_epochs=h.loc[explicit_nonfinite, 'epoch'].tolist(),
                     reason=None if finite.all() else 'No se imputan valores; una curva incompleta no permite comparar losses de las mismas ventanas.')
        availability[col] = entry
        numeric[col] = value.where(finite)  # in-memory only; preserve every original byte
        if not finite.all():
            kind = 'VALIDATION_LOSS_NONFINITE' if col.startswith('val') and explicit_nonfinite.any() else 'OPTIONAL_LOSS_UNAVAILABLE'
            warnings.append(dict(kind=kind, column=col, **entry,
                conclusion=('Anomalia numerica de validacion; no se presume normal ni corrupcion del entrenamiento. Revisar antes de APROBAR.'
                            if explicit_nonfinite.any() else 'Registro ausente; no demuestra divergencia. Diagnostico de sobreajuste por esta loss: ND.')))
    # Do not invent alternative backend column names. Report the names actually found.
    for component in ('box', 'cls', 'dfl'):
        col = 'val/' + component + '_loss'
        if col not in availability:
            availability[col] = dict(status='ND', role='OPTIONAL_DIAGNOSTIC', n_finite=0, n_missing=len(h), n_nonfinite=0,
                                     reason='Columna no publicada por este CSV; no es obligatoria para el objetivo mAP50-95.')
            warnings.append(dict(kind='OPTIONAL_LOSS_COLUMN_ABSENT', column=col, **availability[col]))
    metric_columns = {}
    metric_availability = {}
    for name, aliases in YOLO_METRIC_COLUMNS.items():
        found = [c for c in aliases if c in h]
        if not found:
            if name == 'map50_95' and complete:
                raise Stop('Falta metrica objetiva obligatoria mAP50-95: ' + str(path))
            metric_availability[name] = dict(status='ND', reason='No publicada; no se inventa ni sustituye por otra metrica.')
            warnings.append(dict(kind='DIAGNOSTIC_METRIC_COLUMN_ABSENT', metric=name, **metric_availability[name]))
            continue
        for col in found:
            value, missing = numbers(col)
            if not np.isfinite(value).all() or not value.between(0, 1).all():
                raise Stop('Metrica registrada invalida/no finita/incompleta: ' + str(path) + ' ' + col)
            numeric[col] = value
        chosen = found[0]
        if any(not np.allclose(numeric[chosen], numeric[c], rtol=0, atol=1e-5) for c in found[1:]):
            raise Stop('Aliases de metrica contradictorios: ' + str(path) + ' ' + name)
        metric_columns[name] = chosen
        metric_availability[name] = dict(status='AVAILABLE', column=chosen)
    if 'map50' in metric_columns and 'map50_95' in metric_columns:
        if (numeric[metric_columns['map50_95']] > numeric[metric_columns['map50']] + 1e-5).any():
            raise Stop('Metricas incompatibles: mAP50-95 mayor que mAP50: ' + str(path))
    for col, value in numeric.items():
        h[col] = value
    return h, dict(path=str(path), sha256=digest(path), epochs=len(h), columns=header,
                   loss_availability=availability, metric_availability=metric_availability, warnings=warnings,
                   validation_loss_columns=[c for c in loss_columns if c.startswith('val')],
                   status='VERIFIED_WITH_WARNINGS' if warnings else 'VERIFIED',
                   scope='PARTIAL_PRUNED_DIAGNOSTIC_ONLY' if not complete else 'COMPLETED_FOLD_REQUIRED_CHECKS',
                   raw_csv_modified=False)


def yolo_final_callback_evidence(raw, signed, h, sh):
    """Recognize the b706/8.4.170 early-stop final_eval callback, never a CSV truncation.

    No input is edited. The signed receipt, producer versions, full matching
    prefix and unchanged last training losses bind the extra validation event.
    """
    import numpy as np
    folder = Path(signed).parent
    n = len(h)
    if len(sh) != n + 1 or n < 1:
        raise Stop(f'CSV original truncado/no concuerda con historial firmado: {raw}; raw={n}, firmado={len(sh)}')
    receipt = read(folder/'completed_fold.json'); row = receipt.get('row', {})
    meta = read(folder.parent/'metadata.json'); env = read(folder.parent/'environment.json')
    if (receipt.get('history_sha256') != digest(signed) or row.get('epochs_run') != len(sh)
            or meta.get('status') != 'COMPLETE'
            or meta.get('git_commit') != 'b706f176c8e308969fd3a885d66045ffea027c7e'
            or meta.get('code_hash') != 'e11f772d86f45e5dc3687fce3faa23443a6fcf16146484b3d5f38534cc7695df'
            or env.get('ultralytics') != '8.4.170'):
        raise Stop('Fila adicional sin procedencia final_eval comprobable; no se acepta: '+str(raw))
    from utils.config_loader import load_config
    from utils.architectures import get_arch, section_for
    from utils.experiments import fingerprint
    cfg = load_config(folder.parent/'config.yaml')
    slug = cfg['experiment']['architecture']; spec = get_arch(slug); sec = section_for(spec, cfg)
    if not spec.is_detector or spec.family == 'rtdetr':
        raise Stop('Callback adicional de otro backend; no se acepta como YOLO')
    if meta.get('dataset_hash') != env.get('dataset_hash') or meta.get('split_hash') != env.get('split_hash'):
        raise Stop('Proveniencia dataset/splits contradictoria en callback final')
    if not meta.get('dataset_hash') or not meta.get('split_hash'):
        raise Stop('Faltan hashes dataset/splits del productor del historial')
    hp = meta.get('extra', {}).get('hp', meta.get('hp')); seed = meta.get('seed'); fold = row.get('fold')
    if not isinstance(hp, dict) or seed is None or fold not in (0,1,2):
        raise Stop('Falta contrato de fold/hp/seed para callback adicional')
    signature = fold_signature(fingerprint(cfg), slug, hp, seed, fold,
                               int(sec['max_epochs']), int(sec['early_stopping_patience']))
    if (receipt.get('signature') != signature or not n < int(sec['max_epochs'])
            or row.get('stopped_early') is not (len(sh) < int(sec['max_epochs']))):
        raise Stop('Callback adicional incompatible con firma/presupuesto del fold')
    train = [c for c in h if c.startswith('train/') and c.endswith('_loss') and c in sh]
    if not train or any(not np.isfinite(sh[c].iloc[-2:]).all()
                        or not math.isclose(float(sh[c].iloc[-1]),float(sh[c].iloc[-2]),rel_tol=0,abs_tol=1e-10)
                        for c in train):
        raise Stop('La fila adicional contiene entrenamiento distinto; no es final_eval verificable')
    for name in ('map50_95','map50','precision','recall'):
        aliases = YOLO_METRIC_COLUMNS[name]
        if not any(c in h for c in aliases) or not any(c in sh for c in aliases):
            raise Stop('Faltan metricas para comprobar callback final: '+name)
    metric = next(c for c in YOLO_METRIC_COLUMNS['map50_95'] if c in sh)
    peak = float(sh[metric].iloc[:n].max())
    peak_events = sh.index[:n][sh[metric].iloc[:n] == peak]
    stopper_epoch = int(sh.loc[peak_events[0],'epoch'])
    # Ultralytics keeps the first improving epoch in its stopper; best.pt is
    # rewritten on ties. Both are derived from actual signed training rows.
    checkpoint_epoch = int(sh.loc[peak_events[-1],'epoch'])
    if n - stopper_epoch != int(sec['early_stopping_patience']):
        raise Stop('La parada temprana no se explica por la paciencia original')
    return dict(kind='VERIFIED_POST_TRAIN_BEST_CHECKPOINT_VALIDATION', raw_training_epochs=n,
                signed_events=len(sh), extra_event_index=n+1, extra_event_is_training_epoch=False,
                checkpoint_training_epoch=checkpoint_epoch, online_training_peak=peak,
                final_validation_objective=float(sh[metric].iloc[-1]),
                historical_hpo_score_preserved=True, raw_csv_modified=False,
                receipt_sha256=digest(folder/'completed_fold.json'), producer_ultralytics=env['ultralytics'])


def curve_evidence(path, detector, selected_epoch=None):
    if not detector:
        return _classification_curve_evidence(path, False, selected_epoch)
    import numpy as np
    path = Path(path)
    signed = path if path.name == 'completed_history.csv' else path.with_name('completed_history.csv')
    raw = path.with_name('results.csv')
    source = raw if raw.is_file() else path
    h, inspection = yolo_csv_inspection(source)
    signed_sha = digest(signed) if signed.is_file() else digest(path)
    final_callback = None
    objective_history = h
    if signed.is_file() and signed != source:
        sh, signed_inspection = yolo_csv_inspection(signed)
        compare = sh
        if h.epoch.tolist() != sh.epoch.tolist():
            final_callback = yolo_final_callback_evidence(source, signed, h, sh)
            compare = sh.iloc[:len(h)]
            if h.epoch.tolist() != compare.epoch.tolist():
                raise Stop('Epocas originales desalineadas con prefijo firmado: '+str(source))
        for name, aliases in YOLO_METRIC_COLUMNS.items():
            a = next((c for c in aliases if c in h), None)
            b = next((c for c in aliases if c in compare), None)
            if a and b and not np.allclose(h[a], compare[b], rtol=0, atol=1e-5):
                raise Stop('Metrica original discrepa del historial firmado: ' + str(source) + ' ' + name)
        for col in (c for c in h if c.startswith('train/') and c in sh):
            if not np.allclose(h[col], compare[col], rtol=1e-4, atol=1e-5):
                raise Stop('Loss de entrenamiento discrepa del historial firmado: ' + str(source) + ' ' + col)
        inspection['signed_csv_sha256'] = signed_inspection['sha256']
        if final_callback:
            objective_history = sh
            inspection['warnings'].append({**final_callback, 'kind':'EXTRA_FINAL_VALIDATION_EVENT'})
            emit('CSV_CALLBACK_FINAL_VERIFICADO',path=str(source),training_epochs=len(h),signed_events=len(sh))
    objective = next(c for c in YOLO_METRIC_COLUMNS['map50_95'] if c in objective_history)
    values = objective_history[objective]
    maximum_index = values.idxmax()
    bi = maximum_index
    if selected_epoch is not None:
        candidates = objective_history.index[objective_history.epoch == selected_epoch]
        if len(candidates) != 1:
            raise Stop('Epoca del checkpoint ausente de la curva: ' + str(source))
        bi = candidates[0]
        if not math.isclose(float(values.loc[bi]), float(values.max()), rel_tol=0, abs_tol=1e-5):
            raise Stop('Epoca seleccionada incompatible con maximo registrado')
    best = objective_history.loc[bi]
    selected_event = int(best.epoch)
    checkpoint_epoch = (final_callback['checkpoint_training_epoch'] if final_callback and selected_event > len(h)
                        else selected_event)
    result = dict(path=str(source), sha256=inspection['sha256'], signed_history_sha256=signed_sha, epochs=len(h),
                  losses={}, flags=[], csv_inspection=inspection, loss_availability=inspection['loss_availability'],
                  warnings=inspection['warnings'], train_validation_metric_gap=None,
                  gap_reason='No se registra mAP sobre train por epoca con la misma evaluacion; no se infiere de las losses.')
    for col, entry in inspection['loss_availability'].items():
        if entry['status'] != 'AVAILABLE':
            continue  # incomplete diagnostic: no imputation, interpolation or comparison across missing epochs
        v = h[col]
        n = min(5, max(1, len(v)//3))
        end = float(v.tail(n).mean())
        previous = float(v.iloc[-2*n:-n].mean()) if len(v) >= 2*n else end
        result['losses'][col] = dict(first_window=float(v.head(n).mean()), previous_window=previous, final_window=end,
                                    window=n, late_direction='up' if end > previous else 'down_or_equal')
    for train in [c for c in result['losses'] if c.startswith('train/') or c == 'train_loss']:
        val = 'val/' + train.split('/', 1)[1] if '/' in train else 'val_loss'
        if val in result['losses'] and result['losses'][train]['final_window'] < result['losses'][train]['previous_window'] and result['losses'][val]['final_window'] > result['losses'][val]['previous_window']:
            result['flags'].append(dict(kind='POSSIBLE_LATE_OVERFIT', train_component=train, validation_component=val,
                                        conclusion='Senal orientativa con ventanas completas, no prueba unica de sobreajuste.'))
    raw_objective = next(c for c in YOLO_METRIC_COLUMNS['map50_95'] if c in h)
    result.update(objective=objective, best_epoch=checkpoint_epoch, best_value=float(values.max()), last_value=float(h[raw_objective].iloc[-1]),
                  maximum_metric_event=int(objective_history.loc[maximum_index, 'epoch']), selected_checkpoint_epoch=selected_epoch,
                  best_checkpoint_training_epoch=checkpoint_epoch, final_callback=final_callback,
                  maximum_metric_epoch=(None if final_callback and maximum_index>=len(h) else int(h.loc[maximum_index,'epoch'])),
                  drop_best_to_last=float(values.max()-float(h[raw_objective].iloc[-1])),
                  metrics_at_same_best_epoch={name:metric_number(best, aliases) for name, aliases in YOLO_METRIC_COLUMNS.items()})
    result['overfit_limitations'] = ('No se puede comparar train/validation loss donde falten componentes. Se conservan las curvas y caida de mAP, variabilidad entre folds y brecha del checkpoint final; ninguna demuestra por si sola generalizacion.')
    return result


def inspect_existing_hpo_curves(study, cfg, spec, audit):
    """Inspect every COMPLETE fold and available PRUNED CSV; no training/publication."""
    audit['csv_errors'] = []
    audit['pruned_csv_diagnostics'] = []
    for trial in study.trials:
        if trial.state.name not in ('COMPLETE', 'PRUNED'):
            continue
        if trial.state.name == 'PRUNED' and not trial.user_attrs.get('run_dir'):
            audit['pruned_csv_diagnostics'].append(dict(trial=trial.number, status='ND', reason='PRUNED sin run_dir; no se exige entrenamiento completo.'))
            continue
        folder = allowed_trial_folder(cfg, spec, trial)
        for fold in range(3):
            if trial.state.name == 'PRUNED':
                path = folder/f'fold{fold}/results.csv'
                item = dict(trial=trial.number, fold=fold, path=str(path))
                if path.is_file():
                    try:
                        _, info = yolo_csv_inspection(path, complete=False)
                        item.update(info)
                    except Stop as e:
                        item.update(status='PRUNED_CSV_ANOMALY', error=str(e), selection_eligible=False)
                else:
                    item.update(status='NOT_EXPECTED_COMPLETED', reason='PRUNED puede no iniciar/completar este fold.')
                audit['pruned_csv_diagnostics'].append(item)
                continue
            try:
                epoch = read(folder/f'fold{fold}/completed_fold.json')['row']['best_epoch']
                row = curve_evidence(folder/f'fold{fold}/completed_history.csv', spec.is_detector, selected_epoch=epoch)
                row.update(trial=trial.number, fold=fold, state='COMPLETE')
                audit['curve_diagnostics'].append(row)
                emit('CSV_FOLD_VERIFICADO', trial=trial.number, fold=fold, warnings=row.get('warnings', []),
                     metrics=row['metrics_at_same_best_epoch'], training=False)
            except Stop as e:
                audit['csv_errors'].append(dict(trial=trial.number, fold=fold, error=str(e), action='Conservar originales; restaurar solo copia identica verificada si existe. No reentrenar ni aprobar.'))
    if audit['csv_errors']:
        raise Stop('CSV/evidencia indispensable invalida: ' + json.dumps(audit['csv_errors'], ensure_ascii=False))


def csv_review_summary(audit):
    curves = audit.get('curve_diagnostics', [])
    affected = [dict(path=r['path'], trial=r.get('trial'), fold=r.get('fold'), warnings=r.get('warnings', []),
                     metrics=r.get('metrics_at_same_best_epoch'), limitations=r.get('overfit_limitations'))
                for r in curves if r.get('warnings')]
    return dict(folds_checked=len(curves), folds_with_warnings=len(affected), affected_folds=affected,
                critical_errors=audit.get('csv_errors', []), pruned_csvs=audit.get('pruned_csv_diagnostics', []),
                policy='Loss de validacion auxiliar: ND/advertencia explicita; train y objetivo invalidos bloquean. APROBAR sigue siendo una decision humana, no un resultado automatico.')


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


def job_folder(cfg, spec, job):
    from utils.config_loader import cfg_path
    j=pure_job(job);root=cfg_path(cfg,'paths.runs_dir')/spec.slug
    return {'repeat':lambda:root/'robustness'/j['payload']['config_id']/('seed_'+str(j['payload']['seed'])),
            'aggregate':lambda:root/'robustness/aggregate','outer_train':lambda:root/'final',
            'outer_eval':lambda:root/'outer_validation','thresholds':lambda:root/'thresholds'}[j['kind']]()


def verify_checkpoint_entries(cfg,spec,folder):
    from utils.dataset import recorded_dataset_hash
    from utils.config_loader import cfg_path
    from utils.splits import current_split_hash
    code = __import__('utils.hashing',fromlist=['compute_code_hash']).compute_code_hash(__import__('utils.config_loader',fromlist=['PROJECT_ROOT']).PROJECT_ROOT)
    files=[Path(folder)/'checkpoint_metadata.json',*sorted((Path(folder)/'previous_attempts').glob('*/checkpoint_metadata.json'))]
    expected=dict(code_hash=code,dataset_hash=recorded_dataset_hash(cfg_path(cfg,'paths.artifacts_dir')),
                  split_hash=current_split_hash(cfg_path(cfg,'paths.artifacts_dir'),cfg['outer_split']['manifest_name']))
    found={}
    for p in files:
        if not p.is_file():continue
        row=read(p)
        if any(row.get(k)!=v for k,v in expected.items()):continue
        for entry in row.get('checkpoints',[]):
            role=entry.get('role','')
            if not re.fullmatch('fold[012]_best',role):continue
            if not re.fullmatch('[a-f0-9]{64}',str(entry.get('sha256',''))):raise Stop('Hash de checkpoint invalido: '+str(p))
            if entry.get('weights_discarded') is False:
                path=entry.get('retained_path') or entry.get('path')
                if not path or not Path(path).is_file() or digest(path)!=entry['sha256']:raise Stop('Pesos retenidos alterados/ausentes: '+str(path))
            if role in found and found[role]['sha256']!=entry['sha256']:raise Stop('Metadata de pesos contradictoria: '+role)
            found[role]=entry
    return found


def validate_repeat(cfg,spec,job,cand):
    from utils.architectures import section_for
    from utils.experiments import fingerprint
    folder=job_folder(cfg,spec,job);j=pure_job(job);result=read(folder/'result.json');sec=section_for(spec,cfg)
    if result.get('config_id')!=cand['config_id'] or result.get('hp')!=cand['hp'] or int(result.get('seed',-1))!=j['payload']['seed']:
        raise Stop('Resultado de robustez de otra configuracion/semilla: '+str(folder))
    if result.get('experiment_fingerprint')!=fingerprint(cfg):raise Stop('Robustez de otra ronda')
    scores=[];curves=[];best_epochs=[]
    actual={p.name for p in folder.glob('fold*') if p.is_dir() and re.fullmatch(r'fold\d+',p.name)}
    if actual!={'fold0','fold1','fold2'}:raise Stop('Robustez con folds ausentes/adicionales: '+str(folder))
    for fold in range(3):
        sig=fold_signature(fingerprint(cfg),spec.slug,cand['hp'],j['payload']['seed'],fold,sec['max_epochs'],sec['early_stopping_patience'])
        receipt,history=verify_fold(folder/f'fold{fold}',sig,fold);scores.append(float(receipt['row']['score']))
        selected_epoch=int(receipt['row']['best_epoch'])
        if not 1<=selected_epoch<=len(history):raise Stop('Epoca seleccionada fuera del historial')
        best_epochs.append(selected_epoch)
        curve=curve_evidence(folder/f'fold{fold}/completed_history.csv',spec.is_detector,selected_epoch=selected_epoch)
        if not math.isclose(curve['best_value'],scores[-1],abs_tol=1e-5):raise Stop('Curva/score de robustez no coinciden')
        curves.append(curve)
    if result.get('fold_scores')!=scores or result.get('best_epochs')!=best_epochs or not math.isclose(float(result['score']),sum(scores)/3,abs_tol=1e-8):raise Stop('Media/epocas robustez incorrectas')
    entries=verify_checkpoint_entries(cfg,spec,folder)
    if set(entries)!={'fold0_best','fold1_best','fold2_best'}:raise Stop('Faltan recibos de pesos de robustez: '+str(folder))
    training_epochs = [c.get('best_checkpoint_training_epoch', e) for c,e in zip(curves,best_epochs)]
    return dict(config_id=cand['config_id'],seed=j['payload']['seed'],scores=scores,mean=sum(scores)/3,best_epochs=training_epochs,
                legacy_recorded_best_epochs=best_epochs,curves=curves,
                result_sha256=digest(folder/'result.json'),checkpoints=entries)


def validate_aggregate(cfg,spec,cands,jobs):
    import numpy as np
    from utils.hpo_results import load_best_config
    checks=[];mapping={c['config_id']:c for c in cands}
    for j in map(pure_job,jobs):
        if j['kind']=='repeat':checks.append(validate_repeat(cfg,spec,j,mapping[j['payload']['config_id']]))
    rows=[]
    for c in cands:
        res=[r for r in checks if r['config_id']==c['config_id']]
        if sorted(r['seed'] for r in res)!=sorted(cfg['robustness']['seeds']):raise Stop('Robustez no cubre todas las seeds')
        vals=[r['mean'] for r in res];epochs=[epoch for r in res for epoch in r['best_epochs']]
        rows.append(dict(config_id=c['config_id'],mean=float(np.mean(vals)),sd=float(np.std(vals,ddof=1)),
                         hpo_trial_number=c['trial_number'],final_epochs=max(1,int(round(float(np.mean(epochs)))))))
    expected=sorted(rows,key=lambda r:(-r['mean'],r['sd'],r['hpo_trial_number']))[0];best=load_best_config(cfg,spec)
    if best['config_id']!=expected['config_id'] or best['hp']!=mapping[expected['config_id']]['hp'] or best['final_epochs']!=expected['final_epochs']:
        raise Stop('La agregacion no aplica la seleccion nativa media/SD/numero trial/epocas')
    if not math.isclose(best['score_mean'],expected['mean'],abs_tol=1e-8):raise Stop('Media agregada incorrecta')
    return dict(candidates=rows,selected=best,repeat_evidence=checks)


def validate_final(cfg,spec):
    from utils.hpo_results import final_checkpoint,load_best_config
    from utils.config_loader import cfg_path
    from utils.dataset import recorded_dataset_hash
    from utils.splits import current_split_hash
    best=load_best_config(cfg,spec);cp=final_checkpoint(cfg,spec);meta=read(cp.with_name('final_meta.json'))
    from utils.experiments import fingerprint
    from utils.hashing import compute_code_hash
    from utils.config_loader import PROJECT_ROOT
    expected=dict(slug=spec.slug,config_id=best['config_id'],hp=best['hp'],epochs=best['final_epochs'],seed=cfg['reproducibility']['seed'],
                  dataset_hash=recorded_dataset_hash(cfg_path(cfg,'paths.artifacts_dir')),split_hash=current_split_hash(cfg_path(cfg,'paths.artifacts_dir'),cfg['outer_split']['manifest_name']),
                  code_hash=compute_code_hash(PROJECT_ROOT),early_stopping=False,epoch_selected_on_validation=False,
                  experiment_fingerprint=cfg.get('experiment',{}).get('cohort',{}).get(spec.slug,{}).get('fingerprint',fingerprint(cfg)))
    if not cp.is_file() or digest(cp)!=meta.get('sha256') or any(meta.get(k)!=v for k,v in expected.items()):raise Stop('Checkpoint definitivo incompatible/incompleto: '+str(cp))
    return dict(checkpoint=str(cp),sha256=digest(cp),metadata=meta)


def protect_existing_final(cfg,spec):
    """An old/incomplete final artifact is never an implicit overwrite target."""
    from utils.hpo_results import final_checkpoint
    path=final_checkpoint(cfg,spec);meta=path.with_name('final_meta.json')
    if not path.exists() and not meta.exists():return None
    try:return validate_final(cfg,spec)
    except (Stop,FileNotFoundError,KeyError,ValueError) as e:
        raise Stop('Checkpoint final anterior/incompleto/incompatible conservado; se prohibe sobrescribirlo: '+str(path)+'. '+str(e)) from e


def validate_external(cfg,spec,stage):
    from utils.hpo_results import stage_json_path
    from utils.experiments import fingerprint
    final=validate_final(cfg,spec);p=stage_json_path(cfg,stage,spec);row=read(p)
    expected_fp=cfg.get('experiment',{}).get('cohort',{}).get(spec.slug,{}).get('fingerprint',fingerprint(cfg))
    if row.get('experiment_fingerprint')!=expected_fp or row.get('checkpoint_sha256')!=final['sha256']:raise Stop('Resultado externo de otra ronda/checkpoint: '+str(p))
    for key in ('dataset_hash','split_hash','code_hash'):
        if row.get(key)!=final['metadata'].get(key):raise Stop('Resultado externo incompatible: '+key)
    expected_split='validation' if stage=='outer_validation' else 'outer_validation_20pct'
    if row.get('evaluated_on')!=expected_split or row.get('test_used') is True:raise Stop('Particion externa incorrecta o test utilizado')
    if stage=='outer_validation':
        key='map50_95' if spec.is_detector else 'average_precision_positive'
        if row.get(key) is None or not math.isfinite(float(row[key])):raise Stop('Metrica externa ausente/no finita: '+key)
        from utils.hpo_results import predictions_path
        predictions=predictions_path(cfg,spec)
        if not predictions.is_file():raise Stop('Faltan predicciones crudas de validacion: '+str(predictions))
        if row.get('predictions_sha256')!=digest(predictions):raise Stop('Predicciones crudas alteradas: '+str(predictions))
        from utils.evidence import split_dir
        evidence_root=split_dir(cfg,spec,'validation');index=read(evidence_root/'artifact_index.json')
        if index.get('model')!=spec.name or index.get('split')!='validation' or not index.get('artifacts'):raise Stop('Indice de evidencia externa incompleto')
        for item in index['artifacts']:
            artifact=evidence_root/item['path']
            if not inside(artifact,evidence_root) or not artifact.is_file() or digest(artifact)!=item['sha256']:raise Stop('Artefacto externo alterado/ausente: '+str(artifact))
        report=dict(path=str(p),sha256=digest(p),metrics=row,predictions_sha256=digest(predictions))
        if not spec.is_detector:report['classification_metrics']=classification_prediction_audit(predictions,.5)
        return report
    if row.get('conf' if spec.is_detector else 'threshold') is None:raise Stop('Umbral externo ausente')
    report=dict(path=str(p),sha256=digest(p),metrics=row)
    if not spec.is_detector:
        from utils.hpo_results import predictions_path
        report['classification_metrics']=classification_prediction_audit(predictions_path(cfg,spec),float(row['threshold']))
    return report


def technical_job_result(cfg,spec,job,cands,jobs):
    j=pure_job(job)
    if j['kind']=='repeat':return validate_repeat(cfg,spec,j,next(c for c in cands if c['config_id']==j['payload']['config_id']))
    if j['kind']=='aggregate':return validate_aggregate(cfg,spec,cands,jobs)
    if j['kind']=='outer_train':
        final=validate_final(cfg,spec);train_path=job_folder(cfg,spec,j)/'train_evaluation.json'
        train=read(train_path)
        if train.get('checkpoint_sha256')!=final['sha256']:raise Stop('Falta evaluacion train del mismo checkpoint definitivo')
        return dict(**final,train_evaluation=train)
    return validate_external(cfg,spec,'outer_validation' if j['kind']=='outer_eval' else 'inference_thresholds')


def actual_review(queue,job,evidence,summary):
    """A real operator's acceptance; no programmatic imitation of human review."""
    j=pure_job(job)
    if j['status']=='COMPLETE':return
    if j['status']!='PENDING_REVIEW':raise Stop('No se puede revisar el estado '+j['status'])
    emit('REVISION_TECNICA_OK_REVISION_HUMANA_PENDIENTE',job=j['job_id'],evidence=str(evidence),summary=summary)
    if not sys.stdin.isatty():raise ReviewRequired('La cola exige revision humana de '+j['kind']+'; evidencia lista: '+str(evidence))
    print('Revise el resumen mostrado. APROBAR acepta esta etapa y sus limitaciones tecnicas; no certifica inspeccion visual ni independencia biologica.',flush=True)
    if input('Decision [APROBAR para continuar; otra respuesta conserva PENDING_REVIEW]: ').strip()!='APROBAR':
        raise ReviewRequired('Revision no aprobada; estado original conservado')
    acceptance=Path(evidence).with_name(Path(evidence).stem+'-aceptacion.json')
    atomic_json(acceptance,dict(decision='APPROVED_BY_PRESENT_OPERATOR',operator=os.environ.get('USERNAME',''),
                              job=j['job_id'],evidence=str(evidence),evidence_sha256=digest(evidence),utc=utc(),
                              human_visual_inspection_claimed=False,summary=summary,executor_sha256=EXECUTOR_SHA256))
    queue.approve(j['job_id'],operator=os.environ.get('USERNAME','OPERADOR_PRESENTE'),evidence=str(acceptance))


def seed_reuse(cfg,spec,job,cands,study):
    """Reuse identical HPO fold computations for seed 42; never pretend a new seed run."""
    from utils.architectures import section_for
    from utils.experiments import fingerprint
    j=pure_job(job);seed=j['payload']['seed']
    if seed!=int(cfg['reproducibility']['seed']):return []
    cand=next(c for c in cands if c['config_id']==j['payload']['config_id']);trial=next(t for t in study.trials if t.number==cand['trial_number'])
    source=allowed_trial_folder(cfg,spec,trial);target=job_folder(cfg,spec,j);sec=section_for(spec,cfg);copied=[]
    entries=verify_checkpoint_entries(cfg,spec,source)
    for fold in range(3):
        sig=fold_signature(fingerprint(cfg),spec.slug,cand['hp'],seed,fold,sec['max_epochs'],sec['early_stopping_patience'])
        verify_fold(source/f'fold{fold}',sig,fold)
        if (target/f'fold{fold}/completed_fold.json').is_file():verify_fold(target/f'fold{fold}',sig,fold);continue
        copy_tree_safe(source/f'fold{fold}',target/f'fold{fold}');copied.append(fold)
    if copied:
        # A separate import receipt survives RunContext metadata refreshes.
        atomic_json(target/'reused_hpo_checkpoints.json',dict(source=str(source),source_trial=trial.number,
                    experiment_fingerprint=fingerprint(cfg),seed=seed,checkpoints=list(entries.values()),
                    interpretation='Reutilizacion determinista; no es una repeticion independiente adicional.',executor_sha256=EXECUTOR_SHA256))
        emit('FOLDS_HPO_REUTILIZADOS',config_id=cand['config_id'],seed=seed,folds=copied)
    elif all((target/f'fold{fold}/completed_fold.json').is_file() for fold in range(3)) and not (target/'reused_hpo_checkpoints.json').is_file():
        atomic_json(target/'reused_hpo_checkpoints.json',dict(source=str(source),source_trial=trial.number,
                    experiment_fingerprint=fingerprint(cfg),seed=seed,checkpoints=list(entries.values()),
                    interpretation='Importacion interrumpida verificada; no se repiten folds.',executor_sha256=EXECUTOR_SHA256))
    return copied


class PostHooks(RuntimeHooks):
    def evaluate(self,original,spec,hp,cfg,ctx,run_dir,**kwargs):
        # Native fold receipts remain in place across attempts. Load signed
        # checkpoint metadata BEFORE recording anything in the new RunContext.
        entries=verify_checkpoint_entries(cfg,spec,run_dir)
        reused=Path(run_dir)/'reused_hpo_checkpoints.json'
        if reused.is_file():
            from utils.experiments import fingerprint
            meta=read(reused)
            if meta['experiment_fingerprint']!=fingerprint(cfg) or meta['seed']!=kwargs['seed']:raise RecoveryStop('Importacion HPO de otra ronda/seed')
            for e in meta['checkpoints']:entries.setdefault(e['role'],e)
        for fold in kwargs['folds']:
            if (Path(run_dir)/f'fold{fold}/completed_fold.json').is_file():
                e=entries.get(f'fold{fold}_best')
                if not e:raise RecoveryStop('Fold completado sin metadata de pesos: '+str(run_dir))
                ctx.record_checkpoint(f'fold{fold}_best',**{k:v for k,v in e.items() if k!='role'})
        def invoke(*a,**kw):
            self.context['source']=Path(run_dir)
            return original(*a,**kw)
        return super().evaluate(invoke,spec,hp,cfg,ctx,run_dir,**kwargs)

    def install_post(self,module):
        import utils.hpo_runner as runner
        original=runner.evaluate_config;detector=runner.train_detector;classifier=runner.train_classifier
        module.evaluate_config=lambda *a,**kw:self.evaluate(original,*a,**kw)
        runner.train_detector=lambda *a,**kw:self.detector(detector,*a,**kw)
        runner.train_classifier=lambda *a,**kw:self.classifier(classifier,*a,**kw)
        native_shutil=runner.shutil
        class Retain:
            def __getattr__(inner,name):return getattr(native_shutil,name)
            def rmtree(inner,path,*a,**kw):
                if Path(path).name=='weights' and self.context and inside(path,self.context['run_dir']):return
                return native_shutil.rmtree(path,*a,**kw)
        runner.shutil=Retain()


def child_entry(kind,root,native_args,state_dir):
    if kind in ('hpo','baseline'):
        raise Stop('HPO/baseline terminado: este ejecutor prohibe ejecutarlo otra vez en todas las PCs')
    os.chdir(root);sys.path.insert(0,str(Path(root)/'src'));state_dir.mkdir(parents=True,exist_ok=True)
    if kind=='worker':
        module=load_native(Path(root)/'src/orchestration/worker.py','_bt_post_worker');build=module.build_command;run=module._run_job
        ready=module.local_inputs_ready
        module.local_inputs_ready=lambda cfg,job,runs: (TARGET_JOB_ID is None or job.job_id==TARGET_JOB_ID) and ready(cfg,job,runs)
        def command(job,pc_id,config,sets,python=None,runs_dir=None):
            if job.kind not in POST_KINDS:raise Stop('El ejecutor no inicia baselines ni otras arquitecturas')
            original=build(job,pc_id,config,sets,python=python,runs_dir=runs_dir)
            return [original[0],str(Path(__file__).resolve()),'--executor-sha',EXECUTOR_SHA256,'--_stage',job.kind,
                    '--project',str(root),'--state-dir',str(state_dir),'--',*original[2:]]
        def record_owner(ctx,queue,job,args,log_dir):
            path=state_dir/'owners'/(canonical(job.job_id)[:16]+'.json')
            atomic_json(path,dict(job_id=job.job_id,attempt=job.attempt,owner=own_process(),utc=utc(),executor_sha256=EXECUTOR_SHA256))
            return run(ctx,queue,job,args,log_dir)
        module.build_command=command;module._run_job=record_owner
        sys.argv=[str(Path(root)/'src/orchestration/worker.py'),*native_args]
        return module.run_cli(module.SCRIPT,module.main,description=module.__doc__ or module.SCRIPT,add_arguments=module.add_arguments)
    parser=argparse.ArgumentParser(add_help=False);parser.add_argument('--config');parser.add_argument('--set',action='append',default=[]);parser.add_argument('--architecture')
    a,_=parser.parse_known_args(native_args)
    from utils.config_loader import load_config
    from utils.architectures import get_arch
    cfg=load_config(a.config,a.set)
    if kind not in POST_KINDS:raise Stop('Etapa no admitida')
    spec=get_arch(a.architecture);path=Path(root)/'src'/POST_FILES[kind];module=load_native(path,'_bt_post_'+kind)
    hooks=PostHooks(cfg,spec,state_dir)
    if kind in ('repeat','aggregate'):hooks.install_post(module)
    if kind=='repeat':
        original_run_one=module.run_one
        def documented_run_one(parent,selected,configuration,candidate,seed,**kw):
            from utils.config_loader import cfg_path
            folder=cfg_path(configuration,'paths.runs_dir')/selected.slug/'robustness'/candidate['config_id']/('seed_'+str(seed))
            existed=(folder/'result.json').is_file();res=original_run_one(parent,selected,configuration,candidate,seed,**kw)
            if not existed:
                from utils.io import write_json
                training_seconds=sum(float(read(folder/f'fold{f}/completed_fold.json')['row'].get('train_seconds',0)) for f in range(3))
                res=dict(res,original_fold_training_seconds=training_seconds,current_evaluation_seconds=res['train_seconds'],
                         reused_from_hpo=read(folder/'reused_hpo_checkpoints.json') if (folder/'reused_hpo_checkpoints.json').is_file() else None,
                         executor_sha256=EXECUTOR_SHA256)
                write_json(folder/'result.json',res)
            return res
        module.run_one=documented_run_one
    if kind=='aggregate':
        original_aggregate=module.aggregate
        source_cfg=load_config(a.config)
        baseline_summary=__import__('utils.config_loader',fromlist=['cfg_path']).cfg_path(source_cfg,'paths.runs_dir')/spec.slug/'baseline/summary.json'
        def aggregate_with_baseline(*pos,**kw):
            original_read = module.read_json
            epoch_corrections = []
            def interpreted_result(path):
                row = original_read(path)
                path = Path(path)
                if (spec.is_detector and spec.family != 'rtdetr' and path.name=='result.json'
                        and path.parent.name.startswith('seed_') and path.parent.parent.name.startswith('cfg_')
                        and inside(path, pos[4])):
                    corrected = []
                    for fold, old_epoch in enumerate(row['best_epochs']):
                        curve = curve_evidence(path.parent/f'fold{fold}/completed_history.csv',True,old_epoch)
                        corrected.append(curve['best_checkpoint_training_epoch'])
                    if corrected != row['best_epochs']:
                        epoch_corrections.append(dict(source=str(path),sha256=digest(path),
                            recorded_callback_events=list(row['best_epochs']),training_checkpoint_epochs=corrected,
                            scores_unchanged=True,source_modified=False))
                        row = dict(row,best_epochs=corrected)
                return row
            module.read_json = interpreted_result
            try:result=original_aggregate(*pos,**kw)
            finally:module.read_json=original_read
            changed = False
            if result.get('baseline_score_mean') is None and baseline_summary.is_file():
                result['baseline_score_mean']=read(baseline_summary)['score_mean']
                result['baseline_source_sha256']=digest(baseline_summary)
                changed = True
            if epoch_corrections:
                result['post_training_callback_epoch_interpretations']=epoch_corrections
                result['final_epoch_rule']='Media nativa de mejores epocas reales; final_eval no cuenta como entrenamiento'
                changed = True
            if changed:
                from utils.io import write_json
                from utils.hpo_results import best_config_path
                write_json(best_config_path(cfg,spec),result)
                summary_path=pos[4]/'summary.json';summary=read(summary_path);summary['selected']=result;write_json(summary_path,summary)
            return result
        module.aggregate=aggregate_with_baseline
    if kind=='outer_train':install_outer_recovery(module,cfg,spec,state_dir,hooks)
    sys.argv=[str(path),*native_args]
    layout=module._layout if kind in ('repeat','aggregate') else module.layout_for({'outer_train':'final','outer_eval':'outer_validation','thresholds':'thresholds'}[kind])
    return module.run_cli(module.SCRIPT,module.main,description=module.__doc__ or module.SCRIPT,add_arguments=module.add_arguments,run_layout=layout)


def install_outer_recovery(module,cfg,spec,state_dir,hooks):
    """Preserve optimizer snapshots and use only native resume paths/identities."""
    from utils.config_loader import cfg_path
    from utils.hpo_results import load_best_config,final_checkpoint
    best=load_best_config(cfg,spec);folder=cfg_path(cfg,'paths.runs_dir')/spec.slug/'final';seed=int(cfg['reproducibility']['seed'])
    original_main=module.main
    def main_with_train_metrics(ctx,args):
        original_main(ctx,args)
        target=ctx.run_dir/'train_evaluation.json'
        if not target.is_file():
            from utils.prepared import load_prepared_data
            from utils.guards import GpuJobLock
            from utils.io import write_json
            import importlib
            evaluator=importlib.import_module('10_evaluate_outer_validation');fn=evaluator.evaluate_detector if spec.is_detector else evaluator.evaluate_classifier
            with GpuJobLock(cfg,ctx.pc_id,job='train_metrics_recovery:'+spec.slug):
                payload=fn(ctx,spec,load_prepared_data(cfg,with_folds=True),best,final_checkpoint(cfg,spec),split='train',export_evidence=False,
                           **({} if spec.is_detector else {'gradcam':False}))
            write_json(target,payload)
    module.main=main_with_train_metrics
    if protect_existing_final(cfg,spec) is not None:return  # valid reuse; old/different artifacts cannot be overwritten
    candidates=[]
    for cp,meta in checkpoint_candidates(folder):
        validate_checkpoint(cp,spec.slug,cfg,seed=seed,hp=best['hp'])
        candidates.append((int(meta.get('extra',{}).get('epoch',0)),cp))
    selected=max(candidates,key=lambda t:t[0]) if candidates else None
    if selected and selected[0]>=best['final_epochs']:
        raise RecoveryStop('Outer train termino las epocas sin publicar final.pt; checkpoint conservado. Requiere recuperacion de publicacion, no reentrenamiento.')
    runtime=copy.deepcopy(cfg)
    section='detector' if spec.is_detector else 'classifier'
    if selected:
        cp=selected[1];runtime[section]['resume_checkpoint']=str(cp)
        hooks.event('OUTER_CHECKPOINT_RESUME',checkpoint=str(cp),epoch=selected[0],sha256=digest(cp))
    detector=module.train_detector;classifier=module.train_classifier
    if spec.is_detector:
        def train(*a,**kw):
            # Reuse proven retention logic; fixed-epoch final path is stable.
            hooks.context=dict(source=None,run_dir=folder,options=dict(seed=seed),hp=best['hp'])
            try:return hooks.detector(detector,a[0],a[1],runtime,**kw)
            finally:hooks.context=None
        module.train_detector=train
        import utils.detector_train as native_detector
        factory=native_detector._new_model
        def verbose_factory(*a,**kw):
            model,sha=factory(*a,**kw)
            def progress(tr):
                try:losses={k:float(v) for k,v in tr.label_loss_items(tr.tloss).items()}
                except (AttributeError,TypeError):losses={}
                emit('EPOCA_OUTER_TRAIN',architecture=spec.slug,config_id=best['config_id'],epoch=int(tr.epoch)+1,
                     epochs=best['final_epochs'],training_losses=losses,validation='NO_SE_UTILIZA_PARA_ELEGIR_EPOCA')
            model.add_callback('on_train_epoch_end',progress);return model,sha
        native_detector._new_model=verbose_factory
    else:
        def train(*a,**kw):return classifier(a[0],a[1],runtime,*a[3:],**kw)
        module.train_classifier=train


def launch_visible(root,cfg,spec,config,sets,kind,state_dir,audit_dir,target_job=None):
    """Tail the deterministic native job log WHILE RUNNING (log_path is set only at finish)."""
    import threading
    from utils.config_loader import cfg_path
    if digest(__file__)!=EXECUTOR_SHA256:raise Stop('El ejecutor cambio durante la ejecucion')
    cmd=[sys.executable,str(Path(__file__).resolve()),'--executor-sha',EXECUTOR_SHA256,'--_stage','worker','--project',str(root),'--state-dir',str(state_dir)]
    if target_job:cmd+=['--target-job',target_job]
    cmd+=['--','--config',str(config)]
    for s in sets:cmd+=['--set',s]
    cmd+=['--pc-id',cfg['compute']['pc_id'],'--architectures',spec.slug,'--kinds',kind,'--once']
    emit('EJECUTANDO_ETAPA',kind=kind,model=spec.slug,command=subprocess.list2cmdline(cmd))
    proc=subprocess.Popen(cmd,cwd=root,stdout=subprocess.PIPE,stderr=subprocess.STDOUT,text=True,encoding='utf-8',errors='replace',bufsize=1,
                          creationflags=subprocess.CREATE_NEW_PROCESS_GROUP if os.name=='nt' else 0)
    import psutil
    atomic_json(Path(audit_dir)/('worker-'+kind+'-owner.json'),dict(owner=dict(hostname=socket.gethostname(),pid=proc.pid,created=psutil.Process(proc.pid).create_time()),command=cmd))
    output=Path(audit_dir)/('worker-'+kind+'-'+uuid.uuid4().hex[:6]+'.log')
    def pump():
        with output.open('w',encoding='utf-8') as f:
            for line in proc.stdout:f.write(line);f.flush();print(line,end='',flush=True)
    thread=threading.Thread(target=pump,daemon=True);thread.start();positions={};seen=set()
    def tail():
        for row in jobs_read(cfg,spec.slug):
            if row['kind']!=kind or row['status'] not in ('RUNNING','COMPLETE','PENDING_REVIEW','PUBLICATION_PENDING','FAILED'):continue
            token=(row['job_id'],row['attempt'])
            if row['status']!='RUNNING' and token not in seen:continue
            seen.add(token)
            p=cfg_path(cfg,'paths.queue_dir')/'logs'/(row['job_id'].replace(':','_').replace('/','_')+f"_a{row['attempt']}.log")
            if not p.is_file():continue
            with p.open('rb') as f:f.seek(positions.get(str(p),0));data=f.read();positions[str(p)]=f.tell()
            if data:print(data.decode('utf-8',errors='replace'),end='',flush=True)
    try:
        while proc.poll() is None:tail();time.sleep(.75)
        tail();thread.join(timeout=5)
    except KeyboardInterrupt:
        try:
            proc.send_signal(__import__('signal').CTRL_BREAK_EVENT if os.name=='nt' else __import__('signal').SIGINT);proc.wait(timeout=20)
        except (OSError,subprocess.TimeoutExpired):emit('PROCESO_AUN_ACTIVO',pid=proc.pid)
        raise
    if proc.returncode:raise Stop(f'Worker {kind}: codigo {proc.returncode}; evidencia {output}')


def external_authorization(cfg,spec,evidence):
    from utils.experiments import external_ledger,fingerprint,authorize_external
    p=external_ledger(cfg);ledger=read(p) if p.exists() else {};row=ledger.get(spec.slug)
    if row:
        if row.get('fingerprint')!=fingerprint(cfg):raise Stop('Autorizacion externa pertenece a otra ronda; no se sobrescribe')
        return row
    emit('AUTORIZACION_EXTERNA_REQUERIDA',evidence=str(evidence),model=spec.slug,
         operation='Evaluar candidato fijado sobre validation externo y calibrar inferencia; test permanece cerrado.')
    if not sys.stdin.isatty():raise ReviewRequired('Falta autorizacion predeclarada para validacion externa')
    if input('Escriba AUTORIZAR para esa operacion unica: ').strip()!='AUTORIZAR':raise ReviewRequired('Validacion externa no autorizada')
    acceptance=Path(evidence).with_name('autorizacion-externa.json')
    atomic_json(acceptance,dict(operator=os.environ['USERNAME'],scope='outer_validation + threshold_tuning de candidato fijado; sin acceso al test',
                              report=str(evidence),report_sha256=digest(evidence),utc=utc()))
    return authorize_external(cfg,operator=os.environ['USERNAME'],evidence=str(acceptance))


def recovery_or_reuse(queue,cfg,spec,job,cands,jobs,state_dir,audit_dir):
    """Recover publication without running a stage again; retries require dead-owner proof."""
    from orchestration.exchange import publish_architecture
    from utils.config_loader import cfg_path
    j=pure_job(job);kind=j['kind'];valid=None
    try:valid=technical_job_result(cfg,spec,j,cands,jobs)
    except (Stop,FileNotFoundError,KeyError,ValueError) as exc:
        folder=job_folder(cfg,spec,j)
        if kind=='repeat' and (folder/'result.json').is_file():raise Stop('Resultado publicado de robustez invalido; no se sobrescribe: '+str(exc)) from exc
        if kind=='aggregate':
            from utils.hpo_results import best_config_path
            if best_config_path(cfg,spec).is_file():raise Stop('Configuracion final anterior/incompatible conservada; no se sobrescribe una agregacion no verificada: '+str(exc)) from exc
        if kind in ('outer_eval','thresholds'):
            stage='outer_validation' if kind=='outer_eval' else 'inference_thresholds'
            if (cfg_path(cfg,'paths.artifacts_dir')/stage/(spec.slug+'.json')).is_file():raise Stop('Resultado externo existente invalido; no se repite: '+str(exc)) from exc
    if valid is not None and j['status'] in ('PENDING','RUNNING','FAILED','PUBLICATION_PENDING'):
        atomic_json(Path(audit_dir)/('recuperacion-'+canonical(j['job_id'])[:8]+'.json'),dict(job=j,validated=valid,utc=utc()))
        publish_architecture(cfg,cfg['compute']['pc_id'],spec.slug,cfg_path(cfg,'paths.runs_dir'),stage=kind)
        queue.computed(j['job_id'],published=True);emit('RESULTADO_REUTILIZADO_SIN_ENTRENAR',job=j['job_id']);return True
    if j['status'] in ('COMPLETE','PENDING_REVIEW'):
        if valid is None:raise Stop('Job completado/revisable sin evidencia valida: '+j['job_id'])
        return False
    if j['status']=='PUBLICATION_PENDING':raise Stop('Computo declarado publicado pendiente pero resultado incompleto; no se reentrena: '+j['job_id'])
    if j['status']=='RUNNING':
        owner_file=Path(state_dir)/'owners'/(canonical(j['job_id'])[:16]+'.json')
        if not owner_file.is_file():raise Stop('RUNNING sin propietario verificable; no se recupera por antiguedad: '+j['job_id'])
        owner=read(owner_file)
        if owner['job_id']!=j['job_id'] or owner['attempt']!=j['attempt'] or owner_alive(owner['owner']):raise Stop('YA_ACTIVO: propietario de job vigente/no verificable')
    if j['status'] in ('RUNNING','FAILED'):
        if int(j['attempt'])>=int(cfg.get('operations',{}).get('max_job_attempts',3)):raise Stop('Limite de intentos: '+j['job_id'])
        if kind in ('outer_eval','thresholds'):
            from utils.experiments import external_ledger,retry_external
            stage='outer_validation' if kind=='outer_eval' else 'threshold_tuning';ledger_path=external_ledger(cfg)
            ledger=read(ledger_path) if ledger_path.is_file() else {};use=ledger.get(spec.slug,{}).get('uses',{}).get(stage)
            if use:
                from utils.hpo_results import final_checkpoint
                if use.get('checkpoint_sha256')!=digest(final_checkpoint(cfg,spec)) or use.get('state')!='STARTED':raise Stop('Operacion externa consumida/incompatible; no se repite')
                receipt=Path(audit_dir)/('retry-externo-'+kind+'.json');atomic_json(receipt,dict(job=j,proof='Locks libres y ausencia comprobada de procesos Python',utc=utc()))
                retry_external(cfg,stage,operator='AUTOMATIC_TECHNICAL_RECOVERY',evidence=str(receipt))
        queue.finish(j['job_id'],'PENDING');emit('JOB_RECUPERADO',job=j['job_id'],previous_state=j['status'])
    return False


def ensure_partial_resume(cfg,spec,job):
    """Refuse to replace trained but unsigned/incompatible partial work."""
    from utils.hpo_results import load_best_config
    import pandas as pd
    j=pure_job(job)
    if j['kind'] not in ('repeat','outer_train'):return
    folder=job_folder(cfg,spec,j);candidates=checkpoint_candidates(folder)
    if j['kind']=='repeat':
        hp=None
        from utils.optuna_storage import load_study
        from utils.hpo_results import top_configs
        hp=next(c['hp'] for c in top_configs(load_study(spec,cfg),cfg['robustness']['top_k_configs']) if c['config_id']==j['payload']['config_id'])
        seed=j['payload']['seed'];subfolders=[folder/f'fold{i}' for i in range(3)]
    else:
        hp=load_best_config(cfg,spec)['hp'];seed=cfg['reproducibility']['seed'];subfolders=[folder/'train'] if spec.is_detector else [folder]
        if protect_existing_final(cfg,spec) is not None:return
    # Native classifier checkpoints are stored at the run root (not inside
    # foldN). An unsigned root checkpoint must never be silently overwritten.
    if not spec.is_detector and (j['kind']=='outer_train' or any(not (p/'completed_fold.json').is_file() for p in subfolders)):
        for cp in folder.glob('*.pt'):
            if not cp.with_suffix(cp.suffix+'.resume.json').is_file():raise Stop('Checkpoint parcial de clasificacion sin recibo firmado; se conserva sin reiniciar: '+str(cp))
            validate_checkpoint(cp,spec.slug,cfg,seed=seed,hp=hp)
    for sub in subfolders:
        if (sub/'completed_fold.json').is_file():continue
        history=sub/'results.csv'
        worked=history.is_file() and len(pd.read_csv(history))>0
        if not worked and not list(sub.glob('*.pt')) and not list((sub/'weights').glob('*.pt')):continue
        matching=[]
        for cp,meta in candidates:
            fold=meta.get('extra',{}).get('context',{}).get('fold')
            if j['kind']=='repeat' and sub.name!=f'fold{fold}':
                if spec.is_detector and cp.parent.parent.name==sub.name:pass
                else:continue
            validate_checkpoint(cp,spec.slug,cfg,seed=seed,hp=hp);matching.append(cp)
        if not matching:raise Stop('Trabajo parcial sin checkpoint propio firmado compatible; se conserva sin reiniciarlo: '+str(sub))


def write_pipeline_report(path,audit):
    """Always write a report, including blocked/partial states and unknowns."""
    import html
    audit['csv_validation']=csv_review_summary(audit)
    atomic_json(Path(path).with_suffix('.json'),audit)
    base=audit.get('baseline',{});hpo=audit.get('hpo',{});iso=audit.get('isolation',{})
    rows=[('Baseline CV',base.get('mean'),base.get('sd'),'Preservado; presupuesto baseline puede diferir del HPO'),
          ('Mejor HPO CV',hpo.get('best_value'),hpo.get('best_sd'),'Seleccionado usando folds internos; no es test')]
    agg=audit.get('aggregate',{}).get('selected',{})
    if agg:rows.append(('Robustez configuracion elegida',agg.get('score_mean'),agg.get('score_sd'),'Regla nativa: media entre semillas'))
    outer=audit.get('outer_validation',{}).get('metrics',{})
    if outer:rows.append(('Validacion externa',outer.get('map50_95',outer.get('average_precision_positive')),None,'Solo candidato fijado; no usar test para seleccion'))
    table=''.join('<tr>'+''.join('<td>'+html.escape(str(v))+'</td>' for v in row)+'</tr>' for row in rows)
    conclusion='No se ha determinado un ganador global ni un ganador comparable baseline vs HPO.'
    if audit.get('official_winner'):conclusion='Seleccion oficial previa al test: '+str(audit['official_winner'])
    diagnosis={'confirmados':iso.get('confirmed_relevant_conflicts',[]),'sospechas':iso.get('suspected_relevant_pairs',[]),
               'no_verificable':iso.get('physical_independence','Sin auditar'),'limitacion':iso.get('limitation')}
    title='Botrytis — '+str(audit.get('pc_id',''))+' — '+str(audit.get('model',''))
    page='<!doctype html><html lang="es"><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>'+html.escape(title)+'</title><style>body{max-width:1100px;margin:35px auto;padding:0 18px;font:16px/1.5 system-ui;color:#152a30}table{border-collapse:collapse;width:100%}td,th{border:1px solid #b9c9ce;padding:10px;text-align:left}pre{white-space:pre-wrap;overflow-wrap:anywhere;background:#f0f4f5;padding:14px}h2{margin-top:32px}</style><h1>'+html.escape(title)+'</h1><p>Estado: <b>'+html.escape(str(audit.get('status','EN_CURSO')))+'</b></p><p>'+html.escape(conclusion)+'</p><table><tr><th>Etapa</th><th>Objetivo</th><th>SD</th><th>Interpretacion</th></tr>'+table+'</table><p>Baseline y HPO se conservan. Si difieren epocas, seeds o seleccion, sus scores no demuestran superioridad causal. El framework fija un candidato HPO por arquitectura; no contempla entrenar/evaluar un baseline externo adicional automaticamente.</p><h2>Independencia y fuga</h2><pre>'+html.escape(json.dumps(diagnosis,ensure_ascii=False,indent=2,default=str))+'</pre><h2>Cola y dependencias reales</h2><pre>'+html.escape(json.dumps(audit.get('plan',[]),ensure_ascii=False,indent=2,default=str))+'</pre><h2>Curvas y sobreajuste</h2><p>Las curvas adjuntas proceden de los CSV originales. Losses auxiliares de validacion ausentes o no finitas se detallan como ND/advertencias; no se sustituyen por ceros. Los NaN/inf son anomalias que deben revisarse, no se presumen normales. Se exigen objetivo y losses de entrenamiento validos y concordancia con los recibos. Sin componentes completas no se comparan ventanas train/validation; el analisis de sobreajuste queda limitado. Las senales tardias son orientativas, no una prueba unica. Las diferencias de rendimiento train-validation se calculan solo con evaluaciones del mismo checkpoint.</p><pre>'+html.escape(json.dumps(audit.get('curve_diagnostics',[]),ensure_ascii=False,indent=2,default=str))+'</pre><h2>Evaluaciones, limitaciones y siguientes condiciones</h2><pre>'+html.escape(json.dumps({k:v for k,v in audit.items() if k not in ('curve_diagnostics','plan','baseline','isolation')},ensure_ascii=False,indent=2,default=str))+'</pre></html>'
    images=''.join('<figure><figcaption>'+html.escape(audit['curve_diagnostics'][i]['path'])+'</figcaption><img style="max-width:100%;height:auto" loading="lazy" src="'+html.escape(Path(p).relative_to(Path(path).parent).as_posix(),quote=True)+'" alt="Curvas originales"></figure>' for i,p in enumerate(audit.get('curve_figures',[])))
    page=page.replace('</html>',images+'</html>')
    atomic_bytes(path,page.encode('utf-8'))
    emit('REPORTE_GUARDADO',html=str(path),json=str(Path(path).with_suffix('.json')))


def export_curve_plots(audit,out):
    import pandas as pd
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    out=Path(out);out.mkdir(parents=True,exist_ok=True);files=[]
    for i,row in enumerate(audit.get('curve_diagnostics',[])):
        h=pd.read_csv(row['path']);h.columns=h.columns.str.strip();fig,axes=plt.subplots(1,2,figsize=(11,4))
        columns=list(row['losses'])
        for c,a in row.get('loss_availability',{}).items():
            if c in h and a.get('n_finite',0)>0 and c not in columns:columns.append(c)
        for c in columns:
            v=pd.to_numeric(h[c],errors='coerce');v=v.where(v.map(lambda x:pd.notna(x) and math.isfinite(x)))
            axes[0].plot(h.epoch,v,label=c+(' (parcial)' if row.get('loss_availability',{}).get(c,{}).get('status')=='PARTIAL' else ''))
        if row['losses']:axes[0].legend(fontsize=7)
        axes[0].set(title='Perdidas originales (componentes)',xlabel='Epoca')
        for c in ('metrics/mAP50(B)','metrics/mAP50-95(B)','metrics/precision(B)','metrics/recall(B)','val_ap','val_precision','val_recall'):
            if c in h:axes[1].plot(h.epoch,h[c],label=c)
        if axes[1].lines:axes[1].legend(fontsize=7)
        axes[1].set(title='Metricas de validacion',xlabel='Epoca');fig.tight_layout();p=out/f'c{i:03d}.png';fig.savefig(p,dpi=120);plt.close(fig);files.append(str(p))
    audit['curve_figures']=files


def global_requirements(cfg):
    from utils.architectures import archs
    from utils.config_loader import cfg_path
    art=cfg_path(cfg,'paths.artifacts_dir');available=[];missing=[]
    for spec in archs():
        paths=[art/'best_configs'/f'{spec.slug}.json',art/'checkpoints'/spec.slug/'final.pt',art/'outer_validation'/f'{spec.slug}.json',art/'inference_thresholds'/f'{spec.slug}.json']
        (missing if any(not p.is_file() for p in paths) else available).append(spec.slug)
    return dict(required_architectures=[s.slug for s in archs()],available_locally=available,missing_locally=missing,
                rules='12 requiere cuatro detectores; 13 tres clasificadores; 15 los siete candidatos y decisiones confirmadas. No se usa --allow-partial ni se habilita test antes del freeze.',
                rtdetr_r50='No se inicia entrenamiento ni se resuelve su bloqueo por iniciativa del ejecutor.')


def validate_global_result(cfg,filename):
    from utils.config_loader import cfg_path,PROJECT_ROOT
    from utils.hashing import compute_code_hash
    from utils.dataset import recorded_dataset_hash
    from utils.splits import current_split_hash
    art=cfg_path(cfg,'paths.artifacts_dir')
    expected=dict(code_hash=compute_code_hash(PROJECT_ROOT),dataset_hash=recorded_dataset_hash(art),split_hash=current_split_hash(art,cfg['outer_split']['manifest_name']))
    if filename.startswith(('12','13')):
        task='detection' if filename.startswith('12') else 'classification';target=art/('selected_detector.json' if task=='detection' else 'selected_classifier.json');row=read(target)
        if any(row.get(k)!=v for k,v in expected.items()) or row.get('test_used') is not False:raise Stop('Seleccion oficial obsoleta/no trazable')
        from utils.architectures import get_arch
        from utils.selection import rank_candidates,materialize_policy
        from utils.reporting import collect_validation_table
        spec=get_arch(row['architecture']);final=validate_final(cfg,spec);pol=cfg['selection']['detector' if task=='detection' else 'classifier']
        _,winner=rank_candidates(collect_validation_table(cfg,task),primary=pol['primary_metric'],min_recall=pol.get('minimum_recall' if task=='detection' else 'minimum_recall_positive'),
                                recall_col='recall',tie_margin=pol.get('tie_margin',0),tie_breakers=pol.get('tie_breakers',[]))
        _,policy_sha=materialize_policy(cfg)
        if winner!=spec.name or row['checkpoint_sha256']!=final['sha256'] or row['selection_policy_sha256']!=policy_sha:raise Stop('Seleccion no coincide con politica predeclarada/checkpoint')
    elif filename.startswith('14'):
        row=read(art/'pipeline_validation.json')
        if row.get('status')!='PASS' or not row.get('predicted_crops_used') or any(row.get(k)!=v for k,v in expected.items()):raise Stop('Pipeline integrado sin PASS compatible')
        thresholds=read(art/'final_thresholds.json')
        for role in ('detector','classifier'):
            selected=read(art/('selected_'+role+'.json'))
            if thresholds[role]['checkpoint_sha256']!=selected['checkpoint_sha256']:raise Stop('Umbrales de otro checkpoint')
    elif filename.startswith(('15','16')):
        from utils.freeze import load_freeze,verify_freeze
        from utils.reproducibility import collect_environment
        freeze=load_freeze(cfg);env=collect_environment(cfg=cfg,project_root=PROJECT_ROOT,pc_id=cfg['compute']['pc_id']);verify_freeze(cfg,freeze,expected,env)
        if filename.startswith('16'):
            target=art/'test_results.json';row=read(target);side=target.with_suffix('.sha256')
            if not side.is_file() or side.read_text().split()[0]!=digest(target):raise Stop('Test existente sin sello valido')
            if any(row.get(k)!=v for k,v in expected.items()):raise Stop('Test de otros datos/split/codigo')
            if row.get('thresholds_changed_after_test') is not False or row.get('comparison_use')!='descriptive_only; no selection or tuning from test':raise Stop('Test con uso no permitido')
            if row.get('freeze_sha256')!=digest(cfg_path(cfg,'test.freeze_manifest')):raise Stop('Test de otro freeze')
    elif filename.startswith('17'):
        import pandas as pd
        from utils.architectures import archs
        for task in ('detection','classification'):
            frame=pd.read_csv(art/'benchmark'/('latency_'+task+'.csv'))
            if 'slug' not in frame or set(frame.slug)!={s.slug for s in archs(task)}:raise Stop('Benchmark incompleto: '+task)
    elif filename.startswith('18'):
        row=read(art/'provenance.json')
        if any(row.get(k)!=v for k,v in expected.items()) or 'FAIL' in row.get('audit_status',{}).values() or not row.get('audit_status'):raise Stop('Auditoria cientifica final sin aprobar')
        if not (art/'master_results.csv').is_file():raise Stop('Falta master_results.csv')
    return dict(script=filename,status='VERIFIED',utc=utc())


def maybe_global(root,cfg,pc,short,audit):
    """Run the actual final scripts only for an already assembled, complete official cohort."""
    from utils.config_loader import load_config,cfg_path
    from utils.experiments import verify_round
    candidates=[];registry=Path(cfg['experiment']['registry_root'])/cfg['experiment']['experiment_id']
    for p in registry.glob('*/config.yaml'):
        c=load_config(p)
        if c.get('experiment',{}).get('cohort'):
            try:verify_round(c)
            except (RuntimeError,ValueError):continue
            candidates.append((p,c))
    audit['global_prerequisites']=global_requirements(cfg)
    if pc!='PC-01':audit['global_status']='COORDINACION_GLOBAL_PC01';return
    if len(candidates)!=1:
        audit['global_status']='BLOQUEADO_COHORTE_OFICIAL_AUSENTE_O_AMBIGUA';audit['cohort_configs']=[str(p) for p,_ in candidates];return
    config,cohort=candidates[0];req=global_requirements(cohort);audit['global_prerequisites']=req
    if req['missing_locally']:audit['global_status']='BLOQUEADO_CANDIDATOS_OFICIALES_INCOMPLETOS';return
    if audit.get('isolation',{}).get('suspected_relevant_pairs') or audit.get('isolation',{}).get('confirmed_relevant_conflicts'):
        audit['global_status']='BLOQUEADO_REVISION_INDEPENDENCIA_PENDIENTE';return
    if not cohort.get('decisions',{}).get('confirmed_by_user'):
        audit['global_status']='BLOQUEADO_DECISIONES_METODOLOGICAS_SIN_CONFIRMAR';return
    # Only an explicit assembled cohort links source-round fingerprints. No
    # copying models from unrelated rounds and no new architectures are trained.
    from utils.architectures import archs
    from utils.hpo_results import load_best_config
    for spec in archs():
        load_best_config(cohort,spec);validate_final(cohort,spec)
        for stage in ('outer_validation','inference_thresholds'):validate_external(cohort,spec,stage)
    runs=short/'g';art=cfg_path(cohort,'paths.artifacts_dir');sets=['paths.runs_dir='+str(runs),'paths.results_dir='+str(short/'e')]
    stages=[('12_select_detector.py','selected_detector.json'),('13_select_classifier.py','selected_classifier.json'),
            ('14_validate_integrated_pipeline.py','pipeline_validation.json'),('15_freeze_experiment.py','freeze_manifest.json'),
            ('16_evaluate_test.py','test_results.json'),('17_benchmark_latency.py','benchmark/latency_detection.csv'),('18_export_master_results.py','provenance.json')]
    audit['global_executed']=[]
    for filename,output in stages:
        target=art/output
        if target.is_file():
            # Existing final outputs are reused, never trigger another test.
            validate_global_result(cohort,filename)
            audit['global_executed'].append(dict(script=filename,status='EXISTING_OUTPUT_PRESERVED',path=str(target)));continue
        if filename.startswith('16') and not (art/'freeze_manifest.json').is_file():raise Stop('Test permanece bloqueado: falta freeze')
        if filename.startswith('16') and (art/'test_evaluation_started.json').exists():
            audit['global_status']='BLOQUEADO_TEST_INICIADO_SIN_RESULTADO_REQUIERE_REVISION';return
        cmd=[sys.executable,str(Path(root)/'src'/filename),'--config',str(config),'--pc-id',pc]
        for s in sets:cmd+=['--set',s]
        emit('EJECUTANDO_ETAPA_GLOBAL',script=filename)
        log=short/'audit'/('global-'+filename[:2]+'-'+uuid.uuid4().hex[:6]+'.log')
        with log.open('w',encoding='utf-8') as f:
            proc=subprocess.Popen(cmd,cwd=root,stdout=subprocess.PIPE,stderr=subprocess.STDOUT,text=True,encoding='utf-8',errors='replace')
            for line in proc.stdout:print(line,end='',flush=True);f.write(line);f.flush()
            rc=proc.wait()
        if rc:raise Stop('Etapa global '+filename+' fallo; ver '+str(log))
        if not target.is_file():raise Stop('Etapa global sin salida obligatoria: '+str(target))
        validate_global_result(cohort,filename)
        audit['global_executed'].append(dict(script=filename,status='COMPLETE',path=str(target),sha256=digest(target)))
    audit['official_winner']=dict(detector=read(art/'selected_detector.json')['architecture'],classifier=read(art/'selected_classifier.json')['architecture'],selection_used_test=False)
    audit['global_status']='PIPELINE_OFICIAL_COMPLETO'


def pipeline_parent(args):
    if os.name!='nt':raise Stop('La ejecucion operativa requiere Windows nativo; pruebas CPU disponibles por separado')
    root=find_project(args.project);os.chdir(root);py=root/'.venv/Scripts/python.exe'
    if not py.is_file():raise Stop('Falta entorno virtual: '+str(py))
    if Path(sys.executable).resolve()!=py.resolve():return subprocess.call([str(py),str(Path(__file__).resolve()),*sys.argv[1:]])
    sys.path.insert(0,str(root/'src'))
    from utils.config_loader import load_config,cfg_path
    from orchestration.machine import read_binding
    from utils.architectures import get_arch,section_for
    from utils.durable import FileLock
    from utils.guards import gpu_lock_path
    from orchestration.study_queue import StudyQueue
    from utils.optuna_storage import load_study,budget_used,study_db_path
    from utils.hpo_results import top_configs
    from utils.experiments import fingerprint
    base=load_config(root/'artifacts/v4/local.yaml');pc=base['compute']['pc_id']
    if pc != 'PC-02':raise Stop('Esta correccion es exclusiva de PC-02; no opera en PC-01 ni PC-03')
    if pc not in POST_HPO_REFERENCES or (args.expected_pc and args.expected_pc!=pc):raise Stop('Identidad esperada distinta de configuracion')
    machine,model,budget=MACHINES[pc];binding=read_binding(base)
    if binding['pc_id']!=pc or binding['machine_id'].casefold()!=machine or os.environ.get('USERNAME','').casefold()!=machine:raise Stop('Usuario/binding/PC no coinciden')
    emit('FASE_IDENTIFICACION_OK',pc=pc,architecture=model,project=str(root),policy='SOLO_POST_HPO')
    config,cfg=resolve_round(base,model,Path.home()/'bt-hoy-20261103');spec=get_arch(model)
    short,runs,results,state=make_short_paths(config,Path.home()/'bt-hoy-20261103');state=state/'post';state.mkdir(parents=True,exist_ok=True)
    evidence=short/'audit'/uuid.uuid4().hex[:10];evidence.mkdir(parents=True);report=short/'CONTINUACION.html'
    audit=dict(version=POST_VERSION,started_utc=utc(),pc_id=pc,model=model,config=str(config),project=str(root),
               executor_sha256=EXECUTOR_SHA256,status='IDENTIFICANDO',report=str(report),errors=[],baseline_preserved=True,
               hpo_execution_policy='PROHIBIDO: ambos estudios terminaron; solo trabajos posteriores.',human_approvals_fabricated=False)
    try:
        lock=gpu_lock_path(cfg,pc)
        with OSLock(lock.with_suffix('.final.guard')):
            with FileLock(lock.with_suffix('.worker.guard')),FileLock(lock.with_suffix('.guard')):
                active=python_processes()
                if active:audit.update(status='YA_ACTIVO',processes=active);return 0
                emit('FASE_COMPROBACIONES_CRITICAS',qa='Se reutiliza QA compatible; no se repite entrenamiento ni la suite QA')
                commit,code,exp=protocol_preflight(root,base,cfg,pc,spec,evidence,audit)
                baseline_dir=cfg_path(cfg,'paths.runs_dir')/model/'baseline'
                emit('FASE_REVISION_BASELINE_EXISTENTE',path=str(baseline_dir),training=False)
                audit['baseline']=inspect_baseline(cfg,spec,baseline_dir,code,commit,exp)
                audit['isolation']=partition_evidence(cfg,spec)
                if audit['isolation']['confirmed_relevant_conflicts']:raise Stop('Fuga de contenido/grupo confirmada; no se autoriza continuar ni se alteran splits')
                for fold in range(3):
                    p=baseline_dir/f'fold{fold}'/('results.csv' if spec.is_detector else 'completed_history.csv')
                    epoch=read(baseline_dir/f'fold{fold}/completed_fold.json')['row']['best_epoch']
                    audit.setdefault('curve_diagnostics',[]).append(curve_evidence(p,spec.is_detector,selected_epoch=epoch))
                jobs=list(map(pure_job,jobs_read(cfg,model)));baseline=one_job(jobs,'baseline');hpo=one_job(jobs,'hpo')
                snapshot_sqlite(cfg_path(cfg,'paths.queue_dir')/'queue.sqlite',evidence/'queue-before.sqlite')
                queue=StudyQueue.from_cfg(cfg)
                if baseline['status']!='COMPLETE':raise Stop('Baseline no COMPLETE; el estado confirmado no coincide. No se reentrena ni se cambia su revision: '+baseline['status'])
                sets=['paths.runs_dir='+str(runs),'paths.results_dir='+str(results),'experiment.trial_block_size='+str(budget)]
                effective=load_config(config,sets)
                if fingerprint(effective)!=exp:raise Stop('Las rutas cortas alteraron identidad cientifica')
                audit['effective_overrides']=sets;path_preflight(effective,spec,runs/model/'robustness/cfg_ffffffff/seed_73',results)
                db=study_db_path(effective,spec)
                if db.is_file():
                    snapshot_sqlite(db,evidence/'study-before.sqlite');study=load_study(spec,effective)
                else:study=None
                used=budget_used(study) if study else 0
                if not study or used!=budget:
                    raise Stop('HPO confirmado como terminado pero SQLite/budget incompleto; no se relanza ni se crea otro estudio')
                audit['confirmed_hpo_handoff']=check_completed_hpo(pc,study,audit['baseline'],hpo)
                emit('HPO_CERRADO_CONFIRMADO',pc=pc,trials=budget,reexecution='PROHIBIDA')
            if not study or budget_used(study)!=budget:raise Stop('HPO incompleto o presupuesto incompatible')
            audit['hpo_trials_verified']=verify_trial_evidence(study,effective,spec)
            audit['hpo_summary']=verify_study_summary(study,effective,spec,[runs,cfg_path(cfg,'paths.runs_dir')])
            audit['hpo']=dict(states=study_counts(study),best_trial=study.best_trial.number,best_value=study.best_value,
                              best_sd=study.best_trial.user_attrs.get('score_sd'),objective='map50_95' if spec.is_detector else 'average_precision_positive')
            import statistics
            best_evidence=next(r for r in audit['hpo_trials_verified'] if r['number']==study.best_trial.number)
            audit['hpo']['best_sd']=statistics.stdev(best_evidence['scores'])
            cands=top_configs(study,int(cfg['robustness']['top_k_configs']));audit['top_configs']=cands
            inspect_existing_hpo_curves(study,effective,spec,audit)
            audit['csv_validation']=csv_review_summary(audit)
            emit('REVISION_CSV_COMPLETADA',folds=audit['csv_validation']['folds_checked'],folds_with_warnings=audit['csv_validation']['folds_with_warnings'],critical_errors=0)
            audit['budget_comparison']=dict(baseline_epochs=section_for(spec,cfg)['baseline']['epochs'],hpo_max_epochs=section_for(spec,cfg)['max_epochs'],
                 baseline_minus_best_hpo=audit['baseline']['mean']-study.best_value,
                 conclusion='Comparacion descriptiva; presupuestos y seleccion pueden diferir. Baseline no se descarta ni se declara peor por haber HPO.')
            with FileLock(lock.with_suffix('.worker.guard')),FileLock(lock.with_suffix('.guard')):
                jobs=list(map(pure_job,jobs_read(effective,model)));hpo=one_job(jobs,'hpo')
                validate_queue_plan(effective,spec,cands,jobs,allow_missing=True)
                queue.expand_after_hpo(effective,spec,[c['config_id'] for c in cands])
                audit['plan']=validate_queue_plan(effective,spec,cands,jobs_read(effective,model))
                audit['status']='REVISION_HPO_PREPARADA';write_pipeline_report(report,audit)
                p=evidence/'hpo-review.json';atomic_json(p,dict(hpo=audit['hpo'],candidates=cands,isolation=audit['isolation'],budget_comparison=audit['budget_comparison'],csv_validation=audit['csv_validation']))
                if hpo['status']=='PENDING_REVIEW':actual_review(queue,hpo,p,read(p))
                elif hpo['status']=='PUBLICATION_PENDING':
                    from orchestration.exchange import publish_architecture
                    publish_architecture(effective,pc,model,runs,stage='hpo');queue.computed(hpo['job_id'],published=True)
                    actual_review(queue,one_job(list(map(pure_job,jobs_read(effective,model))),'hpo'),p,read(p))
                elif hpo['status']!='COMPLETE':raise Stop('HPO con estado no revisable: '+hpo['status'])
                audit['plan']=validate_queue_plan(effective,spec,cands,jobs_read(effective,model))
            while True:
                jobs=list(map(pure_job,jobs_read(effective,model)));audit['plan']=validate_queue_plan(effective,spec,cands,jobs)
                remaining=[j for j in jobs if j['kind'] in POST_KINDS and j['status']!='COMPLETE']
                if not remaining:break
                byid={j['job_id']:j for j in jobs};job=next((j for j in remaining if all(byid.get(d,{}).get('status')=='COMPLETE' for d in j['depends_on'])),None)
                if job is None:raise Stop('Jobs pendientes sin dependencias COMPLETE; ver plan del reporte')
                kind=job['kind'];emit('TRABAJO_ACTUAL',job=job['job_id'],kind=kind,payload=job['payload'])
                audit['current_job']=dict(job_id=job['job_id'],kind=kind,payload=job['payload'],observed_utc=utc())
                with FileLock(lock.with_suffix('.worker.guard')),FileLock(lock.with_suffix('.guard')):
                    if python_processes():raise Stop('YA_ACTIVO: aparecio otro proceso Python')
                    if recovery_or_reuse(queue,effective,spec,job,cands,jobs,state,evidence):continue
                    if job['status']=='PENDING_REVIEW':
                        validated=technical_job_result(effective,spec,job,cands,jobs);p=evidence/('review-'+kind+'.json');atomic_json(p,validated)
                        actual_review(queue,job,p,dict(selected=validated.get('selected'),baseline_mean=audit['baseline']['mean'],
                            budget_comparison=audit['budget_comparison'],isolation=audit['isolation']['status'],
                            scope='Seleccion nativa entre configuraciones HPO; baseline preservado, no se declara inferior ni se reentrena.'))
                        if kind=='aggregate':audit['aggregate']=validated
                        continue
                    if kind=='repeat':seed_reuse(effective,spec,job,cands,study)
                    if kind in ('outer_eval','thresholds'):external_authorization(effective,spec,p)
                    ensure_partial_resume(effective,spec,job)
                launch_visible(root,effective,spec,config,sets,kind,state,evidence,target_job=job['job_id'])
                updated=next(j for j in map(pure_job,jobs_read(effective,model)) if j['job_id']==job['job_id'])
                if updated['status'] not in ('COMPLETE','PENDING_REVIEW','PUBLICATION_PENDING'):raise Stop('Etapa no completada: '+str(updated))
                validated=technical_job_result(effective,spec,updated,cands,list(map(pure_job,jobs_read(effective,model))))
                audit.setdefault('completed_jobs',[]).append(dict(job_id=job['job_id'],kind=kind,evidence=validated));atomic_json(evidence/'progress.json',audit)
                if kind=='aggregate':audit['aggregate']=validated
                if kind=='outer_eval':audit['outer_validation']=validated
            # Restore report data even when every job was already COMPLETE.
            jobs=list(map(pure_job,jobs_read(effective,model)));audit['plan']=validate_queue_plan(effective,spec,cands,jobs)
            audit['aggregate']=validate_aggregate(effective,spec,cands,jobs);audit['final']=validate_final(effective,spec)
            known_paths={r['path'] for r in audit['curve_diagnostics']}
            for repeat in audit['aggregate']['repeat_evidence']:
                for curve in repeat['curves']:
                    if curve['path'] not in known_paths:audit['curve_diagnostics'].append(curve);known_paths.add(curve['path'])
            audit['outer_validation']=validate_external(effective,spec,'outer_validation');audit['thresholds']=validate_external(effective,spec,'inference_thresholds')
            if not spec.is_detector:
                audit['classification_report_validation']=dict(reference_threshold_half=audit['outer_validation']['classification_metrics'],
                    calibrated_threshold=audit['thresholds']['classification_metrics'],
                    original_native_metrics=audit['outer_validation']['metrics'],test_used=False)
            train_eval=runs/model/'final/train_evaluation.json'
            if train_eval.is_file():
                train=read(train_eval);val=audit['outer_validation']['metrics'];key='map50_95' if spec.is_detector else 'average_precision_positive'
                if train.get('checkpoint_sha256')==val.get('checkpoint_sha256') and train.get(key) is not None:
                    audit['same_checkpoint_generalization_gap']=dict(metric=key,train=train[key],validation=val[key],gap=train[key]-val[key],
                         note='Diferencia descriptiva para el mismo modelo; no existe un umbral universal de sobreajuste.')
                    audit['train_validation_comparison']=dict(checkpoint_sha256=val['checkpoint_sha256'],
                        train={k:train.get(k) for k in (['map50','map50_95','precision','recall'] if spec.is_detector else ['average_precision_positive','roc_auc','precision','recall','f1','accuracy'])},
                        validation={k:val.get(k) for k in (['map50','map50_95','precision','recall'] if spec.is_detector else ['average_precision_positive','roc_auc','precision','recall','f1','accuracy'])},
                        precision_recall_note='Detector: P/R al maximo F1 de cada conjunto, no un umbral compartido. Clasificador: umbral inicial 0.5.')
                    if not spec.is_detector:
                        train_predictions=Path(train.get('predictions_file',''))
                        if train_predictions.is_file() and inside(train_predictions,runs/model/'final') and digest(train_predictions)==train.get('predictions_sha256'):
                            audit['classification_report_train']=classification_prediction_audit(train_predictions,.5)
                        else:audit['classification_report_train']=dict(status='NOT_CALCULABLE',reason='Faltan probabilidades train firmadas del mismo checkpoint; no se inventan.')
            maybe_global(root,effective,pc,short,audit)
            audit['status']='PIPELINE_COMPLETO' if audit.get('global_status')=='PIPELINE_OFICIAL_COMPLETO' else 'ETAPAS_LOCALES_COMPLETAS_CIERRE_GLOBAL_PENDIENTE'
            return 0
    except ReviewRequired as e:audit.update(status='REVISION_HUMANA_PENDIENTE',error=str(e));return 3
    except KeyboardInterrupt:audit.update(status='INTERRUMPIDO',error='Trabajo conservado. La misma orden verifica antes de reanudar.');return 130
    except (Exception,RecoveryStop) as e:
        audit.update(status='YA_ACTIVO' if 'YA_ACTIVO' in str(e) else 'ERROR_CRITICO',error=str(e),traceback=traceback.format_exc());return 0 if audit['status']=='YA_ACTIVO' else 2
    finally:
        audit['finished_utc']=utc()
        try:
            audit['queue_snapshot']=list(map(pure_job,jobs_read(cfg,model)))
            if 'cands' in locals():audit['plan']=validate_queue_plan(effective,spec,cands,audit['queue_snapshot'],allow_missing=True)
        except Exception as snapshot_error:audit['queue_snapshot_error']=str(snapshot_error)
        try:
            export_curve_plots(audit,evidence/'curves');write_pipeline_report(report,audit);atomic_json(evidence/'summary.json',audit)
        except Exception as e:
            emit('ERROR_EXPORTANDO_REPORTE',error=str(e),audit=str(evidence));atomic_json(evidence/'summary.json',audit)
        emit(audit['status'],report=str(report),error=audit.get('error'))


def main():
    global TARGET_JOB_ID
    p=argparse.ArgumentParser(description='Continua la cola experimental nativa; preserva baseline/HPO, revisiones humanas y test cerrado hasta freeze.')
    p.add_argument('--expected-pc',choices=['PC-02']);p.add_argument('--project')
    p.add_argument('--_stage',choices=['worker','hpo','baseline',*POST_KINDS],help=argparse.SUPPRESS)
    p.add_argument('--state-dir',help=argparse.SUPPRESS);p.add_argument('--executor-sha',help=argparse.SUPPRESS)
    p.add_argument('--target-job',help=argparse.SUPPRESS)
    args,rest=p.parse_known_args()
    if args._stage:
        TARGET_JOB_ID=args.target_job
        if args.executor_sha!=EXECUTOR_SHA256:raise Stop('SHA del ejecutor hijo distinto del autorizado')
        if rest and rest[0]=='--':rest=rest[1:]
        return child_entry(args._stage,Path(args.project).resolve(),rest,Path(args.state_dir))
    if rest:p.error('Argumentos desconocidos: '+' '.join(rest))
    return pipeline_parent(args)


if __name__=='__main__':
    try:sys.exit(main())
    except (Exception,RecoveryStop) as e:
        emit('ERROR_CRITICO',error=str(e));sys.exit(2)
