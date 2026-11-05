"""Reanuda SOLO robustez original y agregacion. Nunca ejecuta 06, 07, 09-16.

Ejecutar fuera del checkout, con sus cuatro dependencias fijadas en esta carpeta.
Cada HPO pendiente de revision conserva la decision presencial APROBAR.
"""
from __future__ import annotations
import argparse
import contextlib
import hashlib
import importlib.util
import json
import math
import os
from pathlib import Path
import sqlite3
import sys
import time
import uuid

HERE = Path(__file__).resolve().parent
SELF = Path(__file__).resolve()
SHA = hashlib.sha256(SELF.read_bytes()).hexdigest()
DEPENDENCIES = {
    'botrytis_robustez_base.py': '453f82b038fac5662a1ec83886e79b5a88ea032778f34d197e4f5d6aa81c699f',
    'botrytis_r50_auditoria.py': '432881891c91bc6e20becc7fec97d245a657ce5c7e6f1677fb99c224d344149e',
    'botrytis_r50_hpo_pc03.py': '160ce95cb02e3ce2a4decf4a2698e5646032b05bbfcbb6b0887b0a8bfef806e7',
    'botrytis_r50_oficial_pinned.py': 'e62e47fdc4cdbe6648d0830a8efac9701ab04ae82b77757d4a0880ec73381df7',
}
MODELS = {'PC-01': ['yolov8n', 'yolov8l'], 'PC-02': ['yolo26l', 'swin_tiny'],
          'PC-03': ['vit_b16', 'rtdetr_r50']}
USERS = {'PC-01': 'i21315', 'PC-02': 'i21314', 'PC-03': 'i21313'}
CODE = 'e11f772d86f45e5dc3687fce3faa23443a6fcf16146484b3d5f38534cc7695df'
COMMIT = 'b706f176c8e308969fd3a885d66045ffea027c7e'
DATA = 'f9cc08785271eecb6566669e1eff7fc1dcb821b8a8a407592a6153fe248018ed'
SPLIT = '617a3fc380835763c6f43f84e770df586d96d06aab7de9101a10c44b072ee64b'


def load_file(name, module_name):
    path = HERE / name
    if hashlib.sha256(path.read_bytes()).hexdigest() != DEPENDENCIES[name]:
        raise RuntimeError('Dependencia ausente/alterada: ' + str(path))
    spec = importlib.util.spec_from_file_location(module_name, path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = module
    spec.loader.exec_module(module)
    return module


bt = load_file('botrytis_robustez_base.py', '_bt_robust_base')
audit = load_file('botrytis_r50_auditoria.py', '_bt_readonly_audit')
bt.__file__ = str(SELF)  # Child workers must enter this restricted dispatcher.
bt.EXECUTOR_SHA256 = SHA
bt.POST_KINDS = ('repeat', 'aggregate')
bt.POST_FILES = {'repeat': '08_repeat_top_configs.py', 'aggregate': '08_repeat_top_configs.py'}
_r50 = None


def official():
    global _r50
    if _r50 is None:
        _r50 = load_file('botrytis_r50_hpo_pc03.py', '_bt_r50_validation')
    return _r50


def restore_paused(database, model, pause_paths):
    """CAS restore only matching original pause; never restore outer stages."""
    restored = []
    for path in pause_paths:
        record = bt.read(path)
        if record.get('architecture') != model or Path(record['database']).resolve() != Path(database).resolve():
            continue
        marker = record.get('marker')
        if marker != 'POSPUESTO_POR_PRIORIDAD ' + str(path):
            raise bt.Stop('Recibo de pausa incompatible: ' + str(path))
        with sqlite3.connect(database, timeout=30) as con:
            con.row_factory = sqlite3.Row
            con.execute('BEGIN IMMEDIATE')
            for old in record.get('postponed', []):
                if old['kind'] not in ('repeat', 'aggregate'):
                    continue
                current = con.execute('SELECT * FROM jobs WHERE job_id=?', (old['job_id'],)).fetchone()
                if current is None:
                    raise bt.Stop('Job pausado ausente; no se recrea: ' + old['job_id'])
                if current['status'] != 'BLOCKED' or current['error'] != marker:
                    continue
                for key in ('kind', 'architecture', 'payload', 'depends_on', 'attempt'):
                    a, b = current[key], old[key]
                    if key in ('payload', 'depends_on'):
                        a = json.loads(a) if isinstance(a, str) else a
                        b = json.loads(b) if isinstance(b, str) else b
                    if a != b:
                        raise bt.Stop('Identidad de job pausado modificada: ' + old['job_id'])
                if old['status'] != 'PENDING':
                    raise bt.Stop('La pausa no provino de PENDING')
                con.execute("UPDATE jobs SET status='PENDING',error=?,updated_at=? WHERE job_id=? AND status='BLOCKED' AND error=?",
                            (old.get('error'), bt.utc(), old['job_id'], marker))
                restored.append(old['job_id'])
    return restored


def wait_idle(cfg, pc):
    """No signal, lock deletion or queue edits while an existing Python works."""
    from utils.durable import FileLock
    from utils.guards import gpu_lock_path
    lock = gpu_lock_path(cfg, pc)
    while True:
        active = bt.python_processes()
        guard = None
        if not active:
            try:
                guard = FileLock(lock.with_suffix('.final.guard'))
                guard.__enter__()
                with FileLock(lock.with_suffix('.worker.guard')), FileLock(lock.with_suffix('.guard')):
                    if not bt.python_processes():
                        return guard
            except RuntimeError:
                pass
            if guard is not None and guard.handle is not None:
                guard.__exit__(None, None, None)
        bt.emit('ESPERANDO_TRABAJO_ACTUAL', pc=pc, processes=active,
                action='No se interrumpe ni se inicia otro entrenamiento')
        time.sleep(15)


def baseline_evidence(source_cfg, cfg, spec, code, commit):
    """Verify baseline closure without assuming every component must decrease."""
    from utils.config_loader import cfg_path, load_config
    from utils.hashing import scientific_config_hash
    from utils.experiments import fingerprint
    from utils.architectures import section_for
    from utils.search_spaces import baseline_hyperparameters
    folder = cfg_path(source_cfg, 'paths.runs_dir') / spec.slug / 'baseline'
    if not (folder / 'summary.json').is_file():
        folder = cfg_path(cfg, 'paths.runs_dir') / spec.slug / 'baseline'
    summary, meta = bt.read(folder / 'summary.json'), bt.read(folder / 'metadata.json')
    if meta.get('status') != 'COMPLETE' or meta.get('code_hash') != code or meta.get('git_commit') != commit:
        raise bt.Stop('Baseline sin cierre o codigo compatible: ' + str(folder))
    if summary.get('dataset_hash') != DATA or summary.get('split_hash') != SPLIT or summary.get('slug') != spec.slug:
        raise bt.Stop('Baseline de otros datos/splits/modelo')
    if scientific_config_hash(load_config(folder / 'config.yaml')) != scientific_config_hash(cfg):
        raise bt.Stop('Configuracion del baseline incompatible')
    sec = section_for(spec, cfg); b = sec['baseline']; hp = baseline_hyperparameters(spec, cfg)
    scores = []
    for fold in range(3):
        patience = b['patience'] if spec.is_detector else sec['early_stopping_patience']
        sig = bt.fold_signature(fingerprint(cfg), spec.slug, hp, int(b['seed']), fold, int(b['epochs']), int(patience))
        receipt, _ = bt.verify_fold(folder / f'fold{fold}', sig, fold)
        scores.append(float(receipt['row']['score']))
    if summary.get('folds') != [0, 1, 2] or len(summary.get('fold_scores', [])) != 3:
        raise bt.Stop('Baseline no tiene exactamente tres folds')
    if any(not math.isclose(a, float(b), abs_tol=1e-8) for a, b in zip(scores, summary['fold_scores'])):
        raise bt.Stop('Scores baseline contradictorios')
    if not math.isclose(sum(scores) / 3, float(summary['score_mean']), abs_tol=1e-8):
        raise bt.Stop('Media baseline contradictoria')
    if set(bt.verify_checkpoint_entries(cfg, spec, folder)) != {'fold0_best', 'fold1_best', 'fold2_best'}:
        raise bt.Stop('Faltan recibos originales de pesos baseline')
    return dict(path=str(folder), mean=summary['score_mean'], folds=scores, preserved=True)


_curve = bt.curve_evidence


def curve_evidence(path, detector, selected_epoch=None):
    if 'rtdetr_r50' not in Path(path).parts:
        return _curve(path, detector, selected_epoch)
    h = official().inspect_curve(path)
    values = h.map50_95
    if selected_epoch is not None:
        selected = h.loc[h.epoch == selected_epoch, 'map50_95']
        if len(selected) != 1 or not math.isclose(float(selected.iloc[0]), float(values.max()), abs_tol=1e-7):
            raise bt.Stop('Epoca R50 seleccionada incompatible')
    return dict(path=str(path), sha256=bt.digest(path), best_value=float(values.max()),
                best_epoch=int(h.loc[values.idxmax(), 'epoch']), epochs=len(h),
                last_value=float(values.iloc[-1]),
                limitation='Curvas verificadas; no certifica ausencia de sobreajuste ni independencia biologica.')


bt.curve_evidence = curve_evidence
_install_post = bt.PostHooks.install_post


def install_post(hooks, module):
    if hooks.spec.slug != 'rtdetr_r50':
        return _install_post(hooks, module)
    adapter = audit.load_adapter()
    adapter.install_hooks(hooks.cfg)
    import utils.hpo_runner as runner
    # Preserve official train/resume/cache logic. Never wrap in YOLO's trainer.
    module.evaluate_config = runner.evaluate_config


bt.PostHooks.install_post = install_post


def validate_existing_partials(cfg, spec, job):
    if spec.slug != 'rtdetr_r50':
        return bt.ensure_partial_resume(cfg, spec, job)
    folder = bt.job_folder(cfg, spec, job)
    from utils.hpo_results import top_configs
    from utils.optuna_storage import load_study
    hp = next(c['hp'] for c in top_configs(load_study(spec, cfg), 3) if c['config_id'] == job['payload']['config_id'])
    for fold in range(3):
        sub = folder / f'fold{fold}'
        if (sub / 'completed_fold.json').is_file():
            continue
        checkpoints = list((sub / 'cp').glob('e*.pt'))
        if sub.is_dir() and any(sub.iterdir()) and not checkpoints:
            raise bt.Stop('Fold R50 parcial sin estado de optimizador; conservado: ' + str(sub))
        for cp in checkpoints:
            bt.validate_checkpoint(cp, spec.slug, cfg, seed=job['payload']['seed'], hp=hp, fold=fold)


def verify_closed_hpo(study, cfg, spec, hpo):
    from utils.architectures import trials_for
    counts = bt.study_counts(study)
    if len(study.trials) != trials_for(spec, cfg) or counts['RUNNING'] or counts['WAITING'] or counts['FAIL']:
        raise bt.Stop('HPO no cerrado; no se relanza ni se inicia robustez: ' + str(counts))
    if hpo['status'] not in ('COMPLETE', 'PENDING_REVIEW', 'PUBLICATION_PENDING') or hpo['payload'].get('partial_budget'):
        raise bt.Stop('Cola HPO no cerrada: ' + hpo['status'])
    verified = (official().verify_trials(study, cfg, spec) if spec.slug == 'rtdetr_r50'
                else bt.verify_trial_evidence(study, cfg, spec))
    curves = []
    for trial in study.trials:
        if trial.state.name != 'COMPLETE':
            continue
        folder = bt.allowed_trial_folder(cfg, spec, trial)
        for fold in range(3):
            receipt = bt.read(folder / f'fold{fold}/completed_fold.json')
            curves.append(curve_evidence(folder / f'fold{fold}/completed_history.csv', spec.is_detector, receipt['row']['best_epoch']))
    return dict(trials=verified, curves=curves, absence_of_overfitting_certified=False)


def process_model(project, base, pc, model, index, out):
    from utils.config_loader import load_config, cfg_path
    from utils.architectures import get_arch
    from utils.experiments import verify_round, fingerprint
    from utils.manual_gates import require_gate
    from utils.optuna_storage import load_study, study_db_path
    from utils.hpo_results import top_configs
    from orchestration.study_queue import StudyQueue
    from orchestration.pc_registry import PCRegistry
    from orchestration.exchange import publish_architecture
    if model not in index:
        raise bt.Stop('Modelo sin ronda indexada; no se crea: ' + model)
    config = Path(index[model]); source_cfg = load_config(config); spec = get_arch(model)
    short, runs, results, state = bt.make_short_paths(config, Path.home() / 'bt-hoy-20261103')
    if model == 'rtdetr_r50':
        short = Path.home() / 'r50'; runs = short / 'r'; results = short / 'e'; state = short / 's'
    state = state / 'post'; state.mkdir(parents=True, exist_ok=True)
    sets = ['paths.runs_dir=' + str(runs), 'paths.results_dir=' + str(results)]
    cfg = load_config(config, sets)
    verify_round(cfg, architecture=model)
    if PCRegistry(cfg).owner(model) != pc or cfg['compute']['pc_id'] != pc:
        raise bt.Stop('Propiedad/configuracion incompatible')
    if cfg['robustness']['seeds'] != [17, 42, 73] or cfg['robustness']['top_k_configs'] != 3 or cfg['inner_hpo']['folds'] != 3:
        raise bt.Stop('Protocolo robustez distinto; se conserva sin cambiarlo')
    require_gate(cfg, pc, 'READY_FOR_EXPERIMENT')
    record = dict(model=model, config=str(config), status='VERIFICANDO', test_opened=False,
                  executor_sha256=SHA, experiment_fingerprint=fingerprint(cfg))
    evidence = out / model; evidence.mkdir(parents=True, exist_ok=True)
    code = audit.readonly_code(project)
    if code['code_hash'] != CODE or code['git_commit'] != COMMIT:
        raise bt.Stop('Version de codigo diferente; no se actualiza ni se cambia la ronda')
    scratch, data = audit.readonly_prepared(cfg, evidence / 'prep')
    if data.dataset_hash != DATA or data.split_hash != SPLIT:
        raise bt.Stop('Datos/splits actuales incompatibles')
    record['isolation'] = audit.partition_evidence(scratch, spec)
    if record['isolation']['confirmed_relevant_conflicts']:
        raise bt.Stop('Fuga confirmada; se conservan splits y resultados')
    queue = StudyQueue.from_cfg(cfg)
    jobs = list(map(bt.pure_job, bt.jobs_read(cfg, model)))
    baseline = bt.one_job(jobs, 'baseline'); hpo = bt.one_job(jobs, 'hpo')
    if baseline['status'] != 'COMPLETE':
        raise bt.Stop('Baseline requiere su revision previa; no se reentrena: ' + baseline['status'])
    record['baseline'] = baseline_evidence(source_cfg, cfg, spec, CODE, COMMIT)
    db = study_db_path(cfg, spec)
    if not db.is_file():
        raise bt.Stop('Falta SQLite HPO original; no se crea otro')
    bt.snapshot_sqlite(db, evidence / 'study-before.sqlite')
    bt.snapshot_sqlite(cfg_path(cfg, 'paths.queue_dir') / 'queue.sqlite', evidence / 'queue-before.sqlite')
    study = load_study(spec, cfg)
    record['hpo_verified'] = verify_closed_hpo(study, cfg, spec, hpo)
    cands = top_configs(study, 3); record['top_configs'] = cands
    bt.validate_queue_plan(cfg, spec, cands, jobs, allow_missing=True)
    queue.expand_after_hpo(cfg, spec, [c['config_id'] for c in cands])
    if hpo['status'] == 'PUBLICATION_PENDING':
        publish_architecture(cfg, pc, model, runs, stage='hpo')
        queue.computed(hpo['job_id'], published=True)
        hpo = bt.pure_job(queue.get(hpo['job_id']))
    bt.atomic_json(evidence / 'revision.json', record)
    if hpo['status'] == 'PENDING_REVIEW':
        bt.actual_review(queue, hpo, evidence / 'revision.json', dict(
            model=model, best_value=study.best_value, baseline=record['baseline']['mean'],
            scope='Solo robustez original y agregacion; TEST cerrado',
            curve_warnings=[dict(path=c['path'], warnings=c.get('warnings', c.get('flags', [])))
                            for c in record['hpo_verified']['curves']
                            if c.get('warnings') or c.get('flags')],
            limitation='No certifica ausencia de sobreajuste; seleccion final externa aun pendiente'))
    paused = (Path.home() / 'bp').glob('*/' + model + '/pause.json')
    record['restored_jobs'] = restore_paused(queue.path, model, list(paused))
    while True:
        jobs = list(map(bt.pure_job, bt.jobs_read(cfg, model)))
        bt.validate_queue_plan(cfg, spec, cands, jobs)
        remaining = [j for j in jobs if j['kind'] in ('repeat', 'aggregate') and j['status'] != 'COMPLETE']
        if not remaining:
            break
        byid = {j['job_id']: j for j in jobs}
        job = next((j for j in remaining if all(byid[d]['status'] == 'COMPLETE' for d in j['depends_on'])), None)
        if job is None:
            raise bt.Stop('Dependencias no habilitadas; no se omiten: ' + str(remaining))
        if job['kind'] == 'aggregate' and job['status'] == 'PENDING_REVIEW':
            record['aggregate'] = bt.validate_aggregate(cfg, spec, cands, jobs)
            break  # Keep review intact; never fall through into 09 or TEST.
        bt.emit('TRABAJO_ROBUSTEZ', model=model, job=job['job_id'], payload=job['payload'])
        if bt.recovery_or_reuse(queue, cfg, spec, job, cands, jobs, state, evidence):
            continue
        if job['status'] == 'BLOCKED':
            raise bt.Stop('Bloqueo distinto de pausa por prioridad; no se elimina: ' + str(job))
        if job['status'] == 'PENDING_REVIEW':
            raise bt.Stop('Revision inesperada de repeat; no se aprueba automaticamente')
        if bt.python_processes():
            raise bt.Stop('Aparecio otro proceso; no se inicia otro entrenamiento')
        if job['kind'] == 'repeat':
            bt.seed_reuse(cfg, spec, job, cands, study)
            validate_existing_partials(cfg, spec, job)
        bt.launch_visible(project, cfg, spec, config, sets, job['kind'], state, evidence, target_job=job['job_id'])
        updated = bt.pure_job(queue.get(job['job_id']))
        if updated['status'] not in ('COMPLETE', 'PENDING_REVIEW', 'PUBLICATION_PENDING'):
            raise bt.Stop('Trabajo sin cierre valido: ' + str(updated))
        bt.technical_job_result(cfg, spec, updated, cands, list(map(bt.pure_job, bt.jobs_read(cfg, model))))
    jobs = list(map(bt.pure_job, bt.jobs_read(cfg, model)))
    record['aggregate'] = bt.validate_aggregate(cfg, spec, cands, jobs)
    record.update(status='ROBUSTEZ_COMPLETA', aggregate_status=bt.one_job(jobs, 'aggregate')['status'],
                  results=str(runs / model / 'robustness'), final_training_started=False, test_opened=False)
    bt.atomic_json(evidence / 'resumen.json', record)
    bt.emit('ROBUSTEZ_COMPLETA', model=model, results=record['results'],
            next='Revision de agregacion -> entrenamiento final -> VAL externo; TEST sigue cerrado')
    return record


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--expected-pc', choices=list(MODELS))
    p.add_argument('--project')
    p.add_argument('--_stage', choices=['worker', 'repeat', 'aggregate'], help=argparse.SUPPRESS)
    p.add_argument('--state-dir', help=argparse.SUPPRESS)
    p.add_argument('--executor-sha', help=argparse.SUPPRESS)
    p.add_argument('--target-job', help=argparse.SUPPRESS)
    args, rest = p.parse_known_args()
    if args._stage:
        if args.executor_sha != SHA:
            raise bt.Stop('SHA del hijo distinto')
        bt.TARGET_JOB_ID = args.target_job
        return bt.child_entry(args._stage, Path(args.project), rest[1:] if rest[:1] == ['--'] else rest, Path(args.state_dir))
    if rest:
        p.error('Argumentos desconocidos')
    if os.name != 'nt':
        raise bt.Stop('Operacion exclusiva Windows; pruebas CPU por separado')
    project = bt.find_project(args.project)
    py = project / '.venv/Scripts/python.exe'
    if Path(sys.executable).resolve() != py.resolve():
        return bt.subprocess.call([str(py), '-u', str(SELF), *sys.argv[1:]])
    os.chdir(project); sys.path.insert(0, str(project / 'src'))
    os.environ.update(PYTHONUTF8='1', PYTHONIOENCODING='utf-8', MPLBACKEND='Agg',
                      BOTRYTIS_BOOTSTRAP_DIR=str(Path.home() / 'btlog'))
    from utils.config_loader import load_config
    from orchestration.machine import read_binding
    cfg = load_config(project / 'artifacts/v4/local.yaml'); pc = cfg['compute']['pc_id']
    binding = read_binding(cfg)
    if pc != args.expected_pc or os.environ.get('USERNAME', '').casefold() != USERS.get(pc) or binding['pc_id'] != pc or binding['machine_id'].casefold() != USERS[pc]:
        raise bt.Stop('Usuario, binding, configuracion y --expected-pc no coinciden')
    index = bt.read(Path.home() / 'bt-hoy-20261103/rondas.json')
    out = Path.home() / 'bg' / uuid.uuid4().hex[:8]; out.mkdir(parents=True)
    results, errors = [], []
    with wait_idle(cfg, pc):
        for model in MODELS[pc]:
            try:
                results.append(process_model(project, cfg, pc, model, index, out))
            except (Exception, bt.RecoveryStop) as exc:
                errors.append(dict(model=model, error=str(exc)))
                bt.emit('ARQUITECTURA_DETENIDA', model=model, error=str(exc), results_preserved=True)
    bt.atomic_json(out / 'resumen.json', dict(pc=pc, completed=results, errors=errors,
                   test_opened=False, protocol_changed=False, baseline_or_hpo_launched=False))
    bt.emit('FIN_ROBUSTEZ', report=str(out / 'resumen.json'), errors=errors)
    return 2 if errors else 0


if __name__ == '__main__':
    try:
        sys.exit(main())
    except KeyboardInterrupt:
        bt.emit('CONTROLADOR_INTERRUMPIDO', action='No borrar locks; comprobar proceso antes de reanudar')
        sys.exit(130)
    except (Exception, bt.RecoveryStop) as exc:
        bt.emit('ERROR_CRITICO', error=str(exc), results_preserved=True)
        sys.exit(2)
