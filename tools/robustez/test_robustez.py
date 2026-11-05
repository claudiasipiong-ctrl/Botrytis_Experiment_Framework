"""CPU tests; synthetic fixtures, never real GPU training or operational approval."""
import importlib.util
import json
from pathlib import Path
from types import SimpleNamespace
import sys

import numpy as np
import optuna
import pandas as pd
import pytest

HERE = Path(__file__).parent
spec = importlib.util.spec_from_file_location('resume_robustness', HERE / 'botrytis_robustez_reanudar.py')
resume = importlib.util.module_from_spec(spec); spec.loader.exec_module(resume)
bt = resume.bt
from utils.config_loader import load_config, cfg_path, PROJECT_ROOT
from utils.architectures import get_arch
from orchestration.study_queue import StudyQueue
from utils.experiments import fingerprint
from utils.hpo_results import top_configs
from utils.search_spaces import config_id


def config(tmp):
    c = load_config()
    for key, rel in dict(artifacts_dir='a', queue_dir='q', runs_dir='r', results_dir='e',
                         best_configs_dir='best', optuna_dir='optuna').items():
        c['paths'][key] = str(tmp / rel)
    c['compute']['pc_id'] = 'PC-02'
    c['experiment'].update(architecture='swin_tiny', experiment_id='botrytis_v4', round_id='fixture')
    return c


def planned(tmp):
    c = config(tmp); sp = get_arch('swin_tiny'); q = StudyQueue.from_cfg(c); q.plan(c, [sp])
    for job in q.jobs():
        q.finish(job.job_id, 'COMPLETE')
    q.expand_after_hpo(c, sp, ['cfg_1', 'cfg_2', 'cfg_3'])
    return c, q


def paused(tmp):
    c, q = planned(tmp)
    jobs = [bt.pure_job(j) for j in q.jobs() if j.kind in ('repeat', 'aggregate', 'outer_train', 'outer_eval', 'thresholds')]
    p = tmp / 'pause.json'; marker = 'POSPUESTO_POR_PRIORIDAD ' + str(p)
    bt.atomic_json(p, dict(database=str(q.path), architecture='swin_tiny', marker=marker, postponed=jobs))
    for job in jobs:
        q.finish(job['job_id'], 'BLOCKED', error=marker)
    return c, q, p


def test_restore_only_robustness_exact_pause_and_idempotent(tmp_path):
    c, q, p = paused(tmp_path)
    ids = resume.restore_paused(q.path, 'swin_tiny', [p])
    assert len(ids) == 10
    assert all(j.status == 'PENDING' for j in q.jobs() if j.kind in ('repeat', 'aggregate'))
    assert all(j.status == 'BLOCKED' for j in q.jobs() if j.kind in ('outer_train', 'outer_eval', 'thresholds'))
    assert resume.restore_paused(q.path, 'swin_tiny', [p]) == []
    assert all(j.status == 'COMPLETE' for j in q.jobs() if j.kind in ('baseline', 'hpo'))


def test_unrelated_block_and_active_work_preserved(tmp_path):
    _, q, p = paused(tmp_path)
    jobs = [j for j in q.jobs() if j.kind == 'repeat']
    q.finish(jobs[0].job_id, 'BLOCKED', error='INVALID_RESULTS')
    with q._conn() as con:
        con.execute("UPDATE jobs SET status='RUNNING' WHERE job_id=?", (jobs[1].job_id,))
    resume.restore_paused(q.path, 'swin_tiny', [p])
    assert q.get(jobs[0].job_id).error == 'INVALID_RESULTS'
    assert q.get(jobs[1].job_id).status == 'RUNNING'


def test_changed_pause_identity_rolls_back(tmp_path):
    _, q, p = paused(tmp_path)
    original = bt.read(p)
    original['postponed'][1]['attempt'] = 888
    bt.atomic_json(p, original)
    with pytest.raises(bt.Stop, match='Identidad'):
        resume.restore_paused(q.path, 'swin_tiny', [p])
    assert all(j.status == 'BLOCKED' for j in q.jobs() if j.kind == 'repeat')


@pytest.mark.parametrize('state', ['RUNNING', 'WAITING', 'FAIL'])
def test_unfinished_hpo_cannot_enable_robustness(tmp_path, state):
    sp = get_arch('swin_tiny'); c = config(tmp_path)
    trials = [SimpleNamespace(state=SimpleNamespace(name='COMPLETE')) for _ in range(19)]
    trials.append(SimpleNamespace(state=SimpleNamespace(name=state)))
    with pytest.raises(bt.Stop, match='no cerrado'):
        resume.verify_closed_hpo(SimpleNamespace(trials=trials), c, sp, dict(status='COMPLETE', payload={}))


def test_dispatch_forbids_baseline_hpo_outer_and_test():
    assert bt.POST_KINDS == ('repeat', 'aggregate')
    for kind in ('baseline', 'hpo', 'outer_train', 'outer_eval', 'thresholds'):
        with pytest.raises((bt.Stop, FileNotFoundError)):
            bt.child_entry(kind, HERE, [], HERE / 'unused')


def test_active_process_wait_has_no_mutation(tmp_path, monkeypatch):
    monkeypatch.setattr(bt, 'python_processes', lambda: [{'pid': 52, 'command': ['python', 'r50_hpo']}])
    monkeypatch.setattr(resume.time, 'sleep', lambda _: (_ for _ in ()).throw(KeyboardInterrupt()))
    c = config(tmp_path)
    before = list(tmp_path.rglob('*'))
    with pytest.raises(KeyboardInterrupt):
        resume.wait_idle(c, 'PC-02')
    assert list(tmp_path.rglob('*')) == before


def test_second_controller_cannot_wait_against_first(tmp_path, monkeypatch):
    import utils.guards as guards
    monkeypatch.setattr(guards, 'gpu_lock_path', lambda cfg, pc: tmp_path / 'PC-02.json')
    with resume.controller_lock({}, 'PC-02'):
        with pytest.raises(RuntimeError, match='ocupado'):
            with resume.controller_lock({}, 'PC-02'):
                pytest.fail('Duplicate controller acquired lock')
    with resume.controller_lock({}, 'PC-02'):
        pass  # Descriptor release, no deletion of the guard file.


def test_official_backend_never_uses_yolo_wrapper(tmp_path, monkeypatch):
    import utils.hpo_runner as runner
    calls = []
    sentinel = lambda *a, **k: 'official'
    fake = SimpleNamespace(install_hooks=lambda c: (calls.append(c), setattr(runner, 'evaluate_config', sentinel)))
    monkeypatch.setattr(resume.audit, 'load_adapter', lambda: fake)
    monkeypatch.setattr(runner, 'evaluate_config', runner.evaluate_config)
    module = SimpleNamespace()
    hooks = SimpleNamespace(spec=get_arch('rtdetr_r50'), cfg=config(tmp_path))
    resume.install_post(hooks, module)
    assert module.evaluate_config is sentinel and len(calls) == 1


def test_manual_approval_not_fabricated(tmp_path, monkeypatch):
    _, q = planned(tmp_path)
    hpo = next(j for j in q.jobs() if j.kind == 'hpo')
    q.finish(hpo.job_id, 'PENDING_REVIEW')
    evidence = tmp_path / 'review.json'; bt.atomic_json(evidence, {'fixture': True})
    monkeypatch.setattr(sys.stdin, 'isatty', lambda: True)
    monkeypatch.setattr('builtins.input', lambda _: 'NO')
    with pytest.raises(bt.ReviewRequired):
        bt.actual_review(q, q.get(hpo.job_id), evidence, {'fixture': True})
    assert q.get(hpo.job_id).status == 'PENDING_REVIEW'
    with q._conn() as con:
        assert con.execute('select count(*) from reviews').fetchone()[0] == 0
    monkeypatch.setattr('builtins.input', lambda _: 'APROBAR')
    bt.actual_review(q, q.get(hpo.job_id), evidence, {'fixture': True})
    assert q.get(hpo.job_id).status == 'COMPLETE'


def make_study():
    study = optuna.create_study(direction='maximize')
    for i in range(3):
        hp = dict(lr=.001 * (i + 1))
        study.add_trial(optuna.trial.create_trial(value=.7 + .02 * i,
            user_attrs=dict(hp=hp, config_id=config_id(hp))))
    return study


def test_native_08_seed_fold_matrix_and_aggregation_no_training(tmp_path, monkeypatch):
    """Run native 08 main/run_one/aggregate. Only GPU evaluate is a fixture."""
    native = bt.load_native(PROJECT_ROOT / 'src/08_repeat_top_configs.py', '_real_08_fixture')
    c = config(tmp_path); sp = get_arch('swin_tiny'); study = make_study(); cands = top_configs(study, 3)
    calls = []
    class Context:
        def __init__(self, folder):
            self.run_dir = folder; self.run_dir.mkdir(parents=True, exist_ok=True)
            self.cfg = c; self.pc_id = 'PC-02'; self.runs_base = cfg_path(c, 'paths.runs_dir')
            self.hashes = dict(dataset_hash='fixture', split_hash='fixture', code_hash='fixture')
            self.logger = SimpleNamespace(info=lambda *a: None)
        def set_seed(self, seed): pass
        def add_metadata(self, **kw): pass
        def log_metrics(self, *a): pass
        def register_artifact(self, *a): pass
        def save_figure(self, *a, **kw): pass
    def evaluate(sp, hp, cfg, ctx, folder, **kw):
        calls.extend((config_id(hp), kw['seed'], fold) for fold in kw['folds'])
        score = .7 + hp['lr'] + kw['seed'] / 10000
        return SimpleNamespace(fold_scores=[score]*3, mean=score,
            folds=[dict(best_epoch=2+i, epochs_run=5) for i in range(3)],
            history=pd.DataFrame(), train_seconds=0, vram_peak_mb=0, gflops=1, params_m=1)
    monkeypatch.setattr(native, 'evaluate_config', evaluate)
    monkeypatch.setattr(native, 'save_training_figures', lambda *a, **k: None)
    monkeypatch.setattr(native, 'plot_robustness', lambda *a, **k: None)
    for candidate in cands:
        for seed in c['robustness']['seeds']:
            ctx = Context(cfg_path(c, 'paths.runs_dir') / sp.slug / 'robustness' / candidate['config_id'] / f'seed_{seed}')
            native.run_one(ctx, sp, c, candidate, seed, nested=False, force=False)
            # Re-enter native stage: result.json must prevent a second call.
            native.run_one(ctx, sp, c, candidate, seed, nested=False, force=False)
    assert len(calls) == len(set(calls)) == 27
    agg = Context(cfg_path(c, 'paths.runs_dir') / sp.slug / 'robustness/aggregate')
    best = native.aggregate(agg, sp, c, cands, agg.run_dir.parent)
    assert best['selected_by_single_seed'] is False and best['seeds'] == [17,42,73]
    assert best['final_epochs'] == 3 and best['config_id'] == config_id(dict(lr=.003))
    # Missing one seed is a real methodology error; never publish partial means.
    p = agg.run_dir.parent / cands[0]['config_id'] / 'seed_73/result.json'
    p.unlink()
    with pytest.raises(Exception, match='(?i)(falta|missing|robustez)'):
        native.aggregate(agg, sp, c, cands, agg.run_dir.parent)


def test_corrupt_cached_fold_cannot_be_reused(tmp_path):
    f = tmp_path / 'fold0'; f.mkdir()
    history = f / 'completed_history.csv'; history.write_text('epoch,fold\n1,0\n')
    bt.atomic_json(f / 'completed_fold.json', dict(signature='fp', history_sha256='wrong', row=dict(fold=0)))
    with pytest.raises(bt.Stop, match='alterado'):
        bt.verify_fold(f, 'fp', 0)


def test_seed42_reuse_checks_signature_preserves_source_and_is_idempotent(tmp_path, monkeypatch):
    c = config(tmp_path); sp = get_arch('swin_tiny'); hp = dict(lr=.003)
    source = cfg_path(c, 'paths.runs_dir') / sp.slug / 'hpo/trial_000'
    candidate = dict(config_id=config_id(hp), hp=hp, trial_number=0)
    trial = SimpleNamespace(number=0, user_attrs=dict(run_dir=str(source)))
    job = dict(kind='repeat', payload=dict(config_id=candidate['config_id'], seed=42))
    for fold in range(3):
        f = source / f'fold{fold}'; f.mkdir(parents=True)
        history = f / 'completed_history.csv'
        pd.DataFrame(dict(epoch=[1,2], fold=[fold]*2, val_ap=[.5,.7])).to_csv(history, index=False)
        signature = bt.fold_signature(fingerprint(c), sp.slug, hp, 42, fold,
                                     c['classifier']['max_epochs'], c['classifier']['early_stopping_patience'])
        bt.atomic_json(f / 'completed_fold.json', dict(signature=signature,
            history_sha256=bt.digest(history), row=dict(fold=fold, epochs_run=2, score=.7)))
    monkeypatch.setattr(bt, 'verify_checkpoint_entries', lambda *a: {
        f'fold{f}_best':dict(role=f'fold{f}_best', sha256=str(f)*64, weights_discarded=True) for f in range(3)})
    original = {str(p):p.read_bytes() for p in source.rglob('*') if p.is_file()}
    assert bt.seed_reuse(c, sp, job, [candidate], SimpleNamespace(trials=[trial])) == [0,1,2]
    assert bt.seed_reuse(c, sp, job, [candidate], SimpleNamespace(trials=[trial])) == []
    assert original == {str(p):p.read_bytes() for p in source.rglob('*') if p.is_file()}
    job['payload']['seed'] = 17
    assert bt.seed_reuse(c, sp, job, [candidate], SimpleNamespace(trials=[trial])) == []
    # Tamper a source, ensure the completed target does not conceal corruption.
    (source / 'fold1/completed_history.csv').write_text('corrupt')
    job['payload']['seed'] = 42
    with pytest.raises(bt.Stop, match='alterado'):
        bt.seed_reuse(c, sp, job, [candidate], SimpleNamespace(trials=[trial]))


def test_r50_missing_guaranteed_loss_stops(tmp_path):
    pd.DataFrame(dict(epoch=[1], map50_95=[.7])).to_csv(tmp_path / 'results.csv', index=False)
    with pytest.raises(Exception, match='columna garantizada'):
        resume.official().inspect_curve(tmp_path / 'results.csv')
