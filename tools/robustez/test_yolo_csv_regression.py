"""Read-only CSV correction tests. Run: python -m unittest -v test_csv_pc02.py"""
import contextlib
import os
import hashlib
import importlib.util
import io
import json
from pathlib import Path
import subprocess
import sys
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch
import zipfile

import pandas as pd

ROOT = Path(__file__).resolve().parent
spec = importlib.util.spec_from_file_location('pc02_fixed', ROOT/'botrytis_robustez_base.py')
bt = importlib.util.module_from_spec(spec)
spec.loader.exec_module(bt)


def frame():
    return pd.DataFrame({'epoch': [1, 2, 3, 4], 'train/box_loss': [1., .7, .4, .2],
        'train/cls_loss': [2., 1., .7, .3], 'train/dfl_loss': [1.2, 1., .9, .8],
        'metrics/mAP50-95(B)': [.7, .8, .9, .85], 'metrics/mAP50(B)': [.8, .9, .97, .95],
        'metrics/precision(B)': [.7, .8, .9, .85], 'metrics/recall(B)': [.7, .8, .9, .85],
        'val/box_loss': [.8, .7, .75, .9], 'val/cls_loss': [1., .7, .8, .9],
        'val/dfl_loss': [1.1, 1., 1.1, 1.2]})


class CSVCorrection(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.root = Path(self.tmp.name)
        self.path = self.root/'results.csv'

    def tearDown(self):
        self.tmp.cleanup()

    def save(self, data):
        data.to_csv(self.path, index=False)
        return self.path

    def check_unchanged(self, func, path=None):
        p = path or self.path
        before = p.read_bytes()
        result = func()
        self.assertEqual(p.read_bytes(), before)
        return result

    def test_reproduces_old_mixed_nan_failure_and_corrects_it(self):
        f = frame(); f.loc[1, 'val/box_loss'] = float('nan'); self.save(f)
        with self.assertRaisesRegex(bt.Stop, 'Loss no finita/incompleta'):
            bt._classification_curve_evidence(self.path, True, 3)
        r = self.check_unchanged(lambda: bt.curve_evidence(self.path, True, 3))
        self.assertEqual(r['best_value'], .9)
        self.assertEqual(r['loss_availability']['val/box_loss']['status'], 'PARTIAL')
        self.assertNotIn('val/box_loss', r['losses'])
        self.assertEqual(r['loss_availability']['val/box_loss']['missing_epochs'], [2])

    def test_absent_all_optional_validation_columns(self):
        self.save(frame().drop(columns=[c for c in frame() if c.startswith('val/')]))
        r = self.check_unchanged(lambda: bt.curve_evidence(self.path, True))
        for c in ('box', 'cls', 'dfl'):
            self.assertEqual(r['loss_availability'][f'val/{c}_loss']['status'], 'ND')
        self.assertEqual(len(r['warnings']), 3)

    def test_entirely_empty_validation_columns_and_empty_aggregates(self):
        f = frame(); f['val/box_loss'] = ''; f['train_loss'] = ''; f['val_loss'] = ''; self.save(f)
        r = bt.curve_evidence(self.path, True)
        self.assertNotIn('train_loss', r['losses'])
        self.assertEqual(r['loss_availability']['val/box_loss']['n_missing'], 4)
        self.assertEqual(r['loss_availability']['train/box_loss']['status'], 'AVAILABLE')

    def test_optional_nan_and_inf_are_diagnosed_not_filled_or_called_normal(self):
        f = frame().astype({'val/box_loss': object})
        f['val/box_loss'] = ['0.8', 'NaN', 'inf', '-inf']; self.save(f)
        r = self.check_unchanged(lambda: bt.curve_evidence(self.path, True))
        a = r['loss_availability']['val/box_loss']
        self.assertEqual((a['n_finite'], a['n_nan'], a['n_nonfinite']), (1, 1, 3))
        self.assertEqual(a['nonfinite_epochs'], [2, 3, 4])
        self.assertTrue(any(w['kind'] == 'VALIDATION_LOSS_NONFINITE' for w in r['warnings']))
        self.assertNotIn('val/box_loss', r['losses'])

    def test_finite_zero_is_not_missing(self):
        f = frame(); f['val/box_loss'] = 0.; self.save(f)
        r = bt.curve_evidence(self.path, True)
        self.assertEqual(r['losses']['val/box_loss']['final_window'], 0)
        self.assertEqual(r['loss_availability']['val/box_loss']['status'], 'AVAILABLE')

    def test_idempotent_inspection_preserves_checkpoint_and_source_bytes(self):
        f = frame(); f.loc[1, 'val/box_loss'] = float('nan'); self.save(f)
        weight = self.root/'best.pt'; weight.write_bytes(b'original checkpoint, not deserialized by CSV review')
        files = {p:p.read_bytes() for p in (self.path, weight)}
        first = bt.curve_evidence(self.path, True, 3)
        second = bt.curve_evidence(self.path, True, 3)
        self.assertEqual(first, second)
        self.assertEqual(files, {p:p.read_bytes() for p in files})

    def test_partial_validation_curve_plot_has_gaps_not_imputed_values(self):
        f = frame().astype({'val/box_loss': object}); f['val/box_loss'] = ['.8', '', 'inf', '.9']; self.save(f)
        audit = {'curve_diagnostics': [bt.curve_evidence(self.path, True)]}
        before = self.path.read_bytes()
        bt.export_curve_plots(audit, self.root/'curves')
        self.assertTrue(Path(audit['curve_figures'][0]).is_file())
        self.assertEqual(self.path.read_bytes(), before)

    def test_uses_actual_alternative_loss_names_without_inventing_alias(self):
        f = frame().rename(columns={'val/box_loss': 'val/giou_loss'}); self.save(f)
        r = bt.curve_evidence(self.path, True)
        self.assertIn('val/giou_loss', r['losses'])
        self.assertEqual(r['loss_availability']['val/box_loss']['status'], 'ND')
        self.assertIn('val/giou_loss', r['csv_inspection']['validation_loss_columns'])

    def test_invalid_training_loss_still_blocks(self):
        for bad in ('', 'NaN', 'inf', '-inf', 'garbage', -1):
            with self.subTest(bad=bad):
                f = frame().astype({'train/box_loss': object}); f.loc[1, 'train/box_loss'] = bad; self.save(f)
                before = self.path.read_bytes()
                with self.assertRaises(bt.Stop): bt.curve_evidence(self.path, True)
                self.assertEqual(before, self.path.read_bytes())

    def test_all_training_components_absent_still_blocks(self):
        self.save(frame().drop(columns=[c for c in frame() if c.startswith('train/')]))
        with self.assertRaisesRegex(bt.Stop, 'Faltan perdidas originales'): bt.curve_evidence(self.path, True)

    def test_invalid_objective_still_blocks(self):
        for bad in ('', 'NaN', 'inf', 'oops', -1, 1.5):
            with self.subTest(bad=bad):
                f = frame().astype({'metrics/mAP50-95(B)': object}); f.loc[2, 'metrics/mAP50-95(B)'] = bad; self.save(f)
                with self.assertRaises(bt.Stop): bt.curve_evidence(self.path, True)

    def test_missing_objective_blocks(self):
        self.save(frame().drop(columns='metrics/mAP50-95(B)'))
        with self.assertRaisesRegex(bt.Stop, 'Falta metrica objetiva'): bt.curve_evidence(self.path, True)

    def test_present_precision_recall_map50_invalid_still_block(self):
        for col in ('metrics/precision(B)', 'metrics/recall(B)', 'metrics/mAP50(B)'):
            for bad in ('', 'nan', 'inf', 'bad'):
                with self.subTest(col=col, bad=bad):
                    f = frame().astype({col: object}); f.loc[2, col] = bad; self.save(f)
                    with self.assertRaises(bt.Stop): bt.curve_evidence(self.path, True)

    def test_missing_diagnostic_metric_is_nd_not_fabricated(self):
        self.save(frame().drop(columns='metrics/precision(B)'))
        r = bt.curve_evidence(self.path, True)
        self.assertEqual(r['csv_inspection']['metric_availability']['precision']['status'], 'ND')
        self.assertIsNone(r['metrics_at_same_best_epoch']['precision'])

    def test_structural_corruption_detected_before_pandas_can_mask_it(self):
        cases = ['epoch,epoch\n1,1\n', 'epoch,x\n1\n', 'epoch,x\n1,2,3\n',
                 'epoch,x\n1,"unfinished\n', '', 'epoch,x\n', 'epoch,\n1,2\n']
        for text in cases:
            with self.subTest(text=text):
                self.path.write_text(text)
                with self.assertRaises(bt.Stop): bt.curve_evidence(self.path, True)

    def test_invalid_epoch_sequences_detected(self):
        for sequence in ([1, 2, 2, 4], [1, 2, 4, 5], [0, 1, 2, 3], [1, 2, 3.5, 4], [1, 3, 2, 4]):
            with self.subTest(sequence=sequence):
                f = frame(); f['epoch'] = sequence; self.save(f)
                with self.assertRaisesRegex(bt.Stop, 'Epocas incompletas'): bt.curve_evidence(self.path, True)

    def test_optional_loss_malformed_text_not_silently_coerced_to_nd(self):
        f = frame().astype({'val/box_loss': object}); f.loc[1, 'val/box_loss'] = '1.2BROKEN'; self.save(f)
        with self.assertRaisesRegex(bt.Stop, 'CSV no numerico'): bt.curve_evidence(self.path, True)

    def test_malformed_optional_negative_is_not_tolerated(self):
        f = frame(); f.loc[1, 'val/box_loss'] = -10; self.save(f)
        with self.assertRaisesRegex(bt.Stop, 'Perdida negativa'): bt.curve_evidence(self.path, True)

    def test_contradictory_metric_aliases_block(self):
        f = frame(); f['map50_95'] = .1; self.save(f)
        with self.assertRaisesRegex(bt.Stop, 'Aliases de metrica contradictorios'): bt.curve_evidence(self.path, True)

    def test_map95_cannot_exceed_map50(self):
        f = frame(); f['metrics/mAP50(B)'] = .1; self.save(f)
        with self.assertRaisesRegex(bt.Stop, 'mAP50-95 mayor'): bt.curve_evidence(self.path, True)

    def signed(self):
        signed = self.root/'completed_history.csv'
        f = frame().drop(columns=[c for c in frame() if c.startswith('val/')])
        f['map50_95'] = f['metrics/mAP50-95(B)']
        f['train_loss'] = ''; f['val_loss'] = ''; f.to_csv(signed, index=False)
        return signed

    def test_signed_and_original_agree_with_optional_gaps(self):
        signed = self.signed(); f = frame(); f.loc[2, 'val/box_loss'] = float('nan'); self.save(f)
        before = signed.read_bytes()
        r = self.check_unchanged(lambda: bt.curve_evidence(signed, True, 3))
        self.assertEqual(r['signed_history_sha256'], hashlib.sha256(before).hexdigest())
        self.assertEqual(before, signed.read_bytes())

    def test_signed_proves_csv_truncated_at_whole_record(self):
        signed = self.signed(); self.save(frame().iloc[:3])
        with self.assertRaisesRegex(bt.Stop, '(truncado/no concuerda|ausente)'): bt.curve_evidence(signed, True, 3)

    def test_signed_proves_valid_numeric_but_wrong_metric(self):
        signed = self.signed(); f = frame(); f.loc[1, 'metrics/mAP50-95(B)'] = .6; self.save(f)
        with self.assertRaisesRegex(bt.Stop, 'discrepa del historial firmado'): bt.curve_evidence(signed, True, 3)

    def test_signed_training_components_remain_checked(self):
        signed = self.signed(); f = frame(); f.loc[1, 'train/box_loss'] = 10; self.save(f)
        with self.assertRaisesRegex(bt.Stop, 'Loss de entrenamiento discrepa'): bt.curve_evidence(signed, True, 3)

    def test_selected_epoch_must_match_receipt_score(self):
        self.save(frame())
        with self.assertRaisesRegex(bt.Stop, 'seleccionada incompatible'): bt.curve_evidence(self.path, True, 1)

    def test_no_trailing_newline_is_a_valid_csv(self):
        self.save(frame()); self.path.write_bytes(self.path.read_bytes().rstrip(b'\n'))
        self.assertEqual(bt.curve_evidence(self.path, True)['best_value'], .9)

    def test_bom_padded_header_supported(self):
        f = frame(); f.columns = [' '+c+' ' for c in f.columns]; self.save(f)
        self.path.write_bytes(b'\xef\xbb\xbf'+self.path.read_bytes())
        self.assertEqual(bt.curve_evidence(self.path, True)['best_value'], .9)

    def test_classification_logic_unchanged(self):
        f = pd.DataFrame({'epoch': [1, 2, 3], 'train_loss': [1., .5, .2], 'val_loss': [1., .6, .4], 'val_ap': [.7, .9, .9]})
        self.save(f)
        self.assertEqual(bt.curve_evidence(self.path, False, 3), bt._classification_curve_evidence(self.path, False, 3))

    def test_no_hpo_or_baseline_entrypoint_and_no_other_pc(self):
        for stage in ('hpo', 'baseline'):
            p = subprocess.run([sys.executable, bt.__file__, '--_stage', stage, '--executor-sha', bt.EXECUTOR_SHA256,
                                '--project', str(self.root), '--state-dir', str(self.root/'not_created')], capture_output=True, text=True)
            self.assertEqual(p.returncode, 2); self.assertIn('prohibe ejecutarlo otra vez', p.stdout)
            self.assertFalse((self.root/'not_created').exists())
        for pc in ('PC-01', 'PC-03'):
            p = subprocess.run([sys.executable, bt.__file__, '--expected-pc', pc], capture_output=True, text=True)
            self.assertEqual(p.returncode, 2); self.assertIn('invalid choice', p.stderr)

    def test_hpo_scan_covers_all_30_complete_folds_and_available_pruned(self):
        trials = []
        for n in range(15):
            folder = self.root/f'trial_{n:03d}'; folder.mkdir()
            trials.append(SimpleNamespace(number=n, state=SimpleNamespace(name='COMPLETE' if n < 10 else 'PRUNED'), user_attrs={'run_dir': str(folder)}))
            for fold in range(3 if n < 10 else 1):
                sub = folder/f'fold{fold}'; sub.mkdir(); f = frame()
                if (n, fold) in ((0, 1), (2, 2), (9, 0)):
                    f.loc[1, 'val/box_loss'] = float('nan')
                f.to_csv(sub/'results.csv', index=False)
                if n < 10:
                    self.path = sub/'results.csv'; self.root_saved = self.root
                    (sub/'completed_fold.json').write_text(json.dumps({'row': {'best_epoch': 3}}))
                    f.drop(columns=['val/box_loss', 'val/cls_loss', 'val/dfl_loss']).to_csv(sub/'completed_history.csv', index=False)
        files = {p:p.read_bytes() for p in self.root.rglob('*') if p.is_file()}
        audit = {'curve_diagnostics': []}
        study = SimpleNamespace(trials=trials)
        with patch.object(bt, 'allowed_trial_folder', side_effect=lambda c, s, t:Path(t.user_attrs['run_dir'])), contextlib.redirect_stdout(io.StringIO()):
            bt.inspect_existing_hpo_curves(study, {}, SimpleNamespace(is_detector=True), audit)
        self.assertEqual(len(audit['curve_diagnostics']), 30)
        self.assertEqual(len(audit['pruned_csv_diagnostics']), 15)
        self.assertEqual(len([r for r in audit['curve_diagnostics'] if r['warnings']]), 3)
        self.assertEqual(audit['csv_errors'], [])
        self.assertEqual(files, {p:p.read_bytes() for p in files})

    def test_hpo_scan_collects_errors_and_does_not_approve(self):
        folder = self.root/'trial_000'; folder.mkdir()
        for fold in range(3):
            sub = folder/f'fold{fold}'; sub.mkdir(); f = frame(); f.loc[1, 'train/box_loss'] = float('inf')
            f.to_csv(sub/'results.csv', index=False); f.to_csv(sub/'completed_history.csv', index=False)
            (sub/'completed_fold.json').write_text(json.dumps({'row': {'best_epoch': 3}}))
        t = SimpleNamespace(number=0, state=SimpleNamespace(name='COMPLETE'), user_attrs={'run_dir': str(folder)})
        audit = {'curve_diagnostics': []}
        with patch.object(bt, 'allowed_trial_folder', return_value=folder):
            with self.assertRaisesRegex(bt.Stop, 'indispensable invalida'):
                bt.inspect_existing_hpo_curves(SimpleNamespace(trials=[t]), {}, SimpleNamespace(is_detector=True), audit)
        self.assertEqual(len(audit['csv_errors']), 3)

    def test_report_contains_loss_warnings_and_real_limitations(self):
        f = frame(); f.loc[1, 'val/box_loss'] = float('nan'); self.save(f)
        audit = {'pc_id': 'PC-02', 'model': 'yolo26l', 'curve_diagnostics': [bt.curve_evidence(self.path, True)]}
        p = self.root/'CONTINUACION.html'
        with contextlib.redirect_stdout(io.StringIO()): bt.write_pipeline_report(p, audit)
        self.assertIn('ND', p.read_text())
        self.assertIn('analisis de sobreajuste queda limitado', p.read_text())
        saved = bt.read(p.with_suffix('.json'))
        self.assertEqual(saved['csv_validation']['folds_with_warnings'], 1)
        json.dumps(saved, allow_nan=False)

    def test_native_queue_still_requires_present_operator(self):
        try:
            from orchestration.study_queue import Job, StudyQueue
        except ImportError:
            self.skipTest('Set PYTHONPATH to real repository src for native queue integration')
        queue = StudyQueue(self.root/'queue.sqlite')
        job = Job(job_id='hpo:yolo26l', kind='hpo', architecture='yolo26l', task='detection', study_name='existing', status='PENDING_REVIEW')
        queue.add_job(job)
        queue.finish(job.job_id, 'PENDING_REVIEW')
        f = frame(); f.loc[1, 'val/box_loss'] = float('nan'); self.save(f)
        evidence = self.root/'review.json'; bt.atomic_json(evidence, bt.curve_evidence(self.path, True))
        with patch.object(sys.stdin, 'isatty', return_value=True), patch('builtins.input', return_value='NO'), contextlib.redirect_stdout(io.StringIO()):
            with self.assertRaises(bt.ReviewRequired): bt.actual_review(queue, queue.get(job.job_id), evidence, bt.read(evidence))
        self.assertEqual(queue.get(job.job_id).status, 'PENDING_REVIEW')
        with patch.object(sys.stdin, 'isatty', return_value=True), patch('builtins.input', return_value='APROBAR'), contextlib.redirect_stdout(io.StringIO()):
            bt.actual_review(queue, queue.get(job.job_id), evidence, bt.read(evidence))
        self.assertEqual(queue.get(job.job_id).status, 'COMPLETE')

    def test_real_pc02_baseline_csvs_remain_valid(self):
        archives = ([Path(os.environ['BOTRYTIS_PRIVATE_PC02_ZIP'])] if os.environ.get('BOTRYTIS_PRIVATE_PC02_ZIP') else [])
        if not archives:
            archives = list((ROOT/'fixtures').glob('PC2.zip'))
        if not archives:
            self.skipTest('Original private PC2 baseline fixture not included in small correction download')
        count = 0
        with zipfile.ZipFile(archives[0]) as z:
            names = [n for n in z.namelist() if '/yolo26l/baseline/fold' in n and n.endswith('/results.csv') and '/previous_attempts/' not in n]
            for n in names:
                self.path.write_bytes(z.read(n))
                before = self.path.read_bytes(); r = bt.curve_evidence(self.path, True)
                self.assertGreater(r['best_value'], .8); self.assertEqual(before, self.path.read_bytes()); count += 1
        self.assertEqual(count, 3)


if __name__ == '__main__':
    unittest.main(verbosity=2)
