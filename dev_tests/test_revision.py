"""DEV only: synthetic fixtures and static contracts, never operational QA."""
from __future__ import annotations
import itertools,json,os,random,subprocess,sys,tempfile,unittest
from pathlib import Path
from unittest.mock import patch
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT/'src'))
from utils.pairing import pair_image,maximum_assignment
def rec(c,o,b=(0.,0.,10.,10.)):
    return dict(class_id=c,ordinal=o,line_no=o+1,x1=b[0],y1=b[1],x2=b[2],y2=b[3],image_id='fixture',image_w=100,image_h=100)
class MatchingTests(unittest.TestCase):
    def test_equality(self):
        d,c,a=pair_image([rec(0,0),rec(1,0)]);self.assertEqual(d[0]['iou'],1);self.assertFalse(a)
    def test_no_sign_not_negative(self):
        import pandas as pd
        from utils.labels import build_fruit_table
        cfg={'pairing':{'min_iou':.5},'classification_labels':{'positive_label':'con_signos','negative_label':'sin_signos'}}
        t=build_fruit_table(pd.DataFrame([rec(0,0),rec(1,0)]),cfg);self.assertEqual(t.label.tolist(),['con_signos'])
    def test_clean_negative(self):
        import pandas as pd
        from utils.labels import build_fruit_table
        cfg={'pairing':{'min_iou':.5},'classification_labels':{'positive_label':'con_signos','negative_label':'sin_signos'}}
        t=build_fruit_table(pd.DataFrame([rec(0,0),rec(2,0)]),cfg);self.assertEqual(t.label.tolist(),['sin_signos'])
    def test_many_to_one(self):
        d,c,a=pair_image([rec(0,0),rec(1,0),rec(1,1)]);self.assertTrue(a);self.assertEqual(sum(r['fruit_ordinal'] is not None for r in d),1)
    def test_tie_deterministic(self):
        rows=[rec(0,0),rec(0,1),rec(1,0)]
        self.assertEqual(pair_image(rows),pair_image(list(reversed(rows))));self.assertIn('global_assignment_tie',str(pair_image(rows)[2]))
    def test_unmatched_positive_blocks_negative(self):
        import pandas as pd
        from utils.labels import build_fruit_table
        cfg={'pairing':{'min_iou':.5},'classification_labels':{'positive_label':'con_signos','negative_label':'sin_signos'}}
        t=build_fruit_table(pd.DataFrame([rec(0,0),rec(1,0,(50,50,60,60))]),cfg);self.assertTrue(t.label.isna().all())
    def test_global_not_greedy(self):
        a,v=maximum_assignment([[.8,.7],[.7,.1]]);self.assertEqual(a,[1,0])
    def test_global_bruteforce(self):
        rng=random.Random(67)
        for n in range(1,4):
            for _ in range(10):
                w=[[rng.choice([0.,.5,.6,1.]) for j in range(n)] for i in range(n)]
                a,_=maximum_assignment(w);best=max(sum((w[i][j] if j<n else 0) for i,j in enumerate(p)) for p in itertools.permutations(range(2*n),n))
                self.assertAlmostEqual(sum(w[i][j] if j<n else 0 for i,j in enumerate(a)),best)
    def test_threshold_boundary(self):
        self.assertFalse(pair_image([rec(0,0),rec(1,0,(0,0,5,10))],.5)[2])
        self.assertTrue(pair_image([rec(0,0),rec(1,0,(0,0,5,10))],.51)[2])
    def test_invalid_threshold(self):
        for v in (0.,-1.,float('nan'),1.1):
            with self.assertRaises(ValueError):pair_image([],v)
    def test_duplicate_id(self):self.assertIn('duplicate_ordinal',str(pair_image([rec(0,0),rec(0,0,(20,20,30,30))])[2]))
    def test_no_fruits(self):self.assertTrue(pair_image([rec(1,0)])[2])
class ConfigAndMetricsTests(unittest.TestCase):
    def test_config_contract(self):
        from utils.config_loader import load_config
        c=load_config();self.assertEqual(c['outer_split']['seed'],67);self.assertEqual(c['inner_hpo']['folds'],3);self.assertEqual(c['robustness']['seeds'],[17,42,73]);self.assertEqual(c['classifier']['objective'],'average_precision_positive')
    def test_seven_models(self):
        from utils.architectures import archs
        self.assertEqual(len(archs('detection')),4);self.assertEqual(len(archs('classification')),3)
        from utils.weights import registry
        self.assertEqual({s.slug for s in archs()},{r['slug'] for r in registry()['models']})
    def test_r50_block(self):
        from utils.weights import verified_local_weight
        from utils.architectures import get_arch
        with self.assertRaisesRegex(RuntimeError,'BLOQUEADO'):verified_local_weight(get_arch('rtdetr_r50'))
    def test_no_weights_assumed(self):
        from utils.weights import registry
        self.assertTrue(all(r['state']=='NO_DISPONIBLE' for r in registry()['models']))
    def test_no_pretrained_scratch_mix(self):
        from utils.config_loader import load_config,FixedDecisionViolation
        with self.assertRaises(FixedDecisionViolation):load_config(overrides=['initialization.mode=scratch'])
    def test_split_leakage(self):
        from utils.splits import assign_outer_split,allocate_counts
        ids=[str(i) for i in range(329)];a=assign_outer_split(ids,{'train':.7,'validation':.2,'test':.1},67,{i:int(i)%2 for i in ids})
        self.assertEqual(len(a),329);self.assertEqual(a,assign_outer_split(list(reversed(ids)),{'train':.7,'validation':.2,'test':.1},67,{i:int(i)%2 for i in ids}));self.assertEqual(allocate_counts(329,{'train':.7,'validation':.2,'test':.1}),{'train':230,'validation':66,'test':33})
    def test_test_closed(self):
        from utils.splits import assert_split_allowed,TestSetLockedError
        from utils.config_loader import load_config
        with self.assertRaises(TestSetLockedError):assert_split_allowed('test','hpo_trial',load_config())
    def test_release_missing(self):
        from utils.manual_gates import require_test_release
        with tempfile.TemporaryDirectory() as d:
            f=Path(d)/'freeze.json';f.write_text('{}')
            with self.assertRaises(RuntimeError):require_test_release({'paths':{'artifacts_dir':d}},f)
    def test_global_error(self):
        from utils.metrics import pipeline_error_rates
        r=pipeline_error_rates(10,2,3,8);self.assertAlmostEqual(r['E_global'],.5);self.assertAlmostEqual(r['E_global'],r['E_det']+(1-r['E_det'])*r['E_cls'])
        self.assertIsNone(pipeline_error_rates(0,0,0,0)['E_global']);self.assertIsNone(pipeline_error_rates(4,4,0,0)['E_cls'])
    def test_owner_mapping(self):
        from orchestration.pc_registry import PCRegistry
        from utils.config_loader import load_config
        r=PCRegistry(load_config());self.assertEqual(len(r.ownership()),7);self.assertEqual(r.owner("yolov8l"),"PC-01");self.assertEqual(r.owner("rtdetr_r50"),"PC-03")
class LoggingTests(unittest.TestCase):
    def test_redaction(self):
        from utils.bootstrap import redact
        self.assertNotIn('s3cr3t',str(redact({'password':'s3cr3t','x':'token=s3cr3t','args':['--token','s3cr3t']})))
    def test_bootstrap_failure_persists(self):
        with tempfile.TemporaryDirectory() as d:
            p=Path(d)/'broken.py';p.write_text('from utils.bootstrap import install\ninstall(__file__)\nprint("stdout before failure")\nimport missing_dependency_for_unit_fixture\n',encoding='utf-8')
            env={**os.environ,'BOTRYTIS_BOOTSTRAP_DIR':str(Path(d)/'logs')}
            proc=subprocess.run([sys.executable,str(p)],capture_output=True,text=True,env=env)
            self.assertNotEqual(proc.returncode,0);files=list((Path(d)/'logs').glob('*/execution.log'));self.assertEqual(len(files),1)
            log=files[0].read_text(encoding='utf-8');self.assertIn('stdout before failure',log);self.assertIn('ModuleNotFoundError',log);self.assertIn('Traceback',log)
            self.assertEqual(json.loads(files[0].with_name('final_status.json').read_text())['status'],'FAILED')
    def test_subprocess_and_returncode(self):
        with tempfile.TemporaryDirectory() as d:
            p=Path(d)/'child.py';p.write_text('from utils.bootstrap import install\ninstall(__file__)\nimport subprocess,sys\nr=subprocess.run([sys.executable,"-c","import sys;print(123);sys.exit(7)"],capture_output=True,text=True)\nprint(r.returncode)\n',encoding='utf-8')
            env={**os.environ,'BOTRYTIS_BOOTSTRAP_DIR':str(Path(d)/'logs')}
            proc=subprocess.run([sys.executable,str(p)],capture_output=True,text=True,env=env,timeout=30);self.assertEqual(proc.returncode,0,proc.stderr)
            log=next((Path(d)/'logs').glob('*/execution.log')).read_text();self.assertIn('123',log)
            events=next((Path(d)/'logs').glob('*/events.jsonl')).read_text();self.assertIn('"returncode": 7',events)
    def test_runcontext_exception_and_reuse(self):
        from utils.logging_utils import RunContext
        with tempfile.TemporaryDirectory() as d:
            c={'paths':{'runs_dir':d,'artifacts_dir':str(Path(d)/'art')},'logging':{'console':False},'reproducibility':{}}
            with patch('utils.logging_utils.RunContext._detect_hashes'):
                with self.assertRaisesRegex(ValueError,'fixture failure'):
                    with RunContext(c,'fixture',runs_dir=d,trial_name='same',reuse_existing=True,env_override={}) as ctx:print('printed evidence');raise ValueError('fixture failure')
                first=ctx.run_id
                self.assertIn('printed evidence',(ctx.run_dir/'execution.log').read_text())
                self.assertEqual(json.loads((ctx.run_dir/'metadata.json').read_text())['status'],'FAILED')
                with RunContext(c,'fixture',runs_dir=d,trial_name='same',reuse_existing=True,env_override={}) as other:pass
                self.assertNotEqual(first,other.run_id);self.assertTrue(list((other.run_dir/'previous_attempts').glob('*/metadata.json')))
    def test_manual_gate_pending(self):
        from utils.manual_gates import require_gate
        with tempfile.TemporaryDirectory() as d:
            with self.assertRaisesRegex(RuntimeError,'QA_MANUAL_PENDIENTE'):require_gate({'manual_qa':{'ledger_dir':d}},'PC-01','READY_FOR_EXPERIMENT')
class FigureTests(unittest.TestCase):
    def test_catalog_states_traceability(self):
        from utils.figures_all import catalog,generate
        with tempfile.TemporaryDirectory() as d:
            specs=catalog();rows=generate(d,catalog_only=True);self.assertEqual(len(rows),60);self.assertEqual(len({r['id'] for r in rows}),60)
            self.assertTrue(all(r['reason'] and r['required_command'] for r in rows));self.assertEqual(sum(r['status'].startswith('PENDIENTE') for r in rows),56)
            for r in rows:
                if r['path']:self.assertTrue(Path(r['path']).is_file());self.assertEqual(len(r['sha256']),64)
    def test_source_hash_and_semantics(self):
        from utils.figures_all import load_source,catalog,InsufficientData
        with self.assertRaises(InsufficientData):load_source(catalog()[0],{})
    def test_synthetic_plot_renderer(self):
        import pandas as pd
        from utils.figures_all import render,catalog
        with tempfile.TemporaryDirectory() as d:
            p=Path(d)/'fixture.csv';df=pd.DataFrame({'architecture':['fixture_A','fixture_B'],'map50_95':[.2,.3]});df.to_csv(p,index=False)
            item={'dataset':'SYNTHETIC_DEV_FIXTURE','split':'synthetic','phase':'DEV_CHECK','metric':'map50_95','unit':'fraction','denominator':'2 synthetic rows','configuration':'fixture'}
            paths=render(next(s for s in catalog() if s['id']=='map_arch'),df,item,p,Path(d));self.assertEqual(len(paths),3);self.assertTrue(all(x.stat().st_size>100 for x in paths))
if __name__=='__main__':unittest.main()


class AddedRegressionTests(unittest.TestCase):
    def test_early_configuration_failure_logged(self):
        from utils.logging_utils import run_cli
        with tempfile.TemporaryDirectory() as d:
            with patch("utils.logging_utils.collect_environment",return_value={}), patch("utils.logging_utils.RunContext._detect_hashes"):
                code=run_cli("unit_bad_config",lambda ctx,args:None,description="DEV fixture",argv=["--config",str(Path(d)/"absent.yaml"),"--runs-dir",d])
            self.assertNotEqual(code,0)
            m=list(Path(d).rglob("metadata.json"));self.assertEqual(len(m),1)
            self.assertEqual(json.loads(m[0].read_text(encoding="utf-8"))["status"],"FAILED")
            self.assertIn("Traceback",m[0].with_name("execution.log").read_text(encoding="utf-8"))
    def test_printed_token_redacted_across_writes(self):
        from utils.logging_utils import RunContext
        with tempfile.TemporaryDirectory() as d:
            cfg={"paths":{"runs_dir":d},"logging":{"console":False},"reproducibility":{},"api_token":"fixture_secret"}
            with patch("utils.logging_utils.RunContext._detect_hashes"):
                with RunContext(cfg,"fixture_secret_test",runs_dir=d,env_override={}) as c:
                    print("token=", "fixture_secret",sep="")
            for name in ("execution.log","config.yaml","metadata.json"):
                self.assertNotIn("fixture_secret",(c.run_dir/name).read_text(encoding="utf-8").replace("fixture_secret_test","run"))
    def test_local_lock_excludes_second_writer(self):
        from utils.durable import FileLock
        with tempfile.TemporaryDirectory() as d:
            p=Path(d)/"fixture.lock"
            with FileLock(p):
                with self.assertRaises(RuntimeError):
                    with FileLock(p):pass
    def test_unc_paths_rejected(self):
        from utils.config_loader import resolve_path,ConfigError
        with self.assertRaises(ConfigError):resolve_path("//server/share/file")
    def test_principal_detector_two_panels_authentic_components(self):
        import pandas as pd
        from utils.figures_all import render,catalog
        with tempfile.TemporaryDirectory() as d:
            p=Path(d)/"synthetic_DEV_history.csv"
            df=pd.DataFrame({"epoch":[1,2],"run_id":["SYNTHETIC_DEV","SYNTHETIC_DEV"],"map50_95":[.1,.2],"train/box_loss":[2.,1.],"val/box_loss":[2.1,1.2]})
            df.to_csv(p,index=False)
            item={"dataset":"SYNTHETIC_DEV_FIXTURE","split":"synthetic","phase":"DEV_CHECK","metric":"map50_95","unit":"fraction","denominator":"2 synthetic epochs","configuration":"fixture"}
            paths=render(next(s for s in catalog() if s["id"]=="det_final_two_panel"),df,item,p,Path(d))
            self.assertEqual(len(paths),3);self.assertTrue((Path(d)/"det_final_two_panel.N_por_epoca.csv").is_file())
