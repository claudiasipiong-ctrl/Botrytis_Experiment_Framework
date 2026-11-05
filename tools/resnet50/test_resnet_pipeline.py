import ast,contextlib,importlib.util
from pathlib import Path
from types import SimpleNamespace
import pytest
P=Path(__file__).parent
s=importlib.util.spec_from_file_location('resnet_entry',P/'botrytis_resnet50_pc01.py');m=importlib.util.module_from_spec(s);s.loader.exec_module(m)

def test_identity_is_only_pc01():
    base={'compute':{'pc_id':'PC-01'}};binding={'pc_id':'PC-01','machine_id':'I21315'}
    m.check_identity(base,binding,'i21315')
    for user in ['i21314','i21313','other']:
        with pytest.raises(RuntimeError):m.check_identity(base,binding,user)


def test_no_old_model_pause_or_execution_paths():
    tree=ast.parse((P/'botrytis_resnet50_pc01.py').read_text())
    attrs={x.attr for x in ast.walk(tree) if isinstance(x,ast.Attribute)}
    assert not {'pause_queue','choose_next','restore_pause'}&attrs
    assert m.MODEL=='resnet50'
    assert "'--only-model',MODEL" in (P/'botrytis_resnet50_pc01.py').read_text()


def test_robustness_dispatch_is_resnet_only(monkeypatch):
    seen=[]
    fake=SimpleNamespace(MODELS={'PC-01':['yolov8n','yolov8l']},SELF='dispatcher.py')
    fake.main=lambda:seen.extend(fake.MODELS['PC-01']) or 0
    monkeypatch.setattr(m,'load',lambda name:fake)
    assert m.robust()==0 and seen==['resnet50']


def test_weight_present_no_download_or_license_prompt(tmp_path,monkeypatch):
    import utils.weights as w
    path=tmp_path/'weights.pt';path.write_bytes(b'fixture')
    monkeypatch.setattr(w,'verified_local_weight',lambda *a:path)
    monkeypatch.setattr(m.subprocess,'run',lambda *a,**k:pytest.fail('unexpected download'))
    assert m.acquire_weight(tmp_path,{},SimpleNamespace(slug='resnet50'))==path


def test_baseline_launch_only_resnet_and_existing_baseline_not_repeated(tmp_path,monkeypatch):
    import utils.guards as guards
    import utils.durable as durable
    from utils.config_loader import load_config
    from utils.architectures import get_arch
    from orchestration.study_queue import StudyQueue
    cfg=load_config();cfg['compute']['pc_id']='PC-01'
    for key,folder in [('queue_dir','q'),('runs_dir','r'),('artifacts_dir','a')]:cfg['paths'][key]=str(tmp_path/folder)
    cfg['experiment'].update(architecture='resnet50',experiment_id='test',round_id='round')
    q=StudyQueue.from_cfg(cfg);calls=[];present=[False]
    pr=SimpleNamespace(wait_idle=lambda p:contextlib.nullcontext(),baseline_present=lambda *a:present[0],emit=lambda *a,**k:None)
    monkeypatch.setattr(guards,'gpu_lock_path',lambda *a:tmp_path/'gpu.json')
    monkeypatch.setattr(durable,'FileLock',lambda *a:contextlib.nullcontext())
    monkeypatch.setattr(m,'prepare_round',lambda *a:(tmp_path/'config',cfg,['paths.runs_dir='+str(tmp_path/'r')],tmp_path))
    import utils.config_loader as cl
    monkeypatch.setattr(cl,'load_config',lambda *a:cfg)
    monkeypatch.setenv('USERNAME','i21315')
    def launch(cmd,**kwargs):
        calls.append(cmd)
        baseline=next(j for j in q.jobs() if j.kind=='baseline')
        q.finish(baseline.job_id,'PENDING_REVIEW');return 0
    monkeypatch.setattr(m.subprocess,'call',launch)
    m.baseline(tmp_path,cfg,tmp_path,pr)
    assert len(calls)==1 and calls[0][calls[0].index('--architecture')+1]=='resnet50'
    assert calls[0][2].endswith('06_run_baseline.py')
    present[0]=True;m.baseline(tmp_path,cfg,tmp_path,pr)
    assert len(calls)==1


def test_outer_child_supports_resnet_only_on_pc01(tmp_path,monkeypatch):
    child=m.load('botrytis_train_val_continuar.py');calls=[]
    monkeypatch.setattr(child.bt,'child_entry',lambda *a:calls.append(a) or 0)
    argv=[str(child.SELF),'--executor-sha',child.SHA,'--_stage','worker','--project',str(tmp_path),'--state-dir',str(tmp_path),'--',
          '--pc-id','PC-01','--architectures','resnet50','--kinds','outer_train']
    monkeypatch.setattr(child.sys,'argv',argv)
    assert child.main()==0 and calls[0][0]=='worker'
    argv[argv.index('PC-01')]='PC-02'
    with pytest.raises(child.bt.Stop,match='fuera de alcance'):child.main()
    assert len(calls)==1


def test_dependencies_are_intact():
    import hashlib,json
    h=json.loads((P/'SHA256SUMS.json').read_text())
    assert all(hashlib.sha256((P/n).read_bytes()).hexdigest()==sha for n,sha in h.items())
    assert m.load('botrytis_resnet_hpo_pc01.py').MODEL=='resnet50'
    assert m.load('botrytis_resnet_hpo_pc01.py').PC=='PC-01'


def test_idle_guard_is_not_reentered_or_released_during_launch():
    calls=[]
    class Guard:
        def __enter__(self):pytest.fail('guard already acquired by wait_idle')
        def __exit__(self,*args):calls.append('released')
    with m.held_idle(SimpleNamespace(wait_idle=lambda lock:Guard()),'lock'):
        assert calls==[]
    assert calls==['released']
