"""CPU control/regression tests. CUDA storage mapping is simulated; no GPU training."""
import contextlib
import hashlib
import importlib.util
import json
from pathlib import Path
import sys
from types import SimpleNamespace

import pytest
import torch

HERE = Path(__file__).parent
spec = importlib.util.spec_from_file_location('rng_resume_driver', HERE/'botrytis_robustez_reanudar.py')
r = importlib.util.module_from_spec(spec); spec.loader.exec_module(r)
bt = r.bt


class MappedCudaState(torch.Tensor):
    """A CPU-backed test double simulating torch.load mapping an RNG to CUDA."""
    @property
    def device(self):
        return torch.device('cuda:0')

    def cpu(self):
        return self.as_subclass(torch.Tensor).clone()


def rng():
    return torch.Generator().manual_seed(73).get_state()


def mapped_rng():
    return rng().as_subclass(MappedCudaState)


def test_gpu_mapped_rng_reproduces_failure_then_restores_identical_sequence(monkeypatch):
    state = mapped_rng(); reference = rng(); calls=[]; events=[]
    def setter(states):
        if any(x.device.type!='cpu' or x.dtype!=torch.uint8 for x in states):
            raise TypeError('RNG state must be a torch.ByteTensor')
        calls.append(states)
    monkeypatch.setattr(torch.cuda,'set_rng_state_all',setter)
    monkeypatch.setattr(torch.cuda,'device_count',lambda:1)
    with pytest.raises(TypeError,match='ByteTensor'):setter([state])
    with bt.classifier_rng_restore(lambda name,**kw:events.append((name,kw))):
        torch.cuda.set_rng_state_all([state])
    actual=calls[0][0]
    assert torch.equal(actual,reference) and state.device.type=='cuda'
    g1=torch.Generator();g2=torch.Generator()
    g1.set_state(reference);g2.set_state(actual)
    assert torch.equal(torch.rand(100,generator=g1),torch.rand(100,generator=g2))
    assert torch.cuda.set_rng_state_all is setter
    assert events[0][1]['reseeded'] is False
    assert events[0][1]['checkpoint_modified'] is False
    assert events[0][1]['state_sha256']==[hashlib.sha256(reference.numpy().tobytes()).hexdigest()]


@pytest.mark.parametrize('states,count',[
    (None,1),([],1),([rng()],2),([rng()],0),
    ([torch.ones(8,dtype=torch.float32)],1),
    ([torch.ones(8,dtype=torch.int64)],1),
    ([torch.ones(2,2,dtype=torch.uint8)],1),
    ([torch.empty(0,dtype=torch.uint8)],1),
    ([list(range(8))],1),
    ([rng(),torch.ones(8,dtype=torch.float32)],2),
])
def test_invalid_rng_never_installed_or_coerced(monkeypatch,states,count):
    calls=[];monkeypatch.setattr(torch.cuda,'device_count',lambda:count)
    original=lambda values:calls.append(values)
    monkeypatch.setattr(torch.cuda,'set_rng_state_all',original)
    with pytest.raises(bt.Stop,match='RNG CUDA'):
        with bt.classifier_rng_restore(lambda *a,**k:None):
            torch.cuda.set_rng_state_all(states)
    assert not calls and torch.cuda.set_rng_state_all is original


def test_multiple_devices_and_noncontiguous_cpu_states_preserve_bytes():
    a=torch.arange(32,dtype=torch.uint8)[::2];b=rng()
    values=bt.classifier_cuda_rng_cpu((a,b),2)
    assert all(v.device.type=='cpu' and v.is_contiguous() for v in values)
    assert torch.equal(values[0],a) and torch.equal(values[1],b)
    assert not a.is_contiguous()  # original unchanged


def test_setter_error_does_not_fallback_to_new_seed(monkeypatch):
    calls=[]
    def reject(states):calls.append(states);raise RuntimeError('invalid native RNG state size')
    monkeypatch.setattr(torch.cuda,'device_count',lambda:1)
    monkeypatch.setattr(torch.cuda,'set_rng_state_all',reject)
    with pytest.raises(RuntimeError,match='native RNG'):
        with bt.classifier_rng_restore(lambda *a,**k:None):torch.cuda.set_rng_state_all([rng()])
    assert len(calls)==1 and torch.cuda.set_rng_state_all is reject


def test_classifier_hook_preserves_checkpoint_and_runtime_config(tmp_path,monkeypatch):
    cp=tmp_path/'state.pt';torch.save({'epoch':8,'cuda_rng':[rng()]},cp)
    receipt=cp.with_suffix('.pt.resume.json');receipt.write_text('{"sha256":"test-fixture"}')
    before={p.name:p.read_bytes() for p in (cp,receipt)}
    hooks=object.__new__(bt.RuntimeHooks);hooks.partial_for=lambda *a:cp
    events=[];hooks.event=lambda name,**kw:events.append((name,kw))
    cfg={'classifier':{'_resume_context':{'fold':1}}};calls=[]
    monkeypatch.setattr(torch.cuda,'device_count',lambda:1)
    original_set=lambda values:calls.append(values)
    monkeypatch.setattr(torch.cuda,'set_rng_state_all',original_set)
    def train(spec,hp,runtime,*args,**kwargs):
        assert runtime['classifier']['resume_checkpoint']==str(cp)
        torch.cuda.set_rng_state_all([mapped_rng()])
        return 'continued epoch 9'
    assert hooks.classifier(train,SimpleNamespace(slug='vit_b16'),{'lr':1e-4},cfg,None,None)=='continued epoch 9'
    assert 'resume_checkpoint' not in cfg['classifier']
    assert before=={p.name:p.read_bytes() for p in (cp,receipt)}
    assert calls[0][0].device.type=='cpu' and torch.cuda.set_rng_state_all is original_set
    assert [e[0] for e in events]==['CHECKPOINT_RESUME','CLASSIFIER_RNG_RESTORE_CPU']
    def failed(*a,**k):raise RuntimeError('unrelated training failure')
    with pytest.raises(RuntimeError,match='unrelated'):hooks.classifier(failed,SimpleNamespace(slug='vit_b16'),{},cfg,None,None)
    assert torch.cuda.set_rng_state_all is original_set


def test_fresh_training_does_not_install_resume_hook(monkeypatch):
    hooks=object.__new__(bt.RuntimeHooks);hooks.partial_for=lambda *a:None
    monkeypatch.setattr(bt,'classifier_rng_restore',lambda *a:pytest.fail('unnecessary resume hook'))
    assert hooks.classifier(lambda *a,**k:'fresh',SimpleNamespace(slug='vit_b16'),{}, {'classifier':{}},None,None)=='fresh'


def test_pc03_only_vit_routes_main_without_hpo_or_rtdetr(tmp_path,monkeypatch):
    class LocalPath(type(tmp_path)):
        @classmethod
        def home(cls):return cls(tmp_path)
    project=tmp_path/'project';py=project/'.venv/Scripts/python.exe'
    py.parent.mkdir(parents=True);py.symlink_to(sys.executable)
    cfg={'compute':{'pc_id':'PC-03'}};calls=[];reports=[]
    import utils.config_loader as cl
    import orchestration.machine as machine
    monkeypatch.setattr(cl,'load_config',lambda p:cfg)
    monkeypatch.setattr(machine,'read_binding',lambda c:{'pc_id':'PC-03','machine_id':'I21313'})
    monkeypatch.setattr(r,'Path',LocalPath)
    monkeypatch.setattr(r,'os',SimpleNamespace(name='nt',environ={'USERNAME':'I21313'},chdir=lambda p:None))
    monkeypatch.setattr(bt,'find_project',lambda p:project)
    monkeypatch.setattr(bt,'read',lambda p:{'vit_b16':'vit.yaml','rtdetr_r50':'r50.yaml'})
    monkeypatch.setattr(bt,'atomic_json',lambda p,d:reports.append(d))
    monkeypatch.setattr(bt,'emit',lambda *a,**k:None)
    monkeypatch.setattr(r,'controller_lock',lambda *a:contextlib.nullcontext())
    monkeypatch.setattr(r,'wait_idle',lambda *a:contextlib.nullcontext())
    monkeypatch.setattr(r,'process_model',lambda *a:calls.append(a[3]) or {'model':a[3]})
    monkeypatch.setattr(sys,'argv',[str(r.SELF),'--expected-pc','PC-03','--only-model','vit_b16'])
    assert r.main()==0 and calls==['vit_b16']
    assert reports[-1]['deferred_models']==['rtdetr_r50']
    assert reports[-1]['baseline_or_hpo_launched'] is False and reports[-1]['test_opened'] is False
    calls.clear()
    monkeypatch.setattr(sys,'argv',[str(r.SELF),'--expected-pc','PC-03','--only-model','yolo26l'])
    with pytest.raises(bt.Stop,match='no pertenece'):r.main()
    assert not calls


def test_actual_native_classifier_resume_block_with_cuda_mapped_rng(tmp_path,monkeypatch):
    """Execute the real b706 resume block, mocking only GPU/model/file transport."""
    import ast, copy
    import utils.classifier_train as native
    import utils.resume as receipt_module
    source=ast.parse(Path(native.__file__).read_text())
    fn=next(x for x in source.body if isinstance(x,ast.FunctionDef) and x.name=='_train_once')
    block=next(x for x in fn.body if isinstance(x,ast.If) and isinstance(x.test,ast.Call)
               and isinstance(x.test.func,ast.Attribute) and x.test.func.attr=='get'
               and x.test.args and isinstance(x.test.args[0],ast.Constant) and x.test.args[0].value=='resume_checkpoint')
    code=compile(ast.fix_missing_locations(ast.Module(body=[block],type_ignores=[])),str(native.__file__),'exec')
    cp=tmp_path/'checkpoint.pt';cp.write_bytes(b'original checkpoint transport fixture')
    original_bytes=cp.read_bytes();restored={};installed=[]
    saved=dict(identity='same-experiment',model={'p':1},optimizer={'p':2},scaler={'p':3},
        history=[{'epoch':i} for i in range(1,9)],best={'epoch':7},bad_epochs=1,step=24,epoch=8,
        python_rng=('original',),numpy_rng=('original',),torch_rng=rng(),cuda_rng=[mapped_rng()],loader_rng=rng())
    def load(path,**kwargs):
        assert path==cp and kwargs['map_location']=='cuda:0' and kwargs['weights_only'] is False
        return dict(saved)
    def setter(states):
        if any(x.device.type!='cpu' for x in states):raise TypeError('RNG state must be a torch.ByteTensor')
        installed.extend(states)
    monkeypatch.setattr(torch,'load',load)
    monkeypatch.setattr(torch,'set_rng_state',lambda state:restored.update(torch_rng=state))
    monkeypatch.setattr(torch.cuda,'is_available',lambda:True)
    monkeypatch.setattr(torch.cuda,'device_count',lambda:1)
    monkeypatch.setattr(torch.cuda,'set_rng_state_all',setter)
    monkeypatch.setattr(receipt_module,'require_checkpoint',lambda *a:cp)
    loader=SimpleNamespace(generator=SimpleNamespace(set_state=lambda state:restored.update(loader_rng=state)))
    loadable=lambda key:SimpleNamespace(load_state_dict=lambda value:restored.update({key:value}))
    def environment():
        return dict(cc={'resume_checkpoint':str(cp)},spec=SimpleNamespace(slug='vit_b16'),cfg={},torch=torch,
            device='cuda:0',identity='same-experiment',model=loadable('model'),opt=loadable('optimizer'),
            scaler=loadable('scaler'),random=SimpleNamespace(setstate=lambda value:restored.update(python_rng=value)),
            np=SimpleNamespace(random=SimpleNamespace(set_state=lambda value:restored.update(numpy_rng=value))),
            train_loader=loader,max_epochs=40)
    with pytest.raises(TypeError,match='ByteTensor'):exec(code,environment())
    env=environment()
    with bt.classifier_rng_restore(lambda *a,**kw:None):exec(code,env)
    assert env['start_epoch']==9 and env['history']==saved['history'] and env['step']==24
    assert restored['model']==saved['model'] and restored['optimizer']==saved['optimizer'] and restored['scaler']==saved['scaler']
    assert torch.equal(installed[0],rng())
    assert original_bytes==cp.read_bytes()
