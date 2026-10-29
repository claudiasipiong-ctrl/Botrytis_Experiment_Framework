"""Stdlib-only trace installed before optional imports, configuration or CLI parsing."""
from __future__ import annotations
import atexit,datetime,hashlib,importlib.metadata,json,os,platform,re,subprocess,sys,threading,time,traceback,uuid
from pathlib import Path
_lock=threading.RLock();_active=None
SENSITIVE=re.compile(r'(password|passwd|token|secret|api[_-]?key|authorization|credential)',re.I)
def redact(value,key=''):
    if SENSITIVE.search(str(key)):return '[REDACTED]'
    if isinstance(value,dict):return {k:redact(v,k) for k,v in value.items()}
    if isinstance(value,(list,tuple)):
        out=[];hide=False
        for v in value:
            out.append('[REDACTED]' if hide else redact(v));hide=isinstance(v,str) and v.startswith('-') and bool(SENSITIVE.search(v))
        return out
    if not isinstance(value,str):return value
    s=re.sub(r'(https?://)[^/@\s]+:[^/@\s]+@',r'\1[REDACTED]@',value)
    s=re.sub(r'(?i)(bearer\s+)[A-Za-z0-9_.+/=-]+',r'\1[REDACTED]',s)
    return re.sub(r'(?i)((?:password|passwd|token|secret|api[_-]?key|authorization)\s*[=:]\s*)[^\s,;\]}]+',r'\1[REDACTED]',s)
def utc():return datetime.datetime.now(datetime.timezone.utc).isoformat()
def _atomic(path,value):
    p=Path(path);tmp=p.with_name(p.name+'.'+uuid.uuid4().hex+'.tmp')
    with tmp.open('x',encoding='utf-8') as f:json.dump(redact(value),f,ensure_ascii=False,indent=2,default=str);f.flush();os.fsync(f.fileno())
    os.replace(tmp,p)
def event(kind,**data):
    if _active:
        with _lock:
            with (_active['dir']/'events.jsonl').open('a',encoding='utf-8') as f:
                f.write(json.dumps(redact({'utc':utc(),'event':kind,'run_id':_active['id'],**data}),ensure_ascii=False,default=str)+'\n');f.flush();os.fsync(f.fileno())
def record_config(config, status="CONFIG_LOADED"):
    if not _active:return
    payload={"status":status,"utc":utc(),"effective_config":redact(config)}
    _atomic(_active["dir"]/"config.json",payload)
    event("effective_configuration",status=status,configuration=redact(config))
def record(text):
    if _active:
        with _lock:
            with (_active['dir']/'execution.log').open('a',encoding='utf-8') as f:f.write(redact(str(text)));f.flush()
class Tee:
    def __init__(self,stream):self.stream=stream;self.pending='';self.encoding='utf-8';self.errors='replace'
    def write(self,s):
        with _lock:
            self.pending+=str(s)
            while '\n' in self.pending:
                line,self.pending=self.pending.split('\n',1);clean=redact(line+'\n');self.stream.write(clean);record(clean)
        return len(s)
    def flush(self):
        with _lock:
            if self.pending:
                clean=redact(self.pending);self.stream.write(clean);record(clean);self.pending=''
            self.stream.flush()
    def isatty(self):return False
    def fileno(self):return self.stream.fileno()
    def reconfigure(self,**kwargs):pass
def _install_subprocess():
    original=subprocess.Popen
    class TracedPopen(original):
        def __init__(self,*args,**kw):
            event('subprocess_start',command=kw.get('args',args[0] if args else None),cwd=kw.get('cwd'))
            self._trace_threads=[];self._trace_inherit=kw.get('stdout') is None and kw.get('stderr') is None
            if self._trace_inherit:kw.update(stdout=subprocess.PIPE,stderr=subprocess.PIPE)
            super().__init__(*args,**kw)
            if self._trace_inherit:
                def pump(pipe,stream):
                    for line in iter(pipe.readline,'' if self.text_mode else b''):
                        stream.write(line if isinstance(line,str) else line.decode('utf-8',errors='replace'));stream.flush()
                for pipe,stream in ((self.stdout,sys.stdout),(self.stderr,sys.stderr)):
                    th=threading.Thread(target=pump,args=(pipe,stream),daemon=True);th.start();self._trace_threads.append(th)
        def communicate(self,*a,**k):
            if self._trace_inherit:
                self.wait(timeout=k.get('timeout'))
                return None,None
            out,err=super().communicate(*a,**k)
            for name,txt in (('stdout',out),('stderr',err)):
                if txt:record(f'\n[subprocess {name}]\n'+(txt.decode('utf-8','replace') if isinstance(txt,bytes) else txt))
            event('subprocess_exit',pid=self.pid,returncode=self.returncode);return out,err
        def wait(self,*a,**k):
            code=super().wait(*a,**k)
            for th in self._trace_threads:th.join(timeout=5)
            event('subprocess_exit',pid=self.pid,returncode=code);return code
    subprocess.Popen=TracedPopen
def install(script,root=None):
    global _active
    if _active:return _active['dir']
    root=Path(root or Path(__file__).resolve().parents[2]);pc=os.environ.get('BOTRYTIS_PC_ID','PC_NO_DECLARADA')
    phase=Path(script).stem;argv=list(sys.argv)
    for flag in ('--pc-id',):
        if flag in argv and len(argv)>argv.index(flag)+1:pc=argv[argv.index(flag)+1]
    tags=[]
    for flag in ('--architecture','--trial','--fold','--seed'):
        if flag in argv and len(argv)>argv.index(flag)+1:tags.append(flag.lstrip('-')+'_'+argv[argv.index(flag)+1])
    safe=lambda s:re.sub('[^A-Za-z0-9_.-]+','-',str(s))[:180]
    rid=safe(datetime.datetime.now(datetime.timezone.utc).strftime('%Y%m%dT%H%M%S.%fZ')+'_'+pc+'_'+phase+'_'+'_'.join(tags)+'_'+uuid.uuid4().hex)
    base=Path(os.environ.get('BOTRYTIS_BOOTSTRAP_DIR',str(root/'runs'/'v4'/'_bootstrap')))
    directory=base/rid;directory.mkdir(parents=True,exist_ok=False)
    _active={'dir':directory,'id':rid,'status':'RUNNING','returncode':0,'started':time.monotonic()}
    sys.stdout=Tee(sys.stdout);sys.stderr=Tee(sys.stderr)
    event('bootstrap_start',command=argv,pc_id=pc,hostname=platform.node(),phase=phase,parent=os.environ.get('BOTRYTIS_PARENT_RUN'))
    os.environ['BOTRYTIS_PARENT_RUN']=rid
    _atomic(directory/'metadata.json',{'run_id':rid,'status':'RUNNING','started_utc':utc(),'argv':redact(argv),'pc_id':pc})
    packages={d.metadata.get('Name','unknown'):d.version for d in importlib.metadata.distributions()}
    _atomic(directory/'environment.json',{'python':sys.version,'executable':sys.executable,'platform':platform.platform(),'packages':packages,'gpu_cuda':'PENDIENTE: entorno completo en RunContext; no se inicializa CUDA en bootstrap'})
    (directory/'execution.log').touch(exist_ok=True)
    _atomic(directory/'config.json',{'status':'PENDIENTE_CONFIG','command':argv})
    old_hook=sys.excepthook
    def hook(typ,exc,tb):
        _active.update(status='INTERRUPTED' if issubclass(typ,KeyboardInterrupt) else 'FAILED',returncode=130 if issubclass(typ,KeyboardInterrupt) else 1)
        record(''.join(traceback.format_exception(typ,exc,tb)));event('unhandled_exception',type=typ.__name__,message=str(exc));old_hook(typ,exc,tb)
    sys.excepthook=hook
    original_exit=sys.exit
    def traced_exit(code=0):
        set_exit(code);original_exit(code)
    sys.exit=traced_exit
    def profile(frame,evt,arg):
        if evt=='return' and frame.f_code.co_name in ('main','run_cli') and isinstance(arg,int):set_exit(arg)
    # Covers `raise SystemExit(main())` without re-running the user's entrypoint.
    sys.setprofile(profile)
    import argparse
    original_parse=argparse.ArgumentParser.parse_args
    def parse_with_trace(parser,*args,**kwargs):
        try:
            parsed=original_parse(parser,*args,**kwargs)
            record_config({"arguments":vars(parsed)},"CLI_ARGUMENTS_EFFECTIVE")
            return parsed
        except BaseException as exc:
            record_config({"argv":argv,"reason":str(exc)},"HELP_OR_ARGUMENT_FAILURE")
            raise
    argparse.ArgumentParser.parse_args=parse_with_trace
    _install_subprocess()
    manual_acquisition = phase=="manage_weights" and "acquire" in argv and "--allow-network" in argv and "--accept-license" in argv
    def network_audit(name,args):
        if name in ("socket.connect","socket.bind") and not manual_acquisition:
            event("network_blocked",operation=name,reason="Solo archivos/terminal locales; adquisicion manual separada")
            raise PermissionError("Operacion de red/puerto bloqueada en este comando local. No se usan equipos remotos.")
    sys.addaudithook(network_audit)
    atexit.register(finish)
    return directory
def set_exit(code):
    if _active:
        c=code if isinstance(code,int) else (0 if code is None else 1)
        _active.update(returncode=c,status='COMPLETE' if c==0 else 'INTERRUPTED' if c==130 else 'FAILED')
def finish():
    if not _active:return
    sys.stdout.flush();sys.stderr.flush()
    if _active['status']=='RUNNING':_active['status']='COMPLETE'
    event('bootstrap_end',status=_active['status'],returncode=_active['returncode'],duration_seconds=time.monotonic()-_active['started'])
    _atomic(_active['dir']/'final_status.json',{'run_id':_active['id'],'status':_active['status'],'returncode':_active['returncode'],'finished_utc':utc()})
