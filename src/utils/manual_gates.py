"""Human approval records. Writing tests never grants operational QA or final-test access."""
from __future__ import annotations
import json,hashlib
from pathlib import Path
from utils.config_loader import cfg_path,PROJECT_ROOT
GATES=('IDENTITY','ENVIRONMENT_CUDA','WEIGHTS','ANNOTATION_REVIEW','SPLIT_FOLDS','DERIVED_DATA','LOGGING_RECOVERY','SMOKE','READY_FOR_EXPERIMENT','SELECTION','THRESHOLDS_PIPELINE','FREEZE','TEST_RELEASE','TEST_FINAL','BENCHMARK','FIGURES')
def ledger_path(cfg,pc_id):
    if pc_id not in ('PC-01','PC-02','PC-03'):raise RuntimeError('PC explicita obligatoria; no se deduce del hostname')
    return PROJECT_ROOT/cfg.get('manual_qa',{}).get('ledger_dir','artifacts/manual_qa')/(pc_id+'.json')
def require_gate(cfg,pc_id,gate):
    p=ledger_path(cfg,pc_id)
    m=json.loads(p.read_text(encoding='utf-8')) if p.is_file() else {}
    row=m.get('gates',{}).get(gate,{})
    if row.get('state')!='APROBADO_POR_USUARIO' or not row.get('evidence') or not row.get('operator'):
        raise RuntimeError(f'QA_MANUAL_PENDIENTE: {pc_id}/{gate}. Complete CHECKLIST_QA_MANUAL y registre personalmente tools/manual_gates.py record.')
    import socket
    if row.get("hostname")!=socket.gethostname():raise RuntimeError("QA_MANUAL_PENDIENTE: aprobacion personal de otro equipo")
    from utils.hashing import compute_code_hash
    from utils.semantics import semantic_fingerprint
    if row.get('code_hash')!=compute_code_hash(PROJECT_ROOT) or row.get('semantic_fingerprint')!=semantic_fingerprint(cfg):raise RuntimeError('QA_MANUAL_PENDIENTE: codigo/semantica cambiaron desde la aprobacion personal')
def require_test_release(cfg,freeze):
    p=cfg_path(cfg,'paths.artifacts_dir')/'manual_test_release.json'
    m=json.loads(p.read_text(encoding='utf-8')) if p.is_file() else {}
    from utils.freeze import freeze_path
    # splits passes the validated manifest dict; authorization binds the actual file.
    actual = freeze_path(cfg) if isinstance(freeze, dict) else Path(freeze)
    if not actual.is_file(): raise RuntimeError('TEST_RELEASE: freeze ausente')
    if isinstance(freeze, dict) and json.loads(actual.read_text(encoding='utf-8')) != freeze:
        raise RuntimeError('TEST_RELEASE: freeze cargado distinto del archivo')
    h=hashlib.sha256(actual.read_bytes()).hexdigest()
    if m.get('state')!='AUTORIZADO_PERSONALMENTE' or m.get('freeze_sha256')!=h or not m.get('operator') or not m.get('evidence'):
        raise RuntimeError('PENDIENTE_USUARIO: test cerrado. Tras freeze, ejecute personalmente tools/manual_gates.py release-test y conserve evidencia.')
