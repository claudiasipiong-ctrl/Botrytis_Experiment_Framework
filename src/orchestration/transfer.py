"""Explicit offline ownership handshake: offer -> receive -> confirm -> activate.

The source is fenced before an offer leaves disk. A destination cannot claim
work until the operator delivers source confirmation. Never restore an old
ownership file to recover a transfer; retained disks are part of the protocol.
"""
from __future__ import annotations
import json
import functools
import uuid
from pathlib import Path
from orchestration.pc_registry import PCRegistry
from orchestration.study_queue import StudyQueue, RUNNING
from orchestration.exchange import pack_results, unpack
from utils.config_loader import cfg_path
from utils.durable import FileLock, atomic_json
from utils.hashing import sha256_file, sha256_json
from utils.experiments import now, fingerprint


def read(path):
    return json.loads(Path(path).read_text(encoding='utf-8'))


def authority(cfg):
    return cfg_path(cfg,'paths.artifacts_dir')/'ownership.json'


def update(cfg,slug,row):
    p=authority(cfg)
    data=read(p) if p.exists() else {}
    data[slug]=row
    atomic_json(p,data)


def offer(cfg,slug,source,target,bundle,*,operator,evidence):
    if not operator.strip() or not evidence.strip(): raise ValueError('Operador y evidencia requeridos')
    reg=PCRegistry(cfg);reg.validate(target)
    if source==target or reg.owner(slug)!=source: raise ValueError('Origen/destino incompatibles')
    q=StudyQueue.from_cfg(cfg)
    from utils.guards import gpu_lock_path
    # Same worker lock as the scheduler, covering claim + subprocess startup.
    with FileLock(gpu_lock_path(cfg,source).with_suffix('.worker.guard')):
        with FileLock(authority(cfg).with_suffix('.guard')):
            if any(j.architecture==slug and j.status==RUNNING for j in q.jobs()):
                raise ValueError('No se transfiere una arquitectura RUNNING')
            pack_results(cfg,bundle,source,cfg_path(cfg,'paths.runs_dir'))
            row=dict(id=uuid.uuid4().hex,slug=slug,source=source,target=target,
                     bundle_sha256=sha256_file(bundle),fingerprint=fingerprint(cfg),
                     operator=operator,evidence=evidence,at=now(),queue_state=q.transfer_snapshot(slug))
            # Durable fence first. Crash here remains safely blocked and operator-recoverable.
            update(cfg,slug,dict(owner=source,state='OFFERED',offer=row))
            atomic_json(Path(bundle).with_suffix('.offer.json'),row)
    return row


def serialized_transfer(function):
    @functools.wraps(function)
    def call(cfg,*args,**kwargs):
        with FileLock(authority(cfg).with_suffix('.guard')):
            return function(cfg,*args,**kwargs)
    return call

@serialized_transfer

def receive(cfg,pc_id,bundle,offer_path,receipt_path,*,operator):
    row=read(offer_path)
    if pc_id!=row['target'] or row['fingerprint']!=fingerprint(cfg) or row['bundle_sha256']!=sha256_file(bundle):
        raise ValueError('Oferta, destino o snapshot incompatibles')
    if not operator.strip(): raise ValueError('Operador obligatorio')
    reg=PCRegistry(cfg)
    prior=read(authority(cfg)).get(row['slug'],{}) if authority(cfg).exists() else {}
    recovering=prior.get('state')=='RECEIVED_PENDING_CONFIRMATION' and prior.get('offer',{}).get('id')==row['id']
    if reg.owner(row['slug'])!=row['source'] and not recovering: raise ValueError('Ownership local conflictivo')
    if any(j.architecture==row['slug'] and j.status==RUNNING for j in StudyQueue.from_cfg(cfg).jobs()):
        raise ValueError('Trabajo RUNNING local; conflicto')
    unpack(cfg,bundle,Path(bundle).parent,runs_dir=cfg_path(cfg,'paths.runs_dir'))
    receipt=dict(offer=row['id'],offer_sha256=sha256_json(row),bundle_sha256=row['bundle_sha256'],
                 target=pc_id,operator=operator,received_at=now())
    update(cfg,row['slug'],dict(owner=pc_id,state='RECEIVED_PENDING_CONFIRMATION',offer=row))
    StudyQueue.from_cfg(cfg).adopt_transfer(row['slug'],row['queue_state'])
    if Path(receipt_path).exists():
        existing=read(receipt_path)
        if existing.get('offer_sha256')!=sha256_json(row):raise ValueError('Recibo local distinto; preservar')
        return existing
    atomic_json(receipt_path,receipt)
    return receipt


@serialized_transfer
def confirm(cfg,slug,receipt_path,confirmation_path,*,operator):
    state=read(authority(cfg))[slug]; row=state['offer']; receipt=read(receipt_path)
    if state['state']=='ACTIVE' and state.get('confirmation',{}).get('receipt_sha256')==sha256_json(receipt):
        atomic_json(confirmation_path,state['confirmation']);return state['confirmation']
    if state['state']!='OFFERED' or receipt.get('offer_sha256')!=sha256_json(row) or receipt.get('target')!=row['target'] or receipt.get('bundle_sha256')!=row['bundle_sha256']:
        raise ValueError('Recibo no corresponde a esta oferta')
    if not operator.strip(): raise ValueError('Operador obligatorio')
    confirmation=dict(offer=row['id'],offer_sha256=sha256_json(row),receipt_sha256=sha256_json(receipt),
                      target=row['target'],operator=operator,confirmed_at=now())
    update(cfg,slug,dict(owner=row['target'],state='ACTIVE',offer=row,confirmation=confirmation))
    atomic_json(confirmation_path,confirmation)
    return confirmation


@serialized_transfer
def activate(cfg,slug,receipt_path,confirmation_path,*,operator):
    state=read(authority(cfg))[slug]; row=state['offer']; receipt=read(receipt_path); conf=read(confirmation_path)
    if state['state']!='RECEIVED_PENDING_CONFIRMATION' or conf.get('offer_sha256')!=sha256_json(row) or conf.get('receipt_sha256')!=sha256_json(receipt) or conf.get('target')!=row['target']:
        raise ValueError('Confirmacion incompatible o replay')
    if not operator.strip(): raise ValueError('Operador obligatorio')
    update(cfg,slug,dict(owner=row['target'],state='ACTIVE',offer=row,confirmation=conf,activated_by=operator))
    return conf
