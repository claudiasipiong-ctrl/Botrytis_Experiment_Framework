"""Adquisicion SOLO bajo invocacion manual explicita; inventory nunca descarga."""
from __future__ import annotations

import sys as _boot_sys
from pathlib import Path as _BootPath
_boot_root=_BootPath(__file__).resolve().parents[1]
_boot_sys.path.insert(0,str(_boot_root/"src"))
from utils import bootstrap as _bootstrap
_bootstrap.install(__file__,_boot_root)
import argparse,json,sys,urllib.request,hashlib,os
from datetime import datetime,timezone
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'src'))
from utils.weights import registry,weight_path,inventory,digest
from utils.durable import atomic_json

def verify(row,path,expected_sha256,license_accepted=False,expected_size=None):
    if not license_accepted:raise RuntimeError('Debe revisar/aceptar la licencia de la fuente antes de registrar pesos')
    expected_sha256=expected_sha256.lower()
    if len(expected_sha256)!=64 or any(c not in '0123456789abcdef' for c in expected_sha256):raise ValueError('SHA256 completo obligatorio, obtenido de fuente fiable o transferencia manual documentada')
    actual=digest(path)
    if actual!=expected_sha256:raise ValueError('Checksum incorrecto; archivo no verificado')
    if row.get('official_sha256') and actual!=row['official_sha256']:raise ValueError('Difiere del checksum oficial')
    if row.get('checksum_prefix') and not actual.startswith(row['checksum_prefix']):raise ValueError('Difiere del prefijo oficial del nombre del checkpoint')
    size=Path(path).stat().st_size
    if row.get("size_bytes") is not None and size!=row["size_bytes"]:raise ValueError("Tamano distinto de metadatos oficiales")
    if expected_size is not None and size!=expected_size:raise ValueError('Tamano distinto del esperado')
    if size<1024:raise ValueError('Respuesta demasiado pequena para un checkpoint; posible pagina de error')
    m=dict(schema=1,slug=row['slug'],state='DESCARGADO_VERIFICADO',sha256=actual,size_bytes=size,source_url=row['url'],
        source_version=row['source_version'],license=row['license'],license_accepted=True,verified_utc=datetime.now(timezone.utc).isoformat(),
        verification_basis=('SHA256 oficial completo' if row.get('official_sha256') else 'Prefijo oficial '+row.get('checksum_prefix','')+'; SHA256 completo calculado localmente. Fortaleza limitada a prefijo oficial; no certifica arquitectura'))
    atomic_json(Path(path).with_suffix(Path(path).suffix+'.provenance.json'),m)
    return m
def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('action',choices=('inventory','acquire','verify','generated'))
    p.add_argument('--architecture');p.add_argument('--sha256');p.add_argument('--expected-size',type=int)
    p.add_argument('--allow-network',action='store_true');p.add_argument('--accept-license',action='store_true')
    p.add_argument('--file',type=Path);p.add_argument('--run-metadata',type=Path)
    a=p.parse_args()
    if a.action=='inventory':print(json.dumps(inventory(),indent=2,ensure_ascii=False));return 0
    from utils.architectures import get_arch
    if not a.architecture:p.error('--architecture requerido')
    spec=get_arch(a.architecture);r=next(r for r in registry()['models'] if r['slug']==spec.slug);dest=weight_path(r)
    if a.action=='generated':
        if not a.file or not a.file.is_file() or not a.run_metadata:p.error('generated requiere checkpoint existente y --run-metadata')
        meta=json.loads(a.run_metadata.read_text(encoding='utf-8'))
        if meta.get('status')!='COMPLETE' or meta.get('architecture')!=spec.slug:raise RuntimeError('Run no terminado o arquitectura distinta')
        result={'state':'GENERADO_POR_ENTRENAMIENTO','path':str(a.file.resolve()),'sha256':digest(a.file),'size_bytes':a.file.stat().st_size,'run_metadata_sha256':digest(a.run_metadata),'run_id':meta['run_id']}
        atomic_json(a.file.with_suffix(a.file.suffix+'.generated.json'),result);print(json.dumps(result,indent=2));return 0
    if not r['compatible_pretrained']:raise RuntimeError('BLOQUEADO: '+r['block_reason'])
    if not a.sha256:
        a.sha256=r.get("official_sha256")
    if not a.sha256 and not r.get("checksum_prefix"):p.error("No existe checksum fiable: obtenga --sha256 independiente antes de adquirir")
    if a.action=='acquire':
        if not a.allow_network or not a.accept_license:p.error('acquire requiere --allow-network y --accept-license personales')
        if dest.exists():raise FileExistsError('Se preservan pesos existentes: '+str(dest))
        dest.parent.mkdir(parents=True,exist_ok=True);tmp=dest.with_suffix(dest.suffix+'.partial')
        if tmp.exists():raise FileExistsError('Descarga parcial anterior conservada; revise '+str(tmp))
        with urllib.request.urlopen(r['url'],timeout=60) as inp,tmp.open('xb') as out:
            while True:
                b=inp.read(1<<20)
                if not b:break
                out.write(b);out.flush()
            os.fsync(out.fileno())
        # Verify .partial before publishing; no false verified state on errors.
        if not a.sha256:
            a.sha256=digest(tmp)
            if not a.sha256.startswith(r["checksum_prefix"]):raise ValueError("Prefijo oficial incorrecto; descarga no publicada")
        verify(r,tmp,a.sha256,a.accept_license,a.expected_size)
        os.replace(tmp,dest);os.replace(tmp.with_suffix(tmp.suffix+'.provenance.json'),dest.with_suffix(dest.suffix+'.provenance.json'))
    if not a.sha256:
        a.sha256=digest(dest)
    print(json.dumps(verify(r,dest,a.sha256,a.accept_license,a.expected_size),indent=2));return 0
if __name__=='__main__':raise SystemExit(main())
