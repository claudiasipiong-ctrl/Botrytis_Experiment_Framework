"""Comprueba incidencias del exportador v1 y crea un complemento; no cambia fuentes ni ZIP original."""
from __future__ import annotations

import argparse
import contextlib
import csv
import hashlib
import io
import json
import os
from pathlib import Path, PurePosixPath
import sqlite3
import sys
import tempfile
from datetime import datetime, timezone
import uuid
import zipfile

USERS = {'i21314': 'PC-02', 'i21313': 'PC-03'}
ORIGINAL_EXPORTER = '4df73954225dc0fe781bd01b451e958ceed6c16db9a5bad96917364a358ad150'
KNOWN_ZIPS = {
    'BOTRYTIS_TESIS_PC-02_20261105T201456Z.zip': 'f90a76a7ba536e87148b984e8bb7f1fe78319aec7927b93c0df148e63783b876',
    'BOTRYTIS_TESIS_PC-03_20261105T201509Z.zip': '6ce242f5c9790011f114ec85e222f2a11c34933958193b6c399cfe43d5f39cf1',
}
CSV_REL = Path('bf/52c7988b/r/yolo26l/final/train/results.csv')
CSV_ARC = '03_RESULTADOS/' + CSV_REL.as_posix()
CSV_ERROR = 'CSV con columnas duplicadas o registros incompletos; revisar fuente'
FIELDS = ['source', 'archive', 'requested_archive', 'source_size', 'source_sha256',
          'export_sha256', 'operation', 'test_rows_removed']


def emit(event, **data):
    print(event + ' ' + json.dumps(data, ensure_ascii=False), flush=True)


def sha_file(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        for chunk in iter(lambda: f.read(4 << 20), b''):
            h.update(chunk)
    return h.hexdigest()


def safe_arc(name):
    p = PurePosixPath(name)
    if not name or '\\' in name or ':' in name or p.is_absolute() or '..' in p.parts:
        raise ValueError('Ruta ZIP no segura: ' + name)
    return name


def same_source(a, b):
    return str(a).replace('\\', '/').casefold() == str(b).replace('\\', '/').casefold()


def sqlite_incident(error):
    return (Path(error.get('source', '')).suffix.casefold() in ('.sqlite', '.sqlite3', '.db')
            and '[WinError 32]' in error.get('error', '')
            and 'snapshot.sqlite' in error.get('error', ''))


def verify_archived_sqlite(z, row):
    """Only reads the archived database; never opens its live source."""
    arc = safe_arc(row['archive'])
    if row['operation'] != 'consistent_sqlite_snapshot':
        raise ValueError('El inventario no identifica un snapshot consistente')
    with tempfile.TemporaryDirectory() as td:
        path = Path(td) / 'check.sqlite'
        h = hashlib.sha256()
        with z.open(arc) as source, path.open('xb') as dest:
            for chunk in iter(lambda: source.read(1 << 20), b''):
                h.update(chunk)
                dest.write(chunk)
        if h.hexdigest() != row['export_sha256']:
            raise ValueError('SHA256 del SQLite exportado no coincide con inventario')
        uri = path.resolve().as_uri() + '?mode=ro&immutable=1'
        with contextlib.closing(sqlite3.connect(uri, uri=True)) as con:
            if con.execute('PRAGMA integrity_check').fetchall() != [('ok',)]:
                raise ValueError('SQLite archivado no supera integrity_check')
    return {'archive': arc, 'sha256': h.hexdigest(), 'integrity_check': 'ok'}


def curve_diagnostic(raw):
    """Describe structure, without deduplication, imputation, or metric certification."""
    table = list(csv.reader(io.StringIO(raw.decode('utf-8-sig')), strict=True))
    if not table:
        raise ValueError('CSV vacio')
    header = table[0]
    fields = [c.strip().casefold() for c in header]
    forbidden = {'image_id', 'stem', 'outer_split', 'split', 'evaluated_on', 'y_true', 'prob', 'prob_positive'}
    if 'epoch' not in fields or not any(c.startswith('train/') for c in fields) or forbidden.intersection(fields):
        raise ValueError('No es una curva de entrenamiento por epoca admisible')
    duplicate = sorted({c for c in fields if fields.count(c) > 1})
    irregular = [{'line': i, 'columns': len(r)} for i, r in enumerate(table[1:], 2) if len(r) != len(header)]
    return {'header_original': header, 'duplicates_normalized': duplicate,
            'data_rows': len(table) - 1, 'expected_columns': len(header),
            'irregular_rows': irregular,
            'conclusion': 'Original integro conservado. Revisar estructura antes de usar esta curva; no certifica metricas.'}


def stable_curve(path):
    if path.is_symlink():
        raise ValueError('No se siguen enlaces para recuperar CSV')
    before = path.stat()
    raw = path.read_bytes()
    after = path.stat()
    if (before.st_size, before.st_mtime_ns) != (after.st_size, after.st_mtime_ns):
        raise ValueError('CSV cambio durante lectura; no se publica')
    return raw, before, curve_diagnostic(raw)


def repair(original, home, pc):
    original, home = Path(original), Path(home)
    before = original.stat()
    emit('VERIFICANDO_ZIP_ORIGINAL', path=str(original))
    base_sha = sha_file(original)
    expected = KNOWN_ZIPS.get(original.name)
    sidecar = original.with_suffix('.zip.sha256.txt')
    if not expected and sidecar.is_file():
        expected = sidecar.read_text().split()[0]
    if not expected or expected != base_sha:
        raise ValueError('ZIP original sin huella verificable o SHA256 distinto; se conserva sin tocar')

    resolved, remaining, new_files = [], [], {}
    with zipfile.ZipFile(original) as z:
        names = z.namelist()
        if len(names) != len(set(names)):
            raise ValueError('ZIP con nombres duplicados; requiere revision')
        summary = json.loads(z.read('00_INDICES/RESUMEN_EXPORTACION.json'))
        if summary.get('pc') != pc or summary.get('code_exporter_sha256') != ORIGINAL_EXPORTER:
            raise ValueError('El ZIP no corresponde a esta PC o al exportador v1 afectado')
        errors = json.loads(z.read('00_INDICES/ERRORES.json'))
        rows = list(csv.DictReader(io.StringIO(z.read('00_INDICES/INVENTARIO.csv').decode('utf-8-sig'))))
        for row in rows:
            safe_arc(row['archive'])
        warnings = json.loads(z.read('00_INDICES/ADVERTENCIAS.json')) if '00_INDICES/ADVERTENCIAS.json' in names else []
        for error in errors:
            try:
                if sqlite_incident(error):
                    matches = [r for r in rows if same_source(r['source'], error['source'])]
                    if len(matches) != 1:
                        raise ValueError('No hay una unica copia inventariada. No se inventa ni reemplaza el estado de la base')
                    verified = verify_archived_sqlite(z, matches[0])
                    resolved.append({'original_incident': error, 'resolution': 'SNAPSHOT_YA_ARCHIVADO_VERIFICADO_ERROR_SOLO_DE_CIERRE_TEMPORAL', **verified})
                    emit('SQLITE_ARCHIVADO_VERIFICADO', archive=verified['archive'])
                elif pc == 'PC-02' and error.get('error') == CSV_ERROR and same_source(error.get('source'), home / CSV_REL):
                    path = home / CSV_REL
                    if not path.resolve().is_relative_to(home.resolve()):
                        raise ValueError('CSV fuera del perfil de esta PC')
                    if any(same_source(r['source'], path) for r in rows) or CSV_ARC in names:
                        raise ValueError('CSV ya inventariado: se conserva para revision manual')
                    raw, stat, diag = stable_curve(path)
                    sha = hashlib.sha256(raw).hexdigest()
                    new_files[CSV_ARC] = raw
                    rows.append(dict(source=str(path), archive=CSV_ARC, requested_archive=CSV_ARC,
                                     source_size=stat.st_size, source_sha256=sha, export_sha256=sha,
                                     operation='curve_csv_structure_warning_original_preserved', test_rows_removed=0))
                    warning = {'source': str(path), 'archive': CSV_ARC, 'sha256': sha,
                               'source_mtime_ns_at_repair': stat.st_mtime_ns,
                               'captured_at_repair_not_original_export': True, **diag}
                    warnings.append(warning)
                    resolved.append({'original_incident': error, 'resolution': 'CSV_ORIGINAL_RECUPERADO_CON_ADVERTENCIA', 'sha256': sha})
                    emit('CSV_CONSERVADO_CON_ADVERTENCIA', **warning)
                else:
                    remaining.append(error)
            except Exception as exc:
                remaining.append({**error, 'repair_error': str(exc)})

    if (before.st_size, before.st_mtime_ns) != (original.stat().st_size, original.stat().st_mtime_ns):
        raise ValueError('ZIP original cambio durante verificacion')
    stamp = datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ')
    status = ('INCOMPLETO_REVISAR_ERRORES' if remaining else
              'EXPORTACION_COMPLETADA_CON_ADVERTENCIAS' if warnings else 'EXPORTACION_COMPLETADA')
    report = {'pc': pc, 'created_utc': stamp, 'status': status,
              'base_zip': original.name, 'base_sha256': base_sha,
              'repair_code_sha256': sha_file(Path(__file__)),
              'original_errors': errors, 'resolved': resolved, 'remaining_errors': remaining,
              'warnings': warnings, 'new_data_files': sorted(new_files),
              'scope': 'Repara incidencias de exportacion; no valida resultados cientificos ni completa etapas pendientes.',
              'original_zip_modified': False, 'source_files_modified': False, 'test_opened': False}
    updated = {**summary, 'original_export_status': summary['status'], 'status': status,
               'repair_created_utc': stamp, 'base_zip': original.name, 'base_sha256': base_sha,
               'files': len(rows), 'errors': len(remaining), 'warnings': len(warnings),
               'requires_original_zip_and_supplement': True,
               'scope': report['scope']}
    buf = io.StringIO(newline='')
    writer = csv.DictWriter(buf, fieldnames=FIELDS, lineterminator='\n')
    writer.writeheader()
    writer.writerows(rows)
    new_files['00_INDICES/INVENTARIO.csv'] = buf.getvalue().encode('utf-8')
    for name, obj in [('ERRORES', remaining), ('ADVERTENCIAS', warnings), ('REPARACION', report), ('RESUMEN_EXPORTACION', updated)]:
        new_files['00_INDICES/' + name + '.json'] = (json.dumps(obj, ensure_ascii=False, indent=2) + '\n').encode('utf-8')
    readme = f'''# Complemento de exportacion {pc}

Conservar y compartir AMBOS ZIP con sus SHA256: el original {original.name} y este complemento.
Extraer primero el original en una carpeta corta (C:\\bt_tesis\\{pc}); luego el complemento en la MISMA carpeta,
aceptando reemplazar los indices exportados. No extraer sobre el proyecto de entrenamiento.
Este complemento solo contiene indices corregidos y los archivos recuperados indicados en REPARACION.json.
El original permanece intacto. Las bases verificadas siguen dentro del ZIP original.
Estado del conjunto: {status}. Incidencias originales: {len(errors)}; resueltas: {len(resolved)};
errores pendientes: {len(remaining)}; advertencias: {len(warnings)}.
Leer 00_INDICES/ADVERTENCIAS.json antes de usar curvas. No se corrigen ni imputan valores del CSV.
La copia recuperada del CSV (si existe) corresponde al momento de esta reparacion, no al de la primera exportacion.
Se comprueba SHA256 e integridad de los SQLite afectados ya archivados; no se reemplazan por bases actuales.
Esta reparacion no certifica todos los datos de la tesis ni termina HPO, robustez, TRAIN/VAL o TEST pendientes.
El historial de errores originales se conserva en 00_INDICES/REPARACION.json.
'''
    new_files['LEEME_REPARACION.md'] = readme.encode('utf-8')
    target = original.parent / ('COMPLEMENTO_TESIS_' + pc + '_' + stamp + '_' + uuid.uuid4().hex[:6] + '.zip')
    partial = target.with_suffix('.zip.partial')
    with zipfile.ZipFile(partial, 'x', compression=zipfile.ZIP_DEFLATED, allowZip64=True) as out:
        for name, raw in new_files.items():
            out.writestr(safe_arc(name), raw)
    with zipfile.ZipFile(partial) as check:
        if check.testzip() is not None:
            raise ValueError('El complemento no supera comprobacion CRC')
    partial.rename(target)
    sha = sha_file(target)
    target.with_suffix('.zip.sha256.txt').write_text(sha + '  ' + target.name + '\n', encoding='utf-8')
    target.with_suffix('.LEEME.txt').write_text(readme, encoding='utf-8')
    emit('COMPLEMENTO_TESIS_CREADO', path=str(target), original=str(original), sha256=sha,
         resolved=len(resolved), errors=len(remaining), warnings=len(warnings), status=status,
         original_zip_modified=False, test_opened=False)
    return target, report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--zip', type=Path)
    args = parser.parse_args()
    if os.name != 'nt':
        raise RuntimeError('Ejecutar en Windows PC-02 o PC-03; pruebas CPU separadas')
    pc = USERS.get(os.environ.get('USERNAME', '').casefold())
    if not pc:
        raise RuntimeError('Este reparador corresponde a PC-02 y PC-03')
    home = Path.home()
    originals = sorted((home / 'Downloads/Botrytis_Tesis').glob(pc + '_*/BOTRYTIS_TESIS_' + pc + '_*.zip'), key=lambda p: p.stat().st_mtime_ns)
    original = args.zip or (originals[-1] if originals else None)
    if original is None:
        raise RuntimeError('No se encuentra ZIP original en Downloads/Botrytis_Tesis')
    _, report = repair(original, home, pc)
    return 2 if report['remaining_errors'] else 0


if __name__ == '__main__':
    try:
        sys.exit(main())
    except Exception as exc:
        emit('REPARACION_DETENIDA', error=str(exc), original_zip_modified=False, source_files_modified=False)
        sys.exit(2)
