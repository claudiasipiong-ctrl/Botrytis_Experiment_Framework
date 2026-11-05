import contextlib
import csv
import hashlib
import importlib.util
import io
import json
from pathlib import Path
import sqlite3
import zipfile

import pytest

p=Path(__file__).with_name('botrytis_reparar_zip_tesis.py')
s=importlib.util.spec_from_file_location('repair',p);r=importlib.util.module_from_spec(s);s.loader.exec_module(r)


def fixture_packet(tmp_path,pc='PC-02',count=23,csv_error=True,*,bad_hash=False,missing_entry=False,bad_sql=False,unknown=False):
    home=tmp_path/'home';home.mkdir()
    db=tmp_path/'db.sqlite'
    with contextlib.closing(sqlite3.connect(db)) as c:
        c.execute('CREATE TABLE trials(number,state)');c.execute("INSERT INTO trials VALUES (6,'RUNNING')");c.commit()
    raw=db.read_bytes();rows=[];errors=[]
    base=home/'Downloads/Botrytis_Tesis'/f'{pc}_fixture'/f'BOTRYTIS_TESIS_{pc}_fixture.zip'
    base.parent.mkdir(parents=True)
    with zipfile.ZipFile(base,'x') as z:
        for i in range(count):
            arc=f'04_AUDITORIA/bv/{i}/queue-before.sqlite'
            payload=b'not a db' if bad_sql and i==0 else raw
            if not missing_entry or i!=0:z.writestr(arc,payload)
            source=str(home/f'bv/{i}/queue-before.sqlite')
            rows.append(dict(source=source,archive=arc,requested_archive=arc,source_size=len(raw),source_sha256='',
                             export_sha256='0'*64 if bad_hash and i==0 else hashlib.sha256(payload).hexdigest(),
                             operation='consistent_sqlite_snapshot',test_rows_removed=0))
            errors.append(dict(source=source,error="[WinError 32] locked: C:\\Temp\\snapshot.sqlite"))
        if csv_error:
            f=home/r.CSV_REL;f.parent.mkdir(parents=True);f.write_bytes(b'epoch,train/cls_loss,train/cls_loss\n1,.3,.4\n2,.2\n')
            errors.append(dict(source=str(f),error=r.CSV_ERROR))
        if unknown:errors.append(dict(source='unknown.json',error='Unrelated failure'))
        buf=io.StringIO();w=csv.DictWriter(buf,fieldnames=r.FIELDS);w.writeheader();w.writerows(rows)
        z.writestr('00_INDICES/INVENTARIO.csv',buf.getvalue())
        z.writestr('00_INDICES/ERRORES.json',json.dumps(errors))
        z.writestr('00_INDICES/RESUMEN_EXPORTACION.json',json.dumps(dict(pc=pc,code_exporter_sha256=r.ORIGINAL_EXPORTER,status='INCOMPLETO_REVISAR_ERRORES',files=count,test_inputs_exported=False)))
        z.writestr('00_INDICES/FALTANTES.json','["rtdetr_hpo_incomplete"]')
    base.with_suffix('.zip.sha256.txt').write_text(r.sha_file(base)+'  '+base.name)
    return home,base,rows


@pytest.mark.parametrize('pc,count,with_csv',[('PC-02',23,True),('PC-03',21,False)])
def test_actual_error_patterns_recovered_without_source_db_access_or_base_modification(tmp_path,pc,count,with_csv,monkeypatch):
    home,base,rows=fixture_packet(tmp_path,pc,count,with_csv)
    before={p:p.read_bytes() for p in home.rglob('*') if p.is_file()}
    real=r.sqlite3.connect
    def only_archive(database,*a,**kw):
        assert 'check.sqlite' in str(database) and 'immutable=1' in str(database)
        return real(database,*a,**kw)
    monkeypatch.setattr(r.sqlite3,'connect',only_archive)
    target,report=r.repair(base,home,pc)
    assert all(p.read_bytes()==b for p,b in before.items())
    assert len(report['resolved'])==count+with_csv and not report['remaining_errors']
    assert report['test_opened'] is False and len(report['original_errors'])==count+with_csv
    with zipfile.ZipFile(target) as supplement,zipfile.ZipFile(base) as original:
        assert supplement.testzip() is None
        assert not any(n.endswith('.sqlite') for n in supplement.namelist())
        assert json.loads(supplement.read('00_INDICES/ERRORES.json'))==[]
        assert '00_INDICES/FALTANTES.json' not in supplement.namelist()
        inventory=list(csv.DictReader(io.StringIO(supplement.read('00_INDICES/INVENTARIO.csv').decode())))
        for row in inventory:
            src=supplement if row['archive'] in supplement.namelist() else original
            assert hashlib.sha256(src.read(row['archive'])).hexdigest()==row['export_sha256']
        if with_csv:
            assert supplement.read(r.CSV_ARC)==(home/r.CSV_REL).read_bytes()
            warning=json.loads(supplement.read('00_INDICES/ADVERTENCIAS.json'))[0]
            assert warning['duplicates_normalized']==['train/cls_loss'] and warning['irregular_rows']==[dict(line=3,columns=2)]
            assert warning['captured_at_repair_not_original_export'] is True
            assert report['status']=='EXPORTACION_COMPLETADA_CON_ADVERTENCIAS'
        else:assert report['status']=='EXPORTACION_COMPLETADA'


@pytest.mark.parametrize('failure',['bad_hash','missing_entry','bad_sql','unknown'])
def test_unverified_or_unknown_error_is_never_cleared(tmp_path,failure):
    home,base,_=fixture_packet(tmp_path,count=1,csv_error=False,**{failure:True})
    _,report=r.repair(base,home,'PC-02')
    assert report['status']=='INCOMPLETO_REVISAR_ERRORES' and len(report['remaining_errors'])==1


def test_foreign_pc_blocked_without_output(tmp_path):
    home,base,_=fixture_packet(tmp_path,count=1,csv_error=False)
    with pytest.raises(ValueError,match='esta PC'):r.repair(base,home,'PC-03')
    assert not list(base.parent.glob('COMPLEMENTO*'))


def test_changed_original_is_never_repaired(tmp_path):
    home,base,_=fixture_packet(tmp_path,count=1,csv_error=False)
    with base.open('ab') as f:f.write(b'changed')
    with pytest.raises(ValueError,match='SHA256'):r.repair(base,home,'PC-02')


@pytest.mark.parametrize('name',['../queue.sqlite','/absolute.sqlite','C:/evil.sqlite','safe\\..\\evil'])
def test_unsafe_archive_paths_rejected(name):
    with pytest.raises(ValueError):r.safe_arc(name)


def test_missing_csv_remains_reported_no_values_invented(tmp_path):
    home,base,_=fixture_packet(tmp_path,count=0)
    (home/r.CSV_REL).unlink()
    target,report=r.repair(base,home,'PC-02')
    assert len(report['remaining_errors'])==1
    with zipfile.ZipFile(target) as z:assert r.CSV_ARC not in z.namelist()


@pytest.mark.parametrize('raw',[b'epoch,image_id,prob,train/loss\n1,private,.9,.3\n',b'epoch,loss\n1,.3\n'])
def test_unexpected_or_sample_csv_not_recovered(raw):
    with pytest.raises(ValueError):r.curve_diagnostic(raw)


def test_archived_sqlite_connection_is_closed_before_cleanup(tmp_path,monkeypatch):
    home,base,rows=fixture_packet(tmp_path,count=1,csv_error=False)
    real=r.sqlite3.connect;connections=[]
    class Tracked(sqlite3.Connection):
        closed=False
        def close(self):self.closed=True;return super().close()
    def connect(*a,**kw):
        c=real(*a,**kw,factory=Tracked);connections.append(c);return c
    monkeypatch.setattr(r.sqlite3,'connect',connect)
    with zipfile.ZipFile(base) as z:r.verify_archived_sqlite(z,rows[0])
    assert connections and all(c.closed for c in connections)
