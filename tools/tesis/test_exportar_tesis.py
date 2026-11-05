import csv,hashlib,importlib.util,io,json,sqlite3,sys,zipfile
from pathlib import Path
from types import SimpleNamespace
import pytest
p=Path(__file__).with_name('botrytis_exportar_tesis.py');s=importlib.util.spec_from_file_location('exporter',p);e=importlib.util.module_from_spec(s);s.loader.exec_module(e)


def test_csv_removes_test_gt_and_preserves_source(tmp_path):
    f=tmp_path/'fruits.csv';f.write_text('image_id,outer_split,x1,y\na,train,1,0\nb,validation,2,1\nc,test,99,1\n')
    before=f.read_bytes();out,n,operation=e.export_csv(f,{'c'})
    assert n==1 and b'99' not in out and f.read_bytes()==before
    assert operation=='test_rows_removed_export_only'


def test_ids_protect_predictions_without_split_column(tmp_path):
    f=tmp_path/'pred.csv';f.write_text('image_id,prob\na,.2\nC,.9\n')
    out,n,_=e.export_csv(f,{'c'});assert n==1 and b'.9' not in out


def test_public_split_manifest_has_only_identifiers_and_split(tmp_path):
    f=tmp_path/'split_manifest.csv';f.write_text('image_id,outer_split,secret_label\na,train,0\nc,test,1\n')
    out,_,op=e.export_csv(f,{'c'});assert b'c,test' in out and b'secret_label' not in out
    assert op=='manifest_ids_splits_only'


@pytest.mark.parametrize('body',['a,b\n1\n','a,b\n1,2,3\n','a,a\n1,2\n'])
def test_broken_csv_is_reportable_not_silently_repaired(tmp_path,body):
    f=tmp_path/'x.csv';f.write_text(body)
    with pytest.raises(ValueError):e.export_csv(f,set())


def test_sqlite_wal_snapshot_is_consistent(tmp_path):
    db=tmp_path/'queue.sqlite';con=sqlite3.connect(db);con.execute('PRAGMA journal_mode=WAL');con.execute('CREATE TABLE jobs (id INTEGER)');con.execute('INSERT INTO jobs VALUES (7)');con.commit()
    copy=tmp_path/'copy.sqlite';e.snapshot(db,copy)
    with sqlite3.connect(copy) as c:assert c.execute('SELECT id FROM jobs').fetchall()==[(7,)]
    assert con.execute('SELECT id FROM jobs').fetchall()==[(7,)]
    con.close()


def test_zip_omits_intermediate_weights_keeps_final_and_compacts_long_paths(tmp_path):
    a=tmp_path/'resume.pt';a.write_bytes(b'optimizer');f=tmp_path/'final.pt';f.write_bytes(b'final')
    doc=tmp_path/'curve.csv';doc.write_text('epoch,loss\n1,.5\n')
    target=tmp_path/'p.zip';pack=e.Packet(target,'PC-01',set())
    pack.add(a,'results/resume.pt');pack.add(f,'results/final.pt');pack.add(doc,'results/'+('long/'*50)+'curve.csv');pack.z.close()
    with zipfile.ZipFile(target) as z:
        assert 'results/final.pt' in z.namelist() and 'results/resume.pt' not in z.namelist()
        assert any(n.startswith('06_RUTAS_LARGAS/') for n in z.namelist())
        for row in pack.rows:assert hashlib.sha256(z.read(row['archive'])).hexdigest()==row['export_sha256']
    assert len(pack.excluded)==1


@pytest.mark.parametrize('name',['test/a.csv','final_test/metrics.json','OFF/a.png','_off/a.png','test_predictions.csv','.env','secrets/key.txt'])
def test_test_and_off_and_private_paths_are_blocked(name):assert e.blocked(Path(name))


def test_live_process_is_not_interrupted(tmp_path,monkeypatch):
    import psutil
    monkeypatch.setattr(psutil,'Process',lambda:SimpleNamespace(pid=1,parents=lambda:[]))
    monkeypatch.setattr(psutil,'process_iter',lambda args:[SimpleNamespace(pid=2,info={'name':'python.exe','cmdline':['python.exe','train.py']})])
    with pytest.raises(RuntimeError,match='otro Python'):e.idle()
    assert list(tmp_path.iterdir())==[]


def test_full_packet_portable_map_raw_train_val_only_and_originals_intact(tmp_path,monkeypatch):
    home=tmp_path/'home';project=home/'project';project.mkdir(parents=True)
    config=home/'bn/botrytis_v4/n_model/config.yaml';config.parent.mkdir(parents=True);config.write_text('fixture: true\n')
    data=home/'data';data.mkdir();(data/'split_manifest.csv').write_text('image_id,outer_split\na,train\nb,validation\nc,test\n')
    for group in ('raw_images_dir','raw_labels_dir'):
        root=home/group;root.mkdir()
        for iid in 'abc':(root/(iid+('.png' if group=='raw_images_dir' else '.txt'))).write_bytes(('PRIVATE_'+iid).encode())
        (root/'_off').mkdir();(root/'_off/ignored.png').write_bytes(b'OFF')
    derived=home/'derived';derived.mkdir();(derived/'manifest.csv').write_text('image_id,outer_split,y\na,train,0\nb,validation,1\nc,test,1\n')
    r=home/'bf/id/r/model/hpo/trial0';r.mkdir(parents=True);(r/'history.csv').write_text('epoch,train_loss\n1,.5\n')
    final=home/'bn/botrytis_v4/n_model/artifacts/checkpoints/model/final.pt';final.parent.mkdir(parents=True);final.write_bytes(b'final model')
    testroot=home/'bf/id/e/classification/model/test';testroot.mkdir(parents=True);(testroot/'predictions.csv').write_text('PRIVATE TEST PREDICTION')
    cfg={'outer_split':{'manifest_name':'split_manifest.csv'},'paths':{k:str(data) for k in ['artifacts_dir','runs_dir','results_dir','reports_dir']}}
    cfg['paths'].update(classification_dir=str(derived),detection_dir=str(derived),raw_images_dir=str(home/'raw_images_dir'),raw_labels_dir=str(home/'raw_labels_dir'))
    monkeypatch.setattr(e,'configs',lambda *a:([('model',config,cfg)],[],lambda c,key:Path(c['paths'][key.split('.')[-1]])))
    originals={str(p):p.read_bytes() for p in home.rglob('*') if p.is_file()}
    dest=tmp_path/'output';target,meta=e.build(project,home,'PC-01',dest,include_inputs=True)
    assert meta['status']=='EXPORTACION_COMPLETADA' and meta['input_counts']=={'raw_images_dir':2,'raw_labels_dir':2}
    assert all(Path(p).read_bytes()==v for p,v in originals.items())
    with zipfile.ZipFile(target) as z:
        assert z.testzip() is None
        names=z.namelist();assert not any('/test/' in n.casefold() or '/_off/' in n.casefold() for n in names)
        assert any(n.endswith('final.pt') for n in names)
        assert any(n.endswith('raw_images_dir/a.png') for n in names)
        assert not any(n.endswith('raw_images_dir/c.png') for n in names)
        rows=list(csv.DictReader(io.StringIO(z.read('00_INDICES/INVENTARIO.csv').decode())))
        for row in rows:assert hashlib.sha256(z.read(row['archive'])).hexdigest()==row['export_sha256']
        assert b'PRIVATE TEST PREDICTION' not in b''.join(z.read(n) for n in names)
    assert target.with_suffix('.zip.sha256.txt').read_text().startswith(e.digest(target))


def test_nested_json_test_annotations_removed_without_source_edits(tmp_path):
    p=tmp_path/'annotations.json';p.write_text(json.dumps({'counts':{'test':32},'items':[{'image_id':'a','boxes':[1]},{'image_id':'c','boxes':['PRIVATE']},{'split':'test','prediction':'PRIVATE'}]}))
    before=p.read_bytes();out,n,op=e.export_json(p,{'c'})
    assert n==2 and b'PRIVATE' not in out and p.read_bytes()==before
    assert json.loads(out)['counts']['test']==32
    packet=e.Packet(tmp_path/'json.zip','PC-01',{'c'});packet.add(p,'data/annotations.json');packet.z.close()
    assert packet.rows[0]['operation']=='test_records_removed_export_only'


def test_test_only_json_is_not_packaged(tmp_path):
    p=tmp_path/'something.json';p.write_text('{"evaluated_on":"test","map":0.99}')
    packet=e.Packet(tmp_path/'json.zip','PC-01',set());packet.add(p,'data/something.json');packet.z.close()
    assert packet.rows==[] and len(packet.excluded)==1


def test_snapshot_closes_every_connection_before_temporary_cleanup(tmp_path,monkeypatch):
    real_connect=sqlite3.connect;connections=[]
    class Tracked(sqlite3.Connection):
        closed=False
        def close(self):
            self.closed=True
            return super().close()
    def connect(*args,**kw):
        c=real_connect(*args,**kw,factory=Tracked);connections.append(c);return c
    db=tmp_path/'queue.sqlite'
    c=real_connect(db);c.execute('create table jobs(id)');c.commit();c.close()
    monkeypatch.setattr(e.sqlite3,'connect',connect)
    pack=e.Packet(tmp_path/'packet.zip','PC-02',set())
    pack.add(db,'02_RONDAS/queue.sqlite');pack.z.close()
    assert len(connections)==3 and all(c.closed for c in connections)


@pytest.mark.parametrize('body',['epoch,train/cls_loss,train/cls_loss\n1,.3,.4\n',
                                 'epoch,train/cls_loss\n1\n'])
def test_irregular_final_training_curve_preserved_with_warning(tmp_path,body):
    path=tmp_path/'train/results.csv';path.parent.mkdir();path.write_text(body)
    raw,n,op=e.export_csv(path,set())
    assert raw==path.read_bytes() and n==0 and 'warning' in op


def test_curve_exception_never_applies_to_sample_table(tmp_path):
    path=tmp_path/'train/results.csv';path.parent.mkdir()
    path.write_text('epoch,train/cls_loss,image_id,image_id\n1,.3,a,c\n')
    with pytest.raises(ValueError):e.export_csv(path,{'c'})


def test_cleanup_failure_occurs_after_sqlite_and_inventory_are_written(tmp_path,monkeypatch):
    import tempfile
    real_temp=tempfile.TemporaryDirectory
    class FailedCleanup:
        def __enter__(self):
            self.inner=real_temp();return self.inner.__enter__()
        def __exit__(self,*args):
            self.inner.__exit__(*args)
            raise PermissionError('[WinError 32] snapshot.sqlite')
    folder=tmp_path/'source';folder.mkdir();db=folder/'queue.sqlite'
    c=sqlite3.connect(db);c.execute('CREATE TABLE jobs(id)');c.commit();c.close()
    monkeypatch.setattr(e.tempfile,'TemporaryDirectory',FailedCleanup)
    target=tmp_path/'packet.zip';pack=e.Packet(target,'PC-03',set())
    pack.tree(folder,'02_RONDAS');pack.z.close()
    assert len(pack.errors)==1 and len(pack.rows)==1
    with zipfile.ZipFile(target) as z:
        assert hashlib.sha256(z.read(pack.rows[0]['archive'])).hexdigest()==pack.rows[0]['export_sha256']
