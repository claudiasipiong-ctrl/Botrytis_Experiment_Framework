"""Regression checks for the archival evidence delivery (no models execute)."""
from pathlib import Path
import csv,hashlib,importlib.util,io,json,sqlite3,sys,types,zipfile
import pytest
SPEC=importlib.util.spec_from_file_location('packet_final',Path(__file__).with_name('botrytis_paquete_tesis_final.py'))
m=importlib.util.module_from_spec(SPEC);SPEC.loader.exec_module(m)

def write(p,value=b'evidence'):
 p.parent.mkdir(parents=True,exist_ok=True);p.write_bytes(value);return p

def inventory(home,pc,paths):
 p=home/'Downloads'/('Inventario_Tesis_'+pc+'_fixture.zip');p.parent.mkdir(parents=True,exist_ok=True)
 buf=io.StringIO();w=csv.DictWriter(buf,fieldnames=['path','kind','bytes']);w.writeheader()
 for x in paths:w.writerow(dict(path=str(x),kind='FILE',bytes=x.stat().st_size))
 with zipfile.ZipFile(p,'w') as z:z.writestr('RESUMEN.json',json.dumps(dict(pc=pc)));z.writestr('ARCHIVOS.csv',buf.getvalue())
 return p

@pytest.mark.parametrize('path,pc,expected',[
 ('bf/17ee1aeb/r/resnet50/hpo/trial_019/metrics.json','PC-01','01_MODELOS/resnet50/03_HPO/'),
 ('bt-hoy-20261103/registro/botrytis_v4/vit_b16-20261103/artifacts/outer_validation/vit_b16.json','PC-03','01_MODELOS/vit_b16/06_VAL/'),
 ('r50/hpo_control/recuperar.py','PC-03','01_MODELOS/rtdetr_r50/03_HPO/'),
 ('bd/dos_test_finales_v1/A/resultado.json','PC-02','02_SISTEMAS_TEST/A_YOLO26l_ResNet50/resultado.json'),
 ('bd/dos_test_finales_v1/B/omisiones.csv','PC-02','02_SISTEMAS_TEST/B_YOLO26l_SwinTiny/omisiones.csv'),
 ('Documents/s2_no_borrar/Entrenamiento/Botrytis_Experiment_Framework_git/src/utils/metrics.py','PC-02','00_PROYECTO_Y_EQUIPO/src/utils/metrics.py'),
])
def test_routing(path,pc,expected):assert expected in m.classify(path,pc)

def test_partial_checkpoint_requires_receipt():
 root='C:\\Users\\I21313\\r50\\r\\rtdetr_r50\\hpo\\trial_006\\fold0\\cp\\'
 paths=[root+'e0051.pt',root+'e0051.pt.resume.json',root+'e0052.pt']
 assert m.choose_partial([dict(path=p,kind='FILE') for p in paths])=={paths[0].casefold()}
 paths.append(root+'e0052.pt.resume.json')
 assert m.choose_partial([dict(path=p,kind='FILE') for p in paths])=={paths[2].casefold()}

def test_sqlite_wal_snapshot_and_irregular_csv(tmp_path):
 db=tmp_path/'source.sqlite';con=sqlite3.connect(db);con.execute('PRAGMA journal_mode=WAL');con.execute('CREATE TABLE evidence(value)');con.execute("INSERT INTO evidence VALUES ('committed')");con.commit()
 raw=b'a,b,c\r\n1,2,3\r\n4,5\r\n';p=write(tmp_path/'results.csv',raw)
 packet=m.Packet(tmp_path/'out.zip','PC-02');packet.add(db,'PC-02/study.sqlite');packet.add(p,'PC-02/results.csv');packet.z.close()
 with zipfile.ZipFile(tmp_path/'out.zip') as z:
  assert z.testzip() is None;assert z.read('PC-02/results.csv')==raw
  q=write(tmp_path/'snapshot.sqlite',z.read('PC-02/study.sqlite'))
  with sqlite3.connect(q) as archived:assert archived.execute('SELECT value FROM evidence').fetchall()==[('committed',)]
 assert packet.rows[0]['operation']=='snapshot_sqlite_consistente';assert packet.rows[0]['source_sha256'] is None
 assert len(packet.warnings)==1;assert packet.warnings[0]['irregular_rows']==[dict(line=3,columns=2)]
 # Connection to original still works; snapshots can be deleted (Windows regression).
 assert con.execute('SELECT COUNT(*) FROM evidence').fetchone()==(1,);con.close();q.unlink()

def test_collisions_and_outside_source(tmp_path):
 packet=m.Packet(tmp_path/'out.zip','PC-01');a=write(tmp_path/'a.json');b=write(tmp_path/'b.json')
 packet.add(a,'PC-01/same.json');packet.attempt(b,'PC-01/SAME.json');packet.attempt(b,'../escape.json')
 assert len(packet.errors)==2;assert len(packet.rows)==1;packet.z.close()
 assert not m.allowed_source(a,tmp_path,tmp_path/'project')

def test_raw_data_once_and_all_final_models(tmp_path,monkeypatch):
 home=tmp_path/'home';project=home/'Documents/s2_no_borrar/Entrenamiento/Botrytis_Experiment_Framework_git'
 files=[]
 for model in m.MODELS['PC-01']:
  files += [write(home/f'bn/botrytis_v4/n_{model}/artifacts/checkpoints/{model}/final.pt'),write(home/f'bn/botrytis_v4/n_{model}/artifacts/outer_validation/{model}.json',b'{}')]
 files.append(write(home/'bf/17ee1aeb/r/resnet50/hpo/trial_000/metrics.json',b'{}'))
 omitted=write(home/'bf/17ee1aeb/r/resnet50/hpo/trial_000/model.pt',b'big intermediate');files.append(omitted)
 write(home/'bt-hoy-20261103/rondas.json',json.dumps({'resnet50':'cfg1','yolov8n':'cfg2'}).encode())
 roots={k:home/'data'/k for k in ['detection_dir','classification_dir','raw_images_dir','raw_labels_dir']}
 for root in roots.values():root.mkdir(parents=True)
 write(roots['detection_dir']/'split.csv',b'id,split\ntrain1,train\ntest1,test\n');write(roots['classification_dir']/'manifest.csv',b'fruit,label\nf1,1\n')
 image=write(roots['raw_images_dir']/'test1.png',b'original test image');label=write(roots['raw_labels_dir']/'test1.txt',b'0 .5 .5 .1 .1\n')
 fake=types.ModuleType('utils.config_loader');fake.load_config=lambda p:{};fake.cfg_path=lambda c,k:roots[k.split('.')[-1]];monkeypatch.setitem(sys.modules,'utils.config_loader',fake)
 inv=inventory(home,'PC-01',files);before={str(p):m.digest(p) for p in files+[image,label]}
 target,result=m.build(home,'PC-01',inv)
 assert result['errors']==0
 with zipfile.ZipFile(target) as z:
  assert z.testzip() is None
  rows=list(csv.DictReader(io.StringIO(z.read('PC-01/00_INDICES/INVENTARIO.csv').decode())))
  assert sum(r['source']==str(image) for r in rows)==1;assert sum(r['source']==str(label) for r in rows)==1
  assert not any(r['source']==str(omitted) for r in rows)
  assert len([r for r in rows if r['source'].endswith('/final.pt')])==3
  for r in rows:assert hashlib.sha256(z.read(r['archive'])).hexdigest()==r['archive_sha256']
  assert json.loads(z.read('PC-01/00_INDICES/ERRORES.json'))==[]
 assert target.with_suffix('.zip.sha256.txt').read_text().startswith(m.digest(target))
 assert before=={str(p):m.digest(p) for p in files+[image,label]}

def test_both_test_results_and_imported_weight(tmp_path,monkeypatch):
 home=tmp_path/'home';files=[]
 for model in m.MODELS['PC-02']:
  files += [write(home/f'bn/n_{model}/artifacts/checkpoints/{model}/final.pt'),write(home/f'bn/n_{model}/artifacts/outer_validation/{model}.json',b'{}')]
 for pair in ['A','B']:files.append(write(home/f'bd/dos_test_finales_v1/{pair}/resultado.json',b'{"test":true}'))
 files.append(write(home/'bd/dos_test_finales_v1/fuentes/resnet50/final.pt'))
 monkeypatch.setattr(m,'input_data',lambda *a:{})
 target,result=m.build(home,'PC-02',inventory(home,'PC-02',files))
 assert result['errors']==0
 with zipfile.ZipFile(target) as z:
  assert 'PC-02/02_SISTEMAS_TEST/A_YOLO26l_ResNet50/resultado.json' in z.namelist()
  assert 'PC-02/02_SISTEMAS_TEST/B_YOLO26l_SwinTiny/resultado.json' in z.namelist()
  assert any('fuentes/resnet50/final.pt' in n for n in z.namelist())
 assert result['test_reexecuted'] is False

def test_partial_checkpoint_hash_receipt_checked(tmp_path):
 p=write(tmp_path/'e0052.pt',b'checkpoint bytes')
 receipt=write(p.with_suffix('.pt.resume.json'),json.dumps(dict(architecture='rtdetr_r50',sha256=m.digest(p))).encode())
 packet=m.Packet(tmp_path/'out.zip','PC-03');packet.add(p,'PC-03/checkpoint.pt','checkpoint_parcial_con_recibo');packet.z.close()
 assert packet.rows[0]['operation']=='checkpoint_parcial_con_recibo'
 receipt.write_text(json.dumps(dict(architecture='rtdetr_r50',sha256='wrong')))
 packet=m.Packet(tmp_path/'bad.zip','PC-03');packet.attempt(p,'PC-03/checkpoint.pt','checkpoint_parcial_con_recibo');packet.z.close()
 assert len(packet.errors)==1;assert not packet.rows

def test_rtdetr_partial_keeps_initial_and_last_signed_only(tmp_path,monkeypatch):
 home=tmp_path/'home';files=[]
 files += [write(home/'bt-hoy-20261103/vit_b16/artifacts/checkpoints/vit_b16/final.pt'),write(home/'bt-hoy-20261103/vit_b16/artifacts/outer_validation/vit_b16.json',b'{}')]
 for epoch in [51,52]:
  p=write(home/f'r50/r/rtdetr_r50/hpo/trial_006/fold0/cp/e{epoch:04d}.pt',f'state {epoch}'.encode());files.append(p)
  files.append(write(p.with_suffix('.pt.resume.json'),json.dumps(dict(architecture='rtdetr_r50',sha256=m.digest(p))).encode()))
 initial=write(home/'r50/weights/rtdetr_r50vd_6x_coco_from_paddle.pth',b'pretrained');files.append(initial)
 monkeypatch.setattr(m,'input_data',lambda *a:{})
 target,result=m.build(home,'PC-03',inventory(home,'PC-03',files))
 assert result['errors']==0;assert len(result['partial_checkpoint_included'])==1;assert result['partial_checkpoint_included'][0].endswith('e0052.pt')
 with zipfile.ZipFile(target) as z:
  assert any('rtdetr_r50vd_6x_coco_from_paddle.pth' in n for n in z.namelist())
  assert any(n.endswith('e0052.pt') for n in z.namelist());assert not any(n.endswith('e0051.pt') for n in z.namelist())
  assert len([n for n in z.namelist() if n.endswith('.resume.json')])==2
  assert not any('/rtdetr_r50/06_VAL/' in n for n in z.namelist())
