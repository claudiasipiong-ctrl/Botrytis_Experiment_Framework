from pathlib import Path
import importlib.util,zipfile,json
p=Path(__file__).with_name('botrytis_inventario_tesis_final.py');s=importlib.util.spec_from_file_location('inventory',p);m=importlib.util.module_from_spec(s);s.loader.exec_module(m)

def test_inventory_contains_final_test_and_partial_model_without_personal_folders(tmp_path):
 home=tmp_path/'home'
 for path in ['bd/dos_test_finales_v1/A/resultado.json','bd/dos_test_finales_v1/B/detecciones.csv','r50/r/rtdetr_r50/hpo/trial_006/history.csv','bn/model/artifacts/checkpoints/model/final.pt','bf/abc/e/detection/model/test/metrics/metrics.json','AppData/private.txt']:
  p=home/path;p.parent.mkdir(parents=True,exist_ok=True);p.write_bytes(b'fixture')
 before={p:p.read_bytes() for p in home.rglob('*') if p.is_file()}
 result=m.build(home,'PC-02')
 with zipfile.ZipFile(result) as z:
  text=z.read('ARBOL.txt').decode('utf-8-sig')
  assert 'resultado.json' in text and 'trial_006' in text and 'final.pt' in text and 'test' in text
  assert 'private.txt' not in text and len(z.namelist())==3
  assert json.loads(z.read('RESUMEN.json'))['errors']==[]
 assert all(p.read_bytes()==v for p,v in before.items())

def test_mass_images_registered_without_enumeration(tmp_path):
 for path in ['bf/id/e/model/crops/crop.png','bf/id/e/model/visualizations/full.png','bf/id/e/model/plots/curve.png','bf/id/e/model/metrics/predictions.csv']:
  p=tmp_path/path;p.parent.mkdir(parents=True,exist_ok=True);p.write_bytes(b'x')
 rows,errors,_=m.gather(tmp_path,'PC-01');names=[r['path'] for r in rows]
 assert any('curve.png' in p for p in names) and any('predictions.csv' in p for p in names)
 assert not any('crop.png' in p or 'full.png' in p for p in names)
 assert sum(r['kind']=='MASS_IMAGE_DIRECTORY' for r in rows)==2

def test_links_and_secrets_excluded(tmp_path):
 root=tmp_path/'bn';root.mkdir();(root/'.env').write_text('private');(root/'config.yaml').write_text('ok')
 ext=tmp_path/'private';ext.mkdir();(ext/'hidden.txt').write_text('secret');(root/'link').symlink_to(ext,target_is_directory=True)
 rows,_,_=m.gather(tmp_path,'PC-03');assert not any('hidden.txt' in r['path'] or r['path'].endswith('.env') for r in rows)
 assert any('config.yaml' in r['path'] for r in rows)
