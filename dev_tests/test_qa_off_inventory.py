"""QA must distinguish explicit OFF images from unexplained orphan labels."""
import importlib.util
import json
import sys
import zipfile
from pathlib import Path
from unittest.mock import patch
import pytest

ROOT = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location('qa_off_checks', ROOT/'tools/qa_checks.py')
qa = importlib.util.module_from_spec(spec)
spec.loader.exec_module(qa)


def inventory(tmp_path, off='OFF', excluded=True, orphan=False, missing=False):
    images, labels = tmp_path/'images', tmp_path/'labels'
    images.mkdir(); labels.mkdir()
    (images/'active.png').write_bytes(b'active image')
    if not missing: (labels/'active.txt').write_text('')
    if excluded:
        (tmp_path/off/'images').mkdir(parents=True)
        (tmp_path/off/'images'/'ignored.png').write_bytes(b'ignored image')
        (labels/'ignored.txt').write_text('')
    if orphan: (labels/'unknown.txt').write_text('')
    (labels/'classes.txt').write_text('fruit')
    return images, labels


@pytest.mark.parametrize('off',['OFF','_off','Off'])
def test_off_label_is_documented_not_an_active_orphan(tmp_path, off):
    images, labels = inventory(tmp_path, off=off)
    before = {p: p.read_bytes() for p in tmp_path.rglob('*') if p.is_file()}
    d = qa.directory_source_inventory(images, labels, ['.png'], 1)
    assert d['images'] == 1 and d['labels'] == 2 and d['active_labels'] == 1
    assert d['ignored_labels'] == ['ignored']
    assert d['without_image'] == d['without_label'] == []
    assert len(d['ignored_image_paths']) == 1
    assert before == {p: p.read_bytes() for p in tmp_path.rglob('*') if p.is_file()}


def test_unexplained_orphan_is_still_rejected(tmp_path):
    images, labels = inventory(tmp_path, orphan=True)
    assert qa.directory_source_inventory(images, labels, ['.png'], 1)['without_image'] == ['unknown']


def test_missing_active_label_is_still_rejected(tmp_path):
    images, labels = inventory(tmp_path, missing=True)
    assert qa.directory_source_inventory(images, labels, ['.png'], 1)['without_label'] == ['active']


def test_off_text_or_wrong_extension_does_not_excuse_orphan(tmp_path):
    images, labels = inventory(tmp_path, excluded=False, orphan=True)
    (tmp_path/'OFF').mkdir(); (tmp_path/'OFF'/'unknown.txt').write_text('')
    assert qa.directory_source_inventory(images, labels, ['.png'], 1)['without_image'] == ['unknown']


@pytest.mark.parametrize('off',['OFF','_off'])
@pytest.mark.parametrize('orphan',[False,True])
def test_dataset_source_cli_result(tmp_path, off, orphan):
    images, labels = inventory(tmp_path, off=off, orphan=orphan)
    out = tmp_path/'out'; out.mkdir()
    cfg = {'paths': {'raw_images_dir': str(images), 'raw_labels_dir': str(labels)},
           'dataset': {'image_extensions': ['.png'], 'n_images_expected': 1}}
    with patch('utils.config_loader.load_config', return_value=cfg), \
         patch('orchestration.machine.read_binding', return_value={}), \
         patch.object(sys, 'argv', ['qa_checks.py','dataset_zip','--config','fixture','--output',str(out)]):
        assert qa.main() == (1 if orphan else 0)
    assert json.loads((out/'dataset_source.json').read_text())['ignored_labels'] == ['ignored']


@pytest.mark.parametrize('off',['OFF','_off'])
@pytest.mark.parametrize('orphan',[False,True])
def test_zip_source_checks_off_and_integrity(tmp_path, off, orphan):
    images, labels = inventory(tmp_path, off=off, orphan=orphan)
    zpath = tmp_path/'dataset.zip'
    with zipfile.ZipFile(zpath,'w') as z:
        for p in tmp_path.rglob('*'):
            if p.is_file() and p != zpath: z.write(p, 'dataset/'+p.relative_to(tmp_path).as_posix())
    out=tmp_path/'out';out.mkdir()
    cfg={'paths': {'raw_images_dir':str(images),'raw_labels_dir':str(labels)},
         'dataset':{'image_extensions':['.png'],'n_images_expected':1}}
    with patch('utils.config_loader.load_config',return_value=cfg), \
         patch('orchestration.machine.read_binding',return_value={'dataset_zip':str(zpath)}), \
         patch.object(sys,'argv',['qa_checks.py','dataset_zip','--config','fixture','--output',str(out)]):
        assert qa.main() == (1 if orphan else 0)
    d=json.loads((out/'dataset_zip.json').read_text())
    assert d['active_images']==1 and d['ignored_labels']==['ignored']
