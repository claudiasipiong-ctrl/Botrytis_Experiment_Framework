"""v4 external launcher/catalog contracts; previous illustrative API is historical only."""
from pathlib import Path
import hashlib,json
import pytest
from utils.figures_all import catalog,load_source,InsufficientData

def test_external_launcher_is_import_safe():
    import importlib.util
    path=Path(__file__).resolve().parents[2]/"generar_graficos_botrytis.py"
    if not path.is_file():pytest.skip("Place the external launcher beside the project as delivered")
    spec=importlib.util.spec_from_file_location("external_botrytis_v4",path)
    module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)
    assert callable(module.main)

def test_estimates_cannot_enter_official_catalog(tmp_path):
    path=tmp_path/"fixture.csv";path.write_text("category,count,is_estimated\na,2,True\n",encoding="utf-8")
    item={"path":str(path),"source_sha256":hashlib.sha256(path.read_bytes()).hexdigest(),"dataset":"fixture","split":"train","phase":"fixture","metric":"count","unit":"count","denominator":"fixture","configuration":"fixture","semantic_version":"botrytis-physical-fruit-pairing-v4","is_estimated":True}
    with pytest.raises(InsufficientData,match="ESTIMADAS"):
        load_source(catalog()[0],{"dataset_counts":item})
