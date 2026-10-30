import ast
import copy
from pathlib import Path
from types import SimpleNamespace

import pytest


@pytest.mark.parametrize("micro_batch,nominal_batch", [(16, 64), (8, 64), (4, 128)])
def test_record_batch_uses_configured_nominal_batch(micro_batch, nominal_batch):
    source = Path(__file__).resolve().parents[1] / "src/utils/detector_train.py"
    tree = ast.parse(source.read_text(encoding="utf-8-sig"))
    train = next(
        node for node in tree.body
        if isinstance(node, ast.FunctionDef) and node.name == "_train_once"
    )
    callback = next(
        node for node in train.body
        if isinstance(node, ast.FunctionDef) and node.name == "record_batch"
    )

    factory = ast.parse(
        "def make_callback(cfg, micro_batch, batch_contract):\n    pass\n"
    ).body[0]
    factory.body = [
        copy.deepcopy(callback),
        ast.Return(value=ast.Name(id="record_batch", ctx=ast.Load())),
    ]
    module = ast.fix_missing_locations(ast.Module(body=[factory], type_ignores=[]))
    namespace = {}
    exec(compile(module, str(source), "exec"), namespace)

    rows = []
    cfg = {"detector": {"nominal_batch_size": nominal_batch}}
    record = namespace["make_callback"](cfg, micro_batch, rows)
    trainer = SimpleNamespace(
        epoch=0,
        accumulate=4,
        optimizer=SimpleNamespace(param_groups=[{"lr": 0.001}]),
    )

    record(trainer)

    assert len(rows) == 1
    assert rows[0]["nominal_batch"] == nominal_batch
    assert rows[0]["micro_batch"] == micro_batch
