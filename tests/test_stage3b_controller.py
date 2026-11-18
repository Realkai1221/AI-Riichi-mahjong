"""Stage 3B authorization and immutable-input controller guards."""
import importlib.util
import json
from pathlib import Path
import sys

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
spec = importlib.util.spec_from_file_location("stage3b_controller", ROOT / "scripts/stage3b.py")
controller = importlib.util.module_from_spec(spec)
spec.loader.exec_module(controller)


def test_authorized_scope():
    _, config, identity = controller.settings(ROOT / "configs/stage3b.json")
    assert len(identity) == 64
    assert config["limit_seconds"] == 300
    assert config["maximum_rows"] == 512
    assert config["probe_rows"] == 32
    assert config["constraints"]["optimizer_updates"] == 0
    assert config["constraints"]["new_matches"] == 0
    assert config["temperatures"] == [1, 10, 100]
    assert len(config["checkpoints"]) == 2


@pytest.mark.parametrize("key,value", [
    ("limit_seconds", 301), ("maximum_rows", 513), ("probe_rows", 33),
    ("temperatures", [1, 2, 4]), ("mixture_epsilon", .1),
    ("torch_threads", 2), ("diagnostic_allowance_seconds", 301),
    ("constraints", {"new_matches": 1, "optimizer_updates": 0}),
])
def test_changed_scope_rejected(tmp_path, key, value):
    config = json.loads((ROOT / "configs/stage3b.json").read_text())
    config[key] = value
    path = tmp_path / "config.json"
    path.write_text(json.dumps(config))
    with pytest.raises(ValueError, match="scope changed"):
        controller.settings(path)


def test_source_cannot_be_output_root(tmp_path):
    config = json.loads((ROOT / "configs/stage3b.json").read_text())
    config["data_root"] = config["source_root"]
    path = tmp_path / "config.json"
    path.write_text(json.dumps(config))
    with pytest.raises(ValueError, match="overwrite"):
        controller.settings(path)


def test_checkpoint_identity_is_fixed(tmp_path):
    config = json.loads((ROOT / "configs/stage3b.json").read_text())
    config["checkpoints"][0]["sha256"] = "0" * 64
    path = tmp_path / "config.json"
    path.write_text(json.dumps(config))
    with pytest.raises(ValueError, match="scope changed"):
        controller.settings(path)
