"""Fail-closed controller checks without launching a model worker."""
import importlib.util
import json
from pathlib import Path
import sys

import pytest


ROOT = Path(__file__).resolve().parents[1]


def controller():
    sys.path.insert(0, str(ROOT / 'scripts'))
    try:
        spec = importlib.util.spec_from_file_location('continuation_controller', ROOT / 'scripts/package_a_continuation.py')
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        return module
    finally:
        sys.path.pop(0)


@pytest.mark.parametrize('key,value', [('limit_seconds', 901), ('maximum_attempts', 5),
    ('starting_step', 3000), ('additional_steps', 5001), ('epsilon', .001),
    ('batch_size', 512), ('memory_bytes', 64 * 1024**3), ('device', 'cpu')])
def test_changed_protocol_refused(tmp_path, key, value):
    config = json.loads((ROOT / 'configs/package_a_continuation.json').read_text())
    config[key] = value
    p = tmp_path / 'config.json'
    p.write_text(json.dumps(config))
    with pytest.raises(ValueError):
        controller().settings(p)


def test_changed_order_or_scope_refused(tmp_path):
    original = json.loads((ROOT / 'configs/package_a_continuation.json').read_text())
    path = tmp_path / 'config.json'
    original['attempts'].reverse()
    path.write_text(json.dumps(original))
    with pytest.raises(ValueError, match='paired order'):
        controller().settings(path)
    original['attempts'].reverse()
    original['constraints']['test_split_loading'] = True
    path.write_text(json.dumps(original))
    with pytest.raises(ValueError, match='Scope'):
        controller().settings(path)


def test_source_and_checkpoint_inputs_match_declared_prior_runs():
    module = controller()
    _, config, identity = module.settings(ROOT / 'configs/package_a_continuation.json')
    assert len(identity) == 64
    required = [Path(config['diagnosis_plan'])]
    for attempt in config['attempts']:
        required.extend([Path(attempt['checkpoint']), ROOT / attempt['source_report']])
    if not all(path.is_file() for path in required):
        pytest.skip('Requires the original continuation archive and frozen checkpoints')
    inputs = module.input_identity(config)
    assert len(inputs) == 4
    for a in config['attempts']:
        assert inputs[a['name']]['checkpoint_sha256'] == a['checkpoint_sha256']
