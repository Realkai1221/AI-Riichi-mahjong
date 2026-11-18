"""Controller invariants; all tests are run under the Stage 3A time lease."""
import importlib.util
import json
from pathlib import Path
import sys

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'scripts'))
spec = importlib.util.spec_from_file_location('stage3a_controller', ROOT / 'scripts/stage3a.py')
controller = importlib.util.module_from_spec(spec)
spec.loader.exec_module(controller)


def test_fixed_plan_and_resource_envelope():
    _, config, identity = controller.settings(ROOT / 'configs/stage3a.json')
    rows = [match for batch in config['batches'] for match in batch['matches']]
    assert len(identity) == 64
    assert len(rows) == len({row['seed'] for row in rows}) == 64
    assert len({row['sampling_seed'] for row in rows}) == 64
    assert not {row['seed'] for row in rows} & {row['sampling_seed'] for row in rows}
    assert [row['learner_seat'] for row in rows] == list(range(4)) * 16
    assert config['limit_seconds'] == 600
    assert config['constraints']['opponent_actions_eligible'] is False


@pytest.mark.parametrize('key,value', [('limit_seconds', 601), ('device', 'mps'),
    ('learning_rate', .001), ('maximum_matches', 65)])
def test_protocol_mutation_rejected(tmp_path, key, value):
    config = json.loads((ROOT / 'configs/stage3a.json').read_text())
    config[key] = value
    path = tmp_path / 'bad.json'
    path.write_text(json.dumps(config))
    with pytest.raises(ValueError):
        controller.settings(path)


def test_failed_formal_batch_stops_chain(tmp_path):
    (tmp_path / 'prepare-attempt.json').write_text(json.dumps({'status': 'completed'}))
    (tmp_path / 'plan.json').write_text('{}')
    controller.check_chain(tmp_path, 0)
    (tmp_path / 'attempt-0.json').write_text(json.dumps({'status': 'failed'}))
    (tmp_path / 'batch-0.json').write_text(json.dumps({'complete': False}))
    with pytest.raises(ValueError):
        controller.check_chain(tmp_path, 0)
    with pytest.raises(ValueError):
        controller.check_chain(tmp_path, 1)


def test_only_next_complete_batch_is_admitted(tmp_path):
    (tmp_path / 'prepare-attempt.json').write_text(json.dumps({'status': 'completed'}))
    (tmp_path / 'plan.json').write_text('{}')
    (tmp_path / 'attempt-0.json').write_text(json.dumps({'status': 'completed'}))
    (tmp_path / 'batch-0.json').write_text(json.dumps({'complete': True}))
    controller.check_chain(tmp_path, 1)
    for invalid in (-1, 0, 2, 4):
        with pytest.raises(ValueError):
            controller.check_chain(tmp_path, invalid)


def test_failed_prepare_never_admits_formal_collection(tmp_path):
    (tmp_path / 'plan.json').write_text('{}')
    for status in ('failed', 'reserved'):
        (tmp_path / 'prepare-attempt.json').write_text(json.dumps({'status': status}))
        with pytest.raises(ValueError, match='Successful preparation'):
            controller.check_chain(tmp_path, 0)
