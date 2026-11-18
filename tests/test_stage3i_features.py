"""Whole-match feature-cache split guards; no model or game execution."""
from copy import deepcopy
import importlib.util
from pathlib import Path
import sys

import pytest

SCRIPTS = Path(__file__).resolve().parents[1] / 'scripts'
sys.path.insert(0, str(SCRIPTS))
spec = importlib.util.spec_from_file_location('stage3i_features_test_module', SCRIPTS / 'stage3i_features.py')
m = importlib.util.module_from_spec(spec)
spec.loader.exec_module(m)


def partitions():
    return {'training': ['train-0', 'train-1'], 'selection': ['dev-0'],
            'calibration': ['dev-1'], 'gate': ['dev-2']}


def test_complete_disjoint_roles_pass_with_arbitrary_fixed_order():
    selected = partitions()
    selected['training'].reverse()
    m.validate_partitions(selected, ['train-0', 'train-1'], ['dev-0', 'dev-1', 'dev-2'])


@pytest.mark.parametrize('change', ['overlap', 'relabel', 'omit', 'empty', 'unexpected', 'invalid'])
def test_feature_roles_cannot_be_changed_by_partition_names(change):
    selected = deepcopy(partitions())
    if change == 'overlap':
        selected['calibration'] = ['dev-0']
    elif change == 'relabel':
        selected['training'][0], selected['selection'][0] = selected['selection'][0], selected['training'][0]
    elif change == 'omit':
        selected['training'].pop()
    elif change == 'empty':
        selected['gate'] = []
    elif change == 'unexpected':
        selected['unknown'] = ['new']
    else:
        selected['gate'] = [None]
    with pytest.raises(ValueError):
        m.validate_partitions(selected, ['train-0', 'train-1'], ['dev-0', 'dev-1', 'dev-2'])
