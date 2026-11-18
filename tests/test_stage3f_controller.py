"""Resource admission checks do not start workers or open a real ledger."""
import importlib.util
from pathlib import Path
import sys
import pytest
SCRIPTS=Path(__file__).resolve().parents[1]/'scripts'
sys.path.insert(0,str(SCRIPTS))
spec=importlib.util.spec_from_file_location('stage3f_controller',SCRIPTS/'stage3f.py')
m=importlib.util.module_from_spec(spec); spec.loader.exec_module(m)


def config(): return dict(reserve_seconds=120,diagnosis_limit_seconds=300)
def snap(remaining=7200,ops=()): return dict(remaining_seconds=remaining,operations=list(ops))


def test_whole_reservation_and_shutdown():
    m.admission(snap(180),'preflight',60,config())
    with pytest.raises(ValueError): m.admission(snap(179),'preflight',60,config())


@pytest.mark.parametrize('name', ['','../escape','x/y','a b','Upper'])
def test_invalid_name(name):
    with pytest.raises(ValueError): m.admission(snap(),name,60,config())


@pytest.mark.parametrize('seconds',[float('nan'),float('inf'),0,14,-1])
def test_invalid_allowance(seconds):
    with pytest.raises(ValueError): m.admission(snap(),'preflight',seconds,config())


def test_active_and_repeat_rejected():
    with pytest.raises(ValueError): m.admission(snap(ops=[dict(name='x',status='active')]),'y',60,config())
    with pytest.raises(ValueError): m.admission(snap(ops=[dict(name='x',status='completed')]),'x',60,config())


def test_diagnostic_cumulative_includes_failures():
    history=[dict(name='diagnosis-tests',status='failed',elapsed_seconds=65)]
    with pytest.raises(ValueError): m.admission(snap(ops=history),'diagnosis-run',240,config())
    m.admission(snap(ops=history),'diagnosis-run',235,config())
