import json

import pytest

from riichi_ai.cli import main


def test_truncated_match_does_not_count_as_completed_throughput(capsys):
    assert main(["smoke", "--games", "1", "--max-events", "1"]) == 1
    report = json.loads(capsys.readouterr().out)
    assert report["completed_games"] == 0
    assert report["truncated_games"] == 1
    assert report["hanchans_per_hour"] == 0
    assert report["decisions"] > 0


def test_entire_seed_range_is_validated_before_simulation(monkeypatch):
    def unexpected_simulation(*args, **kwargs):
        pytest.fail("Seed range must be rejected before starting any match")

    monkeypatch.setattr("riichi_ai.simulator.run_random_hanchan", unexpected_simulation)
    with pytest.raises(SystemExit) as error:
        main(["smoke", "--seed", str(2**64 - 1), "--games", "2"])
    assert error.value.code == 2

