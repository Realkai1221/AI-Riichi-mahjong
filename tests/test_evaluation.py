from dataclasses import replace
import json

import numpy as np
import pytest

from riichi_ai import evaluation


def test_full_match_reproducibility_and_terminal_accounting():
    lineup = ["shanten", "random", "defense", "random"]
    first = evaluation.run_match(11223, lineup)
    second = evaluation.run_match(11223, lineup)
    assert first["completed"] and first["conservation_ok"]
    assert first["digest"] == second["digest"]
    assert first["player_views_hash"] == second["player_views_hash"]
    assert sum(first["scores"]) + first["riichi_sticks"] * 1000 == 100_000
    assert sorted(first["ranks"]) == [1, 2, 3, 4]
    assert sum(first["utilities"]) == 0
    assert first["hands"] == len(first["ordinal_wall_fingerprints"])
    assert sum(seat["decisions"] for seat in first["policy_latency_by_seat"]) == first["decisions"]


def test_identical_policies_have_exactly_zero_paired_difference():
    report = evaluation.evaluate_paired("random", "random", blocks=2, seed=76)
    assert report["valid"] and report["completed_games"] == 16
    assert report["statistics"]["independent_blocks"] == 2
    assert report["statistics"]["mean_utility_difference"] == 0
    for block in report["blocks"]:
        assert block["seat_utility_differences"] == [0, 0, 0, 0]
    for a, b in zip(report["matches"][::2], report["matches"][1::2]):
        assert a["digest"] == b["digest"]
        assert a["paired_common_walls_match"]
    assert {match["focal_seat"] for match in report["matches"]} == {0, 1, 2, 3}


def test_reversing_arms_reverses_differences_and_preserves_pairing():
    forward = evaluation.evaluate_paired("shanten", "random", blocks=1, seed=789)
    reverse = evaluation.evaluate_paired("random", "shanten", blocks=1, seed=789)
    assert forward["valid"] and reverse["valid"]
    assert forward["statistics"]["mean_utility_difference"] == -reverse["statistics"]["mean_utility_difference"]
    for a, b in zip(forward["matches"][::2], reverse["matches"][1::2]):
        assert a["digest"] == b["digest"]


def test_truncation_suppresses_statistics_and_keeps_replay_record(tmp_path):
    report = evaluation.evaluate_paired("shanten", "random", blocks=1, seed=4,
                                        max_steps=1, failure_dir=tmp_path)
    assert not report["valid"]
    assert report["statistics"] is None
    assert report["completed_games"] == 0
    assert report["truncated_games"] == 8
    assert report["hanchans_per_hour"] == 0
    for match in report["matches"]:
        assert match["utilities"] is None and match["ranks"] is None
        artifact = json.loads(open(match["failure_record"]).read())
        assert artifact["record"]["seed"] == match["seed"]
        assert len(artifact["submitted_actions"]) == 1
        assert artifact["auditor_log"][0]["type"] == "start_game"


@pytest.mark.parametrize("invalid_kind", ["forged_token", "none"])
def test_invalid_policy_action_is_reported_without_submitting_it(monkeypatch, tmp_path, invalid_kind):
    class InvalidPolicy:
        def manifest(self):
            return {"policy": "injected_invalid"}

        def choose(self, view):
            return None if invalid_kind == "none" else replace(view.legal_actions[0], token="not-a-legal-action")

    monkeypatch.setattr(evaluation, "make_policy", lambda *args: InvalidPolicy())
    report = evaluation.run_match(4, ["random"] * 4, failure_dir=tmp_path)
    assert not report["completed"]
    assert report["stop_reason"] == "failure"
    assert report["failure"]["type"] in ("ValueError", "AttributeError")
    assert report["steps"] == 0
    assert report["utilities"] is None
    artifact = json.loads(open(report["failure_record"]).read())
    assert artifact["submitted_actions"] == []


def test_block_statistics_do_not_count_rotations_as_independent():
    result = evaluation.summarize_blocks([1, -1, 0.5, -0.5])
    assert result["independent_blocks"] == 4
    assert result["mean_utility_difference"] == 0
    assert result["standard_error"] == pytest.approx(np.std([1, -1, .5, -.5], ddof=1) / 2)
    assert evaluation.summarize_blocks([0])["standard_error"] is None
    for invalid in ([], [float("nan")], [4], [[1]]):
        with pytest.raises(ValueError):
            evaluation.summarize_blocks(invalid)


def test_validation_happens_before_running_matches(monkeypatch):
    monkeypatch.setattr(evaluation, "run_match", lambda *a, **kw: pytest.fail("Unexpected match"))
    for kwargs in ({"blocks": 0}, {"opponents": ["random"]}, {"seed": 2**64}, {"max_steps": 0}):
        with pytest.raises(ValueError):
            evaluation.evaluate_paired("random", "random", **kwargs)


def test_unmatched_wall_stream_invalidates_the_block(monkeypatch):
    real_run = evaluation.run_match
    calls = 0

    def changed_wall(*args, **kwargs):
        nonlocal calls
        result = real_run(*args, **kwargs)
        calls += 1
        if calls == 2:
            result["ordinal_wall_fingerprints"][0] = "injected-wall-mismatch"
        return result

    monkeypatch.setattr(evaluation, "run_match", changed_wall)
    report = evaluation.evaluate_paired("random", "random", blocks=1, seed=15)
    assert not report["valid"]
    assert report["statistics"] is None
    assert report["completed_games"] == 8
    assert not report["matches"][0]["paired_common_walls_match"]
    assert report["blocks"][0]["mean_utility_difference"] is None


def test_cli_marks_incomplete_evaluation_as_failed(capsys, tmp_path):
    from riichi_ai.cli import main
    output = tmp_path / "report.json"
    assert main(["evaluate", "--blocks", "1", "--max-steps", "1",
                 "--output", str(output), "--failure-dir", str(tmp_path / "failures")]) == 1
    report = json.loads(capsys.readouterr().out)
    assert json.loads(output.read_text()) == report
    assert report["statistics"] is None
