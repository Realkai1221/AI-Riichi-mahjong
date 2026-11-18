"""Conditional arithmetic fixtures; root must run these within its numerical lease."""
from copy import deepcopy
import importlib.util
import json
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
SPEC = importlib.util.spec_from_file_location("book_efficiency_probe", ROOT / "scripts/book_efficiency_probe.py")
probe = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(probe)


def record(hand=None):
    value = json.loads((ROOT / "examples/book_scenarios/example.jsonl").read_text().splitlines()[0])
    if hand is not None:
        value["state"]["observed"]["concealed_hand"] = hand
        value["source_answer"]["acceptable_actions"] = [
            {"kind": "DISCARD", "tile": hand[-1], "discard_origin": "any"}]
    return value


def counts(hand):
    result = [0] * 34
    for tile in hand:
        result[probe.tile_type(tile)] += 1
    return result


def test_discarded_copy_stays_known_and_cannot_reappear_in_ukeire():
    hand = ["1m", "2m", "3m", "1p", "2p", "3p", "1s", "2s", "3s", "E", "E", "E", "5p", "5p"]
    own = counts(hand)
    kind = probe.tile_type("5p")
    independent = probe.independent_discard_score(own, own, kind)
    project = probe.project_discard_score(own, own, kind)
    assert independent == project
    assert independent["shanten"] == 0
    assert independent["improving_types"] == ["5p"]
    assert independent["unseen_copies_per_improving_type"] == [2]
    assert independent["ukeire_unseen_copy_upper_bound"] == 2
    wrong_visible = list(own)
    wrong_visible[kind] -= 1
    with pytest.raises(ValueError, match="known-copy"):
        probe.independent_discard_score(own, wrong_visible, kind)
    assert own[kind] == 2


@pytest.mark.parametrize("hand", [
    ["1m", "2m", "3m", "1p", "2p", "3p", "1s", "2s", "3s", "E", "E", "E", "5p", "C"],
    ["1m", "1m", "2m", "2m", "3p", "3p", "4p", "4p", "5s", "5s", "6s", "6s", "E", "E"],
    ["1m", "9m", "1p", "9p", "1s", "9s", "E", "S", "W", "N", "P", "F", "C", "E"],
])
def test_all_candidate_cross_checks_cover_regular_pairs_and_orphans(hand):
    result = probe.audit_record(record(hand))
    assert result["all_calculators_agree"]
    assert len(result["candidates"]) == len(set(hand))
    assert result["project_best_sets"] == result["independent_best_sets"]
    assert result["policy_training_ready"] is False
    assert result["strategic_label_review"].startswith("pending")


def test_red_identity_is_separate_but_shape_arithmetic_is_equal():
    hand = ["1m", "2m", "3m", "1p", "2p", "3p", "1s", "2s", "3s", "E", "E", "E", "5p", "5pr"]
    result = probe.audit_record(record(hand))
    fives = [row for row in result["candidates"] if row["discard_tile"] in ("5p", "5pr")]
    assert len(fives) == 2 and fives[0]["project"] == fives[1]["project"]
    assert fives[0]["project"]["ukeire_unseen_copy_upper_bound"] == 2
    assert result["known_copy_counts_before_and_after_discard"][probe.tile_type("5pr")] == 2


def test_partial_open_or_nonquarantined_records_are_rejected():
    partial = record()
    partial["state"]["observed"]["concealed_hand"].pop(0)
    with pytest.raises(ValueError, match="exactly fourteen"):
        probe.audit_record(partial)
    opened = record()
    opened["rules"]["assumptions"] = ["Open hand with a declared PON."]
    with pytest.raises(ValueError, match="no-meld assertion"):
        probe.audit_record(opened)
    injected = record()
    injected["state"]["observed"]["melds"] = ["PON"]
    with pytest.raises(ValueError, match="state.observed"):
        probe.audit_record(injected)
    heldout = record()
    heldout["split"]["role"] = "heldout"
    with pytest.raises(ValueError, match="quarantined"):
        probe.audit_record(heldout)


def test_answers_change_membership_only_not_candidate_calculations():
    original = record()
    changed = deepcopy(original)
    changed["source_answer"]["acceptable_actions"] = [{"kind": "DISCARD", "tile": "1m", "discard_origin": "any"}]
    changed["source_answer"]["rationale"] = "Fixture answer deliberately differs."
    before, after = probe.audit_record(original), probe.audit_record(changed)
    assert before["candidates"] == after["candidates"]
    assert before["project_best_sets"] == after["project_best_sets"]
    assert before["source_answer_membership"] != after["source_answer_membership"]
    assert original["label_review"]["status"] == changed["label_review"]["status"] == "pending"


def test_disagreement_is_preserved_instead_of_silently_accepting_source(monkeypatch):
    original = probe.project_discard_score

    def corrupt(*args):
        value = original(*args)
        value["ukeire_unseen_copy_upper_bound"] += 1
        return value
    monkeypatch.setattr(probe, "project_discard_score", corrupt)
    result = probe.audit_record(record())
    assert not result["all_calculators_agree"]
    assert all(not row["calculators_agree"] for row in result["candidates"])
    assert result["policy_training_ready"] is False


def test_exclusive_output_input_hash_and_manifest_validation(tmp_path):
    input_path, output_path = tmp_path / "pilot.jsonl", tmp_path / "result.json"
    input_path.write_text(json.dumps(record()) + "\n")
    assert probe.main(["--input", str(input_path), "--output", str(output_path)]) == 0
    report = json.loads(output_path.read_text())
    assert report["input_jsonl"]["sha256"] == probe.sha256(input_path)
    assert report["model_evaluations"] == report["optimizer_updates"] == report["new_matches"] == 0
    assert report["book_pdf_declared_by_manifest"] is None
    assert report["runtime"]["mahjong"] == "2.0.0"
    assert len(report["implementation_files"]) == 4
    with pytest.raises(FileExistsError):
        probe.main(["--input", str(input_path), "--output", str(output_path)])
    manifest = {"pilot": {"sha256": "0" * 64, "bytes": input_path.stat().st_size, "records": 1},
                "source": {"book_id": record()["source"]["book_id"], "sha256": "1" * 64,
                           "path": "not-opened.pdf"}}
    (tmp_path / "source-manifest.json").write_text(json.dumps(manifest))
    with pytest.raises(ValueError, match="manifest does not match"):
        probe.audit_file(input_path)
    manifest["pilot"]["sha256"] = probe.sha256(input_path)
    (tmp_path / "source-manifest.json").write_text(json.dumps(manifest))
    result = probe.audit_file(input_path)
    assert result["book_pdf_declared_by_manifest"]["sha256"] == "1" * 64
    assert result["book_pdf_verified_by_this_probe"] is False
