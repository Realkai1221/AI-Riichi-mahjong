"""Run all arithmetic fixtures inside the root's numerical lease."""
from copy import deepcopy
import importlib.util
import json
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
SPEC = importlib.util.spec_from_file_location("book_count_conventions", ROOT / "scripts/book_count_conventions.py")
probe = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(probe)

HAND = ["1m", "2m", "3m", "1p", "2p", "3p", "1s", "2s", "3s", "E", "E", "E", "5p", "5p"]


def packets():
    """Authored synthetic packets; no book material or actual printed counts."""
    return [{"schema": "raw-source-packet-v1", "packet_id": identifier,
        "source": {"book_id": identifier.rsplit("-q", 1)[0], "path": "fixture-not-a-source-book.pdf", "sha256": "0" * 64},
        "source_fields": {"concealed_hand_before_draw": HAND[:-1], "drawn_tile": HAND[-1],
            "shown_hand_including_draw": HAND[:], "dora_text": {"label": "dora indicator", "tile_type": "5p"}},
        "source_answer": {"reported_acceptance_counts": [
            {"discard": "5p", "reported_count": 3},
            {"discard": "1m", "reported_count": 100},
            {"discard": "E", "reported_count": 100}]},
        "status": {"role": "quarantine", "transcription_only": True, "model_ready": False,
                   "strategic_label_review": "pending"}}
        for identifier in probe.PACKET_IDS]


def test_book_discard_convention_and_indicator_count_have_exact_distinct_effects():
    before = probe.hand_counts(HAND)
    discard = indicator = probe.tile_type("5p")
    book = probe.score_both(before, discard, indicator, "book_post_discard_13")
    physical = probe.score_both(before, discard, indicator, "known_hand_14_plus_indicator")
    assert book["calculators_agree"] and physical["calculators_agree"]
    for calculator in ("project", "independent"):
        assert book[calculator]["shanten"] == physical[calculator]["shanten"] == 0
        assert book[calculator]["improving_types"] == physical[calculator]["improving_types"] == ["5p"]
        assert book[calculator]["counted_ukeire"] == 3
        assert physical[calculator]["counted_ukeire"] == 1
        assert book[calculator]["copies_by_improving_type"] == [
            {"tile_type": "5p", "known_copies": 1, "counted_copies": 3}]
        assert physical[calculator]["copies_by_improving_type"] == [
            {"tile_type": "5p", "known_copies": 3, "counted_copies": 1}]
    assert before[discard] == 2
    # With an unrelated indicator the physical bound is2, so the other unit
    # difference above came specifically from the shown indicator.
    unrelated = probe.score_both(before, discard, probe.tile_type("C"), "known_hand_14_plus_indicator")
    assert unrelated["project"]["counted_ukeire"] == 2


def test_fifth_known_copy_and_unrecognised_convention_are_rejected():
    before = probe.hand_counts(HAND)
    before[probe.tile_type("E")] = 4
    before[probe.tile_type("1m")] = 0
    with pytest.raises(ValueError, match="four physical copies"):
        probe.score_both(before, probe.tile_type("5p"), probe.tile_type("E"), "book_post_discard_13")
    with pytest.raises(ValueError, match="Unknown counting"):
        probe.score_both(probe.hand_counts(HAND), probe.tile_type("5p"), probe.tile_type("C"), "guess")


def test_all_printed_options_are_retained_including_mismatches():
    selected = probe.select_packets(packets())
    report = probe.compare_packet(selected[0])
    assert report["comparison_count"] == 3
    assert [row["discard_tile"] for row in report["comparisons"]] == ["5p", "1m", "E"]
    assert report["all_calculators_agree"]
    assert not report["all_printed_counts_match_book_convention"]
    first = report["comparisons"][0]
    assert first["printed_vs_book_convention"]["project_matches"]
    assert first["printed_vs_book_convention"]["independent_matches"]
    assert first["book_minus_known_hand_plus_indicator"] == {"project": 2, "independent": 2}
    assert report["comparisons"][1]["printed_vs_book_convention"]["project_minus_printed"] < 0
    changed = deepcopy(selected[0])
    changed["source_answer"]["reported_acceptance_counts"][0]["reported_count"] = 77
    alternative = probe.compare_packet(changed)
    assert alternative["comparisons"][0]["conventions"] == first["conventions"]
    assert not alternative["comparisons"][0]["printed_vs_book_convention"]["project_matches"]


@pytest.mark.parametrize("mutation, message", [
    (lambda p: p.pop(), "All four"),
    (lambda p: p.append(deepcopy(p[0])), "Duplicate"),
    (lambda p: p[0]["source_fields"]["dora_text"].update(label="dora"), "indicator"),
    (lambda p: p[0]["source_fields"].update(drawn_tile="C"), "drawn tile disagree"),
    (lambda p: p[0]["source_fields"].update(drawn_tile=[]), "drawn tile disagree"),
    (lambda p: p[0]["source_answer"]["reported_acceptance_counts"].pop(), "all three"),
    (lambda p: p[0]["source_answer"]["reported_acceptance_counts"][0].update(reported_count=True), "candidate count"),
    (lambda p: p[0]["status"].update(role="train"), "quarantined"),
    (lambda p: p[1]["source"].update(sha256="1" * 64), "source identity"),
])
def test_packet_scope_and_provenance_fail_closed(mutation, message):
    value = packets()
    mutation(value)
    with pytest.raises(ValueError, match=message):
        probe.select_packets(value)


def test_interengine_disagreement_remains_visible(monkeypatch):
    actual = probe._discard_score

    def disagree(*args):
        score = actual(*args)
        from riichi_ai.policies import DiscardScore
        return DiscardScore(score.shanten + 1, score.ukeire, score.improving_types)
    monkeypatch.setattr(probe, "_discard_score", disagree)
    result = probe.compare_packet(packets()[0])
    assert not result["all_calculators_agree"]
    assert len(result["comparisons"]) == 3
    assert all(not row["conventions"]["book_post_discard_13"]["calculators_agree"] for row in result["comparisons"])


def test_complete_report_hashes_mismatches_and_exclusive_output(tmp_path):
    source, destination = tmp_path / "packets.jsonl", tmp_path / "report.json"
    source.write_text("\n".join(json.dumps(packet) for packet in packets()) + "\n")
    assert probe.main(["--input", str(source), "--output", str(destination)]) == 0
    report = json.loads(destination.read_text())
    assert report["comparison_count"] == 12 and report["engine_convention_comparisons"] == 24
    assert report["input"]["sha256"] == probe.sha256(source)
    assert len(report["printed_count_mismatches"]) == 8
    assert len(report["implementation_files"]) == 4
    assert report["model_evaluations"] == report["optimizer_updates"] == report["new_matches"] == 0
    assert report["policy_training_ready"] is False
    assert len(report["source_pdf_digests"]) == 2
    with pytest.raises(FileExistsError):
        probe.main(["--input", str(source), "--output", str(destination)])
