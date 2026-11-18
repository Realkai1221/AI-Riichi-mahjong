"""Scope/provenance tests; root executes these within its declared lease.

Fixtures are authored synthetic metadata, never copied book exercises. Numerical
algorithms remain covered by test_book_count_conventions; this file checks the
new frozen selection, identity guards and complete mismatch reporting.
"""
from copy import deepcopy
import hashlib
import importlib.util
import json
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
SPEC = importlib.util.spec_from_file_location("stage3a_book_counts", ROOT / "scripts/stage3a_book_counts.py")
probe = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(probe)

HAND = ["1m", "2m", "3m", "1p", "2p", "3p", "1s", "2s", "3s", "E", "E", "E", "5p", "5p"]
OPTIONS = ["5p", "1m", "2m", "3m", "1p", "2p"]


def packets():
    return [{
        "schema": "raw-source-packet-v1", "packet_id": identifier,
        "source": {"book_id": identifier.rsplit("-q", 1)[0],
                   "path": f"Books/{identifier.rsplit('-q', 1)[0]}-authored-fixture.pdf",
                   "sha256": probe.SOURCE_PINS[identifier.rsplit("-q", 1)[0]]["sha256"]},
        "source_fields": {"shown_hand_including_draw": HAND[:],
                          "concealed_hand_before_draw": HAND[:-1], "drawn_tile": HAND[-1],
                          "dora_text": {"label": "dora indicator", "tile_type": "C"}},
        "source_answer": {"reported_acceptance_counts": [
            {"discard": tile, "reported_count": 100 + index}
            for index, tile in enumerate(OPTIONS[:option_count])]},
        "status": {"role": "quarantine", "transcription_only": True, "model_ready": False,
                   "training_performed": False, "model_evaluation_performed": False,
                   "strategic_label_review": "pending"},
        "grouping": {"intended_use": "development", "holdout_eligible": False},
        "evidence": {"independent_review": {"status": "passed_transcription", "reviewer": "/root"}},
    } for identifier, option_count in probe.PACKET_OPTION_COUNTS.items()]


def test_selection_keeps_all_47_options_in_fixed_order():
    selected = probe.select_packets(list(reversed(packets())) + [{"packet_id": "authored-unselected-example"}])
    assert [row["packet_id"] for row in selected] == list(probe.PACKET_IDS)
    assert sum(len(row["source_answer"]["reported_acceptance_counts"]) for row in selected) == 47


@pytest.mark.parametrize("mutation, message", [
    (lambda rows: rows.pop(), "All ten"),
    (lambda rows: rows.append(deepcopy(rows[0])), "Duplicate selected"),
    (lambda rows: rows[0]["source_answer"]["reported_acceptance_counts"].pop(), "All prespecified"),
    (lambda rows: rows[0]["source_answer"]["reported_acceptance_counts"][1].update(discard="5p"), "duplicate printed"),
    (lambda rows: rows[0]["source_answer"]["reported_acceptance_counts"][0].update(reported_count=True), "printed option"),
    (lambda rows: rows[0]["source"].update(sha256="0" * 64), "source digest"),
    (lambda rows: rows[0]["source_fields"].update(drawn_tile="C"), "draw does not match"),
    (lambda rows: rows[0]["source_fields"]["dora_text"].update(label="dora"), "explicit dora indicator"),
    (lambda rows: rows[0]["status"].update(role="train"), "quarantine"),
    (lambda rows: rows[0]["grouping"].update(holdout_eligible=True), "development"),
    (lambda rows: rows[0]["evidence"]["independent_review"].update(status="pending"), "transcription review"),
])
def test_selection_rejects_scope_or_provenance_mutations(mutation, message):
    rows = packets()
    mutation(rows)
    with pytest.raises(ValueError, match=message):
        probe.select_packets(rows)


def test_changed_input_hash_is_rejected_before_loading_calculators(tmp_path, monkeypatch):
    path = tmp_path / "authored.jsonl"
    raw = ("\n".join(json.dumps(row) for row in packets()) + "\n").encode()
    path.write_bytes(raw)
    monkeypatch.setattr(probe, "INPUT_SHA256", hashlib.sha256(raw).hexdigest())
    monkeypatch.setattr(probe, "INPUT_BYTES", len(raw))
    assert len(probe.verify_input(path)[0]) == 10
    altered = raw.replace(b'"reported_count": 100', b'"reported_count": 101', 1)
    assert altered != raw and len(altered) == len(raw)
    path.write_bytes(altered)
    monkeypatch.setattr(probe, "load_helper", lambda: pytest.fail("Numerical helper loaded before failed input check"))
    with pytest.raises(ValueError, match="input digest"):
        probe.audit_file(path)


def test_source_pdf_bytes_are_verified_not_only_declared_digests(tmp_path, monkeypatch):
    monkeypatch.setattr(probe, "ROOT", tmp_path)
    (tmp_path / "Books").mkdir()
    pins = {}
    for book_id in probe.SOURCE_PINS:
        path = tmp_path / f"Books/{book_id}-authored-fixture.pdf"
        raw = b"Authored identity fixture, not a PDF or book."
        path.write_bytes(raw)
        pins[book_id] = {"sha256": hashlib.sha256(raw).hexdigest(), "bytes": len(raw)}
    monkeypatch.setattr(probe, "SOURCE_PINS", pins)
    selected = probe.select_packets(packets())
    assert len(probe.verify_sources(selected)) == 2
    path = tmp_path / selected[0]["source"]["path"]
    path.write_bytes(path.read_bytes().replace(b"Authored", b"authored", 1))
    with pytest.raises(ValueError, match="PDF digest"):
        probe.verify_sources(selected)


def test_changed_helper_is_rejected(tmp_path, monkeypatch):
    path = tmp_path / "authored_helper.py"
    raw = b"# authored non-executed helper identity fixture\n"
    path.write_bytes(raw)
    monkeypatch.setattr(probe, "HELPER_PATH", path)
    monkeypatch.setattr(probe, "HELPER_SHA256", hashlib.sha256(raw).hexdigest())
    monkeypatch.setattr(probe, "HELPER_BYTES", len(raw))
    assert probe.verify_helper()["sha256"] == hashlib.sha256(raw).hexdigest()
    path.write_bytes(raw.replace(b"authored", b"Authored", 1))
    with pytest.raises(ValueError, match="helper digest"):
        probe.verify_helper()


def test_mismatches_disagreements_and_convention_differences_all_remain_visible():
    # Authored result metadata isolates reporting from the already-tested engines.
    first = {"discard_tile": "1m", "reported_count": 10,
             "conventions": {"book_post_discard_13": {"calculators_agree": True},
                             "known_hand_14_plus_indicator": {"calculators_agree": True}},
             "printed_vs_book_convention": {"project_matches": False, "independent_matches": False,
                                             "project_minus_printed": -2, "independent_minus_printed": -2},
             "book_minus_known_hand_plus_indicator": {"project": 1, "independent": 1}}
    second = deepcopy(first)
    second["discard_tile"] = "2m"
    second["printed_vs_book_convention"].update(project_matches=True, independent_matches=True,
                                              project_minus_printed=0, independent_minus_printed=0)
    second["conventions"]["known_hand_14_plus_indicator"]["calculators_agree"] = False
    second["book_minus_known_hand_plus_indicator"] = {"project": 0, "independent": 0}
    rows = [{"packet_id": "authored-result", "comparisons": [first, second]}]
    before = deepcopy(rows)
    summary = probe.findings(rows)
    assert rows == before
    assert summary["comparison_count"] == 2 and summary["engine_convention_comparisons"] == 4
    assert not summary["all_calculators_agree"] and not summary["all_printed_counts_match_book_convention"]
    assert [x["discard_tile"] for x in summary["printed_count_mismatches"]] == ["1m"]
    assert summary["calculator_disagreements"] == [
        {"packet_id": "authored-result", "discard_tile": "2m", "convention": "known_hand_14_plus_indicator"}]
    assert [x["discard_tile"] for x in summary["count_differences_between_conventions"]] == ["1m"]


def test_existing_output_is_not_overwritten_or_recomputed(tmp_path, monkeypatch):
    output = tmp_path / "existing.json"
    output.write_text('{"retained": true}\n')
    monkeypatch.setattr(probe, "audit_file", lambda path: pytest.fail("Existing output should stop before calculation"))
    with pytest.raises(FileExistsError, match="no silent replacement"):
        probe.main(["--output", str(output)])
    assert output.read_text() == '{"retained": true}\n'
