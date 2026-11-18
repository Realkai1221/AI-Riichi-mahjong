"""Collect or derive one frozen Stage 2B data artifact; never train a model.

Run derivation through the training-budget subprocess wrapper so startup, exact
alignment checks, compression and report writes all count toward its ledger.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import time

from riichi_ai.stage2b_data import derive_efficiency, generate_stage2b_dataset, verify_stage2b_dataset

PROJECT = Path(__file__).resolve().parents[1]
REPORT_RESERVE_BYTES = 20_000_000


def artifact_bytes(data: Path, reports: Path) -> int:
    return sum(path.stat().st_size for root in (data, reports)
               for path in root.rglob('*') if path.is_file())


def save(reports: Path, name: str, value: dict) -> None:
    reports.mkdir(parents=True, exist_ok=True)
    target = reports / name
    if target.exists():
        raise FileExistsError(f'Report already exists: {target}')
    target.write_text(json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + '\n')


def run(command: str, config: dict, *, max_seconds: float | None = None) -> dict:
    data = Path(config['data_root']).expanduser().resolve()
    reports = (PROJECT / config['reports_root']).resolve()
    ceiling = config['limits']['new_artifact_bytes']
    before = artifact_bytes(data, reports)
    remaining = ceiling - before - REPORT_RESERVE_BYTES
    if remaining <= 0:
        raise RuntimeError('Shared Stage 2B artifact budget has no remaining reserved capacity')
    memory_limit = config['limits']['memory_gib'] * 1024**3
    started = time.perf_counter()
    if command == 'collect':
        target = data / 'dataset'
        manifest = generate_stage2b_dataset(target, config, disk_limit_bytes=remaining,
                                           memory_limit_bytes=memory_limit)
        save(reports, 'dataset-manifest.json', manifest)
        if not manifest['valid']:
            raise RuntimeError('Collection failed; inspect the preserved focal manifest')
        verify_started = time.perf_counter()
        audit = verify_stage2b_dataset(target, replay_matches_per_group=config['collection']['replay_matches_per_group'])
        save(reports, 'dataset-audit.json', audit)
        result = {'command': command, 'valid': audit['valid'],
                  'matches': manifest['completed_matches'], 'focal_decisions': manifest['decisions'],
                  'collection_seconds': manifest['elapsed_seconds'],
                  'collection_verification_seconds': time.perf_counter() - verify_started,
                  'replayed_matches': audit['replayed_matches'],
                  'accounting': 'Collection and collection verification; separate from training-related derivation.'}
    else:
        old = command == 'derive-old'
        source = Path(config['stage2a_dataset']) if old else data / 'dataset'
        if old and hashlib.sha256((source / 'manifest.json').read_bytes()).hexdigest() != config['stage2a_manifest_sha256']:
            raise ValueError('Historical Stage 2A source manifest hash differs')
        label = 'derived-stage2a' if old else 'derived-stage2b'
        manifest = derive_efficiency(source, data / label,
                                     splits=('train', 'validation') if old else ('diagnostic', 'validation', 'test'),
                                     disk_limit_bytes=remaining, memory_limit_bytes=memory_limit,
                                     max_seconds=max_seconds)
        save(reports, label + '-manifest.json', manifest)
        result = {'command': command, 'valid': manifest['valid'], 'matches': manifest['completed_matches'],
                  'checked_rows': manifest['checked_rows'], 'checked_candidates': manifest['checked_candidates'],
                  'derivation_seconds': manifest['elapsed_seconds'],
                  'accounting': 'Training-related: caller must charge the complete subprocess duration.'}
    result.update(elapsed_seconds=time.perf_counter() - started, artifact_bytes_before=before,
                  artifact_bytes_before_resources_report=artifact_bytes(data, reports), artifact_limit_bytes=ceiling)
    save(reports, command + '-resources.json', result)
    result['artifact_bytes_after'] = artifact_bytes(data, reports)
    if result['artifact_bytes_after'] > ceiling:
        raise RuntimeError('Shared Stage 2B artifact ceiling exceeded; retain evidence and stop')
    if not result['valid']:
        raise RuntimeError('Data checks failed; retained reports describe the failure')
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('command', choices=('collect', 'derive-old', 'derive-fresh'))
    parser.add_argument('--config', type=Path, default=PROJECT / 'configs/stage2b.json')
    parser.add_argument('--max-seconds', type=float, help='Cooperative derivation limit; entire subprocess remains charged')
    args = parser.parse_args()
    if args.command == 'collect' and args.max_seconds is not None:
        parser.error('--max-seconds is a derivation option')
    print(json.dumps(run(args.command, json.loads(args.config.read_text()), max_seconds=args.max_seconds)), flush=True)


if __name__ == '__main__':
    main()
