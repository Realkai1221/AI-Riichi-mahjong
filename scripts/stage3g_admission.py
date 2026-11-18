"""Admit the whole fixed Stage 3G programme before formal data collection."""
from __future__ import annotations
import json
import math
from pathlib import Path
from stage3g import read_config, digest
from stage3g_campaign import lease, read, write


def main():
    config, config_hash = read_config()
    lease(config)
    reports = Path(config['reports_root'])
    preflight_path = reports / 'optimizer-preflight.json'
    preflight = read(preflight_path)
    if not preflight['complete'] or not preflight['passed']:
        raise ValueError('Real optimizer and resume preflight required')
    mps = next(d for d in preflight['devices'] if d['device'] == 'mps')
    if not mps['available'] or not mps['passed'] or not mps['resume']['passed']:
        raise ValueError('Measured MPS path did not pass; no silent fallback admission')
    root = Path(__file__).resolve().parents[1]
    collector_path = root / 'reports/stage3f/pilot.json'
    collector = read(collector_path)
    if not collector['complete'] or collector['completed_matches'] != 1000:
        raise ValueError('Completed full-collector throughput measurement required')
    # A whole-match metric includes filtering, accumulation, actual Adam and save.
    match_presentations = 3200 * 2 * 4
    learning = 2 * match_presentations / mps['matches_per_second'] + 120
    collection = 2 * 4000 / collector['matches_per_second'] + 30
    # Threefold collection proxy plus startup reserve; evaluation is a different
    # workload and this is a conservative forecast, not a measured candidate rate.
    development = 3 * 1280 / collector['matches_per_second'] + 60
    new_bytes = math.ceil(2 * 4000 * collector['compressed_artifact_bytes'] / 1000 + 2_000_000_000)
    budget = read(reports / 'budget.json')
    limits = config['phase_limits_seconds']
    reservations = dict(collection=900, grp=600, learning=3600, evaluation=700, audit=280, shutdown=120)
    total = sum(reservations.values())
    fits = dict(collection=collection <= reservations['collection'], learning=learning <= reservations['learning'],
                evaluation=development <= reservations['evaluation'],
                storage=new_bytes < config['artifact_bytes'], time=total < budget['remaining_seconds'],
                phase_caps=all(reservations[k] <= limits[k] for k in limits if k in reservations))
    result = dict(schema='stage3g-resource-admission-v1', valid=all(fits.values()), fits=fits,
                  config_sha256=config_hash, plan_sha256=digest(reports / 'plan.json'),
                  preflight_sha256=digest(preflight_path), collector_measurement_sha256=digest(collector_path),
                  assumed_device='mps', corpus_matches=4000, eligible_arms=4, epochs=2,
                  match_presentations=match_presentations, learning_forecast_seconds=learning,
                  collection_forecast_seconds=collection, development_forecast_seconds=development,
                  artifact_forecast_bytes=new_bytes, reservations_seconds=reservations,
                  learning_reservation_per_arm_seconds=900,
                  unused_seconds_after_current_and_reservations=budget['remaining_seconds'] - total,
                  basis='Twofold measured actual optimizer and full-collector time allowance; threefold collector proxy for development games; 120s learning startup/save allowance. Forecasts exclude strength claims.')
    write(reports / 'admission.json', result)
    print(json.dumps(result), flush=True)
    if not result['valid']:
        raise RuntimeError('Whole fixed programme does not fit; formal collection remains unstarted')


if __name__ == '__main__':
    main()
