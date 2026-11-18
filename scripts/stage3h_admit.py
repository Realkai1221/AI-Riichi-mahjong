"""Freeze the largest predeclared panel using measured resource forecasts."""
import hashlib
import json
import os
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
REPORTS = ROOT / 'reports/stage3h'


def read(path):
    return json.loads(Path(path).read_text())


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def write(path, value):
    with path.open('x') as stream:
        json.dump(value, stream, indent=2, allow_nan=False)
        stream.write('\n')


def main():
    if not os.environ.get('RIICHI_STAGE3H_TOKEN'):
        raise RuntimeError('Resource admission requires the active ledger')
    plan = read(REPORTS / 'preflight-plan.json')
    inference = read(REPORTS / 'inference-preflight.json')
    gradients = read(REPORTS / 'gradient-preflight.json')
    config = read(ROOT / 'configs/stage3h.json')
    assert inference['complete'] and gradients['complete'] and gradients['passed']
    assert inference['plan_sha256'] == gradients['plan_sha256'] == sha(REPORTS / 'preflight-plan.json')
    assert all(item['tensors_exact'] and item['scores_exact'] for item in inference['production_path_checks'])
    prior_artifacts = sum(p.stat().st_size for folder in (Path(config['data_root']), REPORTS)
                          for p in folder.rglob('*') if p.is_file())
    preflight_arrays = sum(item['numeric']['bytes'] for item in inference['matches'])
    vector_bytes = gradients['parameter_layout']['trainable_parameters'] * 4 * 2 * 4 * 6
    gradient_forecast = 2 * gradients['projected_formal_compute_seconds_24_groups'] + 20
    choices = []
    for size in config['candidate_split_sizes']:
        names = {split: values[:size] for split, values in plan['selected_matches'].items()}
        rows = sum(plan['optional_rows_by_match'][split][name] for split, values in names.items() for name in values)
        ratio = rows / inference['optional_rows']
        # Scaling all elapsed time is deliberately more conservative than removing startup.
        seconds = 2 * inference['elapsed_seconds'] * ratio + 20
        artifact_forecast = prior_artifacts + 2 * preflight_arrays * ratio + vector_bytes + 20_000_000
        choices.append(dict(matches_per_split=size, optional_rows=rows, inference_seconds=seconds,
                            gradient_seconds=gradient_forecast, artifact_bytes=artifact_forecast,
                            fits=seconds <= 580 and gradient_forecast <= 280 and artifact_forecast < config['artifact_bytes']))
    selected = next((item for item in choices if item['fits']), None)
    if selected is None:
        write(REPORTS / 'admission.json', dict(admitted=False, candidates=choices))
        raise RuntimeError('No predeclared panel fits; formal work must not start')
    admission = dict(schema='stage3h-admission-v1', admitted=True, candidates=choices, selected=selected,
        inference_preflight_sha256=sha(REPORTS / 'inference-preflight.json'),
        gradient_preflight_sha256=sha(REPORTS / 'gradient-preflight.json'),
        measured_initial_cpu_mps_parity=gradients['backend_comparison'],
        forecast_rule='Two times full measured inference elapsed time scaled by exact row count plus20seconds; two times24measured MPS gradient groups plus20seconds. No preflight outcome metrics used.',
        reservations_seconds=dict(inference=600, gradients=300, audit=120, shutdown=60),
        prior_artifact_bytes=prior_artifacts, exact_raw_gradient_bytes=vector_bytes,
        limitation='Short preflight extrapolation; watchdog/resource caps remain authoritative.')
    write(REPORTS / 'admission.json', admission)
    size = selected['matches_per_split']
    plan['selected_matches'] = {split: values[:size] for split, values in plan['selected_matches'].items()}
    plan['optional_rows_by_match'] = {split: {name: plan['optional_rows_by_match'][split][name] for name in names}
                                     for split, names in plan['selected_matches'].items()}
    plan['sources']['scripts/stage3h_admit.py'] = sha(Path(__file__))
    plan['execution_admitted'] = True
    plan['admission_sha256'] = sha(REPORTS / 'admission.json')
    write(REPORTS / 'plan.json', plan)
    print(json.dumps(selected))


if __name__ == '__main__':
    main()
