#!/usr/bin/env python3
"""Execute a frozen, leased Stage 3H saved-state inference plan."""
from __future__ import annotations
import argparse
import hashlib
import json
import math
import os
from pathlib import Path
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'src'))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('phase', choices=('preflight', 'run'))
    parser.add_argument('--plan', required=True)
    parser.add_argument('--output-dir', required=True)
    parser.add_argument('--source-manifest')
    parser.add_argument('--max-seconds', type=float, required=True)
    args = parser.parse_args()
    if not os.environ.get('RIICHI_STAGE3H_OPERATION') or not os.environ.get('RIICHI_STAGE3H_TOKEN'):
        raise RuntimeError('Stage 3H inference requires an active numerical lease')
    from stage3h import read_config, verify_sources
    config, config_sha = read_config()
    operations = json.loads((Path(config['data_root']) / 'budget-ledger.json').read_text())['operations']
    active = [operation for operation in operations if operation['status'] == 'active']
    if (len(active) != 1 or active[0]['name'] != os.environ['RIICHI_STAGE3H_OPERATION']
            or active[0]['token'] != os.environ['RIICHI_STAGE3H_TOKEN']
            or (not active[0]['name'].startswith('preflight') if args.phase == 'preflight' else active[0]['name'] != 'inference')):
        raise RuntimeError('Inference requires its matching active Stage 3H lease')
    remaining = active[0]['reserved_seconds'] - (time.time() - active[0]['started_unix'])
    if remaining <= 12 or not math.isfinite(args.max_seconds) or not 0 < args.max_seconds <= 600:
        raise ValueError('Invalid inference deadline or insufficient lease remains')
    max_seconds = min(args.max_seconds, remaining - 10)
    payload = Path(args.plan).read_bytes()
    plan = json.loads(payload)
    if plan.get('config_sha256') != config_sha:
        raise ValueError('Diagnostic plan configuration differs')
    sources = json.loads(Path(args.source_manifest).read_text())['sources'] if args.source_manifest else plan['sources']
    if not {'src/riichi_ai/stage3h_inference.py', 'scripts/stage3h_inference.py'} <= set(sources):
        raise ValueError('Inference runtime source membership missing')
    verify_sources(sources)
    output = Path(args.output_dir).resolve()
    if output == Path(config['data_root']) or not output.is_relative_to(Path(config['data_root'])):
        raise ValueError('Diagnostic output must stay inside its declared fresh data root')
    from riichi_ai.stage3h_inference import run_plan
    result = run_plan(plan, args.output_dir, phase=args.phase, max_seconds=max_seconds,
                      plan_sha256=hashlib.sha256(payload).hexdigest())
    print(json.dumps({key: result.get(key) for key in ('complete', 'failure', 'optional_rows', 'model_rows', 'elapsed_seconds')}), flush=True)
    if not result['complete']:
        raise RuntimeError('Fixed saved-state inference did not complete')


if __name__ == '__main__':
    main()
