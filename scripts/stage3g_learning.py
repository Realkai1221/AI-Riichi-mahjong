#!/usr/bin/env python3
"""Leased optimizer preflight, fixed offline value training, and validation."""
from __future__ import annotations
import argparse
import json
import os
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'src'))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('phase', choices=['preflight', 'train', 'validate'])
    parser.add_argument('--corpus', required=True)
    parser.add_argument('--reference')
    parser.add_argument('--reference-sha256')
    parser.add_argument('--output-dir', '--output', dest='output_dir', required=True)
    parser.add_argument('--arm', choices=['mc_a0', 'mc_a1', 'delta_a0', 'delta_a1'])
    parser.add_argument('--device', default='mps')
    parser.add_argument('--max-seconds', type=float, default=180)
    parser.add_argument('--threads', type=int, default=1)
    parser.add_argument('--hand-targets')
    parser.add_argument('--resume')
    parser.add_argument('--checkpoint')
    args = parser.parse_args()
    if not os.environ.get('RIICHI_STAGE3G_OPERATION') or not os.environ.get('RIICHI_STAGE3G_TOKEN'):
        raise RuntimeError('Stage 3G numerical work requires an active controller lease')
    if args.max_seconds <= 0 or args.threads <= 0:
        parser.error('Time allowance and threads must be positive')
    if args.phase != 'validate' and (not args.reference or not args.reference_sha256):
        parser.error('Preflight and training require a pinned reference')
    if args.phase == 'train' and not args.arm:
        parser.error('Training requires a fixed arm')
    if args.phase == 'validate' and not args.checkpoint:
        parser.error('Validation requires a completed checkpoint')
    import torch
    from riichi_ai.stage3g_learning import preflight, train, validate
    torch.set_num_threads(args.threads)
    if args.phase == 'preflight':
        result = preflight(args.corpus, args.reference, args.reference_sha256,
                           args.output_dir, max_seconds=args.max_seconds)
    elif args.phase == 'train':
        result = train(args.corpus, args.reference, args.reference_sha256, args.output_dir,
                       arm=args.arm, device=args.device, max_seconds=args.max_seconds,
                       hand_targets_path=args.hand_targets, resume=args.resume)
    else:
        result = validate(args.corpus, args.checkpoint, args.output_dir, device=args.device,
                          hand_targets_path=args.hand_targets, max_seconds=args.max_seconds)
    print(json.dumps({key: result.get(key) for key in ('complete', 'passed', 'seconds',
        'scheduled_updates', 'optimizer_updates', 'initial_weights_sha256', 'final_weights_sha256')}, sort_keys=True))
    if result.get('passed') is False or result.get('complete') is not True:
        raise RuntimeError('Stage 3G learner phase did not pass its declared gate')


if __name__ == '__main__':
    main()
