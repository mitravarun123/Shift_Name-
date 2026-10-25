"""Run principal experiments without shell interpolation; aggregate actual seed metrics."""
import argparse
import json
import subprocess
import sys
from pathlib import Path

import numpy as np
from run_ablation import run_ablation

parser = argparse.ArgumentParser()
parser.add_argument('--config', default='configs/default.yaml')
parser.add_argument('--seeds', type=int, nargs='+', default=[42, 43, 44])
parser.add_argument('--prefix', default='results/reproduction')
args = parser.parse_args()
rows = []
for seed in args.seeds:
    output = Path(f'{args.prefix}_seed{seed}')
    if (output / 'best.pt').exists() and not (output / 'metrics' / 'clean.json').exists():
        stats_path=output/'training.json'
        stats=json.loads(stats_path.read_text()) if stats_path.exists() else {}
        if not stats.get('training_complete', False):
            raise ValueError(f'{output}: incomplete checkpoint; do not silently evaluate an interrupted training run')
    stage = 'evaluate' if (output / 'best.pt').exists() else 'all'
    subprocess.run([sys.executable, 'scripts/run_experiment.py', '--config', args.config,
                    '--seed', str(seed), '--output-dir', str(output), '--stage', stage], check=True)
    rows.append(run_ablation(output))
summary = {'seeds': args.seeds, 'n_runs': len(rows), 'sample_std_ddof': 1, 'methods': {}}
for method in rows[0]['methods']:
    summary['methods'][method] = {}
    for metric in ['auroc', 'aurc', 'risk_at_50', 'risk_at_70', 'risk_at_80', 'risk_at_90']:
        values = [r['methods'][method][metric] for r in rows]
        summary['methods'][method][metric] = dict(
            mean=float(np.mean(values)) if all(v is not None for v in values) else None,
            std=float(np.std(values, ddof=1)) if len(values) > 1 and all(v is not None for v in values) else None)
summary['paired_all_three_minus_uncertainty'] = {
    metric: {'mean': float(np.mean([r['methods']['Combined'][metric] - r['methods']['MSP+Entropy'][metric] for r in rows])),
             'std': float(np.std([r['methods']['Combined'][metric] - r['methods']['MSP+Entropy'][metric] for r in rows], ddof=1)) if len(rows)>1 else None}
    for metric in ['auroc','aurc','risk_at_50','risk_at_70','risk_at_80','risk_at_90']}
Path(args.prefix + '_summary.json').write_text(json.dumps(summary, indent=2, allow_nan=False))
