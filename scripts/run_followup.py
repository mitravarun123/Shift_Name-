"""Sequential, reproducible CPU follow-up; no paid GPU or cloud dependency."""
import json
import subprocess
import sys
from pathlib import Path


def run(*args):
    subprocess.run([sys.executable, *args], check=True)


run('scripts/reproduce_results.py', '--prefix', 'results/full', '--seeds', '42', '43', '44')
for config, output in [('configs/small_cnn_matched_pilot.yaml','results/small_cnn_matched_pilot_seed42'),
                       ('configs/resnet18_pilot.yaml','results/resnet18_pilot_seed42')]:
    folder=Path(output)
    if (folder/'best.pt').exists() and not (folder/'metrics'/'clean.json').exists():
        stats_path=folder/'training.json'
        stats=json.loads(stats_path.read_text()) if stats_path.exists() else {}
        if not stats.get('training_complete',False):
            raise ValueError(f'{output} has an incomplete checkpoint; inspect before resuming')
    stage='evaluate' if (folder/'best.pt').exists() else 'all'
    run('scripts/run_experiment.py','--config',config,'--stage',stage)
    run('scripts/run_ablation.py','--run-dir',output)
run('scripts/summarize_followup.py')
