"""Evaluate synthetic corruptions using a completed clean run, without training."""
import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'src'))
import joblib
import numpy as np
import torch
from torch.utils.data import DataLoader, Subset
from torchvision import datasets, transforms
from shiftguard.corruption import Corrupt
from shiftguard.data import MEAN, STD
from shiftguard.evaluate import evaluate_cache
from shiftguard.features import extract
from shiftguard.models import build_model
from shiftguard.utils import digest, save_json, seed_everything, versions

parser = argparse.ArgumentParser()
parser.add_argument('--run-dir', default='results/full_seed42')
parser.add_argument('--corruptions', nargs='+', default=['noise:1', 'blur:1'])
args = parser.parse_args()
out = Path(args.run_dir)
cfg = json.loads((out / 'config.json').read_text())
if not (out / 'metrics' / 'clean.json').exists():
    raise ValueError('Complete independent clean evaluation first')
seed_everything(cfg['seed'], cfg['threads'])
model = build_model(cfg).to(cfg['device'])
model.load_state_dict(torch.load(out / 'best.pt', map_location=cfg['device'], weights_only=False)['state_dict'])
provenance = digest(out / 'best.pt')
artifact = joblib.load(out / 'reliability.joblib')
if artifact.get('provenance') != provenance:
    raise ValueError('Reliability artifact checkpoint differs')
with np.load(out / 'splits.npz') as z:
    ids = z['test']
for item in args.corruptions:
    kind, severity = item.split(':')
    severity = int(severity)
    name = f'{kind}_{severity}'
    transform = transforms.Compose([Corrupt(kind, severity, cfg['seed']), transforms.ToTensor(),
                                     transforms.Normalize(MEAN, STD)])
    data = datasets.CIFAR10(cfg['data_dir'], train=False, download=False, transform=transform)
    loader = DataLoader(Subset(data, ids), batch_size=cfg['batch_size'], num_workers=cfg['num_workers'])
    cache = extract(model, loader, cfg['device'], out/'predictions'/f'{name}.npz', ids, provenance + name)
    result = evaluate_cache(cache, artifact, cfg, out, name)
    save_json(out/'metrics'/f'{name}_environment.json', {'experiment_config': {'kind':kind, 'severity':severity,
        'seed':cfg['seed'], 'source_run':str(out)}, 'versions':versions()})
    print(name, json.dumps(result['classifier']), flush=True)
