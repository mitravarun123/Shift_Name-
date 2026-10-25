import argparse
import json
import resource
from pathlib import Path

import joblib
import numpy as np
import torch
from torch.utils.data import DataLoader, Subset
from torchvision import datasets, transforms

from .corruption import Corrupt
from .data import MEAN, STD, make_loaders
from .evaluate import evaluate_cache, fit_reliability
from .features import extract
from .models import build_model
from .plotting import plot_saved
from .train import train
from .utils import digest, load_config, save_json, seed_everything, versions


def main(default_config='configs/default.yaml'):
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', default=default_config)
    parser.add_argument('--stage', choices=['all', 'train', 'evaluate', 'plot'], default='all')
    parser.add_argument('--output-dir')
    parser.add_argument('--seed', type=int)
    args = parser.parse_args()
    cfg = load_config(args.config)
    if args.output_dir:
        cfg['output_dir'] = args.output_dir
    if args.seed is not None:
        cfg['seed'] = args.seed
    out = Path(cfg['output_dir'])
    out.mkdir(parents=True, exist_ok=True)
    if args.stage == 'plot':
        for path in (out / 'metrics').glob('*.json'):
            if path.stem != 'environment' and not path.stem.endswith('_environment'):
                plot_saved(out, path.stem)
        return
    saved_cfg = out / 'config.json'
    if saved_cfg.exists() and json.loads(saved_cfg.read_text()) != cfg:
        raise ValueError('Configuration differs from saved run; use a new output directory')
    if args.stage in ('all', 'train') and (out / 'best.pt').exists():
        raise ValueError('Checkpoint already exists; use --stage evaluate or a fresh output directory')
    seed_everything(cfg['seed'], cfg['threads'])
    save_json(saved_cfg, cfg)
    save_json(out / ('evaluation_environment.json' if args.stage == 'evaluate' else 'environment.json'), versions())
    model = build_model(cfg).to(cfg['device'])
    if args.stage in ('all', 'train'):
        loaders, _ = make_loaders(cfg, out, include_test=False)
        train(model, loaders, cfg, out)
        if args.stage == 'train':
            return
    else:
        checkpoint = torch.load(out / 'best.pt', map_location=cfg['device'], weights_only=False)
        model.load_state_dict(checkpoint['state_dict'])
    # Official test loaded only after classifier training/model selection has ended.
    loaders, ids = make_loaders(cfg, out, include_test=True)
    provenance = digest(out / 'best.pt')
    cache_dir = out / 'predictions'
    caches = {name: extract(model, loaders[name], cfg['device'], cache_dir / f'{name}.npz', ids[name],
                           provenance) for name in ['reference', 'fit', 'calibration']}
    if (out / 'reliability.joblib').exists():
        artifact = joblib.load(out / 'reliability.joblib')
        if artifact.get('provenance') != provenance:
            raise ValueError('Reliability artifact checkpoint differs; use a fresh output directory')
    else:
        artifact = fit_reliability(caches, cfg, out)
    test = extract(model, loaders['test'], cfg['device'], cache_dir / 'test.npz', ids['test'], provenance)
    result = evaluate_cache(test, artifact, cfg, out)
    for corruption in cfg.get('corruptions', []):
        kind, severity = corruption['kind'], corruption['severity']
        name = f'{kind}_{severity}'
        transform = transforms.Compose([Corrupt(kind, severity, cfg['seed']), transforms.ToTensor(),
                                        transforms.Normalize(MEAN, STD)])
        data = datasets.CIFAR10(cfg['data_dir'], train=False, transform=transform)
        loader = DataLoader(Subset(data, ids['test']), batch_size=cfg['batch_size'], num_workers=cfg['num_workers'])
        cache = extract(model, loader, cfg['device'], cache_dir / f'{name}.npz', ids['test'], provenance + name)
        evaluate_cache(cache, artifact, cfg, out, name)
    # macOS reports bytes; Linux reports KiB. Lifetime process RSS, not incremental allocation.
    import sys
    peak = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    save_json(out / ('evaluation_resources.json' if args.stage == 'evaluate' else 'resources.json'), {'peak_process_rss_mb': peak / (1024**2 if sys.platform == 'darwin' else 1024),
                                     'device': cfg['device'], 'threads': cfg['threads']})
    print(json.dumps(result, indent=2), flush=True)
