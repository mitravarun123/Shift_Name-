import hashlib
import importlib.metadata
import json
import platform
import random
from pathlib import Path

import numpy as np
import torch
import yaml


def seed_everything(seed, threads=4):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.set_num_threads(threads)
    torch.use_deterministic_algorithms(True)


def load_config(path):
    with open(path) as f:
        cfg = yaml.safe_load(f)
    if cfg['device'] not in ('cpu', 'mps', 'cuda'):
        raise ValueError('device must be cpu, mps or cuda')
    return cfg


def save_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, allow_nan=False))


def versions():
    return {'python': platform.python_version(), 'platform': platform.platform(),
            **{n: importlib.metadata.version(n) for n in
               ['torch', 'torchvision', 'numpy', 'scikit-learn', 'matplotlib', 'PyYAML']}}


def digest(path):
    h = hashlib.sha256()
    with open(path, 'rb') as f:
        for block in iter(lambda: f.read(1024 * 1024), b''):
            h.update(block)
    return h.hexdigest()
