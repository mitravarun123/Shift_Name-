from pathlib import Path

import numpy as np
import torch
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader, Subset
from torchvision import datasets, transforms

MEAN = (0.4914, 0.4822, 0.4465)
STD = (0.2470, 0.2435, 0.2616)


def stratified_split(labels, val_size=5000, seed=42):
    return train_test_split(np.arange(len(labels)), test_size=val_size,
                            stratify=labels, random_state=seed)


def take_stratified(indices, labels, limit, seed):
    indices = np.asarray(indices)
    if limit is None or limit == len(indices):
        return indices
    if not 10 <= limit < len(indices):
        raise ValueError('Subset limit must be >=10 and <= available size')
    selected, _ = train_test_split(indices, train_size=limit,
                                  stratify=np.asarray(labels)[indices], random_state=seed)
    return selected


def partition_validation(indices, labels, seed):
    # 40% checkpoint selection, 20% reference, 20% estimator, 20% thresholds.
    labels = np.asarray(labels)
    selection, remainder = train_test_split(indices, train_size=0.4,
        stratify=labels[indices], random_state=seed)
    reference, remainder = train_test_split(remainder, train_size=1/3,
        stratify=labels[remainder], random_state=seed + 1)
    fit, calibration = train_test_split(remainder, train_size=0.5,
        stratify=labels[remainder], random_state=seed + 2)
    return dict(selection=selection, reference=reference, fit=fit, calibration=calibration)


def make_loaders(cfg, output, include_test=False):
    normalize = transforms.Normalize(MEAN, STD)
    eval_transform = transforms.Compose([transforms.ToTensor(), normalize])
    aug = transforms.Compose([transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip(), transforms.ToTensor(), normalize])
    base = datasets.CIFAR10(cfg['data_dir'], train=True, download=cfg['download'],
                            transform=eval_transform)
    train, val = stratified_split(base.targets, cfg['val_size'], cfg['seed'])
    train = take_stratified(train, base.targets, cfg['train_limit'], cfg['seed'])
    val = take_stratified(val, base.targets, cfg['val_limit'], cfg['seed'] + 1)
    parts = partition_validation(val, base.targets, cfg['seed'])
    indices = dict(train=train, **parts)
    train_base = datasets.CIFAR10(cfg['data_dir'], train=True, transform=aug)
    loaders = {}
    for name, ids in indices.items():
        loaders[name] = DataLoader(Subset(train_base if name == 'train' else base, ids),
            batch_size=cfg['batch_size'], shuffle=name == 'train',
            num_workers=cfg['num_workers'],
            generator=torch.Generator().manual_seed(cfg['seed']), pin_memory=False)
    if include_test:
        test = datasets.CIFAR10(cfg['data_dir'], train=False, download=cfg['download'],
                               transform=eval_transform)
        ids = take_stratified(np.arange(len(test)), test.targets, cfg['test_limit'], cfg['seed'])
        indices['test'] = ids
        loaders['test'] = DataLoader(Subset(test, ids), batch_size=cfg['batch_size'],
                                    num_workers=cfg['num_workers'], shuffle=False)
    path = Path(output) / 'splits.npz'
    # Test IDs live in a separate official index namespace.
    if path.exists():
        old = np.load(path)
        for name, ids in indices.items():
            if name in old and not np.array_equal(old[name], ids):
                raise ValueError('Saved split differs; use a new output directory')
    np.savez_compressed(path, **indices)
    return loaders, indices
