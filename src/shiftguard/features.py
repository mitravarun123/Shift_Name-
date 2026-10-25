import time
from pathlib import Path

import numpy as np
import torch


def extract(model, loader, device, path, indices, provenance):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        with np.load(path) as saved:
            if str(saved['provenance']) == provenance and np.array_equal(saved['indices'], indices):
                return {k: saved[k] for k in saved.files}
        raise ValueError(f'Stale prediction cache: {path}; use a fresh output directory')
    model.eval()
    chunks = {k: [] for k in ['logits', 'features', 'labels']}
    inference_seconds, total_seconds = 0.0, time.perf_counter()
    # A small warmup is excluded from per-example forward latency.
    with torch.inference_mode():
        model(torch.zeros(1, 3, 32, 32, device=device))
        for x, y in loader:
            x = x.to(device)
            if device != 'cpu':
                getattr(torch, device).synchronize()
            start = time.perf_counter()
            logits, features = model(x)
            if device != 'cpu':
                getattr(torch, device).synchronize()
            inference_seconds += time.perf_counter() - start
            chunks['logits'].append(logits.cpu().numpy())
            chunks['features'].append(features.cpu().numpy())
            chunks['labels'].append(y.numpy())
    result = {k: np.concatenate(v) for k, v in chunks.items()}
    result.update(indices=np.asarray(indices), provenance=np.array(provenance),
                  forward_seconds=np.array(inference_seconds),
                  extraction_seconds=np.array(time.perf_counter() - total_seconds))
    np.savez_compressed(path, **result)
    return result
