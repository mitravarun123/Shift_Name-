"""Compare uncertainty-only vs three-signal regression on a frozen run."""
import argparse
import json
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'src'))
import joblib
import numpy as np
from shiftguard.evaluate import evaluate_cache
from shiftguard.metrics import select_threshold
from shiftguard.reliability import fit_estimator, score_inputs
from shiftguard.utils import digest, save_json


def run_ablation(run_dir, output_dir=None):
    source = Path(run_dir)
    out = Path(output_dir) if output_dir else Path('results/ablation') / source.name
    cfg = json.loads((source / 'config.json').read_text())
    artifact = joblib.load(source / 'reliability.joblib')
    fingerprint = digest(source / 'best.pt')
    if artifact.get('provenance') != fingerprint:
        raise ValueError('Reliability checkpoint differs')
    caches = {}
    for name in ['fit', 'calibration']:
        with np.load(source / 'predictions' / f'{name}.npz') as z:
            caches[name] = {k: z[k] for k in z.files}
        if str(caches[name]['provenance']) != fingerprint:
            raise ValueError('Cache checkpoint differs')
    _, pred, x = score_inputs(caches['fit'], artifact['distance'])
    estimator = fit_estimator(x[:, :2], (pred != caches['fit']['labels']).astype(int), cfg['seed'])
    _, pred, x = score_inputs(caches['calibration'], artifact['distance'])
    errors = (pred != caches['calibration']['labels']).astype(int)
    threshold = select_threshold(errors, estimator.predict_proba(x[:, :2])[:, 1],
                                 cfg['target_risk'], cfg['min_accepted'])
    artifact = dict(artifact, uncertainty_estimator=estimator,
                    thresholds=dict(artifact['thresholds'], **{'MSP+Entropy': threshold}))
    (out / 'predictions').mkdir(parents=True, exist_ok=True)
    joblib.dump(artifact, out / 'reliability.joblib')
    save_json(out / 'config.json', dict(cfg, source_run=str(source),
              ablation='C=1 standardized unweighted logistic: MSP+entropy versus all three; no test selection'))
    save_json(out / 'uncertainty_estimator.json', dict(
        feature_names=['MSP','entropy'], coefficients=estimator[-1].coef_[0].tolist(),
        feature_mean=estimator[0].mean_.tolist(), feature_scale=estimator[0].scale_.tolist(),
        fit_n=len(caches['fit']['labels']), threshold_validation_n=len(caches['calibration']['labels']),
        intercept=float(estimator[-1].intercept_[0]), validation_threshold=threshold,
        checkpoint_sha256=fingerprint))
    with np.load(source / 'predictions' / 'test.npz') as z:
        test_cache = {k: z[k] for k in z.files}
    if str(test_cache['provenance']) != fingerprint:
        raise ValueError('Test cache checkpoint differs')
    result = evaluate_cache(test_cache, artifact, cfg, out)
    save_json(out / 'comparison.json', {
        metric: result['methods']['Combined'][metric] - result['methods']['MSP+Entropy'][metric]
        for metric in ['auroc','aurc','risk_at_50','risk_at_70','risk_at_80','risk_at_90']})
    return result


if __name__ == '__main__':
    p = argparse.ArgumentParser()
    p.add_argument('--run-dir', required=True)
    p.add_argument('--output-dir')
    a = p.parse_args()
    print(json.dumps(run_ablation(a.run_dir, a.output_dir), indent=2))
