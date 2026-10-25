import time
from pathlib import Path

import joblib
import numpy as np

from .features import extract
from .metrics import calibration_metrics, classifier_metrics, detection_metrics, select_threshold
from .plotting import plot_saved
from .reliability import CentroidDistance, all_scores, fit_estimator, score_inputs
from .utils import save_json


def fit_reliability(caches, cfg, output):
    ref = caches['reference']
    distance = CentroidDistance().fit(ref['features'], ref['logits'].argmax(1), ref['labels'])
    _, pred, x = score_inputs(caches['fit'], distance)
    estimator = fit_estimator(x, (pred != caches['fit']['labels']).astype(int), cfg['seed'])
    _, pred, x = score_inputs(caches['calibration'], distance)
    errors = (pred != caches['calibration']['labels']).astype(int)
    scores = all_scores(x, estimator)
    thresholds = {name: select_threshold(errors, s, cfg['target_risk'], cfg['min_accepted'])
                  for name, s in scores.items()}
    artifact = dict(distance=distance, estimator=estimator, thresholds=thresholds,
                    provenance=str(ref['provenance']))
    joblib.dump(artifact, Path(output) / 'reliability.joblib')
    scaler, logistic = estimator.steps[0][1], estimator.steps[1][1]
    save_json(Path(output) / 'reliability.json', dict(
        centroid_counts=distance.counts.tolist(), fallback_classes=distance.fallback_classes,
        feature_names=['MSP', 'entropy', 'distance'], feature_mean=scaler.mean_.tolist(),
        feature_scale=scaler.scale_.tolist(), standardized_coefficients=logistic.coef_[0].tolist(),
        intercept=float(logistic.intercept_[0]), thresholds=thresholds,
        calibration_n=len(errors), fit_n=len(caches['fit']['labels']),
        calibration_metrics=calibration_metrics(scores['Combined'], errors)))
    return artifact


def evaluate_cache(cache, artifact, cfg, output, experiment='clean'):
    start = time.perf_counter()
    p, pred, x = score_inputs(cache, artifact['distance'])
    scores = all_scores(x, artifact['estimator'], artifact.get('uncertainty_estimator'))
    score_seconds = time.perf_counter() - start
    errors = (pred != cache['labels']).astype(int)
    methods = {name: detection_metrics(errors, s, cfg['target_risk'], cfg['min_accepted'],
                                       artifact['thresholds'][name]) for name, s in scores.items()}
    result = dict(experiment=experiment, classifier=classifier_metrics(p, cache['labels']),
                  methods=methods, error_probability=calibration_metrics(scores['Combined'], errors),
                  timing=dict(forward_ms_per_image=float(cache['forward_seconds']) / len(errors) * 1000,
                              extraction_seconds=float(cache['extraction_seconds']),
                              all_scores_ms_per_image=score_seconds / len(errors) * 1000),
                  preliminary=True, smoke=cfg['test_limit'] is not None,
                  architecture=cfg.get('architecture', 'small_cnn'),
                  scope=cfg.get('scope', 'smoke' if cfg['test_limit'] is not None else 'full'),
                  training_examples=cfg['train_limit'] or (50000 - cfg['val_size']),
                  target_risk=cfg['target_risk'], min_accepted=cfg['min_accepted'])
    if 'MSP+Entropy' in scores:
        result['error_probability_ablations'] = {name: calibration_metrics(scores[name], errors)
            for name in ['MSP+Entropy', 'Combined']}
    save_json(Path(output) / 'metrics' / f'{experiment}.json', result)
    np.savez_compressed(Path(output) / 'predictions' / f'{experiment}_scores.npz',
        labels=cache['labels'], predictions=pred, errors=errors, probabilities=p,
        indices=cache['indices'], inputs=x, **scores)
    plot_saved(output, experiment)
    return result
