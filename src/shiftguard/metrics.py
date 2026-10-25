import numpy as np
from sklearn.metrics import roc_auc_score


def risk_coverage(errors, scores):
    errors, scores = np.asarray(errors), np.asarray(scores)
    if len(errors) == 0 or len(errors) != len(scores) or not np.isfinite(scores).all():
        raise ValueError('Need nonempty, equal-length arrays and finite scores')
    order = np.argsort(scores, kind='stable')  # lower error score accepted first
    # Expected risk under random ordering within ties; avoids label/order artifacts.
    sorted_scores, e = scores[order], errors[order].astype(float)
    starts = np.r_[0, np.flatnonzero(np.diff(sorted_scores)) + 1]
    ends = np.r_[starts[1:], len(e)]
    for a, b in zip(starts, ends):
        e[a:b] = e[a:b].mean()
    k = np.arange(1, len(e) + 1)
    return k / len(e), np.cumsum(e) / k


def select_threshold(errors, scores, target=0.1, min_accepted=100):
    # Entire ties accepted; minimum sample count. Empirical constraint, no guarantee.
    order = np.argsort(scores, kind='stable')
    s, e = np.asarray(scores)[order], np.asarray(errors)[order]
    ends = np.r_[np.flatnonzero(np.diff(s)) + 1, len(s)]
    risks = np.cumsum(e)[ends - 1] / ends
    valid = ends[(risks <= target) & (ends >= min_accepted)]
    return float(s[valid[-1] - 1]) if len(valid) else None


def detection_metrics(errors, scores, target=0.1, min_accepted=100, threshold=None):
    coverage, risk = risk_coverage(errors, scores)
    n = len(errors)
    result = {'auroc': float(roc_auc_score(errors, scores)) if len(np.unique(errors)) == 2 else None,
              'aurc': float(risk.mean())}
    for c in (0.5, 0.7, 0.8, 0.9):
        k = max(1, int(np.ceil(c * n)))
        result[f'risk_at_{int(c*100)}'] = float(risk[k - 1])
    # Descriptive test curve only: never a deployable threshold selected on test labels.
    good = np.flatnonzero((risk <= target) & (np.arange(1, n + 1) >= min_accepted))
    result['oracle_coverage_at_target_risk'] = float(coverage[good[-1]]) if len(good) else None
    accepted = np.zeros(n, dtype=bool) if threshold is None else scores <= threshold
    result['validation_threshold'] = threshold
    result['threshold_coverage'] = float(accepted.mean())
    result['threshold_risk'] = float(np.mean(errors[accepted])) if accepted.any() else None
    result['threshold_accepted'] = int(accepted.sum())
    return result


def calibration_bins(probabilities, outcomes, bins=15):
    p, y = np.asarray(probabilities), np.asarray(outcomes)
    ids = np.minimum((np.clip(p, 0, 1) * bins).astype(int), bins - 1)
    rows = []
    for b in range(bins):
        mask = ids == b
        if mask.any():
            rows.append(dict(bin=b, count=int(mask.sum()), mean_probability=float(p[mask].mean()),
                             outcome_rate=float(y[mask].mean())))
    return rows


def calibration_metrics(probabilities, outcomes, bins=15):
    rows = calibration_bins(probabilities, outcomes, bins)
    return {'brier': float(np.mean((probabilities - outcomes) ** 2)),
            'ece': float(sum(r['count'] * abs(r['mean_probability'] - r['outcome_rate'])
                             for r in rows) / len(outcomes))}


def classifier_metrics(p, labels):
    pred = p.argmax(axis=1)
    return {'accuracy': float(np.mean(pred == labels)),
            'nll': float(-np.log(np.clip(p[np.arange(len(labels)), labels], 1e-15, 1)).mean()),
            'brier': float(np.mean(np.sum((p - np.eye(p.shape[1])[labels]) ** 2, axis=1))),
            'ece': calibration_metrics(p.max(axis=1), (pred == labels).astype(float))['ece'],
            'correct': int((pred == labels).sum()), 'incorrect': int((pred != labels).sum()),
            'n': len(labels)}
