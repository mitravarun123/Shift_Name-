"""Finite-grid, simultaneous binomial bounds for selective prediction.

Validity requires IID calibration/deployment examples and rules fixed independently
of calibration outcomes. This is an established Bonferroni/Clopper-Pearson
construction, not a new theorem. Conditional accepted risk is not monotone in the
threshold; simultaneous bounds allow searching the entire fixed finite grid.
"""
import numpy as np
from scipy.stats import beta


def upper_binomial(errors, accepted, delta):
    if not 0 < delta < 1 or not 0 <= errors <= accepted:
        raise ValueError('Invalid count or confidence level')
    if accepted == 0 or errors == accepted:
        return 1.0
    return float(beta.isf(delta, errors + 1, accepted - errors))


def candidate_grids(fit_scores, grid_size=21):
    if grid_size < 2:
        raise ValueError('Need at least two prespecified quantiles')
    grids = {}
    for name, scores in fit_scores.items():
        scores = np.asarray(scores)
        if not len(scores) or not np.isfinite(scores).all():
            raise ValueError('Candidate scores must be finite and nonempty')
        grids[name] = np.unique(np.quantile(scores, np.linspace(0, 1, grid_size))).tolist()
    return grids


def certify(calibration_scores, errors, grids, budgets, targets=(.05, .1), delta=.05, min_accepted=20):
    errors = np.asarray(errors)
    if not len(errors) or not np.isin(errors, [0, 1]).all():
        raise ValueError('Errors must be a nonempty binary array')
    if set(calibration_scores) != set(grids):
        raise ValueError('Scores and independent candidate grids must match')
    budgets = sorted(set(int(n) for n in budgets))
    if not budgets or min(budgets) < 1 or max(budgets) > len(errors):
        raise ValueError('Calibration budgets exceed available independent observations')
    if not 0 < delta < 1 or min_accepted < 1 or any(not 0 < t < 1 for t in targets):
        raise ValueError('Invalid targets, confidence, or minimum acceptance')
    for name, scores in calibration_scores.items():
        if len(scores) != len(errors) or not np.isfinite(scores).all():
            raise ValueError(f'Invalid calibration score array: {name}')
        if not len(grids[name]) or not np.isfinite(grids[name]).all():
            raise ValueError('Candidate thresholds must be finite and fixed before calibration')
    # The same upper-bound event covers all risk targets; targets add no tests.
    family_size = len(budgets) * sum(len(g) for g in grids.values())
    per_rule_delta = delta / family_size
    bounds, selections = [], []
    for budget in budgets:
        for name, grid in grids.items():
            candidates = []
            for threshold in grid:
                mask = np.asarray(calibration_scores[name])[:budget] <= threshold
                accepted = int(mask.sum())
                mistakes = int(errors[:budget][mask].sum())
                row = dict(method=name, calibration_n=budget, threshold=float(threshold),
                           accepted=accepted, errors=mistakes,
                           empirical_risk=mistakes / accepted if accepted else None,
                           coverage=accepted / budget,
                           upper_risk=upper_binomial(mistakes, accepted, per_rule_delta))
                candidates.append(row)
                bounds.append(row)
            for target in targets:
                feasible = [r for r in candidates if r['accepted'] >= min_accepted and r['upper_risk'] <= target]
                chosen = max(feasible, key=lambda r: (r['accepted'], -r['upper_risk'], -r['threshold'])) if feasible else None
                selections.append(dict(method=name, calibration_n=budget, target_risk=float(target),
                    status='certified' if chosen else 'not_certified',
                    threshold=chosen['threshold'] if chosen else None,
                    calibration_accepted=chosen['accepted'] if chosen else 0,
                    calibration_errors=chosen['errors'] if chosen else 0,
                    calibration_risk=chosen['empirical_risk'] if chosen else None,
                    calibration_upper_risk=chosen['upper_risk'] if chosen else None))
    return dict(delta=delta, simultaneous_confidence=1-delta, family_size=family_size,
                per_rule_delta=per_rule_delta, targets=list(targets), budgets=budgets,
                selections=selections, candidate_bounds=bounds,
                assumption='IID independent calibration and future clean examples from the same population; fixed pre-calibration model, scores, and candidate grids',
                limitation='CIFAR is a fixed benchmark; the IID population model is an assumption, not empirically verified. No guarantee under arbitrary shift. No certified rule means reject all; its conditional risk is undefined.')


def evaluate_certificates(certificate, test_scores, errors):
    errors = np.asarray(errors)
    rows = []
    for selection in certificate['selections']:
        row = dict(selection)
        threshold = row['threshold']
        mask = np.zeros(len(errors), dtype=bool) if threshold is None else np.asarray(test_scores[row['method']]) <= threshold
        accepted = int(mask.sum())
        mistakes = int(errors[mask].sum())
        row.update(test_n=len(errors), test_accepted=accepted, test_errors=mistakes,
                   test_coverage=accepted / len(errors), test_risk=mistakes / accepted if accepted else None)
        rows.append(row)
    return rows
