import numpy as np
import pytest
from shiftguard.metrics import (risk_coverage, detection_metrics, classifier_metrics,
                               calibration_metrics, select_threshold)


def test_perfect_order_and_direction():
    errors = np.array([1, 0, 1, 0])
    scores = np.array([.9, .1, .8, .2])
    c, r = risk_coverage(errors, scores)
    np.testing.assert_allclose(r, [0, 0, 1/3, 1/2])
    m = detection_metrics(errors, scores, min_accepted=1)
    assert m['auroc'] == 1
    assert m['aurc'] == pytest.approx(5/24)
    assert m['risk_at_50'] == 0
    assert detection_metrics(errors, -scores)['auroc'] == 0


def test_ties_are_order_invariant():
    _, r = risk_coverage(np.array([1, 0, 0, 1]), np.ones(4))
    np.testing.assert_allclose(r, .5)


def test_calibration_and_single_class():
    p = np.eye(2)[[0, 1]]
    m = classifier_metrics(p, np.array([0, 1]))
    assert m['accuracy'] == 1 and m['brier'] == 0 and m['ece'] == 0
    assert detection_metrics(np.zeros(4), np.arange(4))['auroc'] is None
    assert calibration_metrics(np.array([0., 1.]), np.array([0., 1.]))['ece'] == 0


def test_thresholds_use_whole_ties_and_minimum():
    errors = np.array([0, 0, 1, 1])
    scores = np.array([0., 0., 1., 1.])
    assert select_threshold(errors, scores, .1, 2) == 0
    assert select_threshold(errors, scores, .1, 3) is None
    assert detection_metrics(errors, scores, threshold=0)['threshold_coverage'] == .5


def test_empty_rejected():
    with pytest.raises(ValueError):
        risk_coverage([], [])
