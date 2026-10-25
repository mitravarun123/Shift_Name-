import numpy as np
import pytest
import torch
from torch.utils.data import DataLoader, TensorDataset
from shiftguard.models import SmallCNN
from shiftguard.features import extract
from shiftguard.uncertainty import uncertainty
from shiftguard.reliability import CentroidDistance, fit_estimator, all_scores


def test_stable_uncertainty():
    p, msp, entropy = uncertainty(np.array([[1000.,-1000.], [0.,0.]]))
    assert np.isfinite(entropy).all()
    np.testing.assert_allclose(msp, [1, .5])
    np.testing.assert_allclose(entropy, [0, np.log(2)], atol=1e-12)


def test_extraction_and_cache(tmp_path):
    model = SmallCNN(4, 8).eval()
    x = torch.randn(6, 3, 32, 32)
    loader = DataLoader(TensorDataset(x, torch.arange(6)), batch_size=2)
    cache = extract(model, loader, 'cpu', tmp_path/'p.npz', np.arange(6), 'test')
    assert cache['features'].shape == (6, 8)
    assert cache['logits'].shape == (6, 10)
    with torch.inference_mode():
        logits, features = model(x)
    np.testing.assert_allclose(cache['features'], features.numpy(), atol=1e-6)
    saved = extract(model, loader, 'cpu', tmp_path/'p.npz', np.arange(6), 'test')
    np.testing.assert_array_equal(cache['features'], saved['features'])
    with pytest.raises(ValueError):
        extract(model, loader, 'cpu', tmp_path/'p.npz', np.arange(6), 'different')


def test_reference_uses_only_correct_and_predicted_class():
    f = np.array([[0.,0.], [2.,0.], [100.,100.]])
    pred = np.array([0,0,1])
    labels = np.array([0,0,0])
    d = CentroidDistance().fit(f, pred, labels)
    np.testing.assert_allclose(d.mean, [1,0])
    assert d.counts.tolist() == [2,0,0,0,0,0,0,0,0,0]
    assert 1 in d.fallback_classes
    assert d.score(np.array([[1.,0.]]), np.array([0]))[0] == 0


def test_combined_direction():
    x = np.array([[.99,.01,.1],[.9,.1,.2],[.4,1.3,3],[.3,1.5,4]])
    estimator = fit_estimator(x, np.array([0,0,1,1]), 42)
    scores = all_scores(x, estimator)
    assert scores['MSP'][0] < scores['MSP'][3]
    assert scores['Combined'][0] < scores['Combined'][3]
    with pytest.raises(ValueError):
        fit_estimator(x, np.zeros(4), 42)


def test_scoring_does_not_consume_evaluation_labels():
    from shiftguard.reliability import score_inputs
    reference = np.array([[0.,0.], [1.,0.], [3.,1.], [4.,1.]])
    pred = np.array([0,0,1,1])
    distance = CentroidDistance().fit(reference, pred, pred.copy())
    logits = np.zeros((2,10))
    logits[0,0] = 3
    logits[1,1] = 3
    cache = dict(logits=logits, features=np.array([[.5,0.],[3.5,1.]]))
    # No label key needed; distances are determined solely by predicted classes.
    _, predictions, x = score_inputs(cache, distance)
    np.testing.assert_array_equal(predictions, [0,1])
    np.testing.assert_allclose(x[:,2], 0)
    cache['labels'] = np.array([9,9])
    _, _, changed = score_inputs(cache, distance)
    np.testing.assert_array_equal(x, changed)


def test_uncertainty_only_ablation_receives_two_columns():
    x = np.array([[.99,.01,.1],[.9,.1,.2],[.4,1.3,3],[.3,1.5,4]])
    errors = np.array([0,0,1,1])
    full = fit_estimator(x, errors, 42)
    two = fit_estimator(x[:,:2], errors, 42)
    baseline = all_scores(x, full)
    ablation = all_scores(x, full, two)
    assert set(ablation) == set(baseline) | {'MSP+Entropy'}
    for method in baseline:
        np.testing.assert_array_equal(baseline[method], ablation[method])
    np.testing.assert_array_equal(ablation['MSP+Entropy'], two.predict_proba(x[:,:2])[:,1])
