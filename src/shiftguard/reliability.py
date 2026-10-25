import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from .uncertainty import uncertainty


class CentroidDistance:
    """Global feature standardization, class centroids, RMS Euclidean distance."""
    def fit(self, features, predictions, labels):
        correct = predictions == labels
        reference = np.asarray(features, dtype=np.float64)[correct]
        if len(reference) == 0:
            raise ValueError('No correct reference examples; increase training/reference sample size')
        self.mean = reference.mean(axis=0)
        self.scale = np.maximum(reference.std(axis=0), 1e-3)
        z = (reference - self.mean) / self.scale
        self.counts = np.bincount(predictions[correct], minlength=10)
        self.centroids = np.stack([z[predictions[correct] == c].mean(axis=0)
            if self.counts[c] >= 2 else z.mean(axis=0) for c in range(10)])
        self.fallback_classes = np.flatnonzero(self.counts < 2).tolist()
        return self

    def score(self, features, predictions):
        z = (features - self.mean) / self.scale
        return np.sqrt(np.mean((z - self.centroids[predictions]) ** 2, axis=1))


def score_inputs(cache, distance):
    p, msp, entropy = uncertainty(cache['logits'])
    pred = p.argmax(axis=1)
    d = distance.score(cache['features'], pred)
    return p, pred, np.column_stack([msp, entropy, d])


def fit_estimator(x, errors, seed):
    if len(np.unique(errors)) != 2:
        raise ValueError('Estimator fit needs both errors and correct predictions; increase sample size')
    estimator = make_pipeline(StandardScaler(), LogisticRegression(C=1.0, max_iter=1000,
                                                                    random_state=seed))
    estimator.fit(x, errors)
    return estimator


def all_scores(x, estimator, uncertainty_estimator=None):
    scores = {'MSP': 1 - x[:, 0], 'Entropy': x[:, 1], 'Distance': x[:, 2],
            'Combined': estimator.predict_proba(x)[:, 1]}

    if uncertainty_estimator is not None:
        scores['MSP+Entropy'] = uncertainty_estimator.predict_proba(x[:, :2])[:, 1]
    return scores
