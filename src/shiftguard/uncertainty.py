import numpy as np


def softmax(logits):
    shifted = np.asarray(logits, dtype=np.float64) - np.max(logits, axis=1, keepdims=True)
    exp = np.exp(shifted)
    return exp / exp.sum(axis=1, keepdims=True)


def uncertainty(logits):
    probabilities = softmax(logits)
    msp = probabilities.max(axis=1)
    entropy = -(probabilities * np.log(np.clip(probabilities, 1e-15, 1))).sum(axis=1)
    return probabilities, msp, entropy
