import numpy as np
from shiftguard.data import stratified_split, partition_validation, take_stratified


def test_split_integrity_and_reproducibility():
    labels = np.repeat(np.arange(10), 5000)
    train, val = stratified_split(labels)
    train2, val2 = stratified_split(labels)
    assert len(train) == 45000 and len(val) == 5000
    np.testing.assert_array_equal(train, train2)
    np.testing.assert_array_equal(val, val2)
    assert not set(train) & set(val)
    np.testing.assert_array_equal(np.bincount(labels[val]), [500]*10)
    parts = partition_validation(val, labels, 42)
    assert [len(parts[k]) for k in parts] == [2000,1000,1000,1000]
    sets = [set(train)] + [set(p) for p in parts.values()]
    for i, left in enumerate(sets):
        for right in sets[i+1:]:
            assert not left & right
    assert set().union(*sets) == set(range(50000))
    small = take_stratified(train, labels, 1000, 42)
    np.testing.assert_array_equal(np.bincount(labels[small]), [100]*10)
