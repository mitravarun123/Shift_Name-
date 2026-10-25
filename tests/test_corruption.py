import numpy as np
import pytest
from PIL import Image
from shiftguard.corruption import Corrupt


def test_noise_reproducible_and_bounded():
    image = Image.fromarray(np.full((32,32,3), 128, dtype=np.uint8))
    a = np.asarray(Corrupt('noise', 2, 42)(image))
    b = np.asarray(Corrupt('noise', 2, 42)(image))
    c = np.asarray(Corrupt('noise', 2, 43)(image))
    np.testing.assert_array_equal(a, b)
    assert not np.array_equal(a, c)
    assert a.shape == (32,32,3) and a.dtype == np.uint8
    with pytest.raises(ValueError):
        Corrupt('noise', 4, 42)
