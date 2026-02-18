import numpy as np
from ptemcee_bilby.utils import get_max_gradient


def test_get_max_gradient_handles_non_finite_input():
    x = np.array([[1.0, 2.0], [np.inf, 3.0], [4.0, 5.0]])
    assert np.isinf(get_max_gradient(x, axis=0, window_length=3))


def test_get_max_gradient_handles_short_input():
    x = np.array([[1.0, 2.0], [3.0, 4.0]])
    assert np.isinf(get_max_gradient(x, axis=0, window_length=3))
