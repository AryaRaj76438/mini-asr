from unittest import result

import numpy as np
from mini_asr.probability.numerical import (logsumexp, safe_log)

def test_safe_log():
    values = np.array([1.0, 0.5, 0.0])
    result = safe_log(values)
    assert np.isfinite(result).all()

def test_logsumexp():
    values = np.array([1.0, 2.0, 3.0])
    expected = np.log(np.sum(np.exp(values)))
    result = logsumexp(values)
    assert np.allclose(expected, result)

def test_logsumexp_large_negative_values():
    values = np.array([-1000,-1001, -1002])
    result = logsumexp(values)
    assert np.isfinite(result)