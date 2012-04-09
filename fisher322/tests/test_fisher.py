""" Testing standalone Fisher's F implementation
"""

from os.path import dirname, join as pjoin

import numpy as np

from ..fisher import f_sf, f_cdf

from numpy.testing import (assert_array_almost_equal,
                           assert_array_equal, dec)

from nose.tools import assert_true, assert_equal, assert_raises

try:
    from scipy.stats import f
except ImportError:
    have_scipy = False
else:
    have_scipy = True


TEST_DATA_FNAME = pjoin(dirname(__file__), 'fcdf_runs.npz')


def make_test_data():
    # Make test data for stored data test
    # Run with:
    # import fisher322.tests.test_fisher as tf
    # tf.make_test_data()
    if not have_scipy:
        raise RuntimeError("Need scipy to store data")
    N = 10
    mN = 15
    nN = 15
    x = np.random.normal(size=(N,)) ** 2
    fcdf_data = np.zeros((mN, nN, N))
    for m in range(15):
        for n in range(15):
            fcdf_data[m, n][:] = f.cdf(x, m, n)
    np.savez(TEST_DATA_FNAME, fcdf_data=fcdf_data, x=x)


def test_corners():
    # Corner cases
    assert_array_almost_equal(f_cdf(0, 1, 1), 0)
    assert_array_almost_equal(f_sf(0, 1, 1), 1)
    assert_array_almost_equal(f_cdf([-1, 0], 1, 1), [0, 0])
    assert_array_almost_equal(f_sf([-1, 0], 1, 1), [1, 1])
    assert_array_almost_equal(f_cdf(np.inf, 1, 1), 1)


def test_stored_scipy():
    # Test against stored scipy results
    TEST_DATA = np.load(TEST_DATA_FNAME)
    fcdf_data = TEST_DATA['fcdf_data']
    x = TEST_DATA['x']
    mN, nN, N = fcdf_data.shape
    for m in range(15):
        for n in range(15):
            assert_array_almost_equal(f_cdf(x, m, n), fcdf_data[m, n])


@dec.skipif(not have_scipy)
def test_scipy_f():
    rng = np.random.RandomState(20120407)
    x = rng.normal(size=(100)) * 4
    for m in np.arange(1, 15):
        for n in np.arange(1, 15):
            assert_array_almost_equal(f_sf(x, m, n), f.sf(x, m, n))
            assert_array_almost_equal(f_cdf(x, m, n), f.cdf(x, m, n))
