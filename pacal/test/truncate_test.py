import numpy as np
import pytest

from pacal import UniformDistr, ConstDistr
from pacal.rv import TruncRV, RV


class TestTruncate(object):

    def test_trunc_distr(self):
        d = UniformDistr(0, 10)
        truncated = d.trunc(2, 8)

        assert truncated.range() == (2, 8)
        assert truncated.pdf(1.9) == 0
        assert truncated.pdf(8.1) == 0
        assert truncated.pdf(5) == pytest.approx(1 / 6, abs=1e-5)

    def test_trunc_rv(self):
        base_rv = RV(sym="X", a=0, b=10)
        trunc_rv = TruncRV(base_rv, 3, 7)

        assert trunc_rv.getSegments() == (3, 7)
        assert trunc_rv.a == 3
        assert trunc_rv.b == 7

    def test_truncate_method(self):
        d = ConstDistr(5)
        truncated = d.trunc(3, 7)

        assert truncated.pdf(2) == 0
        assert truncated.pdf(8) == 0
        assert truncated.pdf(5) != 0

    def test_truncated_sampling(self):
        d = UniformDistr(0, 10)
        truncated = d.trunc(2, 8)
        samples = truncated.rand(10000)

        assert np.all(samples >= 2)
        assert np.all(samples <= 8)
