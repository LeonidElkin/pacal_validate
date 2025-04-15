import numpy as np

from pacal import ConstDistr, UniformDistr, NormalDistr
from pacal.rv import CensoredRV, RV


class TestCensor(object):
    def test_censored_distr(self):
        d = UniformDistr(0, 10)

        censored = d.censor(2, 8)

        assert censored.range() == (0, 10)
        assert censored.pdf(1.8) == censored.pdf(1.9)
        assert censored.pdf(8.1) == censored.pdf(8.2)

    def test_censored_rv(self):
        base_rv = RV(sym="X", a=0, b=10)
        censored_rv = CensoredRV(base_rv, 4, 6)

        assert censored_rv.getSegmentsCensoredTo() == (4, 6)
        assert censored_rv.censor_a == 4
        assert censored_rv.censor_b == 6

    def test_censor_pdf(self):
        d = NormalDistr(0, 10)
        censored = d.censor(2, 8)

        xs_inside = np.linspace(2, 8, 100)
        xs_outside = np.array([1.0, 1.9, 8.1, 10.0])

        pdf_original = d.pdf(xs_inside)
        pdf_censored = censored.pdf(xs_inside)
        assert np.allclose(pdf_original, pdf_censored, rtol=1e-5)

        pdf_outside = censored.pdf(xs_outside)
        assert np.allclose(pdf_outside, 0)

    def test_censor_cdf(self):
        d = NormalDistr(0, 10)
        censored = d.censor(2, 8)

        xs_inside = np.linspace(2.1, 7.9, 100)
        xs_left = np.array([1.0, 1.9])
        xs_right = np.array([8.1, 10.0])

        cdf_original = d.cdf(xs_inside)
        cdf_censored = censored.cdf(xs_inside)
        assert np.allclose(cdf_original, cdf_censored, rtol=1e-5)

        cdf_left = censored.cdf(xs_left)
        assert np.allclose(cdf_left, 0)

        cdf_right = censored.cdf(xs_right)
        assert np.allclose(cdf_right, 1)

    def test_censored_sampling(self):
        d = UniformDistr(0, 10)
        censored = d.censor(2, 8)
        samples = censored.rand(10000)

        assert np.all(samples >= 0)
        assert np.all(samples <= 10)
        assert np.any(samples == 2)
        assert np.any(samples == 8)
