from pacal import ConstDistr, UniformDistr
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

    def test_censor_method(self):
        d = ConstDistr(5)
        censored = d.censor(3, 7)

        assert censored.pdf(2) == censored.pdf(3)
        assert censored.pdf(8) == censored.pdf(7)
