import numpy as np

from pacal import NormalDistr, MixDistr
from pacal.distr import TruncDistr, CensoredDistr, SumDistr
from pacal.distr_family import DistrFamily


class TestDistrFamily(object):
    def test_basic_instantiation(self):
        P = DistrFamily(['sigma'], lambda p: NormalDistr(0, p['sigma']))
        d = P.instantiate({'sigma': 2.0})
        assert isinstance(d, NormalDistr)
        assert d.sigma == 2.0

    def test_operator_addition(self):
        P1 = DistrFamily(['a'], lambda p: NormalDistr(p['a'], 1.0))
        P2 = DistrFamily(['b'], lambda p: NormalDistr(0, p['b']))
        Psum = P1 + P2
        d = Psum.instantiate({'a': 1.0, 'b': 2.0})
        assert hasattr(d, 'pdf')
        assert isinstance(d, SumDistr)

    def test_refine_and_collapse(self):
        P = DistrFamily(['mu', 'sigma'], lambda p: NormalDistr(p['mu'], p['sigma']))
        P_partial = P.refine({'mu': 1.0})
        assert isinstance(P_partial, DistrFamily)
        assert P_partial.param_names == ['sigma']

        final = P_partial.refine({'sigma': 2.0})
        assert isinstance(final, NormalDistr)
        assert final.mu == 1.0 and final.sigma == 2.0

    def test_params_used_trimming(self):
        P = DistrFamily(['mu', 'sigma', 'unused'], lambda p: NormalDistr(p['mu'], p['sigma']))
        assert set(P.param_names) == {'mu', 'sigma'}

    def test_estimate_normal_sigma(self):
        data = np.random.normal(0, 1.5, 500)
        P = DistrFamily(['sigma'], lambda p: NormalDistr(0, p['sigma']))
        result = P.estimate(data, initial_guess=[1.0], bounds=[(0.1, 5)])
        assert abs(result['sigma'] - 1.5) < 0.2

    def test_mix_DistrFamily(self):
        P = DistrFamily(['s1', 's2', 'w'], lambda p: MixDistr(
            [p['w'], 1 - p['w']],
            [NormalDistr(0, p['s1']), NormalDistr(0, p['s2'])]
        ))
        d = P.instantiate({'s1': 1.0, 's2': 2.0, 'w': 0.3})
        assert hasattr(d, 'pdf') and hasattr(d, 'rand')

    def test_trunc(self):
        P = DistrFamily(['sigma'], lambda p: NormalDistr(0, p['sigma']))
        P_trunc = P.trunc(-3.0, 3.0)
        d = P_trunc.instantiate({'sigma': 2.0})
        assert isinstance(d, TruncDistr)
        assert d.range() == (-3.0, 3.0)

    def test_censor(self):
        P = DistrFamily(['sigma'], lambda p: NormalDistr(0, p['sigma']))
        P_censor = P.censor(-3.0, 3.0)
        d = P_censor.instantiate({'sigma': 2.0})
        assert isinstance(d, CensoredDistr)
        assert d.range() == NormalDistr(0, 2).range()
