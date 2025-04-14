import numpy as np
from scipy.optimize import minimize


class DistrFamily:
    def __init__(self, param_names, expr_func, delete_unused_params=True):
        self._original_param_names = param_names
        self.expr_func = expr_func

        if delete_unused_params:
            self.param_names = self._detect_used_params()
        else:
            self.param_names = self._original_param_names

    def _detect_used_params(self):
        used = []

        class TrackingDict(dict):
            def __getitem__(self, key):
                used.append(key)
                return 1.0

        dummy = {k: 1.0 for k in self._original_param_names}
        try:
            self.expr_func(TrackingDict(dummy))
        except Exception:
            pass

        return list(sorted(set(used)))

    def instantiate(self, param_dict):
        return self.expr_func(param_dict)

    def __add__(self, other):
        return DistrFamily.merge(lambda x, y: x + y, self, other )

    def __mul__(self, other):
        return DistrFamily.merge(lambda x, y: x * y, self, other)

    def __truediv__(self, other):
        return DistrFamily.merge(lambda x, y: x * y, self, other)

    def trunc(self, a, b):
        return DistrFamily.merge(lambda x: x.trunc(a, b), self)

    def censor(self, a, b):
        return DistrFamily.merge(lambda x: x.censor(a, b), self)

    @staticmethod
    def merge(comb_func, *args):
        new_args = []
        for i in args:
            if not isinstance(i, DistrFamily):
                i = DistrFamily([], lambda _: i)
            new_args.append(i)

        param_names = list(set(item for sublist in new_args for item in sublist.param_names))

        def expr(params):
            distrs = [i.instantiate(params) for i in new_args]
            return comb_func(*distrs)

        return DistrFamily(param_names, expr)

    def refine(self, fixed_params):
        remaining = [p for p in self.param_names if p not in fixed_params]

        def new_expr(params):
            full_params = dict(fixed_params)
            full_params.update(params)
            return self.expr_func(full_params)

        if not remaining:
            return self.expr_func(fixed_params)

        return DistrFamily(remaining, new_expr, False)

    def estimate(self, samples, initial_guess, bounds=None, loss_fn=None):

        if loss_fn is None:
            loss_fn = self._default_neg_log_likelihood

        def obj_fn(param_array):
            param_dict = dict(zip(self.param_names, param_array))
            distr = self.instantiate(param_dict)
            return loss_fn(distr, samples)

        res = minimize(obj_fn, initial_guess, bounds=bounds)
        param_values = dict(zip(self.param_names, res.x))
        return param_values

    def _default_neg_log_likelihood(self, distr, data):
        pdf_vals = distr.pdf(data)
        pdf_vals[pdf_vals <= 1e-300] = 1e-300
        return -np.sum(np.log(pdf_vals))

    def __repr__(self):
        return f"<DistributionFamily({', '.join(self.param_names)})>"
