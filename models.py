"""
models.py
---------
Canonical implementations of regression helpers plus a convenient registry.
Each helper still returns a *callable predictor* to keep current code working,
but duplication across scripts is eliminated.
"""

from collections import OrderedDict
from typing import Callable

import numpy as np
from scipy.optimize import curve_fit
from statsmodels.nonparametric.smoothers_lowess import lowess


# ---------------------------------------------------------------------- #
#  Helper builders
# ---------------------------------------------------------------------- #
def poly_fit(degree: int) -> Callable[[np.ndarray, np.ndarray], Callable]:
    """Return <fitter> that fits a degree-`degree` polynomial."""
    def fit(x_arr, y_arr):
        coeffs = np.polyfit(x_arr, y_arr, degree)
        return lambda x_new: np.polyval(coeffs, x_new)
    return fit


def exp_fit(x: np.ndarray, y: np.ndarray) -> Callable:
    """Fit *c·exp(r·(x-x0)) + b* with naive but robust initialisation."""
    x0 = x[0]

    def _model(xs, c, r, b):
        return c * np.exp(r * (xs - x0)) + b

    b0 = float(np.min(y))
    c0 = float(y[0] - b0)
    if c0 > 0 and (y[-1] - b0) > 0:
        r0 = float(np.log((y[-1] - b0) / c0) / (x[-1] - x0))
    else:
        r0 = 1e-3

    p0 = (c0, r0, b0)
    try:
        params, _ = curve_fit(_model, x, y, p0=p0, maxfev=5000)
        return lambda xs: _model(xs, *params)
    except Exception:
        mean_y = float(np.mean(y))
        return lambda xs: np.full_like(xs, mean_y, dtype=float)


def loess_fit(x_arr: np.ndarray, y_arr: np.ndarray, *, frac: float = 0.3) -> Callable:
    """Locally weighted scatter-plot smoothing."""
    sm = lowess(y_arr, x_arr, frac=frac, return_sorted=True)
    xs, ys = sm[:, 0], sm[:, 1]
    return lambda xs_new: np.interp(xs_new, xs, ys)


# ---------------------------------------------------------------------- #
#  Public registry – **import this** instead of hand-rolling buttons
# ---------------------------------------------------------------------- #
REGISTRY: "OrderedDict[str, Callable[[np.ndarray, np.ndarray], Callable]]" = OrderedDict([
    ("Linear",     poly_fit(1)),
    ("Quadratic",  poly_fit(2)),
    ("Cubic",      poly_fit(3)),
    ("Quartic",    poly_fit(4)),
    ("Exponential", exp_fit),
    ("LOESS",      loess_fit),
])
