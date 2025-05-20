import numpy as np
from scipy.optimize import curve_fit
from numpy.polynomial.chebyshev import Chebyshev
from utils import theta_from_r, normalize_r

def fourier_series(theta, *a):
    N = (len(a) - 1) // 2
    result = a[0]
    for n in range(1, N + 1):
        result += a[n] * np.cos(n * theta) + a[n + N] * np.sin(n * theta)
    return result

def fit_fourier(r_vals, k_ref, degree=10):
    theta_vals = np.array([theta_from_r(r) for r in r_vals])
    p0 = np.zeros(2 * degree + 1)
    params, _ = curve_fit(fourier_series, theta_vals, k_ref, p0=p0)
    return params

def k_fourier(r, params):
    return fourier_series(theta_from_r(r), *params)

def fit_chebyshev(r_vals, k_ref, degree=30):
    return Chebyshev.fit(normalize_r(r_vals), k_ref, deg=degree)

def k_chebyshev(r, cheb_fit):
    return cheb_fit(normalize_r(r))
