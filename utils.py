import numpy as np
from math import pi

def theta_from_r(r): return float(pi) * (r - 0.2) / 0.8
def normalize_r(r): return 2 * (r - 0.2) / 0.8 - 1

def relative_error(P_approx, P_true):
    return 100 * np.abs(P_approx - P_true) / P_true

def compute_perimeter(k_func, r_vals, *args):
    return np.array([k_func(ri, *args) * np.pi * (1 + ri) for ri in r_vals])
