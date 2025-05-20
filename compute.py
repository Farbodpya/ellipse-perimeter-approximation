from mpmath import mp, mpf, sqrt, pi, ellipe
from tqdm import tqdm
import numpy as np

def k_true_mp(r, dps=15, desc="Computing k(r)"):
    mp.dps = dps
    results = []
    for ri in tqdm(r, desc=desc):
        a, b = mpf(1), mpf(ri)
        if b > a: a, b = b, a
        e = sqrt(1 - (b / a) ** 2)
        k = 4 * a * ellipe(e ** 2) / (pi * (a + b))
        results.append(float(k))
    return np.array(results)
