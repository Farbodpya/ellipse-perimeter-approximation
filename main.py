import numpy as np
import time
from compute import k_true_mp
from models import fit_fourier, k_fourier, fit_chebyshev, k_chebyshev
from utils import compute_perimeter, relative_error
from ramanujan import ramanujan_perimeter
from plot import plot_errors, plot_times

def main():
    r_vals = np.linspace(0.2, 1.0, 200)
    a_vals = np.ones_like(r_vals)
    b_vals = r_vals

    # Reference calculation
    k_ref = k_true_mp(r_vals, dps=10000, desc="k(r) with 10000 digits")
    k_15 = k_true_mp(r_vals, dps=13, desc="k(r) with 15 digits")

    # Fit Fourier
    start = time.time()
    params_fourier = fit_fourier(r_vals, k_ref)
    time_fourier = time.time() - start

    # Fit Chebyshev
    start = time.time()
    cheb_fit = fit_chebyshev(r_vals, k_ref)
    time_cheb = time.time() - start

    # Ramanujan
    start = time.time()
    P_ramanujan = ramanujan_perimeter(a_vals, b_vals)
    time_ramanujan = time.time() - start

    # Evaluate all
    P_ref = k_ref * np.pi * (1 + r_vals)
    P_15 = k_15 * np.pi * (1 + r_vals)
    P_fourier = compute_perimeter(k_fourier, r_vals, params_fourier)
    P_cheb = compute_perimeter(k_chebyshev, r_vals, cheb_fit)

    # Errors
    error_dict = {
        "Elliptic (15-digit)": relative_error(P_15, P_ref),
        "Fourier": relative_error(P_fourier, P_ref),
        "Chebyshev": relative_error(P_cheb, P_ref),
        "Ramanujan": relative_error(P_ramanujan, P_ref)
    }

    plot_errors(r_vals, error_dict)
    plot_times(["Fourier", "Chebyshev", "Ramanujan"],
               [time_fourier, time_cheb, time_ramanujan])

    # Print summary
    for name, err in error_dict.items():
        print(f"{name:16}: max = {np.max(err):.2e}%, mean = {np.mean(err):.2e}%")

if __name__ == "__main__":
    main()
