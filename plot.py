import matplotlib.pyplot as plt

def plot_errors(r_vals, error_dict):
    plt.figure(figsize=(10, 6))
    for label, error in error_dict.items():
        plt.plot(r_vals, error, label=label)
    plt.yscale('log')
    plt.xlabel(r'Axis Ratio $r = \frac{b}{a}$')
    plt.ylabel('Relative Error (%) (log scale)')
    plt.title('Relative Error Compared to 10000-digit Elliptic Integral')
    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    plt.show()

def plot_times(methods, times):
    plt.figure(figsize=(8, 5))
    plt.bar(methods, times)
    plt.yscale('log')
    plt.ylabel("Execution Time (seconds, log scale)")
    plt.title("Execution Time Comparison")
    plt.grid(True, axis='y', which='both', linestyle='--')
    plt.tight_layout()
    plt.show()
