import numpy as np
import matplotlib.pyplot as plt

from plot_config import setup_plot, save_figure


def henon_map(x, y, a=1.4, b=0.3):
    return 1 - a * x**2 + y, b * x


def compute_orbit(x0, y0, a, b, n_steps, idle):
    x, y = x0, y0
    xs = np.empty(n_steps)
    ys = np.empty(n_steps)
    for i in range(n_steps):
        x, y = henon_map(x, y, a, b)
        xs[i] = x
        ys[i] = y
    return xs[idle:], ys[idle:]


def main():
    setup_plot()

    a_values = [1.0, 1.2, 1.4]
    b = 0.3
    n_steps = 5000
    idle = 500

    fig, axes = plt.subplots(1, 3, figsize=(12, 4))
    for i, a in enumerate(a_values):
        x_data, y_data = compute_orbit(0.1, 0.1, a, b, n_steps, idle)
        axes[i].scatter(x_data, y_data, s=0.3, c="black")
        axes[i].set_xlabel(r"$x_n$")
        axes[i].set_ylabel(r"$y_n$")
        axes[i].set_title(f"$a = {a}$")
        axes[i].grid(True, alpha=0.3)

    plt.tight_layout()
    save_figure(fig, "henon_parameter_scan")
    plt.close(fig)


if __name__ == "__main__":
    main()
