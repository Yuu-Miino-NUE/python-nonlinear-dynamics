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

    a, b = 1.4, 0.3
    n_steps = 10000
    idle = 1000

    x_data, y_data = compute_orbit(0.1, 0.1, a, b, n_steps, idle)

    fig, ax = plt.subplots(figsize=(5, 5))
    ax.scatter(x_data, y_data, s=0.5, c="black")
    ax.set_xlabel(r"$x_n$")
    ax.set_ylabel(r"$y_n$")
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    save_figure(fig, "henon_attractor")
    plt.close(fig)


if __name__ == "__main__":
    main()
