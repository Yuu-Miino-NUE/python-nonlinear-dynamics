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

    initial_conditions = [(0.1, 0.1), (0.2, 0.2), (-0.1, 0.1)]
    colors = ["black", "gray", "lightgray"]
    labels = [r"$(0.1, 0.1)$", r"$(0.2, 0.2)$", r"$(-0.1, 0.1)$"]

    a, b = 1.4, 0.3
    n_steps = 1000
    idle = 100

    fig, ax = plt.subplots(figsize=(6, 6))
    for (x0, y0), color, label in zip(initial_conditions, colors, labels):
        x_data, y_data = compute_orbit(x0, y0, a, b, n_steps, idle)
        ax.scatter(x_data, y_data, s=0.8, c=color, alpha=0.9, label=label)

    ax.set_xlabel(r"$x_n$")
    ax.set_ylabel(r"$y_n$")
    ax.legend()
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    save_figure(fig, "henon_multiple_trajectories")
    plt.close(fig)


if __name__ == "__main__":
    main()
