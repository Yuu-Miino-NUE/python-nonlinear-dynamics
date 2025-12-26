import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import solve_ivp

from plot_config import setup_plot, save_figure


def rossler(t, y, a=0.2, b=0.2, c=5.7):
    x, y_val, z = y
    return np.array([-y_val - z, x + a * y_val, b + z * (x - c)])


def main():
    setup_plot()

    c_values = [4.0, 5.0, 5.7]
    y0 = np.array([1.0, 1.0, 1.0])
    t_eval = np.linspace(0, 100, 2000)

    fig = plt.figure(figsize=(12, 4))
    for i, c in enumerate(c_values):
        def rossler_c(t, y):
            return rossler(t, y, a=0.2, b=0.2, c=c)

        sol = solve_ivp(rossler_c, (0, 100), y0, t_eval=t_eval, rtol=1e-8, atol=1e-10)
        ax = fig.add_subplot(1, 3, i + 1, projection="3d")
        ax.plot(sol.y[0], sol.y[1], sol.y[2], linewidth=0.5, color="black", alpha=0.7)
        ax.set_xlabel(r"$x$")
        ax.set_ylabel(r"$y$")
        ax.set_zlabel(r"$z$")
        ax.set_title(f"$c = {c}$")

    plt.tight_layout()
    save_figure(fig, "rossler_parameter_scan")
    plt.close(fig)


if __name__ == "__main__":
    main()
