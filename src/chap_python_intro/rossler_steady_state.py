import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import solve_ivp

from plot_config import setup_plot, save_figure


def rossler(t, y, a=0.2, b=0.2, c=5.7):
    x, y_val, z = y
    return np.array([-y_val - z, x + a * y_val, b + z * (x - c)])


def main():
    setup_plot()

    y0 = np.array([1.0, 1.0, 1.0])
    t_eval = np.linspace(0, 100, 2000)
    sol = solve_ivp(rossler, (0, 100), y0, t_eval=t_eval, rtol=1e-8, atol=1e-10)

    idle_steps = 500
    x = sol.y[0, idle_steps:]
    y_val = sol.y[1, idle_steps:]
    z = sol.y[2, idle_steps:]

    fig = plt.figure(figsize=(8, 6))
    ax = fig.add_subplot(111, projection="3d")
    ax.plot(x, y_val, z, linewidth=0.5, color="black", alpha=0.7)
    ax.set_xlabel(r"$x$")
    ax.set_ylabel(r"$y$")
    ax.set_zlabel(r"$z$")

    plt.tight_layout()
    save_figure(fig, "rossler_steady_state")
    plt.close(fig)


if __name__ == "__main__":
    main()
