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

    x = sol.y[0]
    y_val = sol.y[1]
    z = sol.y[2]

    fig, ax = plt.subplots(figsize=(10, 3))
    ax.plot(sol.t, x, label=r"$x(t)$", linewidth=0.8, color="black")
    ax.plot(sol.t, y_val, label=r"$y(t)$", linewidth=0.8, color="black", linestyle="--")
    ax.plot(sol.t, z, label=r"$z(t)$", linewidth=0.8, color="gray")
    ax.set_xlabel(r"Time $t$")
    ax.set_ylabel(r"State variables")
    ax.legend()
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    save_figure(fig, "rossler_timeseries")
    plt.close(fig)


if __name__ == "__main__":
    main()
