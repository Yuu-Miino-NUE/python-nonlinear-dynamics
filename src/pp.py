"""
How to use?

```
python duffing.py 01.json
```
"""
import json, sys
from numpy import arange
from scipy.integrate import solve_ivp

from pptools import init_plot
from nonlinear_systems import duffing, duffing_config, duffing_period

# ODE solver initialization
with open(sys.argv[1]) as f:
    data = json.load(f)
y0 = data["y0"]
params = data["params"]
tstart = 0
tend = duffing_period
tspan = arange(tstart, tend, 1e-2)  # tick = 1e-2

# Matplotlib initialization
plt, cfg = init_plot(y0, params, **duffing_config)

# Draw trajectory
while cfg.isRunning:
    # Calculate
    sol = solve_ivp(duffing, (tstart, tend), y0, t_eval=tspan, rtol=1e-5, args=[params])

    # Draw calculated result
    if not cfg.only_map:
        plt.plot(
            sol.y[cfg.xkey, :],
            sol.y[cfg.ykey, :],
            linewidth=cfg.linewidth,
            color=cfg.traj_color,
            ls="-",
            alpha=cfg.alpha,
        )

    plt.plot(
        sol.y[cfg.xkey, -1],
        sol.y[cfg.ykey, -1],
        "o",
        markersize=cfg.pointsize,
        color=cfg.point_color,
        alpha=cfg.alpha,
    )

    # Post process
    y0[:] = sol.y[:, -1]  # Not y0 = sol.y[:, -1], strictly

    # Erase old trajectories
    current_axes = plt.gca()
    number_of_plots = len(current_axes.lines)
    if number_of_plots > cfg.max_plots:
        for line in current_axes.lines[: -cfg.max_plots]:
            line.remove()

    # Event handling
    plt.pause(0.001)  # REQIRED
