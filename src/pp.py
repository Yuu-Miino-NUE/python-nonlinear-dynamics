"""
How to use?

```
python pp.py 01.json
```
"""
import json, sys
from pptools import get_config, init_plot, calc_traj, draw_traj

# Load JSON
with open(sys.argv[1]) as f:
    data = json.load(f)

y0 = data["y0"]
params = data["params"]
system = data["system"]

# Matplotlib initialization
plt, cfg = init_plot(y0, params, **get_config(system))

# Draw trajectory
while cfg.isRunning:
    # Calculate
    sol = calc_traj(system, y0, args=[params])

    # Draw calculated result
    draw_traj(plt, cfg, sol.y)

    # Post process
    y0[:] = sol.y[:, -1]  # Not y0 = sol.y[:, -1], strictly

    # Event handling
    plt.pause(0.001)  # REQIRED
