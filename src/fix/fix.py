import json, sys
import time
import numpy as np
from scipy.integrate import solve_ivp
from nonlinear_systems import duffing, duffing_config, duffing_period
from fixtools import newton

with open(sys.argv[1]) as f:
    data = json.load(f)
y0 = data["y0"]
params = data["params"]
dim = len(y0)

def objective_func(x0, params):
    initial_state = x0[:dim]
    initial_state = np.append(initial_state, np.eye(dim).flatten())
    state = solve_ivp(
        duffing,
        (0, duffing_period),
        initial_state,
        method="RK45",
        rtol=1e-8,
        args=[params],
    )
    state = state.y[:, -1]
    phi = state[:dim]
    dphidx = state[dim : dim * dim + dim].reshape(dim, dim).T

    F = phi - x0
    J = dphidx - np.eye(dim)

    return F, J


# initial vector for first newton method
x0 = np.array([y0[0], y0[1]])

for p in range(data["inc_iter"]):
    result = newton(objective_func, x0, args=(params))
    if result.success:
        print(result.x)
        x0 = result.x
        params["B0"] += 0.0001
        continue
    else:
        print(result.message)
        break
