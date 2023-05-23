import json, sys
import time
import numpy as np
from scipy.integrate import solve_ivp
from nonlinear_systems import duffing, duffing_config, duffing_period
from biftools import newton, det_derivative

with open(sys.argv[1]) as f:
    data = json.load(f)
y0 = data["y0"]
params = data["params"]
dim = len(y0)


# function for newton(): any args -> F, J
def objective_func(x0, params):
    params["B"] = x0[dim]
    initial_state = x0[:dim]
    initial_state = np.append(initial_state, np.eye(dim).flatten())
    initial_state = np.append(
        initial_state, np.zeros(dim + dim**3 + dim**2).flatten()
    )
    state = solve_ivp(
        duffing,
        (0, duffing_period),
        initial_state,
        method="RK45",
        rtol=1e-8,
        args=[params, "B"],
    )
    state = state.y[:, -1]
    idx = 0
    phi = state[idx:dim]
    idx += dim
    dphidx = state[idx : idx + dim * dim].reshape(dim, dim).T
    idx += dim * dim
    dphidlambda = state[idx : idx + dim]
    idx += dim
    dphidxdx = state[idx : idx + dim * dim * dim].reshape(dim, dim, dim)
    idx += dim * dim * dim
    dphidxdlambda = state[idx : idx + dim * dim].reshape(dim, dim).T

    # prepare objective function
    F = phi - initial_state[:dim]
    chi = dphidx + np.eye(dim)
    F = np.append(F, np.linalg.det(chi))

    # prepare jacobian
    J = dphidx - np.eye(dim)
    J = np.insert(J, J.shape[1], dphidlambda, axis=1)
    dchidx = np.array([det_derivative(chi, dphidxdx[i]) for i in range(dim)])
    dchidlambda = det_derivative(chi, dphidxdlambda)
    temp = np.append(dchidx, dchidlambda)
    J = np.insert(J, J.shape[0], temp, axis=0)

    return F, J, np.linalg.eigvals(dphidx)


# initial vector for first newton method
# x0 = np.array([y0[0], y0[1]])
x0 = np.array([y0[0], y0[1], params["B"]])

for p in range(data["inc_iter"]):
    result = newton(objective_func, x0, args=(params))
    if result.success:
        print(result.message)
        print(result.x)
        print(params)
        print(result.eigvals)
        x0 = result.x
        params["B0"] -= 0.001
        continue
    else:
        print(result.message)
        break
