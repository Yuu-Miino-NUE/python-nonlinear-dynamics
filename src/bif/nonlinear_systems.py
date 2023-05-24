from numpy import cos, pi, zeros
from biftools import state_to_derivatives


def duffing(t, state, params, var_param):
    n = 2
    x, y = state[0:n]
    k = params["k"]
    B0 = params["B0"]
    B = params["B"]

    f = [y, -k * y - x**3 + B0 + B * cos(t)]
    dfdx = [[0, 1], [-3 * x**2, -k]]
    dfdxdx = [[[0, 0], [-6 * x, 0]], [[0, 0], [0, 0]]]

    dfdlambda = zeros(n)
    dfdxdlambda = zeros((n, n))
    if var_param == "k":
        dfdlambda[1] = -y
        dfdxdlambda[1, 1] = -1
    elif var_param == "B0":
        dfdlambda[1] = 1
    elif var_param == "B":
        dfdlambda[1] = cos(t)
    else:
        pass

    _, dphidx, dphidlambda, dphidxdx, dphidxdlambda = state_to_derivatives(state, n)

    f.extend((dfdx @ dphidx).T.flatten())
    f.extend(dfdx @ dphidlambda + dfdlambda)
    f.extend((dfdx @ dphidxdx + (dfdxdx @ dphidx).T @ dphidx).flatten())
    f.extend(
        (
            dfdx @ dphidxdlambda
            + ((dfdxdx @ dphidx).T @ dphidlambda).T
            + (dfdxdlambda @ dphidx)
        ).T.flatten()
    )

    return f


duffing_config = {"param_keys": ["B", "B0"], "xrange": (-2, 2), "yrange": (-2, 2)}
duffing_period = 2 * pi
