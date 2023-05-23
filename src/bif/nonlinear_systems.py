from numpy import cos, pi


def duffing(t, state, params, var_param):
    n = 2
    x, y = state[0:n]
    k = params["k"]
    B0 = params["B0"]
    B = params["B"]

    f = [y, -k * y - x**3 + B0 + B * cos(t)]
    dfdx = [[0, 1], [-3 * x**2, -k]]
    dfdxdx = [[[0, 0], [-6 * x, 0]], [[0, 0], [0, 0]]]
    if var_param == "k":
        dfdlambda = [0, -y]
        dfdxdlambda = [[0, 0], [0, -1]]
    elif var_param == "B0":
        dfdlambda = [0, 1]
        dfdxdlambda = [[0, 0], [0, 0]]
    elif var_param == "B":
        dfdlambda = [0, cos(t)]
        dfdxdlambda = [[0, 0], [0, 0]]

    idx = n
    dphidx = state[idx : idx + n * n].reshape(n, n).T
    idx += n * n
    dphidlambda = state[idx : idx + n]
    idx += n
    dphidxdx = state[idx : idx + n * n * n].reshape(n, n, n)
    idx += n * n * n
    dphidxdlambda = state[idx : idx + n * n].reshape(n, n).T

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
