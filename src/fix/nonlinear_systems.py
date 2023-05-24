from sympy import cos, pi


def duffing(t, state, params):
    n = 2
    x, y = state[0:n]
    k = params["k"]
    B = params["B"]
    B0 = params["B0"]

    f = [y, -k * y - x**3 + B0 + B * cos(t)]
    dfdx = [[0, 1], [-3 * x**2, -k]]

    dphidx = state[n : n * n + n].reshape(n, n).T

    f.extend((dfdx @ dphidx).T.flatten())

    return f


duffing_config = {"param_keys": ["B", "B0"], "xrange": (-2, 2), "yrange": (-2, 2)}
duffing_period = 2 * pi
