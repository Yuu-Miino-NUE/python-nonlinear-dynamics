from sympy import cos


def duffing(t, state, params):
    x, y = state
    k = params["k"]
    B = params["B"]
    B0 = params["B0"]

    return [y, -k * y - x**3 + B0 + B * cos(t)]


duffing_config = {"param_keys": ["B", "B0"], "xrange": (-2, 2), "yrange": (-2, 2)}
