from sympy import cos, pi


# def duffing(t, state, params):
#     x, y = state
#     k = params["k"]
#     B = params["B"]
#     B0 = params["B0"]

#     return [y, -k * y - x**3 + B0 + B * cos(t)]


# duffing_config = {"param_keys": ["B", "B0"], "xrange": (-2, 2), "yrange": (-2, 2)}
# duffing_period = 2 * pi


def duffing(t, state, params):
    x, y, z = state
    k = params["k"]
    B = params["B"]
    B0 = params["B0"]

    f1 = y
    f2 = -k * y - (3 * z**2 + x**2) * x / 8 + B * cos(t)
    f3 = -k * (3 * x**2 + z**2) * z / 8 + B0

    return [f1, f2, f3]


duffing_config = {"param_keys": ["B", "B0"], "xrange": (-2, 2), "yrange": (-2, 2)}
duffing_period = 2 * pi
