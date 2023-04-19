from sympy import cos, pi


def ode_func(t, state, params):
    x, y = state
    mu = params["mu"]
    B = params["B"]
    B0 = params["B0"]

    return [y, mu * (1 - x**2) * y - x + B0 + B * cos(t)]


config = {"param_keys": ["B", "B0"], "yrange": (-10, 10)}
period = 2 * pi
