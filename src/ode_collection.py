from sympy import cos, pi


def ode_lambda(func_name):
    match func_name:
        case "duffing":
            return duffing
        case "forced_bvp":
            return forced_bvp
        case "forced_van_der_pol":
            return forced_van_der_pol
        case _:
            raise ValueError(f"Unknown function name: {func_name}")


def duffing(t, state, params):
    x, y = state
    k = params["k"]
    B = params["B"]
    B0 = params["B0"]

    return [y, -k * y - x**3 + B0 + B * cos(t)]


duffing.config = {"param_keys": ["B", "B0"], "xrange": (-2, 2), "yrange": (-2, 2)}
duffing.period = 2 * pi


def forced_bvp(t, state, params):
    x, y = state
    B = params["B"]
    B0 = params["B0"]

    return [y + 1.6 * x - x**3, -x - 0.77 * y + B0 + B * cos(t)]


forced_bvp.config = {"param_keys": ["B", "B0"], "xrange": (0, 2), "yrange": (-1, 1)}
forced_bvp.period = 2 * pi


def forced_van_der_pol(t, state, params):
    x, y = state
    mu = params["mu"]
    B = params["B"]
    B0 = params["B0"]

    return [y, mu * (1 - x**2) * y - x + B0 + B * cos(t)]


forced_van_der_pol.config = {
    "param_keys": ["mu" "B", "B0"],
    "yrange": (-10, 10),
}
forced_van_der_pol.period = 2 * pi
