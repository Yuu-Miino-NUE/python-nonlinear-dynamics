from . import run_pp


def main():
    import sys, json, os
    from importlib import import_module

    # ODE solver initialization
    with open(sys.argv[1]) as f:
        data = json.load(f)
    y0 = data["y0"]
    params = data["params"]
    config = {} if "config" not in data else data["config"]

    dirname = os.path.dirname(sys.argv[1])
    ds = import_module(f"{dirname}.system", package=None)

    tstart = 0
    tend = ds.period

    # Plotting
    run_pp(ds.ode_func, y0, params, (tstart, tend), ds.config | config)


if __name__ == "__main__":
    main()
