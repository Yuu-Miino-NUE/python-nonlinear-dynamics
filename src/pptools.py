class MatplotConfig:
    only_map = False
    figsize = (6, 6)
    xlabel = "x"
    ylabel = "y"
    xrange = (-3, 3)
    yrange = (-3, 3)
    xkey = 0
    ykey = 1
    linewidth = 1
    pointsize = 3
    alpha = 0.3
    traj_color = "black"
    point_color = "red"
    param_keys = []  # For parameter control
    param_idx = 0  # For parameter control
    param_step = 1e-2  # For parameter control
    max_plots = 64

    def __init__(self, **kwargs):
        for k in kwargs:
            if hasattr(self, k):
                setattr(self, k, kwargs[k])
        self.isRunning = True


def init_plot(y0, params, **kwargs):
    from matplotlib import pyplot as plt

    cfg = MatplotConfig(**kwargs)

    plt.figure(figsize=cfg.figsize)

    plt.rcParams["keymap.fullscreen"].remove("f")
    plt.rcParams["keymap.quit"].remove("q")
    plt.connect(
        "key_press_event", lambda event: on_key_pressed(event, plt, cfg, params)
    )
    plt.connect("button_press_event", lambda event: on_click(event, plt, cfg, y0))
    draw_axes(plt, cfg)

    return (plt, cfg)


def get_config(func_name):
    from ode_collection import ode_lambda

    return ode_lambda(func_name).config


def calc_traj(func_name, y0, tspan=None, tick=1e-2, args=[]):
    from scipy.integrate import solve_ivp
    from ode_collection import ode_lambda
    from numpy import arange

    ode_func = ode_lambda(func_name)

    if tspan is None:
        if hasattr(ode_func, "period"):
            tspan = (0, ode_func.period)
        else:
            tspan = (0, 5)

    t_eval = arange(tspan[0], tspan[1], tick)

    sol = solve_ivp(
        ode_lambda(func_name), tspan, y0, t_eval=t_eval, rtol=1e-5, args=args
    )
    return sol


def draw_traj(plt, cfg, soly):
    if not cfg.only_map:
        plt.plot(
            soly[cfg.xkey, :],
            soly[cfg.ykey, :],
            linewidth=cfg.linewidth,
            color=cfg.traj_color,
            ls="-",
            alpha=cfg.alpha,
        )

    plt.plot(
        soly[cfg.xkey, -1],
        soly[cfg.ykey, -1],
        "o",
        markersize=cfg.pointsize,
        color=cfg.point_color,
        alpha=cfg.alpha,
    )

    erase_old_traj(plt, cfg)


def erase_old_traj(plt, cfg):
    current_axes = plt.gca()
    number_of_plots = len(current_axes.lines)
    if number_of_plots > cfg.max_plots:
        for line in current_axes.lines[: -cfg.max_plots]:
            line.remove()


def draw_axes(plt, config):
    plt.xlim(config.xrange)
    plt.ylim(config.yrange)
    plt.xlabel(config.xlabel)
    plt.ylabel(config.ylabel)
    plt.grid(c="gainsboro", zorder=9)


def on_key_pressed(event, plt, config, params):
    match event.key:
        case "q":
            config.isRunning = False
        case " " | "e" | "f":
            if event.key == "f":
                config.only_map = not config.only_map
            plt.cla()
            draw_axes(plt, config)
        case "p":  # For parameter control
            config.param_idx = (config.param_idx + 1) % len(config.param_keys)
            print(f"changable parameter: {config.param_keys[config.param_idx]}")
        case "up" | "down":  # For parameter control
            step = config.param_step * (-1 if event.key == "down" else 1)
            params[config.param_keys[config.param_idx]] = round(
                params[config.param_keys[config.param_idx]] + step, 10
            )
            print(params)


def on_click(event, plt, config, x0):
    if event.xdata == None or event.ydata == None:
        return
    x0[config.xkey] = event.xdata
    x0[config.ykey] = event.ydata

    plt.plot(
        x0[config.xkey],
        x0[config.ykey],
        "o",
        markersize=3,
        color="blue",
        alpha=config.alpha,
    )

    print(x0)
    return
