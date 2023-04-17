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

    def __init__(self, **kwargs):
        for k in kwargs:
            if hasattr(self, k):
                setattr(self, k, kwargs[k])
        self.isRunning = True


def pp(ode_func, tend, y0, params, tstart=0, tick=1e-2, **kwargs):
    from matplotlib import pyplot as plt
    from numpy import arange
    from scipy.integrate import solve_ivp

    # Matplotlib configuration
    cfg = MatplotConfig(**kwargs)

    # Initialization
    plt.figure(figsize=cfg.figsize)

    plt.rcParams["keymap.fullscreen"].remove("f")
    plt.rcParams["keymap.quit"].remove("q")
    plt.connect(
        "key_press_event", lambda event: on_key_pressed(event, plt, cfg, params)
    )
    plt.connect("button_press_event", lambda event: on_click(event, plt, cfg, y0))

    # Draw trajectory
    draw_axes(plt, cfg)
    tspan = arange(tstart, tend, tick)
    while cfg.isRunning:
        sol = solve_ivp(
            ode_func, (tstart, tend), y0, t_eval=tspan, rtol=1e-5, args=[params]
        )

        if not cfg.only_map:
            plt.plot(
                sol.y[cfg.xkey, :],
                sol.y[cfg.ykey, :],
                linewidth=cfg.linewidth,
                color=cfg.traj_color,
                ls="-",
                alpha=cfg.alpha,
            )

        plt.plot(
            sol.y[cfg.xkey, -1],
            sol.y[cfg.ykey, -1],
            "o",
            markersize=cfg.pointsize,
            color=cfg.point_color,
            alpha=cfg.alpha,
        )

        y0 = sol.y[:, -1]

        current_axes = plt.gca()
        number_of_plots = len(current_axes.lines)
        max_plots = 64
        if number_of_plots > max_plots:
            for line in current_axes.lines[:-max_plots]:
                line.remove()

        plt.pause(0.001)  # REQIRED


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
            step = config.param_step * (-1 if event.key == "up" else 1)
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
