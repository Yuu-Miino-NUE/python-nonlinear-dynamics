class MatplotConfig:
    only_map = False
    figsize = (6, 6)
    xlabel = "x"
    ylabel = "y"
    zlabel = "z"
    xrange = (-3, 3)
    yrange = (-3, 3)
    zrange = (-3, 3)
    xkey = 0
    ykey = 1
    zkey = 2
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
    from mpl_toolkits.mplot3d import Axes3D

    cfg = MatplotConfig(**kwargs)

    fig = plt.figure(figsize=cfg.figsize)
    ax = fig.add_subplot(projection="3d")

    plt.rcParams["keymap.fullscreen"].remove("f")
    plt.rcParams["keymap.quit"].remove("q")
    plt.connect(
        "key_press_event", lambda event: on_key_pressed(event, plt, cfg, params)
    )
    plt.connect("button_press_event", lambda event: on_click(event, plt, cfg, y0))
    draw_axes(plt, ax, cfg)

    return (plt, ax, cfg)


def draw_traj(plt, ax, cfg, soly):
    if not cfg.only_map:
        ax.plot(
            soly[cfg.xkey, :],
            soly[cfg.ykey, :],
            soly[cfg.zkey, :],
            linewidth=cfg.linewidth,
            color=cfg.traj_color,
            ls="-",
            alpha=cfg.alpha,
        )

    ax.plot(
        soly[cfg.xkey, -1],
        soly[cfg.ykey, -1],
        soly[cfg.zkey, -1],
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


def draw_axes(plt, ax, cfg):
    import numpy as np

    axis_range = (
        np.array(
            [
                cfg.xrange[1] - cfg.xrange[0],
                cfg.yrange[1] - cfg.yrange[0],
                cfg.zrange[1] - cfg.zrange[0],
            ]
        ).max()
        * 0.5
    )

    mid_x = (cfg.xrange[1] + cfg.xrange[0]) / 2
    mid_y = (cfg.yrange[1] + cfg.yrange[0]) / 2
    mid_z = (cfg.zrange[1] + cfg.zrange[0]) / 2

    ax.set_xlim(mid_x - axis_range, mid_x + axis_range)
    ax.set_ylim(mid_y - axis_range, mid_y + axis_range)
    ax.set_zlim(mid_z - axis_range, mid_z + axis_range)
    ax.set_xlabel(cfg.xlabel)
    ax.set_ylabel(cfg.ylabel)
    ax.set_zlabel(cfg.zlabel)
    # plt.xlim(config.xrange)
    # plt.ylim(config.yrange)
    # plt.xlabel(config.xlabel)
    # plt.ylabel(config.ylabel)
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
