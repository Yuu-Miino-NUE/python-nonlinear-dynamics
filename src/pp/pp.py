def run_pp(ode_func, y0, params, tspan, config, tick=1e-2, rtol=1e-5):
    from scipy.integrate import solve_ivp
    from .pptools import init_plot, draw_traj
    from numpy import arange

    t_eval = arange(tspan[0], tspan[1], tick)

    # Matplotlib initialization
    plt, cfg = init_plot(y0, params, **config)

    # Draw trajectory
    while cfg.isRunning:
        # Calculate
        sol = solve_ivp(ode_func, tspan, y0, t_eval=t_eval, rtol=rtol, args=[params])

        # Draw calculated result
        draw_traj(plt, cfg, sol.y)

        # Post process
        y0[:] = sol.y[:, -1]  # Not y0 = sol.y[:, -1], strictly

        # Event handling
        plt.pause(0.001)  # REQIRED
