import matplotlib.pyplot as plt
import numpy as np
from scipy.integrate import solve_ivp
from matplotlib.backends.backend_pdf import PdfPages

# LaTeX設定
params = {
    "text.usetex": True,
    "text.latex.preamble": r"",
    "legend.fontsize": 16,
    "axes.labelsize": 16,
    "axes.titlesize": 16,
    "xtick.labelsize": 16,
    "ytick.labelsize": 16,
    "font.family": "serif",
    "grid.color": "k",
    "grid.linestyle": ":",
    "grid.linewidth": 0.5,
}
plt.rcParams.update(params)


def vanderpol(t, state, mu):
    x, y = state
    dxdt = y
    dydt = mu * (1 - x**2) * y - x
    return [dxdt, dydt]


def plot_phase_portrait(mu=1.0, t_span=(0, 20), y0=[1, 1], idle_time=10.0):
    # 時間刻みを設定
    t_eval_idle = np.arange(0, idle_time, 0.01)
    t_eval_main = np.arange(0, t_span[1] - idle_time, 0.01)

    # アイドリング（過渡除去）
    sol_idle = solve_ivp(vanderpol, (0, idle_time), y0, args=(mu,), t_eval=t_eval_idle)
    y0_after_idle = [sol_idle.y[0, -1], sol_idle.y[1, -1]]

    # 本計算
    sol = solve_ivp(
        vanderpol,
        (0, t_span[1] - idle_time),
        y0_after_idle,
        args=(mu,),
        t_eval=t_eval_main,
    )

    # 位相ポートレートを描画
    plt.figure(figsize=(8, 8))
    plt.plot(sol.y[0], sol.y[1], "k-", linewidth=2)
    # plt.title(rf"Phase Portrait of the van der Pol Oscillator ($\mu={mu}$)")
    plt.xlabel(r"$x \longrightarrow$")
    plt.ylabel(r"$dx/dt \longrightarrow$")
    plt.grid(True)
    plt.axis("equal")

    # PDFで保存
    pdf = PdfPages("vanderpol.pdf")
    pdf.savefig(dpi=300)
    pdf.close()

    plt.show()


if __name__ == "__main__":
    plot_phase_portrait()
