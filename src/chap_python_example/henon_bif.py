import matplotlib.pyplot as plt
import numpy as np
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


def henon_map(x, y, a=1.4, b=0.3):
    return 1 - a * x**2 + y, b * x


def plot_bifurcation_diagram(a_range, b, num_iter=2000, num_last=100):
    x_vals = []
    y_vals = []

    # 初期条件を前のパラメータの最終状態から継続
    x, y = 0.1, 0.1  # より安定した初期条件

    for a in a_range:
        # 過渡状態を十分に除去
        for _ in range(num_iter):
            x, y = henon_map(x, y, a=a, b=b)
            if _ >= num_iter - num_last:  # 最後のnum_last点を記録
                x_vals.append(a)
                y_vals.append(x)
        # 次のパラメータでは現在の状態を初期条件として使用（連続性の利用）

    plt.figure(figsize=(10, 6))
    plt.plot(x_vals, y_vals, ",k", alpha=0.5)
    # plt.title("Bifurcation Diagram of the Henon Map")
    plt.xlabel(r"$a \longrightarrow$")
    plt.ylabel(r"$x \longrightarrow$")
    plt.grid(True)

    # PDFで保存
    pdf = PdfPages("henon_bif.pdf")
    pdf.savefig(dpi=300)
    pdf.close()

    plt.show()


if __name__ == "__main__":
    a_values = np.linspace(0.5, 1.4, 2000)  # より細かい刻み幅
    plot_bifurcation_diagram(a_values, b=0.3)
