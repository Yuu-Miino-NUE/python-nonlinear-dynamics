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


def plot_phase_portrait(iterations=10000, x0=0.1, y0=0.1, a=1.4, b=0.3):
    x, y = x0, y0
    x_vals, y_vals = [], []
    for _ in range(iterations):
        x, y = henon_map(x, y, a, b)
        x_vals.append(x)
        y_vals.append(y)

    plt.figure(figsize=(8, 8))
    plt.scatter(x_vals, y_vals, s=0.1, color="black")
    # plt.title("Phase Portrait of the Hénon Map")
    plt.xlabel(r"$x \longrightarrow$")
    plt.ylabel(r"$y \longrightarrow$")
    plt.axis("equal")
    plt.grid(True)

    # PDFで保存
    pdf = PdfPages("henon.pdf")
    pdf.savefig(dpi=300)
    pdf.close()

    plt.show()


if __name__ == "__main__":
    plot_phase_portrait()
