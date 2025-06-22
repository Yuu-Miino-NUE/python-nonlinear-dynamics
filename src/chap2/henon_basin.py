import numpy as np
import matplotlib.pyplot as plt
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


def compute_basin_of_attraction(a=1.4, b=0.3, grid_size=1000, iterations=100):
    x_min, x_max = -2.5, 2.5
    y_min, y_max = -2.5, 2.5

    x_values = np.linspace(x_min, x_max, grid_size)
    y_values = np.linspace(y_min, y_max, grid_size)
    basin = np.zeros((grid_size, grid_size))

    for i, x in enumerate(x_values):
        for j, y in enumerate(y_values):
            x_temp, y_temp = x, y
            for _ in range(iterations):
                x_temp, y_temp = henon_map(x_temp, y_temp, a, b)
                if abs(x_temp) > 1.5 or abs(y_temp) > 1.5:
                    basin[j, i] = 1  # Escape condition
                    break

    return x_values, y_values, basin


def plot_basin_of_attraction(x_values, y_values, basin):
    plt.figure(figsize=(8, 8))
    plt.imshow(
        basin,
        extent=(x_values[0], x_values[-1], y_values[0], y_values[-1]),
        origin="lower",
        cmap="binary",
    )
    plt.colorbar()
    plt.title("Basin of Attraction for Henon Map")
    plt.xlabel(r"$x \longrightarrow$")
    plt.ylabel(r"$y \longrightarrow$")

    # PDFで保存
    pdf = PdfPages("henon_basin.pdf")
    pdf.savefig(dpi=300)
    pdf.close()

    plt.show()


if __name__ == "__main__":
    a, b = 1.48, 0.3
    x_values, y_values, basin = compute_basin_of_attraction(a, b)
    plot_basin_of_attraction(x_values, y_values, basin)
