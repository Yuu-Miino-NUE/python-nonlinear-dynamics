import numpy as np
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
import time

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


def henon_map_vectorized(state, a=1.4, b=0.3):
    """ベクトル化されたHenonマップ関数"""
    x, y = state[..., 0], state[..., 1]
    x_next = 1 - a * x**2 + y
    y_next = b * x
    return np.stack([x_next, y_next], axis=-1)


def compute_basin_of_attraction(a=1.4, b=0.3, grid_size=1000, iterations=100):
    x_min, x_max = -2.5, 2.5
    y_min, y_max = -2.5, 2.5

    # メッシュグリッドを作成
    x_values = np.linspace(x_min, x_max, grid_size)
    y_values = np.linspace(y_min, y_max, grid_size)
    X, Y = np.meshgrid(x_values, y_values)

    # 初期状態を3次元配列として設定 (grid_size, grid_size, 2)
    initial_state = np.stack([X, Y], axis=-1)
    current_state = initial_state.copy()

    # 脱出条件を満たした点を追跡
    escaped = np.zeros((grid_size, grid_size), dtype=bool)
    basin = np.zeros((grid_size, grid_size))

    escape_threshold = 1.5

    for iteration in range(iterations):
        # まだ脱出していない点のみを更新
        mask = ~escaped
        if not mask.any():
            break

        # ベクトル化されたHenonマップを適用
        current_state = henon_map_vectorized(current_state, a, b)

        # 脱出条件をチェック
        x_current = current_state[..., 0]
        y_current = current_state[..., 1]

        # 新たに脱出した点を特定
        newly_escaped = mask & (
            (np.abs(x_current) > escape_threshold)
            | (np.abs(y_current) > escape_threshold)
        )

        # 脱出した点をマーク
        escaped |= newly_escaped
        basin[newly_escaped] = 1

        # 進捗表示
        if (iteration + 1) % 20 == 0:
            print(f"\rIteration {iteration + 1}/{iterations}", end="")

    print()  # 改行
    return x_values, y_values, basin


def plot_basin_of_attraction(x_values, y_values, basin):
    plt.figure(figsize=(8, 8))
    plt.imshow(
        basin,
        extent=(x_values[0], x_values[-1], y_values[0], y_values[-1]),
        origin="lower",
        cmap="binary",
    )
    # plt.title("Basin of Attraction for Henon Map")
    plt.xlabel(r"$x \longrightarrow$")
    plt.ylabel(r"$y \longrightarrow$")

    # PDFで保存
    pdf = PdfPages("henon_basin.pdf")
    pdf.savefig(dpi=300, bbox_inches="tight")
    pdf.close()

    plt.show()


if __name__ == "__main__":
    a, b = 1.48, 0.3
    grid_size = 1000
    iterations = 100

    print(f"Computing basin of attraction with grid size {grid_size}x{grid_size}")
    print(f"Parameters: a={a}, b={b}, iterations={iterations}")

    start_time = time.time()
    x_values, y_values, basin = compute_basin_of_attraction(a, b, grid_size, iterations)
    end_time = time.time()

    print(f"Computation completed in {end_time - start_time:.2f} seconds")
    print(f"Escaped points: {np.sum(basin)} ({np.sum(basin)/basin.size*100:.1f}%)")

    plot_basin_of_attraction(x_values, y_values, basin)
