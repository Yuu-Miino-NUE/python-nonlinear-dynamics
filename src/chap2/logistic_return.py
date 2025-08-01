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


def logistic_map(x, a):
    """ロジスティック写像: x_{n+1} = r*x_n*(1-x_n)"""
    return a * x * (1 - x)


def generate_logistic_trajectory(r, x0=0.5, num_points=1000, transient=100):
    """ロジスティック写像の軌道を生成"""
    trajectory = np.zeros(num_points + transient)
    trajectory[0] = x0  # 初期値

    for i in range(num_points + transient - 1):
        trajectory[i + 1] = logistic_map(trajectory[i], r)

    # 過渡状態を除去
    return trajectory[transient:]


def plot_logistic_return_map(a=3.8, x0=0.5, num_points=500, num_iterations=20):
    """ロジスティック写像のリターンマップを描画"""
    plt.figure(figsize=(8, 8))  # 軌道を生成
    trajectory = generate_logistic_trajectory(a, x0, num_points, transient=0)

    # ロジスティック写像の関数 f(x) = a*x*(1-x) を描画
    x_range = np.linspace(0, 1, 1000)
    f_values = logistic_map(x_range, a)
    plt.plot(x_range, f_values, color="black", linewidth=3, label=r"$x_{n+1} = f(x_n)$")

    # 対角線 x_{n+1} = x_n を描画
    plt.plot(
        x_range,
        x_range,
        color="black",
        linestyle="--",
        linewidth=2,
        label=r"$x_{n+1} = x_n$",
    )

    # リターンマップの軌跡を描画（最初のnum_iterations点のみ）
    x_n = trajectory[:num_iterations]
    x_n1 = trajectory[1 : num_iterations + 1]

    # 散布図をプロット
    plt.scatter(
        x_n,
        x_n1,
        s=80,
        alpha=1.0,
        color="black",
        zorder=5,
        edgecolors="white",
        linewidth=1.0,
    )

    # 初期値の打点を追加
    x0_mapped = logistic_map(x0, a)
    plt.plot(
        x0,
        x0_mapped,
        "^",
        color="black",
        markerfacecolor="gray",
        markersize=14,
        markeredgewidth=2,
        label=f"Initial point: {x0:.3f}",
        zorder=6,
    )

    # 軌跡を直角の線で結ぶ
    for i in range(len(x_n)):
        # 最初の垂直線はスキップして、2番目以降から描画
        if i > 0:
            # 垂直線: (x_n[i], x_n[i]) → (x_n[i], x_n1[i])
            plt.plot(
                [x_n[i], x_n[i]],
                [x_n[i], x_n1[i]],
                color="black",
                linewidth=1.2,
                alpha=0.6,
                linestyle=":",
            )
        # 最後の点は次がないので水平線を描かない
        if i < len(x_n) - 1:
            # 水平線: (x_n[i], x_n1[i]) → (x_n1[i], x_n1[i])
            plt.plot(
                [x_n[i], x_n1[i]],
                [x_n1[i], x_n1[i]],
                color="black",
                linewidth=1.2,
                alpha=0.6,
                linestyle=":",
            )

    # 固定点を計算して表示
    if a > 1:
        fixed_point = (a - 1) / a
        plt.plot(
            fixed_point,
            fixed_point,
            "s",
            color="black",
            markerfacecolor="white",
            markersize=12,
            markeredgewidth=2,
            label=rf"Fixed point: {fixed_point:.3f}",
            zorder=7,
        )

    # plt.title(f"Logistic Map Return Map ($a = {a}$)")
    plt.xlabel(r"$x_n \longrightarrow$")
    plt.ylabel(r"$x_{n+1} \longrightarrow$")
    plt.legend()
    plt.grid(True)
    plt.xlim(0, 1)
    plt.ylim(0, 1)

    # PDFで保存
    pdf = PdfPages("logistic_return.pdf")
    pdf.savefig(dpi=300)
    pdf.close()

    plt.show()


if __name__ == "__main__":
    # リターンマップを描画
    plot_logistic_return_map(a=2.8, x0=0.075, num_points=500, num_iterations=32)
