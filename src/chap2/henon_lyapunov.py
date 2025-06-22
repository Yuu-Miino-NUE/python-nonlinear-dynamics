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


def calculate_lyapunov_exponent(a_range, b, num_iter=1000, transient=100, delta=1e-8):
    lyapunov_exponents = []

    for a in a_range:
        x, y = 0.1, 0.1  # 初期条件
        x_perturbed, y_perturbed = x + delta, y + delta  # 微小な摂動を加えた初期条件
        sum_log_divergence = 0

        # 過渡状態をスキップ（アイドリング）
        for _ in range(transient):
            x, y = henon_map(x, y, a=a, b=b)
            x_perturbed, y_perturbed = henon_map(x_perturbed, y_perturbed, a=a, b=b)

        # リヤプノフ指数の計算
        for _ in range(num_iter):
            x, y = henon_map(x, y, a=a, b=b)
            x_perturbed, y_perturbed = henon_map(x_perturbed, y_perturbed, a=a, b=b)

            # 2つの軌道間の距離を計算
            distance = np.sqrt((x_perturbed - x) ** 2 + (y_perturbed - y) ** 2)

            # 距離がゼロに近い場合を防ぐ
            if distance > 0:
                sum_log_divergence += np.log(distance / delta)

                # 距離を正規化
                scale = delta / distance
                x_perturbed = x + scale * (x_perturbed - x)
                y_perturbed = y + scale * (y_perturbed - y)
            else:
                # 距離がゼロの場合、摂動を再初期化
                x_perturbed = x + delta
                y_perturbed = y + delta

        # 最大リヤプノフ指数を計算
        lyapunov_exponent = sum_log_divergence / num_iter
        lyapunov_exponents.append(lyapunov_exponent)

    return lyapunov_exponents


def plot_lyapunov_exponent(a_range, lyapunov_exponents):    
    plt.figure(figsize=(10, 6))
    plt.plot(a_range, lyapunov_exponents, "-k", alpha=0.7)
    plt.title("Maximum Lyapunov Exponent of the Henon Map")
    plt.xlabel(r"$a \longrightarrow$")
    plt.ylabel(r"Lyapunov Exponent $\longrightarrow$")
    plt.axhline(0, color="black", linestyle="--", linewidth=0.8)
    plt.grid(True)
    
    # PDFで保存
    pdf = PdfPages("henon_lyapunov.pdf")
    pdf.savefig(dpi=300)
    pdf.close()
    
    plt.show()


if __name__ == "__main__":
    a_values = np.linspace(0.5, 1.4, 500)  # aの範囲
    lyapunov_exponents = calculate_lyapunov_exponent(a_values, b=0.3)
    plot_lyapunov_exponent(a_values, lyapunov_exponents)
