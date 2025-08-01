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


def henon_map(x, y, a, b):
    return 1 - a * x**2 + y, b * x


def henon_jacobian(x, y, a, b):
    return [[-2 * a * x, 1], [b, 0]]


def calculate_lyapunov_exponent(a, b, x0, y0, num_iter, transient):
    # 初期化
    x, y = x0, y0
    Q = np.eye(2)
    sum_log_R = 0.0

    # 過渡状態をスキップ（アイドリング）
    for _ in range(transient):
        x, y = henon_map(x, y, a=a, b=b)

    # リヤプノフ指数の計算
    for _ in range(num_iter):
        x, y = henon_map(x, y, a=a, b=b)
        J = henon_jacobian(x, y, a, b)
        Q = J @ Q  # リストに@を使うと自動的にnumpy配列に変換される
        Q, R = np.linalg.qr(Q)
        sum_log_R += np.log(np.abs(np.diag(R)))

    # 最大リヤプノフ指数を計算
    lyapunov_exponent = sum_log_R[0] / num_iter

    return lyapunov_exponent


def plot_lyapunov_exponent(a_range, lyapunov_exponents):
    plt.figure(figsize=(10, 6))
    plt.plot(a_range, lyapunov_exponents, "-k", alpha=0.7)
    # plt.title("Maximum Lyapunov Exponent of the Henon Map")
    plt.xlabel(r"$a \longrightarrow$")
    plt.ylabel(r"Lyapunov Exponent $\longrightarrow$")
    plt.axhline(0, color="black", linestyle="--", linewidth=0.8)
    plt.grid(True)

    pdf = PdfPages("henon_lyapunov.pdf")
    pdf.savefig(dpi=300)
    pdf.close()

    plt.show()


if __name__ == "__main__":
    a_values = np.linspace(0.5, 1.4, 500)  # aの範囲

    # Lyapunov指数を計算
    lyapunov_exponents = []
    for a in a_values:
        lyapunov_exponent = calculate_lyapunov_exponent(
            a, b=0.3, x0=0.1, y0=0.1, num_iter=1000, transient=100
        )
        lyapunov_exponents.append(lyapunov_exponent)

    plot_lyapunov_exponent(a_values, lyapunov_exponents)
