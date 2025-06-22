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


def generate_henon_time_series(a, b, num_points=1000, transient=100):
    x, y = 0.1, 0.1  # 初期条件
    x_values = []

    # 過渡状態をスキップ
    for _ in range(transient):
        x, y = henon_map(x, y, a=a, b=b)

    # 時系列データを収集
    for _ in range(num_points):
        x, y = henon_map(x, y, a=a, b=b)
        x_values.append(x)

    return x_values


def compute_power_spectrum(time_series):
    # 時系列データの平均を引いて直流成分を除去
    time_series = time_series - np.mean(time_series)

    # フーリエ変換を実行
    fft_result = np.fft.fft(time_series)
    power_spectrum = np.abs(fft_result) ** 2
    freqs = np.fft.fftfreq(len(time_series))

    return freqs[: len(freqs) // 2], power_spectrum[: len(power_spectrum) // 2]


def plot_power_spectrum(freqs, power_spectrum):
    plt.figure(figsize=(10, 6))
    plt.plot(freqs, power_spectrum, color="black")
    plt.title("Power Spectrum of Henon Map")
    plt.xlabel(r"Frequency $\longrightarrow$")
    plt.ylabel(r"Power $\longrightarrow$")
    plt.grid(True)

    # PDFで保存
    pdf = PdfPages("henon_power.pdf")
    pdf.savefig(dpi=300)
    pdf.close()

    plt.show()


def plot_time_series(time_series):
    plt.figure(figsize=(10, 6))
    plt.plot(time_series, color="black", linewidth=1)
    plt.title("Time Series of Henon Map")
    plt.xlabel(r"Time Step $\longrightarrow$")
    plt.ylabel(r"Value $\longrightarrow$")
    plt.grid(True)

    # PDFで保存
    pdf = PdfPages("henon_time_series.pdf")
    pdf.savefig(dpi=300)
    pdf.close()

    plt.show()


if __name__ == "__main__":
    a, b = 1.4, 0.3  # Henon写像のパラメータ
    time_series = generate_henon_time_series(a, b)
    plot_time_series(time_series)  # 時系列をプロット
    freqs, power_spectrum = compute_power_spectrum(time_series)
    plot_power_spectrum(freqs, power_spectrum)
