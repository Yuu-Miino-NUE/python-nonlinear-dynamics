import numpy as np

from henon import henon, henon_inverse
from tools import DDS, EigenSpace, calc_manifold, save_masked_manifold, ManifoldConfig


def main(x0, param, es_config=[{}], mani_config={}, filename="tmp"):
    # Prepare
    dds = DDS(henon, 2, ["a", "b"])
    cfg = ManifoldConfig(**mani_config)

    # Get fixed point
    xfix = dds.fix(x0, param).x

    # Get eigenspaces
    eigs, vecs = np.linalg.eig(dds.jac(xfix, param))
    espaces = [
        EigenSpace(xfix, eig, vec, **cfg)
        for eig, vec, cfg in zip(eigs, vecs, es_config)
    ]

    # Calculate
    manis = [
        calc_manifold(
            henon if np.abs(e.eigval) > 1 else henon_inverse,
            e,
            param,
            itr=i,
            xlim=cfg.xlim * 1.1,
            ylim=cfg.xlim * 1.1,
            min_images=2000,
        )
        for e, i in zip(espaces, cfg.itrs)
    ]

    # Plot
    plt = plot_manifold(xfix, manis, cfg)
    plt.show()

    # Save
    [
        save_masked_manifold(f"{filename}_manifold_{i:02d}.dat", m.points)
        for i, m in enumerate(manis)
    ]


def plot_manifold(xfix, manis, cfg):
    import matplotlib.pyplot as plt

    plt.figure(figsize=(8, 8))
    plt.grid()
    plt.xlim(cfg.xlim)
    plt.ylim(cfg.ylim)
    [
        plt.plot(m.points[:, 0], m.points[:, 1], "r-" if m.isStable is False else "b-")
        for m in manis
    ]
    plt.plot(xfix[0], xfix[1], c="white", mec="black", mew=1.5, marker="o")
    return plt


if __name__ == "__main__":
    import sys, json

    if len(sys.argv) == 0:
        x0 = np.array([0, 0])
        param = {"a": 1.15, "b": 0.3}
        es_config = []
        mani_config = {"xlim": np.array((-1.5, 1.5)), "ylim": np.array((-1.5, 1.5))}
    else:
        try:
            with open(sys.argv[1], "r") as f:
                data = json.load(f)
            x0 = np.array(data["x0"])
            param = data["param"]
            es_config = [] if "eigenspaces" not in data else data["eigenspaces"]
            mani_config = {} if "manifold" not in data else data["manifold"]
        except:
            print("Error: invalid argument")
            exit()

    main(x0, param, es_config, mani_config, sys.argv[1].split(".")[0])
