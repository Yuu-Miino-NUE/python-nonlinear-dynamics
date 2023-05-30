import numpy as np
import matplotlib.pyplot as plt

from henon import henon, henon_inverse
from tools import DDS, EigenSpace, calc_manifold


def main(x0, param):
    # Discrete-time dynamical systems
    dds = DDS(henon, 2, ["a", "b"])

    xfix = dds.fix(x0, param).x
    print(xfix)
    eigs, vecs = np.linalg.eig(dds.jac(xfix, param))
    print(eigs)

    nums = [1000, 10000]
    espaces = [
        EigenSpace(xfix, eig, vec, num=num) for eig, vec, num in zip(eigs, vecs, nums)
    ]

    itrs = [12, 6]
    manis = [
        calc_manifold(henon if np.abs(e.eigval) > 1 else henon_inverse, e, param, itr=i)
        for e, i in zip(espaces, itrs)
    ]

    plt.figure(figsize=(8, 8))
    plt.grid()
    plt.xlim([-1.5, 1.5])
    plt.ylim([-1.5, 1.5])
    [
        plt.plot(m.points[:, 0], m.points[:, 1], "r-" if m.isStable is False else "b-")
        for m in manis
    ]
    plt.plot(xfix[0], xfix[1], c="white", mec="black", mew=1.5, marker="o")
    plt.show()


if __name__ == "__main__":
    import sys, json

    if len(sys.argv) == 0:
        x0 = np.array([0, 0])
        param = {"a": 1.15, "b": 0.3}
    else:
        try:
            with open(sys.argv[1], "r") as f:
                data = json.load(f)
            x0 = np.array(data["x0"])
            param = data["param"]
        except:
            print("Error: invalid argument")
            exit()

    main(x0, param)
