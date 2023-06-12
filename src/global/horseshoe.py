import numpy as np
from shapely.geometry import Polygon, MultiPolygon

from henon import henon, henon_inverse
from tools import DDS, HorseshoeRect


def main(x0, param, hs_config={}, filename="tmp"):
    # Prepare
    dds = DDS(henon, henon_inverse, 2, ["a", "b"])

    # Get fixed point
    xfix = dds.fix(x0, param).x
    print(xfix)

    # Calculate
    rect = HorseshoeRect(xfix, **hs_config["rect"])
    itr_forward = 1 if "itr_forward" not in hs_config else hs_config["itr_forward"]
    imag_rect = rect.image(dds.image, param, itr=itr_forward)
    itr_backward = 1 if "itr_backward" not in hs_config else hs_config["itr_backward"]
    preimage_rect = rect.image(dds.inverse, param, itr=itr_backward)

    poly_imag = Polygon(imag_rect)
    poly_preimag = Polygon(preimage_rect)
    poly_int = poly_imag.intersection(poly_preimag).difference(Polygon(rect.box))
    print(type(poly_int))
    if isinstance(poly_int, MultiPolygon):
        poly_int_list = np.hstack([np.array(g.boundary.xy) for g in poly_int.geoms]).T
    elif isinstance(poly_int, Polygon):
        poly_int_list = np.array(poly_int.boundary.xy).T
    else:
        print(type(poly_int))
        poly_int_list = None

    hrect = HorseshoeRect(xfix, **hs_config["rect"], box=poly_int_list)
    hrect_forward = [
        hrect.image(dds.image, param, itr=i)
        for i in range(1, hs_config["itr_backward"] + 1)
    ]
    hrect_backward = [
        hrect.image(dds.inverse, param, itr=i)
        for i in range(1, hs_config["itr_forward"] + 1)
    ]

    # Plot
    xlim = (-1.5, 1.5)
    ylim = (-0.5, 0.5)
    xnlim = (rect.ll[0] - rect.width * 0.05, rect.lr[0] + rect.width * 0.05)
    ynlim = (rect.ll[1] - rect.height * 0.05, rect.ul[1] + rect.height * 0.05)
    plt = plot_horseshoe(
        xfix,
        rect.box,
        imag_rect,
        preimage_rect,
        poly_int_list,
        hrect_forward,
        hrect_backward,
        xlim,
        ylim,
        xnlim,
        ynlim,
    )
    plt.show()


def plot_horseshoe(
    xfix,
    rect,
    imag_rect,
    preimag_rect,
    intersection,
    hrect_forward,
    hrect_backward,
    xlim=(-1.5, 1.5),
    ylim=(-0.5, 0.5),
    xnlim=(-1.5, 1.5),
    ynlim=(-0.5, 0.5),
):
    import matplotlib.pyplot as plt

    fig = plt.figure(figsize=(16, 8))

    def _plot(plt):
        plt.plot(xfix[0], xfix[1], "o", c="black")
        plt.fill(imag_rect[:, 0], imag_rect[:, 1], "-", c="red", alpha=0.2)
        plt.fill(preimag_rect[:, 0], preimag_rect[:, 1], "-", c="blue", alpha=0.2)
        plt.fill(rect[:, 0], rect[:, 1], "-", c="black", alpha=0.1)
        plt.fill(intersection[:, 0], intersection[:, 1], "-", c="black", alpha=0.4)
        [
            plt.fill(hf[:, 0], hf[:, 1], "-", c="magenta", alpha=0.4)
            for hf in hrect_forward
        ]
        [
            plt.fill(hb[:, 0], hb[:, 1], "-", c="green", alpha=0.4)
            for hb in hrect_backward
        ]

    all = fig.add_subplot(121)
    all.grid()
    all.set_xlim(xlim)
    all.set_ylim(ylim)
    _plot(all)
    zoom = fig.add_subplot(122)
    zoom.grid()
    zoom.set_xlim(xnlim)
    zoom.set_ylim(ynlim)
    _plot(zoom)

    return plt


if __name__ == "__main__":
    import sys, json

    if len(sys.argv) == 0:
        x0 = np.array([0, 0])
        param = {"a": 1.15, "b": 0.3}
        hs_config = {}
    else:
        try:
            with open(sys.argv[1], "r") as f:
                data = json.load(f)
            x0 = np.array(data["x0"])
            param = data["param"]
            hs_config = {} if "horseshoe" not in data else data["horseshoe"]
        except:
            print("Error: invalid argument")
            exit()

    main(x0, param, hs_config, sys.argv[1].split(".")[0])
