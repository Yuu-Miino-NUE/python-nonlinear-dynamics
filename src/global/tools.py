import sympy as sp
import numpy as np


class FixResult:
    def __init__(self, x, param, itr, tol) -> None:
        self.x = x
        self.param = param
        self.itr = itr
        self.tol = tol

    def __str__(self) -> str:
        return (
            f"FixResult(x={self.x}, param={self.param}, itr={self.itr}, tol={self.tol})"
        )

    def __repr__(self) -> str:
        return self.__str__()


# Discrete-time dynamical systems
class DDS:
    def __init__(self, func, inverse_func, dim, param_keys) -> None:
        self.func = func
        self.inverse_func = inverse_func
        self.dim = dim

        _x = sp.symbols([f"x{i}" for i in range(dim)])
        _p = {k: sp.Symbol(k) for k in param_keys}
        _p_list = list(_p.values())
        self.jac_func = sp.lambdify(
            (_x, _p_list), sp.Matrix(func(_x, _p)).jacobian(_x), "numpy"
        )

    def image(self, x0, param):
        return self.func(x0, param)

    def inverse(self, x0, param):
        return self.inverse_func(x0, param)

    def jac(self, x0, param):
        return self.jac_func(x0, list(param.values()))

    def fix(self, x0, param, max_itr=100, tol=1e-10):
        # newton's method
        try:
            x = x0.copy()
        except AttributeError:
            x = np.array(x0)

        h = 0
        for _ in range(max_itr):
            h = np.linalg.solve(
                self.jac(x, param) - np.eye(self.dim), self.image(x, param) - x
            )
            x_new = x - h
            if np.linalg.norm(h) < tol:
                return FixResult(x_new, param, _, np.linalg.norm(h))
            x = x_new

        return FixResult(x, param, max_itr, np.linalg.norm(h))


class EigenSpace:
    def __init__(self, xfix, eigval, eigvec, radius=1e-2, num=1000) -> None:
        self.xfix = xfix
        self.eigval = eigval
        self.eigvec = eigvec

        _v = eigvec * radius
        self.eigspace = np.linspace(xfix - _v, xfix + _v, num)

    def image(
        self, func, param, itr=2, xlim=(-1.5, 1.5), ylim=(-1.5, 1.5), min_images=1000
    ):
        import sys

        def _f(x):
            for _ in range(itr):
                x = func(x, param)
            return x

        isFin = False

        ret = np.empty(2)
        ll = np.array([xlim[0], ylim[0]])  # lower left
        ur = np.array([xlim[1], ylim[1]])  # upper right

        while not isFin:
            ret = np.array([_f(x) for x in self.eigspace])
            mask = np.all(np.logical_and(ll <= ret, ret <= ur), axis=1)
            ret = np.ma.array(ret, mask=~np.column_stack([mask, mask]))

            if ret.count(axis=0)[0] >= min_images:
                isFin = True
            else:
                self.eigspace = np.linspace(
                    self.eigspace[0],
                    self.eigspace[-1],
                    np.ceil(
                        min_images / max(ret.count(axis=0)[0], 1) * len(self.eigspace)
                        + 1
                    ).astype("int"),
                )
            print(
                f"valid images: {ret.count(axis=0)[0]}, isFin: {isFin}", file=sys.stderr
            )
        print(file=sys.stderr)

        return ret


class Manifold:
    def __init__(self, points, eigval) -> None:
        self.points = points
        self.eigval = eigval
        self.isStable = True if np.abs(eigval) < 1 else False


def calc_manifold(
    func, espace, param, itr=2, xlim=(-1.5, 1.5), ylim=(-1.5, 1.5), min_images=2000
):
    return Manifold(
        espace.image(func, param, itr, xlim, ylim, min_images), espace.eigval
    )


def clip_masked_manifold(masked_manifold: np.ma.MaskedArray):
    ret = []

    _arr = np.array([])
    for i, p in enumerate(masked_manifold.mask):
        if True not in p:
            if _arr.size == 0:
                _arr = np.array([masked_manifold.data[i]])
            else:
                _arr = np.vstack([_arr, masked_manifold.data[i]])
        else:
            if _arr.size != 0:
                ret.append(_arr)
                _arr = np.array([])
    if _arr.size != 0:
        ret.append(_arr)
        _arr = np.array([])

    return ret


def save_masked_manifold(filename: str, masked_manifold: np.ma.MaskedArray):
    data = clip_masked_manifold(masked_manifold)
    with open(filename, "w") as f:
        for arr in data:
            for x, y in arr:
                f.write(f"{x} {y}\n")
            f.write("\n")


class ManifoldConfig:
    def __init__(
        self,
        xlim=np.array([-1.5, 1.5]),
        ylim=np.array([-1.5, 1.5]),
        itrs=(5, 5),
        min_images=(2000, 2000),
    ) -> None:
        self.xlim = np.array(xlim)
        self.ylim = np.array(ylim)
        self.itrs = itrs
        self.min_images = min_images


class HorseshoeRect:
    def __init__(
        self, xfix, ll=(-0.1, -0.1), ur=(0.1, 0.1), num=2000, box=None
    ) -> None:
        self.ll = xfix + np.array(ll)
        self.lr = xfix + np.array([ur[0], ll[1]])
        self.ul = xfix + np.array([ll[0], ur[1]])
        self.ur = xfix + np.array(ur)

        self.num = num

        if box is None:
            self.box = np.vstack(
                [
                    np.linspace(self.ll, self.lr, self.num),
                    np.linspace(self.lr, self.ur, self.num),
                    np.linspace(self.ur, self.ul, self.num),
                    np.linspace(self.ul, self.ll, self.num),
                ]
            )
        else:
            self.box = box

    def image(self, func, param, itr=2):
        def _f(x):
            for _ in range(itr):
                x = func(x, param)
            return x

        return np.array([_f(x) for x in self.box])
