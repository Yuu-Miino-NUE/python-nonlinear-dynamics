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
    def __init__(self, func, dim, param_keys) -> None:
        self.func = func
        self.dim = dim

        _x = sp.symbols([f"x{i}" for i in range(dim)])
        _p = {k: sp.Symbol(k) for k in param_keys}
        _p_list = list(_p.values())
        self.jac_func = sp.lambdify(
            (_x, _p_list), sp.Matrix(func(_x, _p)).jacobian(_x), "numpy"
        )

    def image(self, x0, param):
        return self.func(x0, param)

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
    def __init__(self, xfix, eigval, eigvec, factors=(1e-2, 1e-2), num=1000) -> None:
        self.xfix = xfix
        self.eigval = eigval
        self.eigvec = eigvec

        self.eigspace = np.linspace(
            xfix - eigvec * factors[0], xfix + eigvec * factors[1], num
        )

    def image(
        self, func, param, itr=2, xlim=(-1.5, 1.5), ylim=(-1.5, 1.5), min_images=1000
    ):
        ret = np.empty(2)
        ll = np.array([xlim[0], ylim[0]])  # lower left
        ur = np.array([xlim[1], ylim[1]])  # upper right
        _domain = self.eigspace

        i = 0
        while i < itr:
            ret = np.array([func(x, param) for x in _domain])

            mask = np.all(np.logical_and(ll <= ret, ret <= ur), axis=1)
            ret = np.ma.array(ret, mask=~np.column_stack([mask, mask]))

            if ret.count(axis=0)[0] >= min_images:
                i += 1
                _domain = np.ma.compressed(ret).reshape(-1, 2)
            else:
                # TODO: Add new interporation method for each subspaces
                _domain = np.linspace(
                    _domain[0],
                    _domain[-1],
                    np.ceil(
                        min_images / max(ret.count(axis=0)[0], 1) * len(_domain)
                    ).astype("int"),
                )
                print(ret.count(axis=0)[0], "<", min_images, _domain.shape)

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
