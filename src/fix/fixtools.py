import numpy as np


class NewtonResult:
    x = np.zeros(0)
    success = False
    message = ""

def newton(func, x0, args=(), tol=1e-8, max_iter=16, explode=100):
    result = NewtonResult()
    for i in range(max_iter):
        F, J = func(x0, args)
        try:
            dx = np.linalg.solve(J, -F)
        except np.linalg.LinAlgError:
            result.x = x0
            result.success = False
            result.message = "Singular Jacobian"
            return result
        x = dx + x0
        norm = np.linalg.norm(dx)
        if norm > explode:
            result.x = x
            result.success = False
            result.message = "Solution exploded"
            return result
        elif norm < tol:
            result.x = x
            result.success = True
            result.message = "Converged"
            return result
        x0 = x
    else:
        result.x = x0
        result.success = False
        result.message = "Iteration over"
        return result
