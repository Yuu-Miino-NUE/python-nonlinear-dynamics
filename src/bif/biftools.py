import numpy as np
import itertools as it


class NewtonResult:
    x = np.zeros(0)
    success = False
    message = ""
    eigvals = np.zeros(0)


def newton(func, x0, args=(), tol=1e-8, max_iter=16, explode=100):
    result = NewtonResult()
    for i in range(max_iter):
        F, J, eigvals = func(x0, args)
        result.eigvals = eigvals
        try:
            dx = np.linalg.solve(J, -F)
        except np.linalg.LinAlgError:
            result.x = x0
            result.success = False
            result.message = "Singular Jacobian"
            return result
        x = dx + x0
        if np.linalg.norm(dx) > explode:
            result.x = x
            result.success = False
            result.message = "Solution exploded (norm overs " + str(explode) + ")"
            return result
        elif all(elem < tol for elem in dx):
            result.x = x
            result.success = True
            result.message = "Converged (" + str(i) + " iterations)"
            return result
        x0 = x
    else:
        result.x = x0
        result.success = False
        result.message = "Iteration over (" + str(max_iter) + " iterations)"
        return result


def det_derivative(A, dA):
    dim = A.shape[0]
    ret = 0
    for i in range(dim):
        temp = A.copy()
        temp[:, i] = dA[:, i]
        ret += np.linalg.det(temp)
    return ret


def bialt_square(A):
    n = A.shape[0]
    bialt_dim = sum(range(n))
    result = np.zeros((bialt_dim, bialt_dim))
    temp = np.zeros((2, 2))
    result_idx = ((i, j) for i in range(bialt_dim) for j in range(bialt_dim))
    mul_idx = [(i, j) for i in range(1, n) for j in range(i)]
    for row, col in it.product(mul_idx, mul_idx):
        for i, j in it.product([0, 1], [0, 1]):
            temp[i, j] = A[row[i], col[j]]
        result[next(result_idx)] = np.linalg.det(temp)
    return result


def bialt_square_derivative(A, dA):
    n = A.shape[0]
    bialt_dim = sum(range(n))
    result = np.zeros((bialt_dim, bialt_dim))
    temp = np.zeros((2, 2))
    dtemp = np.zeros((2, 2))
    result_idx = ((i, j) for i in range(bialt_dim) for j in range(bialt_dim))
    mul_idx = [(i, j) for i in range(1, n) for j in range(i)]
    for row, col in it.product(mul_idx, mul_idx):
        for i, j in it.product([0, 1], [0, 1]):
            temp[i, j] = A[row[i], col[j]]
            dtemp[i, j] = dA[row[i], col[j]]
        result[next(result_idx)] = det_derivative(temp, dtemp)
    return result
