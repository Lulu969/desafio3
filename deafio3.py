import numpy as np

def seidel(A, b, tol=1e-10, max_iter=100):
    n = len(b)
    x = np.zeros_like(b)
    for _ in range(max_iter):
        x_new = np.copy(x)
        for i in range(n):
            sum1 = np.dot(A[i, :i], x_new[:i])
            sum2 = np.dot(A[i, i + 1:], x[i + 1:])
            x_new[i] = (b[i] - sum1 - sum2) / A[i, i]
        if np.linalg.norm(x_new - x, ord=np.inf) < tol:
            break
        x = x_new
    return x

A = np.array([[0.52, 0.20, 0.25],
              [0.30, 0.50, 0.20],
              [0.18, 0.20, 0.55]])

b = np.array([4800, 5810, 5690])

x = seidel(A, b)
print("Solución:", x)
