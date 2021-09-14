
import numpy as np
import scipy.linalg as la

def block_chol(L, x):
    B = x[:-1]
    d = x[-1]
    tri = la.solve_triangular(L, B, check_finite=False, lower=True)
    return (np.block([
        [L, np.zeros((len(B), 1))],
        [tri, np.sqrt(d - np.dot(tri, tri))]
    ]))
