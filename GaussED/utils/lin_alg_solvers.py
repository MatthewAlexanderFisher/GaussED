from GaussED.solver.base import Solver
from GaussED.utils.matrix_sqrt import sqrtm, svd_sqrtm
from GaussED.utils.safe_svd import svd_safe, svd_clip
from GaussED.utils.nearest_spd_matrix import nearestSPD

import torch
from torch import cholesky, cholesky_solve


class LinearSolver(Solver):

    def __init__(self):
        self.vectorisable = False

    def solve(self, x):
        raise NotImplementedError


class DefaultSolver(LinearSolver):

    def __init__(self):
        super().__init__()

    @staticmethod
    def square_root(mat):
        return sqrtm(mat)

    @staticmethod
    def inverse(mat):
        try:
            return cholesky(mat)
        except RuntimeError:
            return cholesky(nearestSPD(mat))

    @staticmethod
    def log_det(mat, inverse):
        return torch.log(torch.det(mat))  # to prevent numerical underflow and exploding gradients

    @staticmethod
    def solve(mat1, mat2):
        return cholesky_solve(mat2, mat1).squeeze(-1)

    @staticmethod
    def vec_solve(mat1, mat2):
        return cholesky_solve(mat2, mat1)

class SVDSolver(LinearSolver):

    def __init__(self):
        super().__init__()

    @staticmethod
    def square_root(mat):
        return svd_sqrtm(mat)

    @staticmethod
    def inverse(mat):
        return cholesky(mat)

    @staticmethod
    def log_det(mat, inverse):
        return torch.log(torch.clamp(torch.det(mat), min=1e-30))  # to prevent numerical underflow and exploding gradients

    @staticmethod
    def solve(mat1, mat2):
        return cholesky_solve(mat2, mat1).squeeze(-1)

    @staticmethod
    def vec_solve(mat1, mat2):
        return cholesky_solve(mat2, mat1)

class SafeSVDSolver(LinearSolver):

    def __init__(self):
        super().__init__()

    @staticmethod
    def square_root(mat):
        u, s = svd_safe(mat)
        return u.mul(torch.sqrt(s))

    @staticmethod
    def inverse(mat):
        return cholesky(mat)

    @staticmethod
    def log_det(mat, inverse):
        return torch.log(torch.clamp(torch.det(mat), min=1e-30))  # to prevent numerical underflow and exploding gradients

    @staticmethod
    def solve(mat1, mat2):
        return cholesky_solve(mat2, mat1).squeeze(-1)

    @staticmethod
    def vec_solve(mat1, mat2):
        return cholesky_solve(mat2, mat1)

class ClipSVDSolver(LinearSolver):

    def __init__(self):
        super().__init__()

    @staticmethod
    def square_root(mat):
        u, s = svd_clip(mat)
        return u.mul(torch.sqrt(s))

    @staticmethod
    def inverse(mat):
        return cholesky(mat)

    @staticmethod
    def log_det(mat, inverse):
        return torch.log(torch.clamp(torch.det(mat), min=1e-30))  # to prevent numerical underflow and exploding gradients

    @staticmethod
    def solve(mat1, mat2):
        return cholesky_solve(mat2, mat1).squeeze(-1)

    @staticmethod
    def vec_solve(mat1, mat2):
        return cholesky_solve(mat2, mat1)

class SafeCholeskySolver(LinearSolver):

    def __init__(self, nugget=1e-10, print_errors=True):
        super().__init__()
        self.vectorisable = True
        self.sqrt_nugget = nugget
        self.inverse_nugget = nugget
        self.print_errors = print_errors
        self.n_errors = 0

    def square_root(self, mat):
        try:
            sqrt = cholesky(mat + torch.eye(mat.size(1)) * self.sqrt_nugget)
            self.n_errors = 0
            return sqrt
        except RuntimeError:
            self.n_errors += 1
            if self.print_errors is True:
                print("Square root failed, nugget value: ", self.sqrt_nugget)
            self.sqrt_nugget = self.sqrt_nugget + 1e-10 * self.n_errors ** 2
            return self.square_root(mat)

    def inverse(self, mat):
        try:
            inverse = cholesky(mat)
            self.n_errors = 0
            return inverse
        except RuntimeError:
            self.n_errors += 1
            if self.print_errors is True:
                print("Inverse failed, inverse nugget value: ", self.inverse_nugget)
            self.inverse_nugget = self.inverse_nugget + 1e-11 * self.n_errors ** 2
            return self.inverse(mat + torch.eye(mat.size(1)) * self.inverse_nugget)

    @staticmethod
    def log_det(mat, inverse):
        return 2 * torch.sum(torch.log(torch.diag(inverse)))

    @staticmethod
    def solve(mat1, mat2):
        return cholesky_solve(mat2, mat1).squeeze(-1)

    @staticmethod
    def vec_solve(mat1, mat2):
        return cholesky_solve(mat2, mat1)


class SafeInverseSolver(LinearSolver):

    def __init__(self, nugget=None, print_errors=True):
        super().__init__()

        if nugget is None:
            self.inverse_nugget = 1e-10
            self.cholesky_nugget = 1e-10
        else:
            self.inverse_nugget = nugget
            self.cholesky_nugget = nugget

        self.print_errors = print_errors
        self.n_errors = 0
        self.vectorisable = True

    @staticmethod
    def square_root(mat):
        return sqrtm(mat)

    def inverse(self, mat):
        try:
            inverse = torch.inverse(mat)
            self.n_errors = 0
            return inverse
        except RuntimeError:
            self.n_errors += 1
            if self.print_errors is True:
                print("Inverse failed, inverse nugget value: ", self.inverse_nugget)
            self.inverse_nugget = self.inverse_nugget + 1e-11 * self.n_errors ** 2
            return self.inverse(mat + torch.eye(mat.size(1)) * self.inverse_nugget)

    def log_det(self, mat, inverse):
        return 2 * torch.sum(torch.log(torch.diag(self.cholesky(mat))))

    @staticmethod
    def solve(mat1, mat2):
        return torch.matmul(mat1, mat2).squeeze(-1)

    @staticmethod
    def vec_solve(mat1, mat2):
        return torch.matmul(mat1, mat2)

    def cholesky(self, mat):
        try:
            chol = cholesky(mat)
            self.n_errors = 0
            return chol
        except RuntimeError:
            self.n_errors += 1
            if self.print_errors is True:
                print("Cholesky failed, nugget value: ", self.cholesky_nugget)
            self.cholesky_nugget = self.cholesky_nugget + 1e-10 * self.n_errors ** 2
            return self.cholesky(mat + torch.eye(mat.size(1)) * self.cholesky_nugget)
