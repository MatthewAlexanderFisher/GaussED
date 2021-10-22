import math
from functools import lru_cache
import torch

from gaussed.basis.base import Basis
from gaussed.utils.summation_tensor_gen import sum_tensor_gen


def diff_basis_func_gen(order):
    """Generates t

    Args:
        order ([type]): [description]

    Raises:
        TypeError: [description]

    Returns:
        [type]: [description]
    """

    if order % 4 == 0:

        def func(x, i):
            return torch.sin(i * x)

    elif order % 4 == 1:

        def func(x, i):
            return torch.cos(i * x)

    elif order % 4 == 2:

        def func(x, i):
            return -torch.sin(i * x)

    elif order % 4 == 3:

        def func(x, i):
            return -torch.cos(i * x)

    else:
        raise TypeError("order should be of type int")

    return func


class Laplace(Basis):
    def __init__(self, dim, a=None, b=math.pi, c=0, d=None, dims=None):
        """
        General form of basis function is of the form

        a(x,i) * sin(i(bx + c)) or a(x,i) * cos(i(bx + c))
        :param dim:
        """

        super().__init__()

        self.bases = [self]  # multiple bases

        self.dim = dim  # dimension of the basis functions generated
        self.input_dim = dim  # dimension of the expected x input

        if dims is None:
            self.dims = [
                i for i in range(self.dim)
            ]  # list of the dimensions the laplace basis takes as input
            self.full_dims = True
        else:
            self.dims = sorted(dims)
            if set(dims) == set([i for i in range(self.dim)]):
                self.full_dims = True
            else:
                self.full_dims = False

        if a is None:

            def a(x, i):
                return torch.Tensor([1])

            self.a = a
        else:
            self.a = a

        self.b = torch.ones(self.dim) * b
        self.c = torch.ones(self.dim) * c
        self.norm = (2 * self.b / math.pi).pow(1 / 2)

        self.integrable = True
        self.fully_integrated = False
        self.integral_dims = []
        self.differentiable = True
        self.diff_orders = torch.zeros(self.dim)

        def integral_const(m, i):
            return torch.Tensor([1])

        self.integral_const = integral_const

        self.index_func = None
        self.integrated_index_func = None

        def func(x, i):
            return torch.sin(i * x)

        self.func = func
        self.integrated_func = None

        def basis_func(x, i):
            x_t = x.unsqueeze(1)
            return self.func(self.b * x_t + self.c, i)

        self.basis_func = basis_func
        self.integrated_basis_func = None

        self.func_vector = [self.func for i in range(self.dim)]
        self.integrated_func_vector = []

        self.laplace_list = (
            []
        )  # this keeps track of laplace that are built from this object

    @lru_cache(maxsize=32)
    def eval(self, x, m):
        """
        Computes the tensor with Phi_ij = phi_j(x_i). m is the number of basis functions.

        :param x: Tensor of size n x d
        :param m: Tensor or integer value
        :return: Tensor of size n x m
        """
        if x is not None:
            n, dim = x.shape

        if self.full_dims is False:
            x_r = x.T[self.dims].T
        else:
            x_r = x

        j_mat = sum_tensor_gen(self.dim, m)

        if (
            self.fully_integrated
        ):  # TODO: include normalisation constant when integrated
            # integral_transform const includes m for line_integral
            return self.integral_const(m, j_mat).unsqueeze(0)
        elif self.integral_dims:
            j_mat_int = (j_mat.T[self.integral_dims]).T
            j_mat_other = (
                j_mat.T[
                    list(set([i for i in range(self.dim)]) - set(self.integral_dims))
                ]
            ).T

            return (
                self.integral_const(m, j_mat_int)
                * self.a(x, j_mat_other)
                * torch.prod(self.basis_func(x_r, j_mat_other), dim=2).reshape(
                    n, m ** self.dim
                )
            )

        else:
            return self.a(x, j_mat) * torch.prod(
                self.norm * self.basis_func(x_r, j_mat), dim=2
            ).reshape(n, m ** self.dim)

    def set_new_basis(self, a, b, c, diff_orders, integral_dims, basis_func=None):

        self.a = a
        self.b = b
        self.c = c
        self.norm = (2 * self.b / math.pi).pow(1 / 2)

        self.diff_orders = diff_orders
        self.integral_dims = integral_dims

        non_integral_dims = list(set([i for i in range(self.dim)]) - set(integral_dims))

        diff_orders_integral, diff_orders_other = (
            diff_orders[integral_dims],
            diff_orders[non_integral_dims],
        )
        b_integral, b_other = b[integral_dims], b[non_integral_dims]
        c_integral, c_other = c[integral_dims], c[non_integral_dims]

        if (
            basis_func is None and non_integral_dims
        ):  # if not all dimensions have been integrated

            def index_func(i):
                return (i * b_other).pow(diff_orders_other)

            self.index_func = index_func

            if torch.equal(diff_orders_other, torch.zeros(self.dim)):
                pass  # if all derivative orders are the same
            elif torch.equal(
                diff_orders_other, (torch.ones(self.dim) * diff_orders_other[0])
            ):

                self.func = diff_basis_func_gen(diff_orders_other[0])
                self.func_vector = [self.func for i in range(len(non_integral_dims))]

                # the following is faster than using the more general torch.stack(...) function below
                def basis_func(x, i):
                    x_t = x.unsqueeze(1)
                    return index_func(i) * self.func(b_other * x_t + c_other, i)

                self.basis_func = basis_func
            else:
                new_func_vector = []

                for i in range(len(non_integral_dims)):
                    new_func_vector.append(diff_basis_func_gen(diff_orders_other[i]))

                self.func_vector = new_func_vector

                def basis_func(x, i):
                    return (
                        index_func(i)
                        * torch.stack(
                            [
                                f(b_other[k] * x.T[k] + c_other[k], i.T[k].unsqueeze(1))
                                for k, f in enumerate(self.func_vector)
                            ]
                        ).T
                    )

                self.basis_func = basis_func

        elif basis_func is not None:
            self.basis_func = basis_func
        else:  # if we have integrated all dimension, basis function just returns 1

            self.func_vector = []
            self.fully_integrated = True

            def basis_func(x, i):
                return torch.Tensor([1])

            self.basis_func = basis_func

        if (
            integral_dims
        ):  # if integral_dims is not the empty list (i.e. we have an integral_transform)

            def integrated_index_func(i):
                return (i * b_integral).pow(diff_orders_integral)

            self.integrated_index_func = integrated_index_func

            if torch.equal(
                diff_orders_integral, (torch.ones(self.dim) * diff_orders_integral[0])
            ):

                self.integrated_func = diff_basis_func_gen(diff_orders_integral[0])
                self.integrated_func_vector = [
                    self.integrated_func for i in range(len(integral_dims))
                ]

                def integrated_basis_func(x, i):
                    x_t = x.unsqueeze(1)
                    return integrated_index_func(i) * self.integrated_func(
                        b_integral * x_t + c_integral, i
                    )

                self.integrated_basis_func = integrated_basis_func

            else:
                new_integrated_func_vector = []

                for i in range(len(integral_dims)):
                    new_integrated_func_vector.append(
                        diff_basis_func_gen(diff_orders_integral[i])
                    )

                self.integrated_func_vector = new_integrated_func_vector

                def integrated_basis_func(x, i):
                    return (
                        integrated_index_func(i)
                        * torch.stack(
                            [
                                f(
                                    b_integral[k] * x.T[k] + c_integral[k],
                                    i.T[k].unsqueeze(1),
                                )
                                for k, f in enumerate(self.integrated_func_vector)
                            ]
                        ).T
                    )

                self.integrated_basis_func = integrated_basis_func

    def set_differentiable(self, laplace):
        if laplace.differentiable:
            self.differentiable = True
        else:
            self.differentiable = False

    def set_integrable(self, laplace):
        if laplace.integrable:
            self.integrable = True
        else:
            self.integrable = False

    def set_domain(self, domain):
        relative_domain = domain[self.dims]
        new_b = math.pi / (relative_domain.T[1] - relative_domain.T[0])
        new_c = -relative_domain.T[0] * new_b
        self.b = new_b
        self.c = new_c
        self.norm = (2 * self.b / math.pi).pow(1 / 2)

    @staticmethod
    def affine(laplace, linear):
        """
        Function valued affine transform of basis function :- basis_func(x, i) * linear(x) + translation(x)
        :param laplace:
        :param linear: linear function
        :param translation: translation function
        """

        def new_a(x, i):
            return laplace.a(x, i) * linear(x)

        new_laplace = Laplace(laplace.dim)
        new_laplace.set_new_basis(
            new_a,
            laplace.b,
            laplace.c,
            laplace.diff_orders,
            laplace.integral_dims,
            laplace.normalisation,
        )

        new_laplace.integrable = False
        new_laplace.differentiable = False

        return new_laplace

    @staticmethod
    def simple_affine(laplace, A=1, B=0):
        """
        Overrides eval with simple affine transform of the form A * f(x) + B
        :param laplace:
        :param A: Tensor of dimension 1
        :param B: Tensor of dimension 1
        """

        def new_a(x, i):
            return A * laplace.a(x, i)

        new_laplace = Laplace(laplace.dim)
        new_laplace.set_new_basis(
            new_a, laplace.b, laplace.c, laplace.diff_orders, laplace.integral_dims
        )
        new_laplace.set_dims(laplace.dims)
        new_laplace.set_differentiable(laplace)
        new_laplace.set_integrable(laplace)

        return new_laplace

    @staticmethod
    def input_affine(
        laplace, b, c
    ):  # TODO: we are not performing affine transform - just overwriting b and c

        new_laplace = Laplace(laplace.dim)
        new_laplace.set_new_basis(
            laplace.a, b, c, laplace.diff_orders, laplace.integral_dims
        )

        new_laplace.integrable = laplace.integrable
        new_laplace.differentiable = laplace.differentiable

        return new_laplace

    @staticmethod
    def input_warp(laplace, func):
        """
        Composes function to input of basis function :- basis_func(objective(x),i)
        :param laplace:
        :param func: function to be composed
        """

        if laplace.full_dims is True:

            def new_a(x, i):
                return laplace.a(func(x), i)

            def new_basis_func(x, i):
                return laplace.basis_func(func(x), i)

        else:

            def new_a(x, i):
                return laplace.a(func(x).T[[laplace.dims]].T, i)

            def new_basis_func(x, i):
                return laplace.basis_func(func(x).T[[laplace.dims]].T, i)

        new_laplace = Laplace(laplace.dim)

        new_laplace.set_new_basis(
            new_a,
            laplace.b,
            laplace.c,
            laplace.diff_orders,
            laplace.integral_dims,
            new_basis_func,
        )
        new_laplace.set_dims(laplace.dims)
        new_laplace.set_full_dims(laplace.full_dims)

        new_laplace.integrable = False
        new_laplace.differentiable = False

        return new_laplace

    @staticmethod
    def differentiate(laplace, diff_dims, orders):
        """
        Overrides basis_func with differentiated basis
        :param laplace:
        :param diff_dims: diff_dims
        :param orders: derivative orders
        """

        diff_orders = laplace.diff_orders

        relative_diff_dims, relative_diff_orders = laplace.get_relative_dims(
            diff_dims, diff_orders=orders
        )
        for j in range(len(relative_diff_dims)):
            diff_orders[relative_diff_dims[j]] = (
                diff_orders[relative_diff_dims[j]] + relative_diff_orders[j]
            )

        if laplace.differentiable:
            new_laplace = Laplace(laplace.dim)
            new_laplace.set_new_basis(
                laplace.a, laplace.b, laplace.c, diff_orders, laplace.integral_dims
            )
            new_laplace.set_differentiable(
                laplace
            )  # this is unnecessary since laplace is automatically differentiable
            new_laplace.set_integrable(laplace)
            new_laplace.set_dims(laplace.dims)
        else:
            raise NotImplementedError("basis object not differentiable")

        return new_laplace

    @staticmethod
    def integrate(laplace, integral_dims, limits, method=None):

        # TODO: FIX NORMALISATION FOR INTEGRALS
        integral_dims, relative_limits = laplace.get_relative_dims(integral_dims)
        limits = torch.stack(relative_limits)
        integral_dims = (
            laplace.integral_dims + integral_dims
        )  # check for repeated integral_transform dims?
        integral_dims.sort()

        if laplace.integrable:

            new_laplace = Laplace(laplace.dim)
            new_laplace.input_dim = laplace.dim - len(integral_dims)

            new_diff_orders = laplace.diff_orders
            new_diff_orders[integral_dims] = new_diff_orders[integral_dims] - 1

            def integral_const(m, i):
                return torch.prod(
                    new_laplace.integrated_basis_func(limits.T, i)[1]
                    - new_laplace.integrated_basis_func(limits.T, i)[0],
                    dim=1,
                )

            new_laplace.integral_const = integral_const

            new_laplace.set_new_basis(
                laplace.a, laplace.b, laplace.c, new_diff_orders, integral_dims
            )
            new_laplace.set_differentiable(laplace)
            new_laplace.set_integrable(laplace)
            new_laplace.set_dims(laplace.dims)

        else:
            raise NotImplementedError
        # TODO: include method in argument to perform numerical integration for non-integrable bases

        return new_laplace

    @staticmethod
    def line_integral(laplace, r, method, dr=None):
        def laplace_eval(x, m):
            return laplace.eval(r(x), m)

        new_laplace = Laplace(laplace.dim)
        new_laplace.set_new_basis(
            laplace.a,
            laplace.b,
            laplace.c,
            laplace.diff_orders,
            laplace.integral_dims,
            laplace.basis_func,
        )

        if dr is None:

            def integral_const(m, i):
                return method.line_integral_basis(laplace_eval, m)

        else:
            raise NotImplementedError  # TODO: implement line_integral with dr/dx (or r')

        new_laplace.integral_const = integral_const
        new_laplace.fully_integrated = True
        new_laplace.set_dims(laplace.dims)

        return new_laplace

    def set_dims(self, dims):
        self.dims = sorted(dims)

    def set_full_dims(self, full_dims):
        self.full_dims = full_dims

    def get_relative_dims(self, new_dims, diff_orders=None, integral_limits=None):
        relative_dims = []

        if diff_orders is not None:
            relative_diff_orders = []
        if integral_limits is not None:
            relative_integral_limits = []

        for i in range(self.dim):
            if self.dims[i] in new_dims:
                relative_dims.append(i)
                index = new_dims.index(self.dims[i])
                if diff_orders is not None:
                    relative_diff_orders.append(diff_orders[index])
                if integral_limits is not None:
                    relative_integral_limits.append(integral_limits[index])

        if diff_orders is not None:
            return relative_dims, relative_diff_orders
        elif integral_limits is not None:
            return relative_dims, relative_integral_limits
        else:
            return relative_dims
