from typing import Callable, Protocol
from gaussed.gp.backends.base import Backend
from gaussed.gp.kernels import Kernel
from gaussed.gp.means import MeanFunction


class GP:
    def __init__(self, mean: MeanFunction, kernel: Kernel, backend: Backend):
        self.mean, self.kernel = mean, kernel
        self.backend = backend