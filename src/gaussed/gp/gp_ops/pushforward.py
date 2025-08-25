from gaussed.gp.base import GP
from gaussed.gp_ops.base import GPOperator

def pushforward(L: GPOperator, gp: GP) -> GP:
    return GP(
        mean=L.apply_mean(gp.mean),
        kernel=L.apply_kernel(gp.kernel),
        backend=gp.backend,
    )