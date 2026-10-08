import numpy as np


def logistic_map(x: float, r: float) -> float:
    """
    Logistic map: x_{n+1} = r * x_n * (1 - x_n)
    """
    return r * x * (1.0 - x)


def orbit_logistic(x0: float, r: float, n_iters: int) -> np.ndarray:
    """
    Generate orbit [x0, x1, ..., x_n_iters].
    """
    xs = np.empty(n_iters + 1, dtype=float)
    xs[0] = x0
    x = x0
    for i in range(1, n_iters + 1):
        x = logistic_map(x, r)
        xs[i] = x
    return xs
