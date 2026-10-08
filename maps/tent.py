import numpy as np


def tent_map(x: float, mu: float) -> float:
    """
    Tent map (mapa de la tienda/carpa):

        x_{n+1} = mu * x_n         if x_n < 0.5
                = mu * (1 - x_n)   if x_n >= 0.5

    Parámetros
    ----------
    x : float
        Estado actual (en [0, 1]).
    mu : float
        Parámetro de control (0 < mu <= 2, típico mu≈2 para caos).
    """
    if x < 0.5:
        return mu * x
    else:
        return mu * (1.0 - x)


def orbit_tent(x0: float, mu: float, n_iters: int) -> np.ndarray:
    """
    Genera la órbita del mapa de tent:
        [x0, x1, ..., x_n_iters]
    """
    xs = np.empty(n_iters + 1, dtype=float)
    xs[0] = x0
    x = x0
    for i in range(1, n_iters + 1):
        x = tent_map(x, mu)
        xs[i] = x
    return xs
