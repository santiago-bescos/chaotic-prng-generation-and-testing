import numpy as np


def henon_map(x: float, y: float, a: float, b: float) -> tuple[float, float]:
    """
    Mapa de Hénon:

        x_{n+1} = 1 - a * x_n^2 + y_n
        y_{n+1} = b * x_n

    Parámetros típicos caóticos:
      a = 1.4, b = 0.3
    """
    x_next = 1.0 - a * x * x + y
    y_next = b * x
    return x_next, y_next


def orbit_henon(
    x0: float,
    y0: float,
    a: float,
    b: float,
    n_iters: int,
) -> np.ndarray:
    """
    Genera la órbita del mapa de Hénon:
        [(x0, y0), (x1, y1), ..., (x_n, y_n)]

    Devuelve
    --------
    xs : np.ndarray de shape (n_iters+1, 2)
    """
    xs = np.empty((n_iters + 1, 2), dtype=float)
    x, y = x0, y0
    xs[0, 0] = x
    xs[0, 1] = y

    for i in range(1, n_iters + 1):
        x, y = henon_map(x, y, a, b)
        xs[i, 0] = x
        xs[i, 1] = y

    return xs
