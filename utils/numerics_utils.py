import numpy as np

def float_spacing_around_05() -> dict[str, float]:
    """
    Información sobre la separación entre números float64 alrededor de x = 0.5.
    """
    x = 0.5
    x_next = np.nextafter(x, np.inf)
    x_prev = np.nextafter(x, -np.inf)

    eps_plus = x_next - x       # x_next - x
    eps_minus = x - x_prev      # x - x_prev

    n_vals_interval = 1.0 / eps_plus
    log2_n_vals = np.log2(n_vals_interval)

    return {
        "x": x,
        "x_prev": x_prev,
        "x_next": x_next,
        "eps_plus": eps_plus,
        "eps_minus": eps_minus,
        "n_vals_interval": n_vals_interval,
        "log2_n_vals_interval": log2_n_vals,
    }


def interval_capacity(L: float) -> dict[str, float]:
    """
    Estima cuántos float64 distintos caben en un intervalo de longitud L.
    """
    info = float_spacing_around_05()
    eps = info["eps_plus"]
    N_vals = L / eps
    log2_N_vals = np.log2(N_vals)

    return {
        "L": L,
        "eps": eps,
        "N_vals": N_vals,
        "log2_N_vals": log2_N_vals,
    }


def bits_from_interval(L: float) -> float:
    """
    Bits de entropía aproximados aportados por una variable real en un intervalo de longitud L.
    """
    info = interval_capacity(L)
    return info["log2_N_vals"]
