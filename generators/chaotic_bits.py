from typing import Iterator
import numpy as np

from maps.logistic import logistic_map
from maps.tent import tent_map
from maps.henon import henon_map


# ============================================================
#  Bitstreams para cada mapa
# ============================================================

def logistic_bit_stream(
    r: float,
    x0: float,
    n_transient: int,
    threshold: float = 0.5,
) -> Iterator[int]:
    """
    Generador infinito de bits a partir del mapa logístico.

    Cada paso:
      - x_{n+1} = r * x_n * (1 - x_n)
      - bit = 1 si x_n > threshold, 0 en caso contrario
    """
    if not (0.0 < x0 < 1.0):
        raise ValueError("x0 must be in (0,1).")
    if not (0.0 < threshold < 1.0):
        raise ValueError("threshold must be in (0,1).")

    x = x0

    # Quemamos transitorio
    for _ in range(n_transient):
        x = logistic_map(x, r)

    while True:
        x = logistic_map(x, r)
        yield 1 if x > threshold else 0


def tent_bit_stream(
    mu: float,
    x0: float,
    n_transient: int,
    threshold: float = 0.5,
) -> Iterator[int]:
    """
    Generador infinito de bits a partir del mapa de tent.

    Cada paso:
      - x_{n+1} = tent_map(x_n, mu)
      - bit = 1 si x_n > threshold, 0 en caso contrario
    """
    if not (0.0 < x0 < 1.0):
        raise ValueError("x0 must be in (0,1).")
    if not (0.0 < threshold < 1.0):
        raise ValueError("threshold must be in (0,1).")

    x = x0

    # Quemamos transitorio
    for _ in range(n_transient):
        x = tent_map(x, mu)

    while True:
        x = tent_map(x, mu)
        yield 1 if x > threshold else 0


def henon_bit_stream(
    a: float,
    b: float,
    x0: float,
    y0: float,
    n_transient: int,
    threshold: float = 0.5,
    coord: str = "x",
) -> Iterator[int]:
    """
    Generador infinito de bits a partir del mapa de Hénon.

    Estrategia:
      - Evolucionamos (x_n, y_n) con el mapa de Hénon.
      - Tomamos una coordenada (por defecto 'x').
      - Aplicamos parte fraccionaria para llevarla a [0,1):
            u = x - floor(x)
      - Generamos bit = 1 si u > threshold, 0 en caso contrario.

    Esto evita problemas de que x,y puedan ser negativos o grandes.
    """
    if coord not in ("x", "y"):
        raise ValueError("coord must be 'x' or 'y'.")
    if not (0.0 < threshold < 1.0):
        raise ValueError("threshold must be in (0,1).")

    x, y = x0, y0

    # Quemamos transitorio
    for _ in range(n_transient):
        x, y = henon_map(x, y, a, b)

    while True:
        x, y = henon_map(x, y, a, b)
        z = x if coord == "x" else y
        u = z - np.floor(z)  # parte fraccionaria en [0,1)
        yield 1 if u > threshold else 0


# ============================================================
#  Utilidad genérica
# ============================================================

def generate_bits_from_stream(stream: Iterator[int], n_bits: int) -> np.ndarray:
    """
    Consume n_bits de cualquier generador de bits y devuelve np.array de 0/1.
    """
    return np.fromiter(
        (next(stream) for _ in range(n_bits)),
        dtype=np.uint8,
        count=n_bits,
    )


def generate_chaotic_bits(
    map_name: str,
    n_bits: int,
    **params,
) -> np.ndarray:
    """
    Interfaz unificada para obtener bits caóticos de distintos mapas.

    map_name:
      - "logistic"
      - "tent"
      - "henon"

    Parámetros esperados en **params**:

    - map_name == "logistic":
        r, x0, n_transient, (optional) threshold

    - map_name == "tent":
        mu, x0, n_transient, (optional) threshold

    - map_name == "henon":
        a, b, x0, y0, n_transient,
        (optional) threshold, (optional) coord ('x' o 'y')
    """
    if map_name == "logistic":
        stream = logistic_bit_stream(
            r=params["r"],
            x0=params["x0"],
            n_transient=params["n_transient"],
            threshold=params.get("threshold", 0.5),
        )
        return generate_bits_from_stream(stream, n_bits)

    if map_name == "tent":
        stream = tent_bit_stream(
            mu=params["mu"],
            x0=params["x0"],
            n_transient=params["n_transient"],
            threshold=params.get("threshold", 0.5),
        )
        return generate_bits_from_stream(stream, n_bits)

    if map_name == "henon":
        stream = henon_bit_stream(
            a=params["a"],
            b=params["b"],
            x0=params["x0"],
            y0=params["y0"],
            n_transient=params["n_transient"],
            threshold=params.get("threshold", 0.5),
            coord=params.get("coord", "x"),
        )
        return generate_bits_from_stream(stream, n_bits)

    raise ValueError(f"Unsupported map: {map_name}")
