import math
import numpy as np


def basic_bit_stats(bits: np.ndarray) -> dict:
    """
    Estadísticas básicas de una secuencia de bits 0/1.

    Devuelve
    --------
    dict con:
      - 'n'  : nº total de bits
      - 'n0' : nº de ceros
      - 'n1' : nº de unos
      - 'p0' : proporción de ceros
      - 'p1' : proporción de unos
    """
    bits = np.asarray(bits, dtype=np.uint8)
    n = bits.size
    n1 = int(bits.sum())
    n0 = int(n - n1)
    p1 = n1 / n if n > 0 else float("nan")
    p0 = n0 / n if n > 0 else float("nan")

    print(f"Total bits: {n}")
    print(f"0s: {n0}")
    print(f"1s: {n1}")
    print(f"p(1): {p1:.4f}")
    print(f"p(0): {p0:.4f}")

    return {"n": n, "n0": n0, "n1": n1, "p0": p0, "p1": p1}


def _normal_cdf(z: float) -> float:
    """
    Función de distribución acumulada de la normal estándar N(0,1).
    """
    return 0.5 * (1.0 + math.erf(z / math.sqrt(2.0)))


def frequency_chi_square_test(bits: np.ndarray) -> dict:
    """
    Test de frecuencia (monobit) para H0: p(1) = 0.5.

    Devuelve
    --------
    dict con:
      - 'n', 'n0', 'n1'
      - 'chi2'
      - 'p_value'  (aprox, usando normal para 1 g.l.)
    """
    bits = np.asarray(bits, dtype=np.uint8)
    n = bits.size

    if n == 0:
        print("Empty sequence; chi-square test not applicable.")
        return {"n": 0, "n0": 0, "n1": 0, "chi2": None, "p_value": None}

    n1 = int(bits.sum())
    n0 = n - n1
    expected = n / 2

    chi2 = (n0 - expected) ** 2 / expected + (n1 - expected) ** 2 / expected

    # Para 1 g.l., chi2 = Z^2 con Z ~ N(0,1)
    z = math.sqrt(chi2)
    p_value = 2.0 * (1.0 - _normal_cdf(z))

    print(f"n: {n}, n0: {n0}, n1: {n1}")
    print(f"chi2: {chi2:.3f}")
    print(f"p-value ≈ {p_value:.4f} (H0: p(1)=0.5)")

    return {
        "n": n,
        "n0": n0,
        "n1": n1,
        "chi2": chi2,
        "p_value": p_value,
    }


def runs_test(bits: np.ndarray) -> dict:
    """
    Test de rachas (Wald–Wolfowitz).

    H0: bits i.i.d. Bernoulli(0.5).

    Devuelve
    --------
    dict con:
      - 'R'       : nº de rachas observado
      - 'E'       : nº de rachas esperado bajo H0
      - 'var'     : varianza bajo H0
      - 'z'       : estadístico z
      - 'p_value' : p-valor bilateral (aprox normal)
    """
    bits = np.asarray(bits, dtype=np.uint8)
    n = bits.size

    if n == 0:
        print("Empty sequence; runs test not applicable.")
        return {"R": None, "E": None, "var": None, "z": None, "p_value": None}

    n1 = int(bits.sum())
    n0 = n - n1
    if n0 == 0 or n1 == 0:
        print("All bits equal; runs test undefined.")
        return {"R": 1, "E": None, "var": None, "z": None, "p_value": None}

    # Contar rachas
    R = 1
    for i in range(1, n):
        if bits[i] != bits[i - 1]:
            R += 1

    # Esperanza y varianza bajo H0
    E = 1 + 2 * n0 * n1 / n
    var = (2 * n0 * n1 * (2 * n0 * n1 - n)) / (n**2 * (n - 1))

    z = (R - E) / math.sqrt(var)
    # p-valor bilateral
    p_value = 2.0 * (1.0 - _normal_cdf(abs(z)))

    print(f"n: {n}, n0: {n0}, n1: {n1}")
    print(f"Runs observed R: {R}")
    print(f"Runs expected E: {E:.2f}")
    print(f"Variance: {var:.2f}")
    print(f"z-stat: {z:.3f}")
    print(f"p-value ≈ {p_value:.4f} (runs test)")

    return {
        "R": R,
        "E": E,
        "var": var,
        "z": z,
        "p_value": p_value,
    }
