import numpy as np
import matplotlib.pyplot as plt

from maps.tent import tent_map  # asumimos que ya tienes maps/tent.py


# ============================================================
# 1. Órbita (serie temporal)
# ============================================================

def orbit_tent(x0: float, a: float, n_iters: int) -> np.ndarray:
    """
    Genera la órbita del mapa tent para un parámetro a y x0 dados.

    Devuelve un array [x0, x1, ..., x_n_iters].

    Mapa tent típico:
        x_{n+1} = a x_n            si x_n < 0.5
        x_{n+1} = a (1 - x_n)      si x_n >= 0.5
    con 0 < a <= 2.
    """
    xs = np.empty(n_iters + 1, dtype=float)
    x = x0
    xs[0] = x
    for i in range(1, n_iters + 1):
        x = tent_map(x, a)
        xs[i] = x
    return xs


def plot_orbit_tent(x0: float, a: float, n_iters: int = 200) -> None:
    """
    Dibuja la serie temporal x_n de la órbita del mapa tent.
    """
    xs = orbit_tent(x0, a, n_iters)
    n = np.arange(xs.size)

    plt.figure(figsize=(8, 4))
    plt.plot(n, xs, marker=".", linestyle="-")
    plt.xlabel("n")
    plt.ylabel("x_n")
    plt.title(f"Órbita mapa tent (a = {a:.3f}, x0 = {x0:.3f})")
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.show()


# ============================================================
# 2. Atractor numérico
# ============================================================

def attractor_points_tent(
    a: float,
    x0: float = 0.5,
    n_transient: int = 1000,
    n_points: int = 500,
) -> np.ndarray:
    """
    Extrae puntos del atractor para un valor de a:

    - Primero descarta n_transient iteraciones (transitorio).
    - Luego devuelve n_points puntos de la órbita.
    """
    x = x0

    # Quemamos el transitorio
    for _ in range(n_transient):
        x = tent_map(x, a)

    xs = np.empty(n_points, dtype=float)
    for i in range(n_points):
        x = tent_map(x, a)
        xs[i] = x

    return xs


def plot_attractor_tent(
    a: float,
    x0: float = 0.5,
    n_transient: int = 1000,
    n_points: int = 500,
) -> None:
    """
    Dibuja un scatter de los últimos n_points de la órbita
    tras descartar el transitorio.
    """
    xs = attractor_points_tent(
        a=a,
        x0=x0,
        n_transient=n_transient,
        n_points=n_points,
    )
    n = np.arange(xs.size)

    plt.figure(figsize=(8, 4))
    plt.scatter(n, xs, s=5)
    plt.xlabel("n (después del transitorio)")
    plt.ylabel("x_n")
    plt.title(f"Atractor numérico mapa tent (a = {a:.3f})")
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.show()


# ============================================================
# 3. Diagrama de bifurcación
# ============================================================

def bifurcation_data_tent(
    a_min: float = 0.5,
    a_max: float = 2.0,
    a_points: int = 1000,
    x0: float = 0.5,
    n_transient: int = 1000,
    n_plot: int = 80,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Datos para el diagrama de bifurcación del mapa tent.

    Para cada a en [a_min, a_max]:
      - descarta n_transient iteraciones;
      - guarda los siguientes n_plot valores.
    """
    as_ = np.linspace(a_min, a_max, a_points)
    a_list: list[float] = []
    x_list: list[float] = []

    for a in as_:
        x = x0
        # Transitorio
        for _ in range(n_transient):
            x = tent_map(x, a)

        # Puntos del atractor para este a
        for _ in range(n_plot):
            x = tent_map(x, a)
            a_list.append(a)
            x_list.append(x)

    a_vals = np.array(a_list, dtype=float)
    x_vals = np.array(x_list, dtype=float)
    return a_vals, x_vals


def plot_bifurcation_tent(
    a_min: float = 0.5,
    a_max: float = 2.0,
    a_points: int = 1000,
    x0: float = 0.5,
    n_transient: int = 1000,
    n_plot: int = 80,
) -> None:
    """
    Dibuja el diagrama de bifurcación del mapa tent.
    """
    a_vals, x_vals = bifurcation_data_tent(
        a_min=a_min,
        a_max=a_max,
        a_points=a_points,
        x0=x0,
        n_transient=n_transient,
        n_plot=n_plot,
    )

    plt.figure(figsize=(8, 6))
    plt.plot(a_vals, x_vals, ",", markersize=0.5)
    plt.xlabel("a")
    plt.ylabel("x")
    plt.title("Diagrama de bifurcación – mapa tent")
    plt.tight_layout()
    plt.show()


# ============================================================
# 4. Exponente de Lyapunov
# ============================================================

def lyapunov_exponent_tent(
    a: float,
    x0: float = 0.5,
    n_transient: int = 1000,
    n_iters: int = 5000,
) -> float:
    """
    Calcula el exponente de Lyapunov aproximado para el mapa tent.

    Para el mapa tent clásico, el módulo de la derivada es |a| casi en todas
    partes, por lo que teóricamente lambda = log(|a|). Aquí se sigue el mismo
    esquema numérico que en el caso logístico para mantener la simetría.
    """
    x = x0

    # Transitorio
    for _ in range(n_transient):
        x = tent_map(x, a)

    lyap_sum = 0.0
    abs_slope = abs(a)
    if abs_slope == 0.0:
        abs_slope = 1e-16

    for _ in range(n_iters):
        lyap_sum += np.log(abs_slope)
        x = tent_map(x, a)

    return lyap_sum / n_iters


def lyapunov_vs_a_tent(
    a_min: float = 0.5,
    a_max: float = 2.0,
    a_points: int = 200,
    x0: float = 0.5,
    n_transient: int = 1000,
    n_iters: int = 3000,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Calcula lambda(a) para muchos valores de a.
    """
    as_ = np.linspace(a_min, a_max, a_points)
    lambdas = np.empty_like(as_)

    for i, a in enumerate(as_):
        lambdas[i] = lyapunov_exponent_tent(
            a=a,
            x0=x0,
            n_transient=n_transient,
            n_iters=n_iters,
        )

    return as_, lambdas


def plot_lyapunov_vs_a_tent(
    a_min: float = 0.5,
    a_max: float = 2.0,
    a_points: int = 200,
    x0: float = 0.5,
    n_transient: int = 1000,
    n_iters: int = 3000,
) -> None:
    """
    Dibuja lambda(a) para el mapa tent.
    """
    as_, lambdas = lyapunov_vs_a_tent(
        a_min=a_min,
        a_max=a_max,
        a_points=a_points,
        x0=x0,
        n_transient=n_transient,
        n_iters=n_iters,
    )

    plt.figure(figsize=(8, 4))
    plt.axhline(0.0, color="black", linewidth=0.8)
    plt.plot(as_, lambdas)
    plt.xlabel("a")
    plt.ylabel("lambda(a)")
    plt.title("Exponente de Lyapunov vs a – mapa tent")
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.show()


# ============================================================
# 5. Exploración del transitorio
# ============================================================

def explore_transient_effect_tent(
    a: float = 2.0,
    x0: float = 0.2,
    n_total: int = 20_000,
    step: int = 500,
) -> None:
    """
    Estudia cómo cambia la media de x_n según el número de iteraciones
    transitorias descartadas (igual que en el caso logístico).
    """
    xs = orbit_tent(x0=x0, a=a, n_iters=n_total)
    Ts = list(range(0, n_total, step))
    means: list[float] = []

    for T in Ts:
        tail = xs[T + 1 :]
        means.append(tail.mean())

    plt.figure(figsize=(8, 4))
    plt.plot(Ts, means, marker=".")
    plt.xlabel("T (iteraciones descartadas)")
    plt.ylabel("Media de x_n desde T")
    plt.title(f"Efecto del transitorio en la media (a = {a:.3f})")
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.show()


# ============================================================
# 6. Efecto del umbral de binarización en el mapa tent
# ============================================================

def threshold_effect_tent(
    a: float,
    x0: float = 0.5,
    n_transient: int = 5000,
    n_points: int = 1_000_000,
    thresholds: list[float] | None = None,
) -> dict[float, float]:
    """
    Estudia el efecto del umbral de binarización en el mapa tent.

    Para cada threshold t:
      - Genera n_points puntos del atractor (tras descartar n_transient).
      - Convierte a bits con la regla: bit = 1 si x > t, 0 en caso contrario.
      - Devuelve p(1) = numero_de_unos / n_points.

    No dibuja nada, solo devuelve un diccionario {threshold: p_ones}.
    """
    if thresholds is None:
        thresholds = [0.1, 0.3, 0.5, 0.7, 0.9]

    xs = attractor_points_tent(
        a=a,
        x0=x0,
        n_transient=n_transient,
        n_points=n_points,
    )

    results: dict[float, float] = {}
    for th in thresholds:
        bits = (xs > th).astype(np.uint8)
        p_ones = float(bits.mean())
        results[th] = p_ones

    return results


def plot_threshold_effect_tent(
    a: float,
    x0: float = 0.5,
    n_transient: int = 5000,
    n_points: int = 1_000_000,
    thresholds: list[float] | None = None,
) -> None:
    """
    Dibuja p(1) en función del threshold para el mapa tent.
    """
    res = threshold_effect_tent(
        a=a,
        x0=x0,
        n_transient=n_transient,
        n_points=n_points,
        thresholds=thresholds,
    )

    ts = np.array(sorted(res.keys()))
    p1 = np.array([res[t] for t in ts])

    plt.figure(figsize=(6, 4))
    plt.plot(ts, p1, marker="o")
    plt.axhline(0.5, linestyle="--", alpha=0.5)
    plt.xlabel("threshold")
    plt.ylabel("p(1)")
    plt.title(f"Efecto del threshold en el mapa tent (a = {a:.3f})")
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.show()


# ============================================================
# 7. Efectos numéricos de la precisión en coma flotante (tent)
# ============================================================

def detect_cycle_tent(
    a: float,
    x0: float,
    n_iters_max: int = 1_000_000,
) -> tuple[int | None, int | None]:
    """
    Intenta detectar si la órbita del mapa tent entra en ciclo
    dentro de las primeras n_iters_max iteraciones.

    Devuelve (n0, n1) donde:
      - n0 es el índice de la primera aparición del valor repetido,
      - n1 es el índice donde se repite.
    Si no detecta repetición, devuelve (None, None).

    Igual que en el caso logístico, se mira igualdad exacta de coma flotante,
    de modo que si se detecta ciclo es porque la órbita ha cerrado debido
    a la finitud del espacio de estados.
    """
    x = x0
    seen: dict[float, int] = {x: 0}

    for n in range(1, n_iters_max + 1):
        x = tent_map(x, a)
        if x in seen:
            return seen[x], n
        seen[x] = n

    return None, None


def divergence_two_orbits_tent(
    a: float,
    x0: float,
    delta: float,
    n_iters: int = 2_000,
) -> np.ndarray:
    """
    Compara dos órbitas que parten de condiciones iniciales muy cercanas:
      x0 y x0 + delta.

    Devuelve un array con |x_n^(1) - x_n^(2)| para n = 0..n_iters.

    Sirve para ilustrar:
      - la sensibilidad a condiciones iniciales,
      - y que debido al redondeo finito puede llegar un momento en que
        ambas órbitas colapsan en el mismo valor.
    """
    xs1 = orbit_tent(x0=x0, a=a, n_iters=n_iters)
    xs2 = orbit_tent(x0=x0 + delta, a=a, n_iters=n_iters)
    diffs = np.abs(xs1 - xs2)
    return diffs


def plot_divergence_two_orbits_tent(
    a: float = 2.0,
    x0: float = 0.314159,
    delta: float = 1e-15,
    n_iters: int = 2000,
    log_scale: bool = True,
) -> None:
    """
    Representa gráficamente la divergencia de dos órbitas próximas
    del mapa tent. Por defecto usa escala logarítmica en el eje Y.
    """
    diffs = divergence_two_orbits_tent(
        a=a,
        x0=x0,
        delta=delta,
        n_iters=n_iters,
    )

    n = np.arange(diffs.size)

    plt.figure(figsize=(6, 4))
    if log_scale:
        plt.semilogy(n, diffs + 1e-30)
        plt.ylabel("|x_n^(1) - x_n^(2)| (escala log)")
    else:
        plt.plot(n, diffs)
        plt.ylabel("|x_n^(1) - x_n^(2)|")

    plt.xlabel("n")
    plt.title(
        "Divergencia de dos órbitas cercanas (mapa tent)\n"
        f"a = {a:.3f}, x0 = {x0:.6f}, delta = {delta:.1e}"
    )
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.show()

    # ============================================================
# A. Funciones auxiliares para los apéndices
#    (sensibilidad al parámetro y caso especial a = 2)
# ============================================================

def _tent_step_scalar(x: float, a: float) -> float:
    """
    Un solo paso del mapa tent para un escalar x.

        x_{n+1} = a x_n            si x_n < 1/2
        x_{n+1} = a (1 - x_n)      si x_n >= 1/2
    """
    if x < 0.5:
        return a * x
    else:
        return a * (1.0 - x)


def orbit_tent_scalar(
    x0: float,
    a: float,
    n_iters: int,
) -> np.ndarray:
    """
    Genera la órbita escalar del mapa tent para n_iters iteraciones.

    Devuelve un array xs de longitud n_iters + 1 con x_0, x_1, ..., x_{n_iters}.
    """
    xs = np.empty(n_iters + 1, dtype=float)
    x = x0
    xs[0] = x
    for i in range(1, n_iters + 1):
        x = _tent_step_scalar(x, a)
        xs[i] = x
    return xs


def plot_param_sensitivity_tent(
    a: float = 1.999,
    delta_a: float = 1e-6,
    x0: float = 0.314159,
    n_iters: int = 2_000,
    log_scale: bool = True,
) -> None:
    """
    Representa cómo cambia la órbita del mapa tent al variar ligeramente
    el parámetro a.

    Se comparan dos órbitas:
      - con parámetro a
      - con parámetro a + delta_a

    y se dibuja la diferencia d_n = |x_n(a) - x_n(a + delta_a)|.
    """
    xs_a = orbit_tent_scalar(x0, a, n_iters)
    xs_ap = orbit_tent_scalar(x0, a + delta_a, n_iters)

    diffs = np.abs(xs_a - xs_ap)
    n = np.arange(diffs.size)

    plt.figure(figsize=(6, 4))
    if log_scale:
        plt.semilogy(n, diffs + 1e-30)
        plt.ylabel(r"$d_n = |x_n(a) - x_n(a + \Delta a)|$ (escala log)")
    else:
        plt.plot(n, diffs)
        plt.ylabel(r"$d_n = |x_n(a) - x_n(a + \Delta a)|$")

    plt.xlabel("n")
    plt.title(
        "Sensibilidad al parámetro en el mapa tent\n"
        f"a = {a:.6f}, a + Δa = {a + delta_a:.6f}, x0 = {x0:.6f}"
    )
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.show()


def plot_orbit_a_equals_2_tent(
    x0: float = 0.2,
    n_iters: int = 200,
) -> None:
    """
    Ilustra el comportamiento especial del mapa tent para a = 2.0.

    - Genera la órbita con a = 2.0 desde x0.
    - Busca el primer índice n en el que x_n es (numéricamente) cero.
    - Imprime ese índice y dibuja la serie temporal.
    """
    a = 2.0
    xs = np.empty(n_iters + 1, dtype=float)
    x = x0
    xs[0] = x

    hit_zero_at: int | None = None
    tol = 0.0  # en double, normalmente será exactamente 0.0

    for n in range(1, n_iters + 1):
        x = _tent_step_scalar(x, a)
        xs[n] = x
        if hit_zero_at is None and (x == 0.0 or abs(x) <= tol):
            hit_zero_at = n

    print("Mapa tent con a = 2.0")
    print(f"Condición inicial x0 = {x0}")
    print(f"Primer índice n en el que x_n == 0.0 :", hit_zero_at)

    n_vals = np.arange(xs.size)

    plt.figure(figsize=(7, 3))
    plt.plot(n_vals, xs, marker="o", ms=3)
    plt.xlabel("n")
    plt.ylabel(r"$x_n$")
    plt.title("Órbita del mapa tent con a = 2.0 (doble precisión)")
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.show()

