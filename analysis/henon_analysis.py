import numpy as np
import matplotlib.pyplot as plt

# Utilidades numéricas genéricas (precisión, keyspace, etc.)
from utils.numerics_utils import (
    float_spacing_around_05,
    interval_capacity,
    bits_from_interval,
)

# Para este módulo de análisis, ignoramos avisos de overflow/invalid
# (que simplemente indican que la órbita se ha ido al infinito).
np.seterr(over="ignore", invalid="ignore")


# ============================================================
# Núcleo del mapa de Hénon (versión "limpia" para análisis)
# ============================================================

def henon_step_analysis(x: float, y: float, a: float, b: float) -> tuple[float, float]:
    """
    Un paso del mapa de Hénon clásico:

        x_{n+1} = 1 - a x_n^2 + y_n
        y_{n+1} = b x_n

    Parámetros típicos caóticos: a = 1.4, b = 0.3
    """
    x_next = 1.0 - a * x * x + y
    y_next = b * x
    return x_next, y_next


def _escaped(x: float, y: float, threshold: float = 1e6) -> bool:
    """
    Devuelve True si (x, y) parece haberse ido 'al infinito' o no es finito.
    Se usa solo para cortar las iteraciones en los gráficos de análisis.
    """
    if not np.isfinite(x) or not np.isfinite(y):
        return True
    if abs(x) > threshold or abs(y) > threshold:
        return True
    return False


# ============================================================
# 1. Órbita (serie temporal)
# ============================================================

def orbit_henon(
    x0: float,
    y0: float,
    a: float,
    b: float,
    n_iters: int,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Genera la órbita del mapa de Hénon para (x0, y0, a, b).

    Devuelve dos arrays: xs, ys de longitud n_iters+1.
    """
    xs = np.empty(n_iters + 1, dtype=float)
    ys = np.empty(n_iters + 1, dtype=float)

    x, y = x0, y0
    xs[0], ys[0] = x, y

    for i in range(1, n_iters + 1):
        x, y = henon_step_analysis(x, y, a, b)
        if _escaped(x, y):
            # rellenamos con NaN a partir de aquí
            xs[i:] = np.nan
            ys[i:] = np.nan
            break
        xs[i], ys[i] = x, y

    return xs, ys


def plot_time_series_henon(
    x0: float,
    y0: float,
    a: float,
    b: float,
    n_iters: int = 1_000,
) -> None:
    """
    Dibuja la serie temporal de x_n del mapa de Hénon.
    """
    xs, ys = orbit_henon(x0, y0, a, b, n_iters)
    n = np.arange(xs.size)

    plt.figure(figsize=(10, 4))
    plt.plot(n, xs, lw=0.8)
    plt.xlabel("n")
    plt.ylabel("x_n")
    plt.title(f"Serie temporal x_n – mapa de Hénon (a = {a:.3f}, b = {b:.3f})")
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.show()


# ============================================================
# 2. Atractor numérico (retrato de fase)
# ============================================================

def attractor_points_henon(
    a: float,
    b: float,
    x0: float = 0.0,
    y0: float = 0.0,
    n_transient: int = 1_000,
    n_points: int = 10_000,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Extrae puntos del atractor de Hénon:

    - Primero descarta n_transient iteraciones (transitorio).
    - Luego devuelve n_points puntos de la órbita (x_n, y_n).
    """
    x, y = x0, y0

    # Quemamos transitorio
    for _ in range(n_transient):
        x, y = henon_step_analysis(x, y, a, b)
        if _escaped(x, y):
            break

    xs = np.empty(n_points, dtype=float)
    ys = np.empty(n_points, dtype=float)

    for i in range(n_points):
        x, y = henon_step_analysis(x, y, a, b)
        if _escaped(x, y):
            xs[i:] = np.nan
            ys[i:] = np.nan
            break
        xs[i], ys[i] = x, y

    return xs, ys


def plot_attractor_henon(
    a: float,
    b: float,
    x0: float = 0.0,
    y0: float = 0.0,
    n_transient: int = 1_000,
    n_points: int = 50_000,
    xlim: tuple[float, float] | None = None,
    ylim: tuple[float, float] | None = None,
) -> None:
    """
    Dibuja el retrato de fase (x_n, y_n) del atractor de Hénon.

    Parámetros extra para "zoom":
      - xlim: tupla (xmin, xmax) para limitar el eje X (opcional).
      - ylim: tupla (ymin, ymax) para limitar el eje Y (opcional).

    Si xlim/ylim son None, se usan los límites automáticos de matplotlib.
    """
    xs, ys = attractor_points_henon(
        a=a,
        b=b,
        x0=x0,
        y0=y0,
        n_transient=n_transient,
        n_points=n_points,
    )

    plt.figure(figsize=(6, 6))
    # Muchos puntos pequeños para apreciar bien la estructura fractal
    plt.scatter(xs, ys, s=0.3, alpha=0.6, linewidths=0)

    if xlim is not None:
        plt.xlim(*xlim)
    if ylim is not None:
        plt.ylim(*ylim)

    plt.xlabel("x_n")
    plt.ylabel("y_n")
    plt.title(f"Atractor de Hénon (a = {a:.3f}, b = {b:.3f})")
    plt.grid(True, alpha=0.2)
    plt.tight_layout()
    plt.show()


# ============================================================
# 3. "Diagrama de bifurcación" en función de a (proyección en x)
# ============================================================

def bifurcation_data_henon_a(
    a_min: float = 1.0,
    a_max: float = 1.5,
    a_points: int = 300,
    b: float = 0.3,
    x0: float = 0.0,
    y0: float = 0.0,
    n_transient: int = 1_000,
    n_plot: int = 100,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Genera datos para un diagrama de tipo "bifurcación" del mapa de Hénon
    en función del parámetro a, proyectando sobre la coordenada x.

    Para cada a en [a_min, a_max]:
      - se descartan n_transient iteraciones,
      - se guardan los siguientes n_plot valores de x_n.

    Si la órbita se escapa (overflow), los valores se marcan como NaN y
    simplemente no aparecen en el gráfico.
    """
    a_values = np.linspace(a_min, a_max, a_points)
    a_list: list[float] = []
    x_list: list[float] = []

    for a in a_values:
        x, y = x0, y0

        # Transitorio
        escaped = False
        for _ in range(n_transient):
            x, y = henon_step_analysis(x, y, a, b)
            if _escaped(x, y):
                escaped = True
                break

        if escaped:
            continue  # no añadimos puntos para este a

        # Puntos "en régimen" para este a
        for _ in range(n_plot):
            x, y = henon_step_analysis(x, y, a, b)
            if _escaped(x, y):
                break
            a_list.append(a)
            x_list.append(x)

    return np.array(a_list), np.array(x_list)


def plot_bifurcation_henon_a(
    a_min: float = 1.0,
    a_max: float = 1.5,
    a_points: int = 300,
    b: float = 0.3,
    x0: float = 0.0,
    y0: float = 0.0,
    n_transient: int = 1_000,
    n_plot: int = 100,
    ylim: tuple[float, float] | None = None,
) -> None:
    """
    Dibuja un "diagrama de bifurcación" proyectando x_n frente a a.

    Parámetro extra:
      - ylim: tupla (xmin, xmax) para recortar el rango vertical
        y hacer zoom en la zona más interesante del diagrama.
    """
    a_vals, x_vals = bifurcation_data_henon_a(
        a_min=a_min,
        a_max=a_max,
        a_points=a_points,
        b=b,
        x0=x0,
        y0=y0,
        n_transient=n_transient,
        n_plot=n_plot,
    )

    plt.figure(figsize=(8, 6))
    plt.plot(a_vals, x_vals, ',', markersize=0.5)
    if ylim is not None:
        plt.ylim(*ylim)
    plt.xlabel("a")
    plt.ylabel("x_n")
    plt.title(f"Diagrama tipo bifurcación – mapa de Hénon (b = {b:.3f})")
    plt.tight_layout()
    plt.show()


# ============================================================
# 4. Exponentes de Lyapunov
# ============================================================

def lyapunov_exponents_henon(
    a: float,
    b: float,
    x0: float = 0.0,
    y0: float = 0.0,
    n_transient: int = 1_000,
    n_iters: int = 5_000,
) -> tuple[float, float]:
    """
    Calcula los dos exponentes de Lyapunov del mapa de Hénon
    usando un método de vector tangente.

    Idea:
      - Se propaga un vector v por la jacobiana en cada paso,
      - se normaliza y se acumula log(norm),
      - eso da una aproximación del exponente máximo λ1.
      - Para el mapa de Hénon, det(J) = -b, por lo que:
            λ1 + λ2 = log |b|
        y así obtenemos λ2 = log|b| - λ1.

    Si la órbita se escapa (overflow), se devuelven NaN para ese (a, b).
    """
    # Estado inicial
    x, y = x0, y0

    # Quemamos transitorio
    for _ in range(n_transient):
        x, y = henon_step_analysis(x, y, a, b)
        if _escaped(x, y):
            return np.nan, np.nan

    # Vector tangente inicial
    v = np.array([1.0, 0.0], dtype=float)
    sum_log_norm = 0.0

    for _ in range(n_iters):
        if _escaped(x, y):
            return np.nan, np.nan

        # Jacobiana del mapa de Hénon en (x, y)
        J = np.array([
            [-2.0 * a * x, 1.0],
            [b,          0.0],
        ])

        # Propagamos el vector tangente
        v = J @ v
        norm_v = np.linalg.norm(v)

        # Protección extra: norma no finita o muy pequeña
        if (not np.isfinite(norm_v)) or (norm_v == 0.0):
            return np.nan, np.nan

        sum_log_norm += np.log(norm_v)
        v /= norm_v

        # Siguiente paso del sistema
        x, y = henon_step_analysis(x, y, a, b)

    lambda1 = sum_log_norm / n_iters
    # Propiedad: λ1 + λ2 = log|det(J)| = log|b|
    lambda2 = np.log(abs(b)) - lambda1
    return lambda1, lambda2


def lyapunov_vs_a_henon(
    a_min: float = 1.0,
    a_max: float = 1.5,
    a_points: int = 100,
    b: float = 0.3,
    x0: float = 0.0,
    y0: float = 0.0,
    n_transient: int = 1_000,
    n_iters: int = 5_000,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Calcula λ1(a) (exponente máximo) para muchos valores de a.
    """
    a_values = np.linspace(a_min, a_max, a_points)
    lambda1_vals = np.empty_like(a_values)

    for i, a_val in enumerate(a_values):
        lambda1, lambda2 = lyapunov_exponents_henon(
            a=a_val,
            b=b,
            x0=x0,
            y0=y0,
            n_transient=n_transient,
            n_iters=n_iters,
        )
        lambda1_vals[i] = lambda1

    return a_values, lambda1_vals


def plot_lyapunov_vs_a_henon(
    a_min: float = 1.0,
    a_max: float = 1.5,
    a_points: int = 100,
    b: float = 0.3,
    x0: float = 0.0,
    y0: float = 0.0,
    n_transient: int = 1_000,
    n_iters: int = 5_000,
) -> None:
    """
    Dibuja λ1(a) para el mapa de Hénon.
    """
    a_vals, lambda1_vals = lyapunov_vs_a_henon(
        a_min=a_min,
        a_max=a_max,
        a_points=a_points,
        b=b,
        x0=x0,
        y0=y0,
        n_transient=n_transient,
        n_iters=n_iters,
    )

    plt.figure(figsize=(8, 4))
    plt.axhline(0.0, color="black", linewidth=0.8)
    plt.plot(a_vals, lambda1_vals)
    plt.xlabel("a")
    plt.ylabel("λ₁(a)")
    plt.title(f"Exponente de Lyapunov máximo – mapa de Hénon (b = {b:.3f})")
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.show()


# ============================================================
# 5. Exploración del transitorio 
# ============================================================

def explore_transient_effect_henon(
    a: float = 1.4,
    b: float = 0.3,
    x0: float = 0.0,
    y0: float = 0.0,
    n_total: int = 50_000,
    step: int = 1_000,
) -> None:
    """
    Estudia cómo cambia la media de la norma ||(x_n, y_n)|| según
    el número de iteraciones descartadas T.

    Sirve como guía para elegir un transitorio razonable.
    """
    xs, ys = orbit_henon(x0, y0, a, b, n_total)
    radii = np.sqrt(xs**2 + ys**2)

    Ts = list(range(0, n_total, step))
    means: list[float] = []

    for T in Ts:
        tail = radii[T + 1 :]
        tail = tail[np.isfinite(tail)]
        if tail.size == 0:
            means.append(np.nan)
        else:
            means.append(tail.mean())

    plt.figure(figsize=(8, 4))
    plt.plot(Ts, means, marker=".")
    plt.xlabel("T (iteraciones descartadas)")
    plt.ylabel("Media de ||(x_n, y_n)|| desde T")
    plt.title(f"Efecto del transitorio – mapa de Hénon (a = {a:.3f}, b = {b:.3f})")
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.show()
    
# ============================================================
# 6. Efecto del umbral de binarización en el mapa de Hénon
# ============================================================

def threshold_effect_henon(
    a: float,
    b: float,
    x0: float = 0.0,
    y0: float = 0.0,
    n_transient: int = 5_000,
    n_points: int = 1_000_000,
    thresholds: list[float] | np.ndarray = (0.1, 0.3, 0.5, 0.7, 0.9),
) -> tuple[np.ndarray, np.ndarray]:
    """
    Estudia el efecto del umbral de binarización en el mapa de Hénon.

    Estrategia:
      - Se genera una órbita larga (x_n, y_n) con parámetros (a, b).
      - Se descartan n_transient iteraciones.
      - Sobre los siguientes n_points, se toma la parte fraccionaria de x_n:
            u_n = x_n - floor(x_n)  in [0, 1)
      - Para cada umbral θ en 'thresholds' se calcula la proporción de unos
        al aplicar la regla de binarización:
            bit_n(θ) = 1 si u_n > θ, 0 en otro caso.

    Devuelve:
      - array de thresholds (float)
      - array p_ones, donde p_ones[i] ≈ p(1) para θ = thresholds[i]
    """
    thresholds = np.array(thresholds, dtype=float)

    # Órbita suficientemente larga
    n_total = n_transient + n_points
    xs, ys = orbit_henon(x0, y0, a, b, n_total)

    # Nos quedamos con la parte "en régimen"
    xs = xs[n_transient + 1 :]
    xs = xs[np.isfinite(xs)]
    if xs.size == 0:
        return thresholds, np.full_like(thresholds, np.nan)

    # Parte fraccionaria de x_n, mapeando a [0, 1)
    u = xs - np.floor(xs)

    p_ones = np.empty_like(thresholds)
    for i, theta in enumerate(thresholds):
        bits = u > theta
        p_ones[i] = bits.mean()

    return thresholds, p_ones


def plot_threshold_effect_henon(
    a: float = 1.4,
    b: float = 0.3,
    x0: float = 0.0,
    y0: float = 0.0,
    n_transient: int = 5_000,
    n_points: int = 1_000_000,
    thresholds: list[float] | np.ndarray = (0.1, 0.3, 0.5, 0.7, 0.9),
) -> None:
    """
    Dibuja p(1) en función del umbral de binarización θ para el mapa de Hénon.
    """
    thetas, p_ones = threshold_effect_henon(
        a=a,
        b=b,
        x0=x0,
        y0=y0,
        n_transient=n_transient,
        n_points=n_points,
        thresholds=thresholds,
    )

    plt.figure(figsize=(6, 4))
    plt.plot(thetas, p_ones, marker="o")
    plt.axhline(0.5, linestyle="--", linewidth=1.0)
    plt.xlabel("threshold θ")
    plt.ylabel("p(1)")
    plt.title(f"Efecto del threshold en el mapa de Hénon (a = {a:.3f}, b = {b:.3f})")
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.show()
    
# ============================================================
# 7. Limitaciones numéricas y dinámica en coma flotante (Hénon)
# ============================================================

# Nota: las utilidades genéricas de precisión y keyspace
# (float_spacing_around_05, interval_capacity, bits_from_interval)
# se importan desde analysis.numerics_utils al principio del módulo.


# 7.2 Búsqueda aproximada de ciclos en una órbita larga
# ----------------------------------------------------

def detect_cycle_henon(
    x0: float,
    y0: float,
    a: float,
    b: float,
    n_total: int = 100_000,
    n_transient: int = 10_000,
    max_period: int = 10_000,
    tol: float = 1e-12,
) -> tuple[int | None, int | None]:
    """
    Intenta detectar un ciclo periódico en la órbita del mapa de Hénon.

    Estrategia:
      - Se genera una órbita larga de longitud n_total.
      - Se descartan n_transient iteraciones iniciales (transitorio).
      - Se toma el primer punto tras el transitorio como referencia.
      - Se busca el menor k (1 <= k <= max_period) tal que la órbita
        vuelve a un punto muy cercano (distancia < tol) a ese punto de referencia.

    Parámetros:
      x0, y0 : condición inicial.
      a, b   : parámetros del mapa de Hénon.
      n_total: longitud total de la órbita generada.
      n_transient: iteraciones descartadas como transitorio.
      max_period: máximo periodo que se intenta detectar.
      tol: tolerancia para considerar que dos puntos son "el mismo".

    Devuelve:
      (periodo, indice_global_ref) si se detecta un ciclo,
      (None, None) en caso contrario.

      El índice global ref es el índice (en la órbita completa) del primer
      punto utilizado como referencia (después del transitorio).
    """
    xs, ys = orbit_henon(x0, y0, a, b, n_total)

    # Nos quedamos con la parte después del transitorio
    xs_tail = xs[n_transient:]
    ys_tail = ys[n_transient:]

    # Por seguridad, eliminamos puntos no finitos
    mask = np.isfinite(xs_tail) & np.isfinite(ys_tail)
    xs_tail = xs_tail[mask]
    ys_tail = ys_tail[mask]

    if xs_tail.size == 0:
        return None, None

    x_ref = xs_tail[0]
    y_ref = ys_tail[0]

    ref = np.array([x_ref, y_ref], dtype=float)
    max_k = min(max_period, xs_tail.size - 1)

    for k in range(1, max_k + 1):
        p = np.array([xs_tail[k], ys_tail[k]], dtype=float)
        dist = np.linalg.norm(p - ref)
        if dist < tol:
            # periodo k, y el índice global del punto de referencia
            idx_global_ref = n_transient + np.where(mask)[0][0]
            return k, idx_global_ref

    return None, None


# 7.3 Divergencia de dos órbitas con condiciones iniciales próximas
# ----------------------------------------------------------------

def divergence_two_orbits_henon(
    a: float,
    b: float,
    x0: float,
    y0: float,
    delta0: float = 1e-12,
    n_iters: int = 5_000,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Calcula la divergencia entre dos órbitas del mapa de Hénon con
    condiciones iniciales muy próximas.

    Órbita 1: (x0, y0)
    Órbita 2: (x0 + delta0, y0)

    Devuelve:
      n_vals : array de índices n (0, 1, ..., n_iters-1)
      d_vals : array con las distancias ||(x_n^(1), y_n^(1)) - (x_n^(2), y_n^(2))||
    """
    xs1, ys1 = orbit_henon(x0, y0, a, b, n_iters)
    xs2, ys2 = orbit_henon(x0 + delta0, y0, a, b, n_iters)

    # Distancia euclídea entre las dos órbitas en cada paso
    d_vals = np.sqrt((xs1 - xs2) ** 2 + (ys1 - ys2) ** 2)

    n_vals = np.arange(xs1.size, dtype=int)
    return n_vals, d_vals


def plot_divergence_two_orbits_henon(
    a: float = 1.4,
    b: float = 0.3,
    x0: float = 0.0,
    y0: float = 0.0,
    delta0: float = 1e-12,
    n_iters: int = 5_000,
    log_scale: bool = True,
) -> None:
    """
    Representa la divergencia de dos órbitas próximas del mapa de Hénon.

    Si log_scale=True, se usa escala semilogarítmica en el eje Y (semilogy),
    lo que permite visualizar mejor la fase de crecimiento exponencial.
    """
    n_vals, d_vals = divergence_two_orbits_henon(
        a=a,
        b=b,
        x0=x0,
        y0=y0,
        delta0=delta0,
        n_iters=n_iters,
    )

    plt.figure(figsize=(7, 4))
    if log_scale:
        plt.semilogy(n_vals, d_vals, "-")
        plt.ylabel("||Δ(x_n, y_n)|| (escala log)")
    else:
        plt.plot(n_vals, d_vals, "-")
        plt.ylabel("||Δ(x_n, y_n)||")

    plt.xlabel("n")
    plt.title(
        f"Divergencia de dos órbitas – mapa de Hénon\n"
        f"(a = {a:.3f}, b = {b:.3f}, delta0 = {delta0:.1e})"
    )
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.show()

# ============================================================
# Anexo – Sensibilidad al parámetro a
# ============================================================

def divergence_two_params_henon(
    a: float,
    delta_a: float,
    b: float,
    x0: float,
    y0: float,
    n_iters: int = 5_000,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Calcula la divergencia entre dos órbitas del mapa de Hénon con
    el mismo estado inicial pero parámetros ligeramente distintos.

    Órbita 1: parámetros (a, b)
    Órbita 2: parámetros (a + delta_a, b)

    Devuelve:
      n_vals : array de índices n (0, 1, ..., n_iters-1)
      d_vals : array con las distancias ||(x_n^(a), y_n^(a)) - (x_n^(a+Δa), y_n^(a+Δa))||
    """
    xs1, ys1 = orbit_henon(x0, y0, a, b, n_iters)
    xs2, ys2 = orbit_henon(x0, y0, a + delta_a, b, n_iters)

    d_vals = np.sqrt((xs1 - xs2) ** 2 + (ys1 - ys2) ** 2)
    n_vals = np.arange(xs1.size, dtype=int)
    return n_vals, d_vals


def plot_divergence_two_params_henon(
    a: float = 1.4,
    delta_a: float = 1e-6,
    b: float = 0.3,
    x0: float = 0.0,
    y0: float = 0.0,
    n_iters: int = 5_000,
    log_scale: bool = True,
) -> None:
    """
    Representa la divergencia de dos órbitas del mapa de Hénon que
    comparten condición inicial pero difieren ligeramente en el parámetro a.

    Si log_scale=True, se usa escala semilogarítmica en el eje Y (semilogy),
    lo que permite visualizar mejor la fase de crecimiento exponencial.
    """
    n_vals, d_vals = divergence_two_params_henon(
        a=a,
        delta_a=delta_a,
        b=b,
        x0=x0,
        y0=y0,
        n_iters=n_iters,
    )

    plt.figure(figsize=(7, 4))
    if log_scale:
        plt.semilogy(n_vals, d_vals, "-")
        plt.ylabel(r"$\|\Delta(x_n, y_n)\|$ (escala log)")
    else:
        plt.plot(n_vals, d_vals, "-")
        plt.ylabel(r"$\|\Delta(x_n, y_n)\|$")

    plt.xlabel("n")
    plt.title(
        "Divergencia de órbitas con parámetros próximos – mapa de Hénon\n"
        f"(a = {a:.6f}, a+Δa = {a+delta_a:.6f}, b = {b:.3f})"
    )
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.show()
