import numpy as np
import matplotlib.pyplot as plt

from maps.logistic import logistic_map


# ============================================================
# 1. Órbita (serie temporal)
# ============================================================

def orbit_logistic(x0: float, r: float, n_iters: int) -> np.ndarray:
    """
    Genera la órbita del mapa logístico para un r y x0 dados.

    Devuelve un array [x0, x1, ..., x_n_iters].
    """
    xs = np.empty(n_iters + 1, dtype=float)
    x = x0
    xs[0] = x
    for i in range(1, n_iters + 1):
        x = logistic_map(x, r)
        xs[i] = x
    return xs


def plot_orbit_logistic(x0: float, r: float, n_iters: int = 200) -> None:
    """
    Dibuja la serie temporal x_n de la órbita del mapa logístico.
    """
    xs = orbit_logistic(x0, r, n_iters)
    n = np.arange(xs.size)

    plt.figure(figsize=(8, 4))
    plt.plot(n, xs, marker=".", linestyle="-")
    plt.xlabel("n")
    plt.ylabel("x_n")
    plt.title(f"Órbita mapa logístico (r = {r:.3f}, x0 = {x0:.3f})")
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.show()


# ============================================================
# 2. Atractor numérico
# ============================================================

def attractor_points_logistic(
    r: float,
    x0: float = 0.5,
    n_transient: int = 1_000,
    n_points: int = 500,
) -> np.ndarray:
    """
    Extrae puntos del atractor para un valor de r:

    - Primero descarta n_transient iteraciones (transitorio).
    - Luego devuelve n_points puntos de la órbita.
    """
    x = x0

    # Quemamos el transitorio
    for _ in range(n_transient):
        x = logistic_map(x, r)

    xs = np.empty(n_points, dtype=float)
    for i in range(n_points):
        x = logistic_map(x, r)
        xs[i] = x

    return xs


def plot_attractor_logistic(
    r: float,
    x0: float = 0.5,
    n_transient: int = 1_000,
    n_points: int = 500,
) -> None:
    """
    Dibuja un scatter de los últimos n_points de la órbita
    tras descartar el transitorio.
    """
    xs = attractor_points_logistic(
        r=r,
        x0=x0,
        n_transient=n_transient,
        n_points=n_points,
    )
    n = np.arange(xs.size)

    plt.figure(figsize=(8, 4))
    plt.scatter(n, xs, s=5)
    plt.xlabel("n (después del transitorio)")
    plt.ylabel("x_n")
    plt.title(f"Atractor numérico mapa logístico (r = {r:.3f})")
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.show()


# ============================================================
# 3. Diagrama de bifurcación
# ============================================================

def bifurcation_data_logistic(
    r_min: float = 2.5,
    r_max: float = 4.0,
    r_points: int = 2_000,
    x0: float = 0.5,
    n_transient: int = 1_000,
    n_plot: int = 100,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Datos para el diagrama de bifurcación del mapa logístico.

    Para cada r en [r_min, r_max]:
      - descarta n_transient iteraciones;
      - guarda los siguientes n_plot valores.
    """
    rs = np.linspace(r_min, r_max, r_points)
    r_list: list[float] = []
    x_list: list[float] = []

    for r in rs:
        x = x0
        # Transitorio
        for _ in range(n_transient):
            x = logistic_map(x, r)

        # Puntos del atractor para este r
        for _ in range(n_plot):
            x = logistic_map(x, r)
            r_list.append(r)
            x_list.append(x)

    r_vals = np.array(r_list, dtype=float)
    x_vals = np.array(x_list, dtype=float)
    return r_vals, x_vals


def plot_bifurcation_logistic(
    r_min: float = 2.5,
    r_max: float = 4.0,
    r_points: int = 2_000,
    x0: float = 0.5,
    n_transient: int = 1_000,
    n_plot: int = 100,
) -> None:
    """
    Dibuja el diagrama de bifurcación del mapa logístico.
    """
    r_vals, x_vals = bifurcation_data_logistic(
        r_min=r_min,
        r_max=r_max,
        r_points=r_points,
        x0=x0,
        n_transient=n_transient,
        n_plot=n_plot,
    )

    plt.figure(figsize=(8, 6))
    plt.plot(r_vals, x_vals, ",", markersize=0.5)
    plt.xlabel("r")
    plt.ylabel("x_n")
    plt.title("Diagrama de bifurcación – mapa logístico")
    plt.tight_layout()
    plt.show()


# ============================================================
# 4. Exponente de Lyapunov
# ============================================================

def lyapunov_exponent_logistic(
    r: float,
    x0: float = 0.5,
    n_transient: int = 1_000,
    n_iters: int = 5_000,
) -> float:
    """
    Calcula el exponente de Lyapunov aproximado para el mapa logístico:

        λ ≈ (1/N) * Σ log |f'(x_n)|,  con f'(x) = r (1 - 2x)

    Si λ > 0 -> comportamiento caótico.
    """
    x = x0

    # Transitorio
    for _ in range(n_transient):
        x = logistic_map(x, r)

    lyap_sum = 0.0
    for _ in range(n_iters):
        derivative = abs(r * (1.0 - 2.0 * x))
        if derivative == 0.0:
            derivative = 1e-16
        lyap_sum += np.log(derivative)
        x = logistic_map(x, r)

    return lyap_sum / n_iters


def lyapunov_vs_r_logistic(
    r_min: float = 2.5,
    r_max: float = 4.0,
    r_points: int = 300,
    x0: float = 0.5,
    n_transient: int = 1_000,
    n_iters: int = 3_000,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Calcula λ(r) para muchos valores de r.
    """
    rs = np.linspace(r_min, r_max, r_points)
    lambdas = np.empty_like(rs)

    for i, r in enumerate(rs):
        lambdas[i] = lyapunov_exponent_logistic(
            r=r,
            x0=x0,
            n_transient=n_transient,
            n_iters=n_iters,
        )

    return rs, lambdas


def plot_lyapunov_vs_r_logistic(
    r_min: float = 2.5,
    r_max: float = 4.0,
    r_points: int = 300,
    x0: float = 0.5,
    n_transient: int = 1_000,
    n_iters: int = 3_000,
) -> None:
    """
    Dibuja λ(r) para el mapa logístico.
    """
    rs, lambdas = lyapunov_vs_r_logistic(
        r_min=r_min,
        r_max=r_max,
        r_points=r_points,
        x0=x0,
        n_transient=n_transient,
        n_iters=n_iters,
    )

    plt.figure(figsize=(8, 4))
    plt.axhline(0.0, color="black", linewidth=0.8)
    plt.plot(rs, lambdas)
    plt.xlabel("r")
    plt.ylabel("λ(r)")
    plt.title("Exponente de Lyapunov vs r – mapa logístico")
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.show()


# ============================================================
# 5. Exploración del transitorio
# ============================================================

def explore_transient_effect_logistic(
    r: float = 3.99,
    x0: float = 0.2,
    n_total: int = 20_000,
    step: int = 500,
) -> None:
    """
    Estudia cómo cambia la media de x_n según el número de iteraciones
    transitorias descartadas.

    - Genera una órbita larga (n_total).
    - Para distintos T, calcula la media de x_n desde T hasta el final.
    """
    xs = orbit_logistic(x0=x0, r=r, n_iters=n_total)
    Ts = list(range(0, n_total, step))
    means: list[float] = []

    for T in Ts:
        tail = xs[T + 1 :]  # desde T+1 hasta el final
        means.append(tail.mean())

    plt.figure(figsize=(8, 4))
    plt.plot(Ts, means, marker=".")
    plt.xlabel("T (iteraciones descartadas)")
    plt.ylabel("Media de x_n desde T")
    plt.title(f"Efecto del transitorio en la media (r = {r:.3f})")
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.show()


# ============================================================
# 6. Efecto del threshold en el mapa logístico
# ============================================================

def threshold_effect_logistic(
    r: float,
    x0: float = 0.5,
    n_transient: int = 5_000,
    n_points: int = 1_000_000,
    thresholds: list[float] | None = None,
) -> dict[float, float]:
    """
    Calcula la proporción de unos p(1) para varios thresholds en el mapa logístico.

    Para cada threshold t:
      - Genera n_points puntos del atractor (tras descartar n_transient).
      - Convierte a bits con la regla: bit = 1 si x > t, 0 en caso contrario.
      - Devuelve p(1) = #1 / n_points.

    NO dibuja nada, solo devuelve un diccionario {threshold: p_ones}.
    """
    if thresholds is None:
        thresholds = [0.3, 0.5, 0.7]

    xs = attractor_points_logistic(
        r=r,
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


def plot_threshold_effect_logistic(
    r: float,
    x0: float = 0.5,
    n_transient: int = 5_000,
    n_points: int = 1_000_000,
    thresholds: list[float] | None = None,
) -> None:
    """
    Dibuja p(1) en función del threshold para el mapa logístico.
    """
    if thresholds is None:
        thresholds = [0.1, 0.3, 0.5, 0.7, 0.9]

    res = threshold_effect_logistic(
        r=r,
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
    plt.title(f"Efecto del threshold en el mapa logístico (r = {r:.3f})")
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.show()


# ============================================================
# 7. Efectos numéricos de la precisión en coma flotante
# ============================================================

def detect_cycle_logistic(
    r: float,
    x0: float,
    n_iters_max: int = 1_000_000,
) -> tuple[int | None, int | None]:
    """
    Intenta detectar si la órbita del mapa logístico entra en ciclo
    dentro de las primeras n_iters_max iteraciones.

    Devuelve (n0, n1) donde:
      - n0 es el índice de la primera aparición del valor repetido,
      - n1 es el índice donde se repite (periodo ≈ n1 - n0).

    Si no detecta repetición, devuelve (None, None).

    IMPORTANTE: solo mira igualdad EXACTA de coma flotante, así que
    si detecta ciclo es porque realmente la órbita ha cerrado debido
    a la finitud del espacio de estados.
    """
    x = x0
    seen: dict[float, int] = {x: 0}

    for n in range(1, n_iters_max + 1):
        x = logistic_map(x, r)
        if x in seen:
            return seen[x], n
        seen[x] = n

    return None, None


def divergence_two_orbits_logistic(
    r: float,
    x0: float,
    delta0: float = 1e-12,
    n_iters: int = 2_000,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Compara dos órbitas que parten de condiciones iniciales muy cercanas:
      x0 y x0 + delta0.

    Devuelve:
      - n_vals: índices n = 0..n_iters
      - d_vals: |x_n^{(1)} - x_n^{(2)}| en cada paso

    Esto sirve para ilustrar:
      - la sensibilidad a condiciones iniciales (el error crece),
      - y el hecho de que, debido al redondeo finito, puede llegar un
        momento en que ambas órbitas colapsan en el mismo valor.
    """
    xs1 = orbit_logistic(x0=x0, r=r, n_iters=n_iters)
    xs2 = orbit_logistic(x0=x0 + delta0, r=r, n_iters=n_iters)
    d_vals = np.abs(xs1 - xs2)
    n_vals = np.arange(xs1.size, dtype=int)
    return n_vals, d_vals


def plot_divergence_two_orbits_logistic(
    r: float = 3.99,
    x0: float = 0.2,
    delta0: float = 1e-12,
    n_iters: int = 2_000,
    log_scale: bool = True,
) -> None:
    """
    Representa la divergencia de dos órbitas próximas del mapa logístico.

    Si log_scale=True, usa escala semilogarítmica en Y (semilogy),
    lo que permite visualizar mejor la fase de crecimiento exponencial.
    """
    n_vals, d_vals = divergence_two_orbits_logistic(
        r=r,
        x0=x0,
        delta0=delta0,
        n_iters=n_iters,
    )

    plt.figure(figsize=(7, 4))
    if log_scale:
        plt.semilogy(n_vals, d_vals, "-")
        plt.ylabel("|Δx_n| (escala log)")
    else:
        plt.plot(n_vals, d_vals, "-")
        plt.ylabel("|Δx_n|")

    plt.xlabel("n")
    plt.title(
        f"Divergencia de dos órbitas – mapa logístico\n"
        f"(r = {r:.3f}, x0 = {x0:.3f}, delta0 = {delta0:.1e})"
    )
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.show()

# ============================================================
# Anexo – Sensibilidad al parámetro r
# ============================================================

def divergence_two_params_logistic(
    r: float,
    delta_r: float,
    x0: float,
    n_iters: int = 5_000,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Calcula la divergencia entre dos órbitas del mapa logístico con
    el mismo estado inicial pero parámetros ligeramente distintos.

    Órbita 1: parámetro r
    Órbita 2: parámetro r + delta_r

    Devuelve:
      n_vals : array de índices n (0, 1, ..., n_iters)
      d_vals : array con las distancias |x_n^(r) - x_n^(r+Δr)|
    """
    xs1 = orbit_logistic(r=r, x0=x0, n_iters=n_iters)
    xs2 = orbit_logistic(r=r + delta_r, x0=x0, n_iters=n_iters)

    d_vals = np.abs(xs1 - xs2)
    n_vals = np.arange(xs1.size, dtype=int)
    return n_vals, d_vals


def plot_divergence_two_params_logistic(
    r: float = 3.99,
    delta_r: float = 1e-8,
    x0: float = 0.2,
    n_iters: int = 5_000,
    log_scale: bool = True,
) -> None:
    """
    Representa la divergencia de dos órbitas del mapa logístico que
    comparten condición inicial pero difieren ligeramente en el parámetro r.

    Si log_scale=True, se usa escala semilogarítmica en el eje Y (semilogy),
    lo que permite visualizar mejor la fase de crecimiento exponencial.
    """
    n_vals, d_vals = divergence_two_params_logistic(
        r=r,
        delta_r=delta_r,
        x0=x0,
        n_iters=n_iters,
    )

    plt.figure(figsize=(7, 4))
    if log_scale:
        plt.semilogy(n_vals, d_vals, "-")
        plt.ylabel(r"$|\Delta x_n|$ (escala log)")
    else:
        plt.plot(n_vals, d_vals, "-")
        plt.ylabel(r"$|\Delta x_n|$")

    plt.xlabel("n")
    plt.title(
        "Divergencia de órbitas con parámetros próximos – mapa logístico\n"
        f"(r = {r:.8f}, r+Δr = {r+delta_r:.8f}, x0 = {x0:.3f})"
    )
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.show()

