from __future__ import annotations

import time
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import pandas as pd
from typing import Dict, Tuple, Optional, List
import numpy as np

from generators.chaotic_bits import generate_chaotic_bits
from generators.extractors import von_neumann, hash_extractor

N_TRANSIENT = 20_000
TH = 0.5

MAP_STREAM_CONFIGS = {
    "logistic": [
        dict(r=3.80, x0=0.123456, n_transient=N_TRANSIENT, threshold=TH),
        dict(r=3.85, x0=0.234567, n_transient=N_TRANSIENT, threshold=TH),
        dict(r=3.90, x0=0.345678, n_transient=N_TRANSIENT, threshold=TH),
        dict(r=3.94, x0=0.456789, n_transient=N_TRANSIENT, threshold=TH),
        dict(r=3.97, x0=0.567891, n_transient=N_TRANSIENT, threshold=TH),
    ],
    "tent": [
        dict(mu=1.85, x0=0.123579, n_transient=N_TRANSIENT, threshold=TH),
        dict(mu=1.90, x0=0.246800, n_transient=N_TRANSIENT, threshold=TH),
        dict(mu=1.95, x0=0.357911, n_transient=N_TRANSIENT, threshold=TH),
        dict(mu=1.98, x0=0.468022, n_transient=N_TRANSIENT, threshold=TH),
        dict(mu=1.97, x0=0.579133, n_transient=N_TRANSIENT, threshold=TH),
    ],
    "henon": [
        dict(a=1.36, b=0.30, x0=0.0, y0=0.0, n_transient=N_TRANSIENT, threshold=TH, coord="x"),
        dict(a=1.38, b=0.30, x0=0.01, y0=0.0, n_transient=N_TRANSIENT, threshold=TH, coord="x"),
        dict(a=1.40, b=0.30, x0=0.02, y0=0.0, n_transient=N_TRANSIENT, threshold=TH, coord="x"),
        dict(a=1.42, b=0.30, x0=0.03, y0=0.0, n_transient=N_TRANSIENT, threshold=TH, coord="x"),
        dict(a=1.37, b=0.30, x0=0.05, y0=0.0, n_transient=N_TRANSIENT, threshold=TH, coord="x"),
    ],
}

def time_raw(map_name: str, params: Dict[str, Any], n_bits: int = 1_000_000) -> Tuple[int, float]:
    t0 = time.perf_counter()
    bits = generate_chaotic_bits(map_name, n_bits=n_bits, **params)
    t1 = time.perf_counter()
    return int(bits.size), (t1 - t0)


def time_vn_until(
    map_name: str,
    params: Dict[str, Any],
    n_bits_out: int = 1_000_000,
    chunk_raw: int = 2_000_000,
    max_rounds: int = 50,
) -> Tuple[int, int, float]:
    """
    Genera bits crudos en chunks, aplica Von Neumann y concatena hasta alcanzar n_bits_out.

    Esto evita depender de un raw_factor fijo (VN puede descartar demasiados pares y quedarse corto).
    Devuelve:
      - n_bits_obtenidos (debería ser n_bits_out salvo que max_rounds sea insuficiente)
      - rounds (nº de iteraciones/chunks usados)
      - segundos (tiempo total)

    Parámetros:
      - chunk_raw: nº de bits raw generados por ronda.
      - max_rounds: límite de rondas para evitar bucles infinitos.
    """
    t0 = time.perf_counter()

    total = 0
    rounds = 0

    # Guardamos solo lo necesario para completar; así evitamos memoria extra
    out_parts = []

    while total < n_bits_out and rounds < max_rounds:
        rounds += 1

        raw = generate_chaotic_bits(map_name, n_bits=chunk_raw, **params)
        vn = von_neumann(raw)

        if vn.size == 0:
            continue

        need = n_bits_out - total
        out_parts.append(vn[:need])
        total += int(min(vn.size, need))

    t1 = time.perf_counter()

    # No necesitamos devolver el array final; solo tamaño y tiempo.
    return int(total), int(rounds), (t1 - t0)



def time_hash(
    map_name: str,
    params: Dict[str, Any],
    n_bits_out: int = 1_000_000,
    hash_name: str = "sha256",
    block_in_bits: int = 1024,
) -> Tuple[int, float]:
    t0 = time.perf_counter()
    raw = generate_chaotic_bits(map_name, n_bits=n_bits_out, **params)
    out = hash_extractor(raw, n_bits_out=n_bits_out, hash_name=hash_name, block_in_bits=block_in_bits)
    t1 = time.perf_counter()
    return int(out.size), (t1 - t0)


def plot_tradeoff(
    df_eval: pd.DataFrame,
    label_mode: str = "hash_and_baseline",
    offsets: Optional[Dict[Tuple[str, str], Tuple[float, float]]] = None,
) -> None:
    """
    Scatter: seconds vs pass_rate.
    Cada punto representa una configuración (generator, variant).

    label_mode:
      - "none"
      - "all"
      - "hash_and_baseline" (recomendado): etiqueta sha2/sha3 y chacha/raw
      - "baseline_only"
    offsets: {(generator, variant): (dx, dy)} para evitar solapes puntuales.
    """
    df = df_eval.copy().dropna(subset=["seconds", "pass_rate"])

    variant_order = ["raw", "vn", "sha2", "sha3"]
    df["variant"] = pd.Categorical(df["variant"], categories=variant_order, ordered=True)
    df = df.sort_values(["variant", "generator"])

    plt.figure()

    for v, sub in df.groupby("variant", dropna=False, observed=False):
        if sub.empty:
            continue

        is_chacha = (sub["generator"] == "chacha") & (sub["variant"] == "raw")
        if is_chacha.any():
            sub_base = sub[is_chacha]
            sub_rest = sub[~is_chacha]
            if not sub_rest.empty:
                plt.scatter(sub_rest["seconds"], sub_rest["pass_rate"], label=str(v), alpha=0.9)
            plt.scatter(
                sub_base["seconds"],
                sub_base["pass_rate"],
                label=f"{v} (baseline)",
                alpha=1.0,
                s=140,
                edgecolors="black",
                linewidths=1.5,
            )
        else:
            plt.scatter(sub["seconds"], sub["pass_rate"], label=str(v), alpha=0.9)

    def want_label(gen: str, var: str) -> bool:
        if label_mode == "none":
            return False
        if label_mode == "all":
            return True
        if label_mode == "baseline_only":
            return gen == "chacha" and var == "raw"
        # hash_and_baseline
        if gen == "chacha" and var == "raw":
            return True
        return var in ("sha2", "sha3")

    offsets = offsets or {}
    for _, r in df.iterrows():
        gen = str(r["generator"])
        var = str(r["variant"])
        if not want_label(gen, var):
            continue
        dx, dy = offsets.get((gen, var), (0.03, 0.02))
        plt.text(r["seconds"] + dx, r["pass_rate"] + dy, gen, fontsize=9)

    plt.title("Trade-off: rendimiento vs calidad")
    plt.xlabel("segundos (1e6 bits)")
    plt.ylabel("pass_rate (NIST STS)")
    plt.ylim(-0.02, 1.05)
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.show()


def build_eval_table(df_summary: pd.DataFrame, df_bench: pd.DataFrame) -> pd.DataFrame:
    """
    Une la tabla resumen de NIST (df_summary) con el benchmark (df_bench) mediante (generator, variant).
    Devuelve una tabla ordenada lista para exportar/plotear.
    """
    out = (
        df_summary.merge(df_bench[["generator", "variant", "seconds"]], on=["generator", "variant"], how="left")
                 .sort_values(["generator", "variant"])
                 .reset_index(drop=True)
    )
    return out
    
import time
from typing import Tuple
import secrets

def time_chacha(n_bits_out: int = 1_000_000) -> Tuple[int, float]:
    """
    Mide tiempo de generar n_bits_out usando ChaCha20 (cryptography).
    Se mide únicamente la generación de keystream (no conversión a ASCII).
    Devuelve (n_bits_generados, segundos).
    """
    from cryptography.hazmat.primitives.ciphers import Cipher, algorithms
    from cryptography.hazmat.backends import default_backend

    n_bytes = n_bits_out // 8
    key = secrets.token_bytes(32)
    nonce = secrets.token_bytes(16)  # cryptography ChaCha20 requiere 16 bytes

    t0 = time.perf_counter()
    algo = algorithms.ChaCha20(key, nonce)
    enc = Cipher(algo, mode=None, backend=default_backend()).encryptor()
    out = enc.update(b"\x00" * n_bytes) + enc.finalize()
    t1 = time.perf_counter()

    return int(len(out) * 8), (t1 - t0)


def plot_tradeoff_zoom_hashes(
    df_eval: pd.DataFrame,
    min_y: float = 0.80,
    title: str = "Trade-off (zoom): hashes vs baseline ChaCha20",
) -> pd.DataFrame:
    """
    Plot trade-off (seconds vs pass_rate) centrado en configuraciones candidatas:
      - sha2/sha3 de los mapas caóticos
      - chacha/raw como baseline

    Estética:
      - Color = generador
      - Marker = variante
      - ChaCha resaltado con borde negro y tamaño mayor
      - Dos leyendas: generador y variante
      - Jitter visual en X para evitar solapes (sin modificar datos reales)

    Devuelve df_zoom (filtrado) por si quieres mostrarlo/guardarlo.
    """
    # Filtrar candidatos: hashes + baseline
    df_zoom = df_eval[
        (df_eval["variant"].isin(["sha2", "sha3"])) |
        ((df_eval["generator"] == "chacha") & (df_eval["variant"] == "raw"))
    ].dropna(subset=["seconds", "pass_rate"]).copy()

    # Orden útil (calidad desc, tiempo asc)
    df_zoom = df_zoom.sort_values(["pass_rate", "seconds"], ascending=[False, True]).reset_index(drop=True)

    # Columnas para plot (permiten jitter sin alterar datos originales)
    df_zoom["seconds_plot"] = df_zoom["seconds"].astype(float)
    df_zoom["pass_plot"] = df_zoom["pass_rate"].astype(float)

    # ---- Jitter visual: separa puntos casi idénticos para que no se tapen ----
    # Agrupamos por (pass_rate redondeado, seconds redondeado) y aplicamos offsets si hay colisiones
    # seconds: redondeo a 3 decimales ≈ milisegundos; suficiente para detectar solapes como tent sha2/sha3
    key = list(zip(df_zoom["pass_plot"].round(6), df_zoom["seconds_plot"].round(3)))
    groups = {}
    for i, k in enumerate(key):
        groups.setdefault(k, []).append(i)

    for _, idxs in groups.items():
        if len(idxs) <= 1:
            continue
        # offsets simétricos: -0.02, 0, +0.02 (o más si hay más puntos)
        center = (len(idxs) - 1) / 2.0
        for j, idx in enumerate(idxs):
            shift = (j - center) * 0.02  # 0.02s = 20ms (solo visual)
            df_zoom.loc[idx, "seconds_plot"] = df_zoom.loc[idx, "seconds_plot"] + shift
    # ------------------------------------------------------------------------

    # Mapeos de marcador por variante
    marker_map = {"raw": "o", "sha2": "s", "sha3": "^"}

    # Colores automáticos por generador (sin fijarlos manualmente)
    gens = df_zoom["generator"].astype(str).unique().tolist()
    cmap = plt.get_cmap("tab10")
    color_map = {g: cmap(i % 10) for i, g in enumerate(gens)}

    fig, ax = plt.subplots()

    # Pintar puntos
    for _, r in df_zoom.iterrows():
        g = str(r["generator"])
        v = str(r["variant"])
        x = float(r["seconds_plot"])
        y = float(r["pass_plot"])

        is_chacha = (g == "chacha") and (v == "raw")

        ax.scatter(
            x, y,
            marker=marker_map.get(v, "o"),
            color=color_map[g],
            s=140 if is_chacha else 60,
            edgecolors="black" if is_chacha else "none",
            linewidths=2.0 if is_chacha else 0.0,
            alpha=0.95,
            zorder=3 if is_chacha else 2
        )

    ax.set_title(title)
    ax.set_xlabel("segundos (1e6 bits)")
    ax.set_ylabel("pass_rate (NIST STS)")
    ax.set_ylim(min_y, 1.05)
    ax.set_xlim(left=0)
    ax.grid(True, alpha=0.3)

    # Leyenda 1: generadores (colores)
    gen_handles = [
        Line2D([0], [0], marker="o", color="none",
               markerfacecolor=color_map[g], markersize=8, label=g)
        for g in gens
    ]
    leg1 = ax.legend(handles=gen_handles, title="generador", loc="lower left")
    ax.add_artist(leg1)

    # Leyenda 2: variantes (marcadores)
    present_vars = [vv for vv in ["raw", "sha2", "sha3"] if vv in set(df_zoom["variant"].astype(str))]
    var_handles = [
        Line2D([0], [0], marker=marker_map[vv], color="black",
               linestyle="none", markersize=8, label=vv)
        for vv in present_vars
    ]
    ax.legend(handles=var_handles, title="variante", loc="lower right")

    plt.show()

    # Devuelve el df sin perder las columnas reales, pero incluyendo *_plot por si quieres debug
    return df_zoom
