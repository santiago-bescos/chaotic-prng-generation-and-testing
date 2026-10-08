# analysis/nist_sts.py
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import re


# =============================================================================
# Modelo de "run" detectado en nist_results/
# =============================================================================

@dataclass(frozen=True)
class NistRun:
    generator: str   # e.g. "logistic", "tent", "henon", "chacha"
    variant: str     # e.g. "raw", "vn", "sha2", "sha3"
    report_path: Path


# =============================================================================
# Detección de ejecuciones y parsing del report final
# =============================================================================

def _find_final_report(run_dir: Path) -> Optional[Path]:
    """
    Dado un directorio de ejecución (experiments_*), intenta localizar el fichero
    finalAnalysisReport.txt dentro de AlgorithmTesting/.
    """
    p = run_dir / "AlgorithmTesting" / "finalAnalysisReport.txt"
    if p.exists():
        return p
    # fallback: por si la estructura difiere
    candidates = list(run_dir.rglob("finalAnalysisReport.txt"))
    return candidates[0] if candidates else None
from pathlib import Path

def resolve_final_analysis_report_path(
    report_path: str | Path,
    *,
    start_dir: str | Path | None = None,
    max_levels_up: int = 8,
) -> Path:
    """
    Resuelve automáticamente la ruta a finalAnalysisReport.
    - Si report_path existe tal cual -> ok.
    - Si no, sube niveles desde start_dir (o cwd) buscando 'nist_results'
      y luego busca 'finalAnalysisReport*' dentro.
    """
    p = Path(report_path)

    # 1) si ya es una ruta absoluta/relativa válida
    if p.exists():
        return p

    # 2) punto de partida para subir
    base = Path(start_dir) if start_dir is not None else Path.cwd()

    # 3) subimos por parents buscando carpeta nist_results
    cur = base.resolve()
    checked = 0
    while True:
        nist_dir = cur / "nist_results"
        if nist_dir.exists() and nist_dir.is_dir():
            # Si report_path parece una subruta útil, probamos a colgarla de nist_results
            candidate = nist_dir / p
            if candidate.exists():
                return candidate

            # fallback: buscar cualquier finalAnalysisReport dentro de nist_results
            hits = list(nist_dir.rglob("finalAnalysisReport*"))
            if not hits:
                raise FileNotFoundError(f"Encuentro nist_results en {nist_dir}, pero no hay finalAnalysisReport* dentro.")
            return hits[0]

        checked += 1
        if checked >= max_levels_up or cur.parent == cur:
            break
        cur = cur.parent

    raise FileNotFoundError(
        f"No pude resolver '{report_path}'. "
        f"Empecé en '{base.resolve()}' y subí {checked} niveles buscando 'nist_results'."
    )



def discover_runs(nist_results_dir: Path) -> List[NistRun]:
    """
    Descubre ejecuciones STS dentro de nist_results_dir.

    Espera carpetas tipo:
      nist_results/experiments_<generator>_<variant>/

    Devuelve una lista de NistRun con generator/variant y ruta al report final.
    """
    nist_results_dir = Path(nist_results_dir)
    if not nist_results_dir.exists():
        raise FileNotFoundError(f"No existe nist_results_dir: {nist_results_dir}")

    runs: List[NistRun] = []
    for d in sorted(nist_results_dir.glob("experiments_*")):
        if not d.is_dir():
            continue

        name = d.name  # experiments_<generator>_<variant>...
        parts = name.split("_")
        if len(parts) < 3:
            continue

        # experiments, <generator>, <variant>, ...
        generator = parts[1].strip().lower()
        variant = parts[2].strip().lower()

        report = _find_final_report(d)
        if report is None:
            continue

        runs.append(NistRun(generator=generator, variant=variant, report_path=report))

    return runs

def parse_final_analysis_report(report_path: Path) -> pd.DataFrame:
    """
    Parsea la tabla "RESULTS FOR THE UNIFORMITY..." del finalAnalysisReport.txt (NIST STS).

    Devuelve un DataFrame con columnas:
      - c1..c10 (int, opcional; si no se pueden parsear quedan NaN)
      - p_value (float o NaN si '----')
      - proportion (str, ej. '10/10')
      - test (str, ej. 'Frequency', 'NonOverlappingTemplate', ...)
    """
    report_path = Path(report_path)
    lines = report_path.read_text(errors="ignore").splitlines()

    # 1) localizar cabecera
    header_idx = None
    for i, line in enumerate(lines):
        if ("P-VALUE" in line) and ("PROPORTION" in line) and ("STATISTICAL TEST" in line):
            header_idx = i
            break
    if header_idx is None:
        raise ValueError(f"No puedo localizar cabecera de tabla en: {report_path}")

    rows = []

    # 2) leer filas de datos
    for line in lines[header_idx + 1 :]:
        s = line.strip()
        if not s:
            continue

        # separadores
        if set(s) <= {"-"}:
            continue

        # texto final típico del report
        if s.startswith("The minimum pass rate") or s.startswith("For further guidelines"):
            break

        parts = s.split()
        if len(parts) < 4:
            continue

        test = parts[-1]

        # proporción: buscar token d/d desde la derecha
        prop_idx = None
        for k in range(len(parts) - 2, -1, -1):
            if re.match(r"^\d+/\d+$", parts[k]):
                prop_idx = k
                break
        if prop_idx is None:
            continue
        proportion = parts[prop_idx]

        # p-value: token inmediatamente a la izquierda de proportion (o el más cercano válido)
        p_tok = None
        for j in range(prop_idx - 1, -1, -1):
            tok = parts[j]
            if tok == "----" or re.match(r"^\d+(\.\d+)?$", tok):
                p_tok = tok
                break
        if p_tok is None:
            continue

        p_value = np.nan if p_tok == "----" else float(p_tok)

        # c1..c10: normalmente son los 10 primeros enteros antes del p-value
        # (si no encaja perfecto, intentamos parsear los primeros 10 tokens como ints)
        c_vals = [np.nan] * 10
        for idx in range(min(10, len(parts))):
            try:
                c_vals[idx] = int(parts[idx])
            except Exception:
                c_vals[idx] = np.nan

        rows.append(
            {
                **{f"c{i+1}": c_vals[i] for i in range(10)},
                "p_value": p_value,
                "proportion": proportion,
                "test": test,
            }
        )

    df = pd.DataFrame(rows)

    # Normalizar tipos
    for i in range(10):
        col = f"c{i+1}"
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")

    if "p_value" in df.columns:
        df["p_value"] = pd.to_numeric(df["p_value"], errors="coerce")

    return df


# =============================================================================
# Criterios: bounds para proportion y flags de paso
# =============================================================================

def nist_proportion_bounds(alpha: float = 0.01, m: int = 10) -> Tuple[float, float]:
    """
    Banda de aceptación aproximada para la proporción de secuencias que pasan.
    En NIST STS se usa típicamente:
      p_hat = 1 - alpha
      sigma = sqrt(p_hat*(1-p_hat)/m)
      bounds = p_hat +/- 3*sigma
    """
    p_hat = 1.0 - float(alpha)
    sigma = np.sqrt(p_hat * (1.0 - p_hat) / float(m))
    return (p_hat - 3.0 * sigma, p_hat + 3.0 * sigma)


def evaluate_run(run: NistRun, alpha: float = 0.01, m: int = 10) -> pd.DataFrame:
    """
    Convierte un NistRun en tabla a nivel test con flags de aprobación.

    Columnas:
      - generator, variant
      - test
      - p_value
      - proportion_num (int)
      - proportion_den (int)
      - pass_uniformity (bool)   p_value >= alpha
      - pass_proportion (bool)   proportion dentro de bounds
      - pass_test (bool)         ambas
      - alpha, m, prop_lo, prop_hi
    """
    df = parse_final_analysis_report(run.report_path).copy()
    df["generator"] = run.generator
    df["variant"] = run.variant
    df["alpha"] = float(alpha)
    df["m"] = int(m)

    # proportion: "k/m"
    k = df["proportion"].astype(str).str.split("/", expand=True)
    df["proportion_num"] = pd.to_numeric(k[0], errors="coerce")
    df["proportion_den"] = pd.to_numeric(k[1], errors="coerce")

    prop_lo, prop_hi = nist_proportion_bounds(alpha=alpha, m=m)
    df["prop_lo"] = prop_lo
    df["prop_hi"] = prop_hi

    df["pass_uniformity"] = df["p_value"] >= float(alpha)
    # si no tenemos num/den, no podemos evaluar proportion
    df["pass_proportion"] = (
        df["proportion_num"].notna()
        & df["proportion_den"].notna()
        & (df["proportion_den"] > 0)
        & ((df["proportion_num"] / df["proportion_den"]) >= prop_lo)
        & ((df["proportion_num"] / df["proportion_den"]) <= prop_hi)
    )

    df["pass_test"] = df["pass_uniformity"] & df["pass_proportion"]
    return df


def build_full_table(runs: Sequence[NistRun], alpha: float = 0.01, m: int = 10) -> pd.DataFrame:
    """
    Construye df_all concatenando evaluate_run() para todas las ejecuciones.
    """
    parts = []
    for r in runs:
        parts.append(evaluate_run(r, alpha=alpha, m=m))
    return pd.concat(parts, ignore_index=True)


def summarize_runs(df_all: pd.DataFrame) -> pd.DataFrame:
    """
    Resume por configuración (generator, variant).

    Nota importante:
    - df_all está a nivel "test", no por bitstream.
    - tests_total = número de filas (tests) para esa config.
    - tests_passed = número de tests que pasan (pass_test True).
    - pass_rate = tests_passed / tests_total.
    """
    g = df_all.groupby(["generator", "variant"], as_index=False)
    out = g.agg(
        tests_total=("test", "size"),
        tests_passed=("pass_test", "sum"),
        worst_p=("p_value", "min"),
    )
    out["pass_rate"] = out["tests_passed"] / out["tests_total"].replace(0, np.nan)

    # Un indicador conservador extra: peor proportion observada (como fracción)
    if "proportion_num" in df_all.columns and "proportion_den" in df_all.columns:
        frac = df_all["proportion_num"] / df_all["proportion_den"].replace(0, np.nan)
        worst_prop = (
            df_all.assign(_prop_frac=frac)
            .groupby(["generator", "variant"], as_index=False)["_prop_frac"]
            .min()
            .rename(columns={"_prop_frac": "worst_prop"})
        )
        out = out.merge(worst_prop, on=["generator", "variant"], how="left")
    else:
        out["worst_prop"] = np.nan

    return out


# =============================================================================
# Helpers: fallos por test y comparativas a nivel test
# =============================================================================

def failures_for(df_all: pd.DataFrame, generator: str, variant: str) -> pd.DataFrame:
    """
    Devuelve subset df_all para una configuración con pass/fail por test.
    """
    generator = str(generator).lower()
    variant = str(variant).lower()
    return df_all[(df_all["generator"].str.lower() == generator) & (df_all["variant"].str.lower() == variant)].copy()


def failed_tests_set(df_all: pd.DataFrame, generator: str, variant: str) -> set:
    """
    Conjunto de tests fallidos (pass_test==False) para una configuración.
    """
    df = failures_for(df_all, generator, variant)
    return set(df.loc[~df["pass_test"].astype(bool), "test"].astype(str).tolist())


def compare_failed_tests(df_all: pd.DataFrame, a: Tuple[str, str], b: Tuple[str, str]) -> Dict[str, set]:
    """
    Compara tests fallidos entre dos configs (generator, variant).
    """
    A = failed_tests_set(df_all, a[0], a[1])
    B = failed_tests_set(df_all, b[0], b[1])
    return {
        "a_failed": A,
        "b_failed": B,
        "common_failed": A & B,
        "only_a_failed": A - B,
        "only_b_failed": B - A,
    }


# =============================================================================
# Familias de tests (agregación)
# =============================================================================

_TEST_FAMILY_MAP = {
    # Nombres típicos STS
    "Frequency": "Frequency",
    "BlockFrequency": "BlockFrequency",
    "CumulativeSums": "CumulativeSums",
    "Runs": "Runs",
    "LongestRun": "LongestRun",
    "Rank": "Rank",
    "FFT": "FFT",
    "NonOverlappingTemplate": "NonOverlappingTemplate",
    "OverlappingTemplate": "OverlappingTemplate",
    "Universal": "Universal",
    "ApproximateEntropy": "ApproximateEntropy",
    "LinearComplexity": "LinearComplexity",
    "Serial": "Serial",
}


def _test_family_name(test: str) -> str:
    t = str(test)
    return _TEST_FAMILY_MAP.get(t, t)


def add_family_columns(df: pd.DataFrame) -> pd.DataFrame:
    """
    Añade columnas:
      - test_family: familia agregada del test (si existe df['test'])
      - kind: 'baseline' si generator == 'chacha', si no 'chaos'

    Nota: df_summary no tiene columna 'test', así que ahí solo añade 'kind'
    y deja 'test_family' como NaN si no aplica.
    """
    df = df.copy()

    if "test" in df.columns:
        df["test_family"] = df["test"].map(_TEST_FAMILY_MAP).fillna(df["test"])
    else:
        df["test_family"] = np.nan

    df["kind"] = np.where(df["generator"].astype(str).str.lower() == "chacha", "baseline", "chaos")
    return df


def fail_rate_by_test_family(df_all: pd.DataFrame, generator: str, variant: str) -> pd.DataFrame:
    """
    Fail-rate por familia para una config:
      fail_rate = (#tests con pass_test==False) / (#tests totales) por familia
    """
    df = failures_for(df_all, generator, variant)
    df = add_family_columns(df)

    g = df.groupby("test_family", as_index=False).agg(
        total=("test", "size"),
        fails=("pass_test", lambda s: (~s.astype(bool)).sum()),
    )
    g["fail_rate"] = g["fails"] / g["total"].replace(0, np.nan)
    return g.sort_values("fail_rate", ascending=False)


# =============================================================================
# Builders para plots (lo que te está faltando en el notebook)
# =============================================================================

def build_failrate_family_matrix(
    df_all: pd.DataFrame,
    variant: str,
    generators: Sequence[str],
) -> pd.DataFrame:
    """
    Devuelve una matriz wide:
      index = test_family
      columns = generator
      values = fail_rate

    Filtra df_all por variant y por generators.
    """
    variant = str(variant).lower()
    gens = [str(g).lower() for g in generators]

    df = df_all.copy()
    df = df[df["variant"].astype(str).str.lower() == variant]
    df = df[df["generator"].astype(str).str.lower().isin(gens)]

    df = add_family_columns(df)

    grp = df.groupby(["test_family", "generator"], as_index=False).agg(
        total=("test", "size"),
        fails=("pass_test", lambda s: (~s.astype(bool)).sum()),
    )
    grp["fail_rate"] = grp["fails"] / grp["total"].replace(0, np.nan)

    M = grp.pivot(index="test_family", columns="generator", values="fail_rate").fillna(0.0)

    cols = [g for g in gens if g in M.columns]
    return M[cols]


# =============================================================================
# Plots (sin defs en el notebook)
# =============================================================================

def plot_pass_rate_by_variant_for_generator(
    df_summary: pd.DataFrame,
    generator: str,
    variants_order: Sequence[str] = ("raw", "vn", "sha2", "sha3"),
    color: str = "0.4",
    figsize: Tuple[int, int] = (6, 4),
    title: Optional[str] = None,
):
    """
    Bar plot de pass_rate por variante para un generator.
    Acepta figsize para que no vuelva a romper el notebook.
    """
    gen = str(generator).lower()
    d = df_summary[df_summary["generator"].astype(str).str.lower() == gen].copy()
    if d.empty:
        print(f"[plot_pass_rate_by_variant_for_generator] No hay filas para generator={generator}")
        return

    # orden de variantes
    d["variant"] = d["variant"].astype(str).str.lower()
    order = [str(v).lower() for v in variants_order]
    d["variant"] = pd.Categorical(d["variant"], categories=order, ordered=True)
    d = d.sort_values("variant")

    fig = plt.figure(figsize=figsize)
    plt.bar(d["variant"].astype(str), d["pass_rate"].astype(float), color=color)
    plt.ylim(0, 1.05)
    plt.ylabel("tests_passed / tests_total")
    plt.title(title or f"NIST STS – pass_rate por variante ({gen})")
    plt.grid(axis="y", alpha=0.3)
    plt.tight_layout()
    plt.show()

import numpy as np
import pandas as pd

def build_failrate_family_matrix(
    df_all: pd.DataFrame,
    variant: str,
    generators: list[str] | None = None,
    *,
    pass_col: str = "pass_test",
    family_col: str = "test_family",
    generator_col: str = "generator",
) -> pd.DataFrame:
    """
    Construye una matriz (DataFrame) con la tasa de fallo por familia de tests.

    Devuelve un DataFrame S tal que:
      - filas   -> familias (family_col)
      - columnas-> generadores/mapas (generator_col)
      - valores -> fail_rate = (#subtests que fallan) / (#subtests totales)

    Requisitos:
      - df_all contiene columnas: variant, generator, test_family, pass_test (bool)
      - pass_col debe ser booleano o convertible a booleano

    Parámetros:
      - variant: 'raw' | 'vn' | 'sha2' | 'sha3'
      - generators: lista opcional para filtrar y forzar orden de columnas
      - pass_col: por defecto 'pass_test'
    """
    if "variant" not in df_all.columns:
        raise KeyError("df_all no tiene columna 'variant'")
    for c in [generator_col, family_col, pass_col]:
        if c not in df_all.columns:
            raise KeyError(f"df_all no tiene columna '{c}'. Columnas: {list(df_all.columns)}")

    df = df_all[df_all["variant"].astype(str) == str(variant)].copy()

    if generators is not None:
        gens_norm = [str(g) for g in generators]
        df = df[df[generator_col].astype(str).isin(gens_norm)].copy()

    # Asegurar booleano
    p = df[pass_col].astype(bool)

    g = (
        df.assign(_pass=p)
          .groupby([family_col, generator_col], as_index=False)
          .agg(
              total=("_pass", "size"),
              fails=("_pass", lambda s: (~s).sum()),
          )
    )
    g["fail_rate"] = g["fails"] / g["total"]

    S = (
        g.pivot(index=family_col, columns=generator_col, values="fail_rate")
         .fillna(0.0)
    )

    # Ordenar columnas si se pidió explícitamente
    if generators is not None:
        # Solo las que existen
        cols = [c for c in gens_norm if c in S.columns]
        S = S[cols]

    return S

def plot_failrate_family_grouped(
    M: pd.DataFrame,
    title: str = "Fail rate por familia (comparativa de mapas)",
    top_k: int = 12,
    figsize: Tuple[int, int] = (18, 5),
    legend_title: str = "mapa",
    title_suffix: str = "",
    cmap_name: Optional[str] = None,
    cmap_range: Tuple[float, float] = (0.35, 0.85),
    colors: Optional[Sequence[str]] = None,
    # --- NUEVO: control fino de layout ---
    bar_width: float = 0.6,        # ancho total del grupo
    group_spacing: float = 1.4,    # separación entre familias
    rotation: int = 45,            # rotación de etiquetas
    tick_fontsize: int = 9,
):
    """
    Barplot agrupado (comparativa de mapas) a partir de una matriz wide M:
      index: test_family
      columns: generators (mapas)
      values: fail_rate
    """

    if M is None or len(M) == 0:
        print("[plot_failrate_family_grouped] Matriz vacía")
        return

    # Top-k por máximo entre columnas
    order = (
        M.max(axis=1)
         .sort_values(ascending=False)
         .head(int(top_k))
         .index
         .tolist()
    )
    S = M.loc[order].copy()

    families = list(S.index.astype(str))
    cols = list(S.columns.astype(str))

    # Centros de cada grupo (con separación configurable)
    x = np.arange(len(families), dtype=float) * float(group_spacing)

    # Ancho de cada barra dentro del grupo
    n = max(len(cols), 1)
    w = float(bar_width) / n

    # Paleta (azules por defecto)
    if cmap_name:
        cm = plt.get_cmap(cmap_name)
        a, b = float(cmap_range[0]), float(cmap_range[1])
        if n == 1:
            palette = [cm((a + b) / 2)]
        else:
            palette = [cm(a + (b - a) * (i / (n - 1))) for i in range(n)]
    elif colors:
        palette = list(colors)
    else:
        palette = ["#08306B", "#2171B5", "#6BAED6", "#BDD7E7"][:n]

    plt.figure(figsize=figsize)

    # Barras agrupadas centradas en x
    for i, col in enumerate(cols):
        offset = (i - (n - 1) / 2) * w
        plt.bar(
            x + offset,
            S[col].values,
            width=w,
            label=col,
            color=palette[i] if i < len(palette) else None,
        )

   
    plt.xticks(x, families, rotation=rotation, ha="right", fontsize=tick_fontsize)

    plt.ylim(0, 1.05)
    plt.ylabel("fail_rate")

    final_title = title + (f" {title_suffix}" if title_suffix else "")
    plt.title(final_title)

    plt.grid(axis="y", alpha=0.3)
    plt.legend(title=legend_title)

    plt.tight_layout()
    plt.gcf().subplots_adjust(bottom=0.28)
    plt.show()


def plot_failrate_family_logistic_sha2_sha3(
    df_all: pd.DataFrame,
    top_k: int = 5,
    title: str = "Fail rate por familia – Logistic (sha2 vs sha3)",
    cmap: Sequence[str] = ("#08306B", "#6BAED6"),
    figsize: Tuple[int, int] = (10, 4),
):
    """
    Caso específico que quieres:
    - SOLO logistic
    - comparar sha2 vs sha3
    - mostrar top_k familias con mayor fail_rate (por max entre sha2/sha3)
    """
    M = build_failrate_family_matrix(df_all=df_all, variant="sha2", generators=["logistic"])
    M2 = build_failrate_family_matrix(df_all=df_all, variant="sha3", generators=["logistic"])

    # matrices son family x ["logistic"]; convertimos a columnas sha2/sha3
    s_sha2 = M["logistic"] if ("logistic" in M.columns) else pd.Series(dtype=float)
    s_sha3 = M2["logistic"] if ("logistic" in M2.columns) else pd.Series(dtype=float)

    U = pd.concat([s_sha2.rename("sha2"), s_sha3.rename("sha3")], axis=1).fillna(0.0)

    if U.empty:
        print("[plot_failrate_family_logistic_sha2_sha3] Sin datos")
        return

    order = U.max(axis=1).sort_values(ascending=False).head(int(top_k)).index.tolist()
    S = U.loc[order].copy()

    x = np.arange(len(S.index))
    w = 0.35

    plt.figure(figsize=figsize)
    plt.bar(x - w/2, S["sha2"].values, width=w, label="sha2", color=cmap[0])
    plt.bar(x + w/2, S["sha3"].values, width=w, label="sha3", color=cmap[1])

    plt.xticks(x, S.index.astype(str), rotation=60, ha="right")
    plt.ylim(0, 1.05)
    plt.ylabel("fail_rate")
    plt.title(title)
    plt.grid(axis="y", alpha=0.3)
    plt.legend(title="hash")
    plt.tight_layout()
    plt.show()
