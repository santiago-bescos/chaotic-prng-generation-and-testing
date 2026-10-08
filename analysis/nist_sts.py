from pathlib import Path
import re
from dataclasses import dataclass
import numpy as np


@dataclass
class NistTestResult:
    test_name: str
    p_value: float | None
    passed: bool
    proportion: str | None
    num_sequences: int | None

_WORST_RE = re.compile(
    r"^\s*worst\s+p\s*=\s*(?P<worst_p>\d+(?:\.\d+)?)\s+worst\s+prob\s*=\s*(?P<worst_prob>\d+(?:\.\d+)?)",
    re.IGNORECASE,
)

def parse_worst_metrics(report_path: Path) -> tuple[float | None, float | None]:
    """
    Extrae 'worst p' y 'worst prob' del finalAnalysisReport.txt si existen.
    Devuelve (worst_p, worst_prob) o (None, None).
    """
    text = Path(report_path).read_text(encoding="utf-8", errors="ignore")
    for line in reversed(text.splitlines()):
        m = _WORST_RE.match(line.strip())
        if m:
            return float(m.group("worst_p")), float(m.group("worst_prob"))
    return None, None


def parse_final_analysis_report(report_path: Path) -> list[NistTestResult]:
    results = []

    with open(report_path, "r", encoding="utf-8", errors="ignore") as f:
        lines = f.readlines()

    for line in lines:
        if "*" not in line:
            continue

        parts = line.split()
        if len(parts) < 12:
            continue

        try:
            test_name = parts[-1]
            p_value = None if parts[-4] == "----" else float(parts[-4])
            proportion = parts[-3]
            passed = "*" not in parts[-4]
            num_sequences = int(proportion.split("/")[1]) if "/" in proportion else None

            results.append(
                NistTestResult(
                    test_name=test_name,
                    p_value=p_value,
                    passed=passed,
                    proportion=proportion,
                    num_sequences=num_sequences,
                )
            )
        except Exception:
            continue

    return results,


def parse_experiment_folder(experiment_path: Path) -> dict:
    report = experiment_path / "AlgorithmTesting" / "finalAnalysisReport.txt"
    worst_p, worst_prob = parse_worst_metrics(report)

    test_results = parse_final_analysis_report(report)

    total = len(test_results)
    passed = sum(1 for t in test_results if t.passed)

    return {
        "experiment": experiment_path.name,
        "total_tests": total,
        "passed_tests": passed,
        "pass_ratio": passed / total if total else 0.0,
        "rows": rows,
        "tests": [],
        "worst_p": worst_p,
        "worst_prob": worst_prob,
    }

def parse_all_experiments(nist_results_root: Path) -> list[dict]:
    experiments = []

    for exp in sorted(nist_results_root.iterdir()):
        if not exp.is_dir():
            continue
        if not exp.name.startswith("experiments_"):
            continue

        experiments.append(parse_experiment_folder(exp))

    return experiments
import pandas as pd

def experiments_to_dataframe(experiments: list[dict]) -> pd.DataFrame:
    rows = []

    for exp in experiments:
        for test in exp["tests"]:
            rows.append({
                "experiment": exp["experiment"],
                "test": test.test_name,
                "p_value": test.p_value,
                "passed": test.passed,
            })

    return pd.DataFrame(rows)

import re

_NIST_TABLE_ROW_RE = re.compile(
    r"""
    ^\s*
    (?:\d+\s+){10}                  # C1..C10
    (?P<pval>----|\d+\.\d+)\s*      # P-VALUE or ----
    (?P<pval_star>\*)?\s+           # optional '*' right after P-VALUE
    (?P<prop>\d+/\d+|----)\s+       # PROPORTION or ----
    (?P<col_star>\*)?\s*            # optional '*' column before test name
    (?P<test>[A-Za-z0-9_]+)\s*$     # TEST NAME
    """,
    re.VERBOSE,
)


def parse_final_analysis_report_table_rows(report_path):
    """
    Devuelve una lista de dicts con filas de la tabla principal del finalAnalysisReport.txt.
    Cada fila corresponde a 1 'test instance' (por ejemplo NonOverlappingTemplate repetido muchas veces).
    """
    rows = []
    text = Path(report_path).read_text(encoding="utf-8", errors="ignore")

    for line in text.splitlines():
        m = _NIST_TABLE_ROW_RE.match(line)
        if not m:
            continue

        pval_raw = m.group("pval")
        star = (m.group("pval_star") is not None) or (m.group("col_star") is not None)
        prop_raw = m.group("prop")
        test = m.group("test")

        p_value = None if pval_raw == "----" else float(pval_raw)

        # proportion as float
        prop = None
        if prop_raw and prop_raw != "----" and "/" in prop_raw:
            a, b = prop_raw.split("/", 1)
            try:
                prop = int(a) / int(b)
            except Exception:
                prop = None

        rows.append({
            "test": test,
            "p_value": p_value,
            "passed": (not star),          # '*' => fail
            "proportion": prop,            # float 0..1
            "proportion_raw": None if prop_raw == "----" else prop_raw,
        })

    return rows

from pathlib import Path

def parse_experiment_folder(experiment_path: Path) -> dict:
    """
    Versión compatible:
    - Si existe parse_final_analysis_report_table_rows -> usa 'rows' (lista de dicts)
    - Si no, usa parse_final_analysis_report -> usa 'tests' (lista de NistTestResult)
    Devuelve SIEMPRE ambas claves: 'rows' y 'tests' (cuando aplica).
    """
    report = experiment_path / "AlgorithmTesting" / "finalAnalysisReport.txt"

    # preferimos el parser de tabla si está disponible
    if "parse_final_analysis_report_table_rows" in globals():
        rows = parse_final_analysis_report_table_rows(report)
        total = len(rows)
        passed = sum(1 for r in rows if r.get("passed") is True)

        return {
            "experiment": experiment_path.name,
            "total_tests": total,
            "passed_tests": passed,
            "pass_ratio": passed / total if total else 0.0,
            "rows": rows,
            "tests": [],  # para compatibilidad
        }

    # fallback al parser antiguo
    test_results = parse_final_analysis_report(report)
    total = len(test_results)
    passed = sum(1 for t in test_results if getattr(t, "passed", False))

    return {
        "experiment": experiment_path.name,
        "total_tests": total,
        "passed_tests": passed,
        "pass_ratio": passed / total if total else 0.0,
        "tests": test_results,
        "rows": [],  # para compatibilidad
    }



def experiments_to_dataframe(experiments: list[dict]) -> pd.DataFrame:
    """
    Acepta experiments con:
      - exp["rows"] : lista de dicts (test, p_value, passed, ...)
      - exp["tests"]: lista de NistTestResult (test_name, p_value, passed, ...)
    """
    out = []

    for exp in experiments:
        exp_name = exp.get("experiment", "")

        # Caso nuevo: rows (dicts)
        if exp.get("rows"):
            for r in exp["rows"]:
                out.append({
                    "experiment": exp_name,
                    "test": r.get("test"),
                    "p_value": r.get("p_value"),
                    "passed": r.get("passed"),
                })
            continue

        # Caso antiguo: tests (dataclass)
        if exp.get("tests"):
            for t in exp["tests"]:
                out.append({
                    "experiment": exp_name,
                    "test": getattr(t, "test_name", None),
                    "p_value": getattr(t, "p_value", None),
                    "passed": getattr(t, "passed", None),
                })
            continue

    return pd.DataFrame(out)

def add_test_family_column(df):
    """
    Añade columna 'test_family' al DataFrame en base al nombre del test NIST.
    """
    def family_from_test(test):
        t = str(test)

        if t in {"Frequency", "BlockFrequency", "CumulativeSums"}:
            return "Frequency"

        if t in {"Runs", "LongestRun"}:
            return "Runs"

        if "Template" in t:
            return "Templates"

        if t in {"FFT", "Spectral"}:
            return "Spectral"

        if "Excursions" in t:
            return "RandomWalk"

        if t in {"ApproximateEntropy", "Serial", "LinearComplexity", "Universal", "Rank"}:
            return "Complexity"

        return "Other"

    df = df.copy()
    df["test_family"] = df["test"].apply(family_from_test)
    return df
def build_passrate_family_matrix(df_all, variant, generators):
    """
    Devuelve DataFrame wide:
      index   = test_family
      columns = generator
      values  = pass_rate

    Filtra por variante y mapas dados.
    """
    variant = str(variant).lower()
    gens = [str(g).lower() for g in generators]

    df = df_all.copy()
    df = df[df["variant"].astype(str).str.lower() == variant]
    df = df[df["generator"].astype(str).str.lower().isin(gens)]

    df = add_test_family_column(df)

    grp = (
        df.groupby(["test_family", "generator"])
          .agg(
              total=("passed", "size"),
              passed=("passed", "sum"),
          )
          .reset_index()
    )

    grp["pass_rate"] = grp["passed"] / grp["total"].replace(0, float("nan"))

    M = grp.pivot(
        index="test_family",
        columns="generator",
        values="pass_rate"
    ).fillna(0.0)

    # orden columnas como el usuario pide
    cols = [g for g in gens if g in M.columns]
    return M[cols]

_WORST_RE = re.compile(
    r"^\s*worst\s+p\s*=\s*(?P<worst_p>\d+(?:\.\d+)?)\s+worst\s+prob\s*=\s*(?P<worst_prob>\d+(?:\.\d+)?)",
    re.IGNORECASE,
)

def parse_worst_metrics(report_path: Path) -> tuple[float | None, float | None]:
    """
    Extrae 'worst p' y 'worst prob' del finalAnalysisReport.txt si existen.
    Devuelve (worst_p, worst_prob) o (None, None).
    """
    text = Path(report_path).read_text(encoding="utf-8", errors="ignore")
    for line in text.splitlines()[::-1]:  # buscar desde abajo (suele estar al final)
        m = _WORST_RE.match(line.strip())
        if m:
            return float(m.group("worst_p")), float(m.group("worst_prob"))
    return None, None


def build_experiments_summary_table(
    experiments: list[dict],
    *,
    include_min_prop: bool = True,
    include_worst_prop: bool = True,
    include_worst_p: bool = True,
) -> pd.DataFrame:
    """
    Construye una tabla 'Word-friendly' por experimento (generator, variant) a partir de:
      - experiments: salida de parse_all_experiments()

    Devuelve columnas:
      generator, variant,
      tests_total, tests_passed, pass_rate,
      fail_count, min_p_value,
      (opcional) min_prop,
      (opcional) worst_prop, worst_prop_raw,
      (opcional) worst_p

    Notas:
      - min_p_value se calcula desde df largo (p_value mínimo observado).
      - worst_prop se calcula desde experiment["rows"] (NIST table rows) usando el mínimo proportion.
      - worst_p (si include_worst_p) se calcula desde experiment["rows"] como p_value mínimo NO nulo.
    """
    # 1) df largo
    df = experiments_to_dataframe(experiments)

    # 2) resumen base por experimento (conteos y pass_rate)
    summary = (
        df.groupby("experiment", as_index=False)
          .agg(
              tests_total=("passed", "size"),
              tests_passed=("passed", lambda s: s.astype(bool).sum()),
          )
    )
    summary["pass_rate"] = summary["tests_passed"] / summary["tests_total"].replace(0, np.nan)

    # 3) generator/variant desde nombre experiments_<gen>_<var>
    tmp = summary.copy()
    base = tmp["experiment"].astype(str).str.replace("experiments_", "", regex=False).str.split("_", expand=True)
    tmp["generator"] = base[0]
    tmp["variant"] = base[1]

    # 4) métricas extra desde df: min_p_value, fail_count, (min_prop)
    df_work = df.copy()

    # fail_count / min_p_value
    agg_dict = {
        "min_p_value": ("p_value", "min"),
        "fail_count": ("passed", lambda s: (~s.astype(bool)).sum()),
    }

    # min_prop si existe proportion o proportion_raw
    if include_min_prop:
        prop_float_col: Optional[str] = None

        if "proportion" in df_work.columns:
            prop_float_col = "proportion"
        elif "proportion_raw" in df_work.columns:
            parts = df_work["proportion_raw"].astype(str).str.split("/", expand=True)
            df_work["_prop_float"] = pd.to_numeric(parts[0], errors="coerce") / pd.to_numeric(parts[1], errors="coerce")
            prop_float_col = "_prop_float"

        if prop_float_col is not None:
            agg_dict["min_prop"] = (prop_float_col, "min")

    extra_df = df_work.groupby("experiment", as_index=False).agg(**agg_dict)

    # 5) worst_prop / worst_p desde experiments["rows"]
    rows_out = []
    for e in experiments:
        exp = e.get("experiment")
        rr = e.get("rows", []) or []

        out = {"experiment": exp}

        if include_worst_prop:
            props = [
                r.get("proportion")
                for r in rr
                if isinstance(r.get("proportion"), (int, float)) and not np.isnan(r.get("proportion"))
            ]
            if props:
                worst_prop = float(np.min(props))
                out["worst_prop"] = worst_prop

                raw = None
                for r in rr:
                    if r.get("proportion") == worst_prop:
                        raw = r.get("proportion_raw")
                        break
                out["worst_prop_raw"] = raw
            else:
                out["worst_prop"] = np.nan
                out["worst_prop_raw"] = None

        if include_worst_p:
            pvals = [
                r.get("p_value")
                for r in rr
                if isinstance(r.get("p_value"), (int, float)) and not np.isnan(r.get("p_value"))
            ]
            out["worst_p"] = float(np.min(pvals)) if pvals else np.nan

        rows_out.append(out)

    rows_df = pd.DataFrame(rows_out) if rows_out else pd.DataFrame({"experiment": []})

    # 6) merge final
    tmp2 = (
        tmp.merge(extra_df, on="experiment", how="left")
           .merge(rows_df, on="experiment", how="left")
    )

    # 7) columnas finales
    cols = [
        "generator", "variant",
        "tests_total", "tests_passed", "pass_rate",
        "fail_count", "min_p_value",
    ]
    if include_min_prop and "min_prop" in tmp2.columns:
        cols.append("min_prop")
    if include_worst_p and "worst_p" in tmp2.columns:
        cols.append("worst_p")
    if include_worst_prop and "worst_prop" in tmp2.columns:
        cols += ["worst_prop", "worst_prop_raw"]

    out = (
        tmp2[cols]
        .sort_values(["generator", "variant"])
        .reset_index(drop=True)
    )
    return out

