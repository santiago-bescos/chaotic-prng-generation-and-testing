from pathlib import Path
import re
from dataclasses import dataclass

@dataclass
class NistTestResult:
    test_name: str
    p_value: float | None
    passed: bool
    proportion: str | None
    num_sequences: int | None


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

    return results

def parse_experiment_folder(experiment_path: Path) -> dict:
    report = experiment_path / "AlgorithmTesting" / "finalAnalysisReport.txt"

    test_results = parse_final_analysis_report(report)

    total = len(test_results)
    passed = sum(1 for t in test_results if t.passed)

    return {
        "experiment": experiment_path.name,
        "total_tests": total,
        "passed_tests": passed,
        "pass_ratio": passed / total if total > 0 else 0.0,
        "tests": test_results,
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
    (?:\d+\s+){10}                 # C1..C10
    (?P<pval>----|\d+\.\d+)\s*     # P-VALUE or ----
    (?P<star>\*)?\s+              # optional '*'
    (?P<prop>\d+/\d+|----)\s*      # PROPORTION or ----
    \*?\s+                         # sometimes an extra '*'
    (?P<test>[A-Za-z0-9_]+)\s*$    # TEST NAME
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
        star = m.group("star") is not None
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


