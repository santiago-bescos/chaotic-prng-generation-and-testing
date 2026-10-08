from __future__ import annotations

from pathlib import Path
from typing import Dict, Any, List

from generators.export_helpers import (
    generate_and_save_for_map,
    save_manifest,
)

# ============================================================
# CONFIGURACIÓN GENERAL DEL EXPERIMENTO
# ============================================================

OUTPUT_ROOT = Path("outputs_randomness")  

N_BITS_TARGET = 1_000_000
RAW_BITS_FACTOR = 4  # 4x para alimentar bien Von Neumann

SAVE_ASCII01 = True
SAVE_BIN_PACKED = True
SAVE_UINT32 = True

HASH_NAME_SHA2 = "sha256"
HASH_NAME_SHA3 = "sha3_256"

# ============================================================
# CONFIGURACIONES EXPLÍCITAS DE STREAMS POR MAPA
# 10 streams por mapa, cada uno con parámetros y condición inicial propios
# (todos en zonas que hemos considerado caóticas en el análisis).
# ============================================================

N_TRANSIENT = 20_000
TH = 0.5

MAP_STREAM_CONFIGS: Dict[str, List[Dict[str, Any]]] = {
    "logistic": [
        # r en zona caótica, evitando cosas muy raras, x0 en (0,1)
        dict(r=3.80,  x0=0.123456, n_transient=N_TRANSIENT, threshold=TH),
        dict(r=3.84,  x0=0.234567, n_transient=N_TRANSIENT, threshold=TH),
        dict(r=3.88,  x0=0.345678, n_transient=N_TRANSIENT, threshold=TH),
        dict(r=3.92,  x0=0.456789, n_transient=N_TRANSIENT, threshold=TH),
        dict(r=3.94,  x0=0.567890, n_transient=N_TRANSIENT, threshold=TH),
        dict(r=3.96,  x0=0.678901, n_transient=N_TRANSIENT, threshold=TH),
        dict(r=3.97,  x0=0.789012, n_transient=N_TRANSIENT, threshold=TH),
        dict(r=3.98,  x0=0.890123, n_transient=N_TRANSIENT, threshold=TH),
        dict(r=3.985, x0=0.135791, n_transient=N_TRANSIENT, threshold=TH),
        dict(r=3.99,  x0=0.246802, n_transient=N_TRANSIENT, threshold=TH),
    ],
    "tent": [
        # mu cerca de 2 (zona caótica), evitando mu=2 exacto, x0 en (0,1)
        dict(mu=1.80,  x0=0.13579,  n_transient=N_TRANSIENT, threshold=TH),
        dict(mu=1.85,  x0=0.24680,  n_transient=N_TRANSIENT, threshold=TH),
        dict(mu=1.90,  x0=0.35791,  n_transient=N_TRANSIENT, threshold=TH),
        dict(mu=1.93,  x0=0.46802,  n_transient=N_TRANSIENT, threshold=TH),
        dict(mu=1.95,  x0=0.57913,  n_transient=N_TRANSIENT, threshold=TH),
        dict(mu=1.97,  x0=0.69024,  n_transient=N_TRANSIENT, threshold=TH),
        dict(mu=1.98,  x0=0.80135,  n_transient=N_TRANSIENT, threshold=TH),
        dict(mu=1.985, x0=0.41246,  n_transient=N_TRANSIENT, threshold=TH),
        dict(mu=1.99,  x0=0.52357,  n_transient=N_TRANSIENT, threshold=TH),
        dict(mu=1.999, x0=0.63468,  n_transient=N_TRANSIENT, threshold=TH),
    ],
    # mass_export_bits.py

    "henon": [
        # a en [1.36, 1.42], b=0.3, x0,y0 muy cercanos al origen
        dict(a=1.36, b=0.30, x0=0.00,  y0=0.00, n_transient=N_TRANSIENT, threshold=TH, coord="x"),
        dict(a=1.37, b=0.30, x0=0.01,  y0=0.00, n_transient=N_TRANSIENT, threshold=TH, coord="x"),
        dict(a=1.38, b=0.30, x0=-0.01, y0=0.00, n_transient=N_TRANSIENT, threshold=TH, coord="x"),
        dict(a=1.39, b=0.30, x0=0.02,  y0=0.01, n_transient=N_TRANSIENT, threshold=TH, coord="x"),
        dict(a=1.40, b=0.30, x0=-0.02, y0=-0.01, n_transient=N_TRANSIENT, threshold=TH, coord="x"),
        dict(a=1.41, b=0.30, x0=0.03,  y0=0.00, n_transient=N_TRANSIENT, threshold=TH, coord="x"),
        dict(a=1.42, b=0.30, x0=-0.03, y0=0.00, n_transient=N_TRANSIENT, threshold=TH, coord="x"),
        dict(a=1.385, b=0.30, x0=0.015, y0=-0.005, n_transient=N_TRANSIENT, threshold=TH, coord="x"),
        dict(a=1.395, b=0.30, x0=-0.015,y0=0.005, n_transient=N_TRANSIENT, threshold=TH, coord="x"),
        dict(a=1.405, b=0.30, x0=0.005, y0=0.01, n_transient=N_TRANSIENT, threshold=TH, coord="x"),
    ],


}


# ============================================================
# PROGRAMA PRINCIPAL
# ============================================================

def main() -> None:
    print("=== Exportador masivo de bitstreams caóticos ===\n")
    print(f"Carpeta de salida       : {OUTPUT_ROOT.resolve()}")
    print(f"Bits útiles por stream  : {N_BITS_TARGET}")
    print(f"Factor de sobremuestreo : {RAW_BITS_FACTOR}")
    print(f"Formatos: ASCII={SAVE_ASCII01}, "
          f"PACKED={SAVE_BIN_PACKED}, UINT32={SAVE_UINT32}")
    print(f"Hashes: SHA-2='{HASH_NAME_SHA2}', SHA-3='{HASH_NAME_SHA3}'\n")

    manifest_rows: List[Dict[str, Any]] = []

    for map_name, stream_configs in MAP_STREAM_CONFIGS.items():
        generate_and_save_for_map(
            map_name=map_name,
            stream_configs=stream_configs,
            output_root=OUTPUT_ROOT,
            n_bits_target=N_BITS_TARGET,
            raw_bits_factor=RAW_BITS_FACTOR,
            hash_name_sha2=HASH_NAME_SHA2,
            hash_name_sha3=HASH_NAME_SHA3,
            save_ascii01=SAVE_ASCII01,
            save_bin_packed=SAVE_BIN_PACKED,
            save_uint32=SAVE_UINT32,
            manifest_rows=manifest_rows,
            save_concatenated=True,
        )

    save_manifest(OUTPUT_ROOT, manifest_rows, prefix="manifest")

    print("\n[+] Generación masiva completada.")
    print("    - Usa *_seqXX_ascii01.txt para Dieharder / TestU01.")
    print("    - Usa *_allseq_ascii01.txt para NIST STS "
          f"(con m = 10 bitstreams de 1e6 bits).")


if __name__ == "__main__":
    main()
