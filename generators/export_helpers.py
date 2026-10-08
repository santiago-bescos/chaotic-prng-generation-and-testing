# generators/export_helpers.py

from __future__ import annotations

from pathlib import Path
from datetime import datetime
from typing import Dict, Any, List, Optional

import numpy as np

from generators.chaotic_bits import generate_chaotic_bits
from generators.extractors import hash_extractor, von_neumann
from utils.io_bits import (
    save_bits_ascii01,
    save_bits_binary_packed,
    save_bits_uint32_binary,
)


def ensure_dir(path: Path) -> None:
    """Crea la carpeta (y padres) si no existe."""
    path.mkdir(parents=True, exist_ok=True)


# ============================================================
#  Construcción de variantes a partir de bits brutos
# ============================================================

def compute_variants_from_raw(
    raw_big: np.ndarray,
    n_bits_target: int,
    hash_name_sha2: str,
    hash_name_sha3: str,
) -> Dict[str, np.ndarray]:
    """
    A partir de una secuencia bruta larga genera las variantes:

      - "raw"  : primeros n_bits_target bits tal cual.
      - "sha2" : hash_extractor con hash_name_sha2.
      - "sha3" : hash_extractor con hash_name_sha3.
      - "vn"   : Von Neumann (se recorta a n_bits_target si hace falta).

    Devuelve un diccionario {nombre_variante: bits}.
    """
    raw_big = np.asarray(raw_big, dtype=np.uint8)

    # RAW: truncar
    raw_out = raw_big[:n_bits_target]

    # SHA-2
    sha2_out = hash_extractor(
        raw_big,
        n_bits_out=n_bits_target,
        hash_name=hash_name_sha2,
        block_in_bits=1024,
    )

    # SHA-3
    sha3_out = hash_extractor(
        raw_big,
        n_bits_out=n_bits_target,
        hash_name=hash_name_sha3,
        block_in_bits=1024,
    )

    # Von Neumann
    vn_full = von_neumann(raw_big)

    variants: Dict[str, np.ndarray] = {
        "raw": np.asarray(raw_out, dtype=np.uint8),
        "sha2": np.asarray(sha2_out, dtype=np.uint8),
        "sha3": np.asarray(sha3_out, dtype=np.uint8),
    }

    if vn_full.size > 0:
        variants["vn"] = np.asarray(vn_full, dtype=np.uint8)

    return variants


# ============================================================
#  Generación y guardado multi-stream por mapa
# ============================================================

def generate_and_save_for_map(
    map_name: str,
    stream_configs: List[Dict[str, Any]],
    output_root: Path,
    n_bits_target: int,
    raw_bits_factor: int,
    hash_name_sha2: str,
    hash_name_sha3: str,
    save_ascii01: bool,
    save_bin_packed: bool,
    save_uint32: bool,
    manifest_rows: List[Dict[str, Any]],
    save_concatenated: bool = True,
) -> None:
    """
    Para un mapa:

      - Recorre una lista de configuraciones de stream (parámetros + x0,y0).
      - Para cada config:
          * genera bits brutos,
          * construye variantes raw/sha2/sha3/vn,
          * guarda ficheros por stream (seqXX_*).
      - Opcionalmente concatena streams por variante (allseq_*).
      - Añade entradas al manifest por cada fichero generado.

    stream_configs es una lista de dicts, cada uno con todos los
    parámetros necesarios para generate_chaotic_bits(...) de ese mapa.
    """
    print(f"\n[+] Mapa: {map_name}")
    print(f"    Número de streams: {len(stream_configs)}")

    map_dir = output_root / map_name
    ensure_dir(map_dir)

    # Para concatenar: acumulamos una lista de arrays por variante.
    concat_buffers: Dict[str, List[np.ndarray]] = {}

    # ========================================================
    #  Bucle sobre streams independientes
    # ========================================================
    for stream_idx, params_stream in enumerate(stream_configs, start=1):
        print(f"    - Stream {stream_idx:02d} con params: {params_stream}")

        n_bits_raw = n_bits_target * raw_bits_factor
        print(f"      Generando {n_bits_raw} bits brutos...")

        raw_big = generate_chaotic_bits(
            map_name,
            n_bits=n_bits_raw,
            **params_stream,
        )
        raw_big = np.asarray(raw_big, dtype=np.uint8)

        # Variantes para ESTE stream
        variants = compute_variants_from_raw(
            raw_big,
            n_bits_target=n_bits_target,
            hash_name_sha2=hash_name_sha2,
            hash_name_sha3=hash_name_sha3,
        )

        params_repr = str(params_stream)

        for variant_name, bits_full in variants.items():
            bits = np.asarray(bits_full, dtype=np.uint8)
            bits = bits[:n_bits_target]
            n_bits_variant = int(bits.size)

            if n_bits_variant == 0:
                print(
                    f"      [!] Stream {stream_idx:02d}, variante {variant_name} "
                    f"vacía, se omite."
                )
                continue

            if variant_name == "vn" and n_bits_variant < n_bits_target:
                print(
                    f"      [!] Stream {stream_idx:02d}, variante VN sólo tiene "
                    f"{n_bits_variant} bits; objetivo eran {n_bits_target}."
                )

            base_name = f"{map_name}_{variant_name}_seq{stream_idx:02d}"

            # ASCII 0/1
            if save_ascii01:
                ascii_path = map_dir / f"{base_name}_ascii01.txt"
                save_bits_ascii01(bits, str(ascii_path))
                manifest_rows.append(
                    dict(
                        map_name=map_name,
                        variant=variant_name,
                        format="ascii01",
                        filepath=str(ascii_path),
                        n_bits=n_bits_variant,
                        params_repr=params_repr,
                        stream_index=stream_idx,
                        concatenated=False,
                    )
                )

            # Binario empaquetado
            if save_bin_packed:
                bin_packed_path = map_dir / f"{base_name}_packed.bin"
                save_bits_binary_packed(bits, str(bin_packed_path))
                manifest_rows.append(
                    dict(
                        map_name=map_name,
                        variant=variant_name,
                        format="bin_packed",
                        filepath=str(bin_packed_path),
                        n_bits=n_bits_variant,
                        params_repr=params_repr,
                        stream_index=stream_idx,
                        concatenated=False,
                    )
                )

            # uint32 little-endian
            if save_uint32:
                uint32_path = map_dir / f"{base_name}_uint32_le.bin"
                save_bits_uint32_binary(bits, str(uint32_path), endian="little")
                manifest_rows.append(
                    dict(
                        map_name=map_name,
                        variant=variant_name,
                        format="uint32_le",
                        filepath=str(uint32_path),
                        n_bits=n_bits_variant,
                        params_repr=params_repr,
                        stream_index=stream_idx,
                        concatenated=False,
                    )
                )

            # Guardamos para concatenar luego
            concat_buffers.setdefault(variant_name, []).append(bits)

    # ========================================================
    #  Exportar versiones concatenadas (allseq) por variante
    # ========================================================
    if save_concatenated:
        for variant_name, list_bits in concat_buffers.items():
            if not list_bits:
                print(
                    f"    [!] Variante {variant_name}: sin bits para concatenar, "
                    f"se omite allseq."
                )
                continue

            bits_concat = np.concatenate(list_bits, axis=0)
            n_bits_concat = int(bits_concat.size)
            base_name_all = f"{map_name}_{variant_name}_allseq"

            print(
                f"    [+] Variante {variant_name}: fichero concatenado allseq "
                f"con {n_bits_concat} bits."
            )

            # ASCII 0/1
            if save_ascii01:
                ascii_path = map_dir / f"{base_name_all}_ascii01.txt"
                save_bits_ascii01(bits_concat, str(ascii_path))
                manifest_rows.append(
                    dict(
                        map_name=map_name,
                        variant=variant_name,
                        format="ascii01",
                        filepath=str(ascii_path),
                        n_bits=n_bits_concat,
                        params_repr="concatenated",
                        stream_index=None,
                        concatenated=True,
                    )
                )

            # Binario empaquetado
            if save_bin_packed:
                bin_packed_path = map_dir / f"{base_name_all}_packed.bin"
                save_bits_binary_packed(bits_concat, str(bin_packed_path))
                manifest_rows.append(
                    dict(
                        map_name=map_name,
                        variant=variant_name,
                        format="bin_packed",
                        filepath=str(bin_packed_path),
                        n_bits=n_bits_concat,
                        params_repr="concatenated",
                        stream_index=None,
                        concatenated=True,
                    )
                )

            # uint32 little-endian
            if save_uint32:
                uint32_path = map_dir / f"{base_name_all}_uint32_le.bin"
                save_bits_uint32_binary(bits_concat, str(uint32_path), endian="little")
                manifest_rows.append(
                    dict(
                        map_name=map_name,
                        variant=variant_name,
                        format="uint32_le",
                        filepath=str(uint32_path),
                        n_bits=n_bits_concat,
                        params_repr="concatenated",
                        stream_index=None,
                        concatenated=True,
                    )
                )


# ============================================================
#  Manifest en CSV
# ============================================================

def save_manifest(
    output_root: Path,
    manifest_rows: List[Dict[str, Any]],
    prefix: str = "manifest",
) -> Optional[Path]:
    """
    Guarda un CSV con la información de todos los ficheros generados.
    """
    if not manifest_rows:
        return None

    ensure_dir(output_root)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    manifest_path = output_root / f"{prefix}_{timestamp}.csv"

    import csv

    fieldnames = list(manifest_rows[0].keys())
    with open(manifest_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(manifest_rows)

    print(f"\n[+] Manifest guardado en: {manifest_path}")
    return manifest_path
