import numpy as np

from generators.chaotic_bits import generate_chaotic_bits
from generators.extractors import hash_extractor
from utils.io_bits import (
    save_bits_ascii01,
    save_bits_binary_packed,
    save_bits_uint32_binary,
)


def main() -> None:
    # Parámetros del generador caótico
    map_name = "logistic"
    params = dict(
        r=3.99,
        x0=0.314159,
        n_transient=5000,
        threshold=0.5,
    )

    # Nº de bits a exportar (para NIST suelen pedir largo, p.ej. 1e6)
    n_bits = 1_000_000

    print(f"Generando {n_bits} bits con mapa {map_name} (HASH extractor)...")

    # 1) Bits crudos del mapa
    raw_bits = generate_chaotic_bits(
        map_name,
        n_bits=n_bits,
        **params,
    )

    # 2) Aplicamos extractor basado en hash para mejorar la aleatoriedad
    bits = hash_extractor(
        raw_bits,
        n_bits_out=n_bits,
        hash_name="sha256",
        block_in_bits=1024,
    )

    print("Ejemplos de exportación en carpeta actual:")

    # a) Texto '0'/'1'
    ascii_path = "bits_logistic_hash_ascii01.txt"
    save_bits_ascii01(bits, ascii_path)
    print(f"  - ASCII 0/1  : {ascii_path}")

    # b) Binario empaquetado en bytes
    bin_packed_path = "bits_logistic_hash_packed.bin"
    save_bits_binary_packed(bits, bin_packed_path)
    print(f"  - Bin packed : {bin_packed_path}")

    # c) Binario como uint32 (para Dieharder/TestU01)
    uint32_path = "bits_logistic_hash_uint32.bin"
    save_bits_uint32_binary(bits, uint32_path, endian="little")
    print(f"  - uint32 bin : {uint32_path}")

if __name__ == "__main__":
    main()
