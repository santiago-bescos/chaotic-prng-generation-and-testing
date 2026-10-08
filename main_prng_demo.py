import numpy as np

from generators.chaotic_bits import generate_chaotic_bits
from generators.extractors import von_neumann, hash_extractor
from randomness_tests.basic_tests import (
    basic_bit_stats,
    frequency_chi_square_test,
    runs_test,
)


def analyze_stream(bits, label: str) -> None:
    """
    Ejecuta los tests básicos sobre una secuencia de bits con una etiqueta.
    """
    n = len(bits)
    print(f"\n=== {label} ===")

    if n == 0:
        print("Sequence is empty; skipping tests.")
        return

    basic_bit_stats(bits)
    frequency_chi_square_test(bits)
    runs_test(bits)


def main() -> None:
    # Configuraciones para cada mapa
    configs = [
        {
            "map_name": "logistic",
            "params": dict(
                r=3.99,
                x0=0.314159,
                n_transient=2000,
                threshold=0.5,
            ),
        },
        {
            "map_name": "tent",
            "params": dict(
                mu=1.999,  # evitar mu=2.0 por colapso numérico
                x0=0.314159,
                n_transient=2000,
                threshold=0.5,
            ),
        },
        {
            "map_name": "henon",
            "params": dict(
                a=1.36,
                b=0.3,
                x0=0,
                y0=0,
                n_transient=5000,
                threshold=0.5,
                coord="x",  # usamos la coordenada x para extraer bits
            ),
        },
    ]

    for cfg in configs:
        map_name = cfg["map_name"]
        params = cfg["params"]

        print("\n" + "#" * 60)
        print(f"MAP: {map_name.upper()}")
        print("#" * 60)

        # 1) Bits caóticos crudos
        raw_bits = generate_chaotic_bits(
            map_name,
            n_bits=100_000,
            **params,
        )

        # 2) Extractor de Von Neumann
        vn_bits = von_neumann(raw_bits)

        # 3) Extractor basado en hash (SHA-256)
        hash_bits = hash_extractor(
            raw_bits,
            n_bits_out=100_000,
            hash_name="sha256",
            block_in_bits=1024,
        )

        # 4) Referencia Bernoulli(0.5)
        ref_bits = np.random.randint(0, 2, size=100_000, dtype=np.uint8)

        print("Lengths:")
        print("  RAW   :", raw_bits.size)
        print("  VN    :", vn_bits.size)
        print("  HASH  :", hash_bits.size)
        print("  REF   :", ref_bits.size)

        # Tests para cada flujo
        analyze_stream(raw_bits,  f"{map_name} RAW")
        analyze_stream(vn_bits,   f"{map_name} Von Neumann")
        analyze_stream(hash_bits, f"{map_name} HASH (SHA-256)")
        analyze_stream(ref_bits,  f"{map_name} Bernoulli(0.5)")


if __name__ == "__main__":
    main()
