"""
Derivación de claves usando mapas caóticos.

Esquema general:

1) A y B ejecutan Diffie–Hellman y comparten un secreto entero S.
2) A partir de S se deriva una semilla con SHA-256.
3) Esa semilla fija los parámetros del mapa caótico (logistic, tent, henon).
4) Con esos parámetros se genera un flujo de bits caótico.
5) Según el "pipeline" elegido:
   - 'raw'         : se usan directamente los bits crudos del mapa.
   - 'von_neumann' : se aplica el extractor de Von Neumann.
   - 'hash'        : se aplica un extractor criptográfico basado en hash (SHA-256).
"""

import hashlib
from typing import Dict, Literal

import numpy as np

from generators.chaotic_bits import generate_chaotic_bits
from generators.extractors import hash_extractor, von_neumann


PipelineName = Literal["raw", "von_neumann", "hash"]


# ------------------------------------------------------------
# Helpers internos
# ------------------------------------------------------------

def _floats_from_seed(seed: bytes, n: int) -> list[float]:
    """
    Genera n flotantes en (0,1) a partir de una semilla de bytes.

    Se usa SHA-256 de forma iterativa: seed || contador.
    """
    floats: list[float] = []
    counter = 0
    while len(floats) < n:
        h = hashlib.sha256(seed + counter.to_bytes(4, "big")).digest()
        # 32 bytes -> 4 enteros de 64 bits
        for i in range(0, 32, 8):
            if len(floats) >= n:
                break
            u64 = int.from_bytes(h[i : i + 8], "big")
            x = (u64 + 0.5) / 2**64  # en (0,1)
            floats.append(x)
        counter += 1
    return floats


def _bits_to_bytes(bits: np.ndarray) -> bytes:
    """
    Convierte un array/lista de bits 0/1 en bytes.

    - Se empaquetan 8 bits por byte.
    - Si la longitud no es múltiplo de 8, se rellena con ceros al final.
    """
    arr = np.asarray(bits, dtype=np.uint8)
    n_bits = arr.size
    pad = (-n_bits) % 8
    if pad:
        arr = np.concatenate([arr, np.zeros(pad, dtype=np.uint8)])
    packed = np.packbits(arr)
    return packed.tobytes()


# ------------------------------------------------------------
# Derivación de parámetros por mapa a partir del secreto DH
# ------------------------------------------------------------

def derive_map_params(map_name: str, shared_secret: int) -> Dict:
    """
    A partir del secreto compartido (entero DH) deriva los parámetros
    del mapa indicado.

    map_name:
      - "logistic"
      - "tent"
      - "henon"

    Devuelve un dict listo para pasarlo a generate_chaotic_bits.
    """
    # 1) Normalizamos el secreto con SHA-256
    secret_bytes = shared_secret.to_bytes(
        (shared_secret.bit_length() + 7) // 8,
        byteorder="big",
    )
    seed = hashlib.sha256(secret_bytes).digest()

    # 2) Obtenemos flotantes en (0,1) para parametrizar
    u1, u2, u3, u4 = _floats_from_seed(seed, 4)

    if map_name == "logistic":
        # r en [3.8, 3.999...)
        r = 3.8 + 0.199 * u1
        # x0 en (0,1), evitando exactamente 0 ó 1
        x0 = 1e-6 + (1.0 - 2e-6) * u2
        # transitorio entre 3000 y 8000
        n_transient = 3000 + int(u3 * 5000)

        return dict(
            r=r,
            x0=x0,
            n_transient=n_transient,
            threshold=0.5,
        )

    if map_name == "tent":
        # mu en [1.8, 1.999) (evitamos 2.0 por temas numéricos)
        mu = 1.8 + 0.199 * u1
        x0 = 1e-6 + (1.0 - 2e-6) * u2
        n_transient = 3000 + int(u3 * 5000)

        return dict(
            mu=mu,
            x0=x0,
            n_transient=n_transient,
            threshold=0.5,
        )

    if map_name == "henon":
        # Parámetros alrededor de (1.4, 0.3), pero en un rango más estrecho
        # para reducir problemas numéricos (órbitas que se disparan).
        a = 1.4 + 0.05 * (u1 - 0.5)   # ~[1.375, 1.425]
        b = 0.3  + 0.02 * (u2 - 0.5)  # ~[0.29, 0.31]

        # Condiciones iniciales en una zona razonable del atractor
        x0 = -1.0 + 2.0 * u3    # [-1,1]
        y0 = -0.5 + 1.0 * u4    # [-0.5,0.5]
        n_transient = 5000

        return dict(
            a=a,
            b=b,
            x0=x0,
            y0=y0,
            n_transient=n_transient,
            threshold=0.5,
            coord="x",
        )

    raise ValueError(f"Unsupported map for key derivation: {map_name}")


# ------------------------------------------------------------
# Generación de keystream a partir del secreto compartido
# ------------------------------------------------------------

def generate_keystream_bits_from_shared_secret(
    map_name: str,
    shared_secret: int,
    n_bits: int,
    pipeline: PipelineName = "hash",
) -> np.ndarray:
    """
    Genera un flujo de n_bits a partir del secreto compartido, usando
    un mapa caótico y el pipeline especificado.

    pipeline:
      - 'raw'         : bits crudos del mapa caótico.
      - 'von_neumann' : mapa + extractor de Von Neumann.
      - 'hash'        : mapa + extractor hash (SHA-256).
    """
    params = derive_map_params(map_name, shared_secret)

    if pipeline == "raw":
        # Simplemente generamos n_bits directamente del mapa
        bits = generate_chaotic_bits(
            map_name,
            n_bits=n_bits,
            **params,
        )
        return bits

    if pipeline == "von_neumann":
        # Von Neumann reduce la longitud (~1 bit de salida por cada 4 de entrada),
        # así que generamos bastantes más bits de entrada.
        factor = 10
        n_raw = max(n_bits * factor, 4096)

        raw_bits = generate_chaotic_bits(
            map_name,
            n_bits=n_raw,
            **params,
        )
        vn_bits = von_neumann(raw_bits)

        if vn_bits.size < n_bits:
            raise ValueError(
                f"Von Neumann output too short ({vn_bits.size} bits) "
                f"for requested {n_bits} bits. Increase factor or adjust parameters."
            )

        return vn_bits[:n_bits]

    if pipeline == "hash":
        # Generamos varias veces la longitud necesaria para alimentar el hash
        n_raw = max(n_bits * 4, 4096)

        raw_bits = generate_chaotic_bits(
            map_name,
            n_bits=n_raw,
            **params,
        )

        key_bits = hash_extractor(
            raw_bits,
            n_bits_out=n_bits,
            hash_name="sha256",
            block_in_bits=1024,
        )
        return key_bits

    raise ValueError(f"Unsupported pipeline: {pipeline}")


def generate_keystream_bytes_from_shared_secret(
    map_name: str,
    shared_secret: int,
    n_bits: int,
    pipeline: PipelineName = "hash",
) -> bytes:
    """
    Igual que generate_keystream_bits_from_shared_secret, pero devuelve bytes.
    """
    bits = generate_keystream_bits_from_shared_secret(
        map_name=map_name,
        shared_secret=shared_secret,
        n_bits=n_bits,
        pipeline=pipeline,
    )
    return _bits_to_bytes(bits)


# ------------------------------------------------------------
# Wrappers "clave" para compatibilidad con lo que ya teníamos
# ------------------------------------------------------------

def derive_key_bits_from_shared_secret(
    map_name: str,
    shared_secret: int,
    n_key_bits: int = 256,
) -> np.ndarray:
    """
    Mantiene la antigua interfaz: siempre usa el pipeline 'hash'.
    """
    return generate_keystream_bits_from_shared_secret(
        map_name=map_name,
        shared_secret=shared_secret,
        n_bits=n_key_bits,
        pipeline="hash",
    )


def derive_key_bytes_from_shared_secret(
    map_name: str,
    shared_secret: int,
    n_key_bits: int = 256,
) -> bytes:
    """
    Mantiene la antigua interfaz: siempre usa el pipeline 'hash'.
    """
    return generate_keystream_bytes_from_shared_secret(
        map_name=map_name,
        shared_secret=shared_secret,
        n_bits=n_key_bits,
        pipeline="hash",
    )
