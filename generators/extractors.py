import hashlib
from typing import Literal
import numpy as np


def von_neumann(bits: np.ndarray) -> np.ndarray:
    """
    Von Neumann extractor over 0/1 bits.
    """
    bits = np.asarray(bits, dtype=np.uint8)
    n = bits.size
    if n < 2:
        return np.array([], dtype=np.uint8)

    if n % 2 == 1:
        bits = bits[:-1]
        n -= 1

    pairs = bits.reshape(-1, 2)
    out = []
    for b0, b1 in pairs:
        if b0 == 0 and b1 == 1:
            out.append(1)
        elif b0 == 1 and b1 == 0:
            out.append(0)
    return np.array(out, dtype=np.uint8)


def bits_to_bytes(bits: np.ndarray) -> bytes:
    """
    Pack a 0/1 bit array into bytes, padding with zeros if needed.
    """
    bits = np.asarray(bits, dtype=np.uint8)
    n_bits = bits.size
    pad = (-n_bits) % 8
    if pad:
        bits = np.concatenate([bits, np.zeros(pad, dtype=np.uint8)])
    packed = np.packbits(bits)
    return packed.tobytes()


HashName = Literal["sha256", "sha3_256"]


def _get_hash_func(name: HashName):
    if name == "sha256":
        return hashlib.sha256
    elif name == "sha3_256":
        return hashlib.sha3_256
    else:
        raise ValueError(f"Unsupported hash: {name}")


def hash_extractor(
    bits: np.ndarray,
    n_bits_out: int,
    hash_name: HashName = "sha256",
    block_in_bits: int = 1024,
) -> np.ndarray:
    """
    Hash-based extractor: blocks -> hash -> output bits.
    """
    bits = np.asarray(bits, dtype=np.uint8)
    hash_func = _get_hash_func(hash_name)

    out_bits: list[int] = []
    idx = 0
    n = bits.size

    while len(out_bits) < n_bits_out:
        if idx + block_in_bits <= n:
            block = bits[idx:idx + block_in_bits]
            idx += block_in_bits
        else:
            remaining = block_in_bits - (n - idx)
            block = np.concatenate([bits[idx:], bits[:remaining]])
            idx = remaining

        block_bytes = bits_to_bytes(block)
        digest = hash_func(block_bytes).digest()
        digest_bits = np.unpackbits(np.frombuffer(digest, dtype=np.uint8))
        out_bits.extend(digest_bits.tolist())

    return np.array(out_bits[:n_bits_out], dtype=np.uint8)
