import numpy as np
from pathlib import Path


def _ensure_bits_array(bits) -> np.ndarray:
    """
    Convierte la entrada a np.ndarray de 0/1 (uint8).
    """
    arr = np.asarray(bits, dtype=np.uint8)
    if arr.ndim != 1:
        raise ValueError("Bits must be a 1D sequence.")
    if not np.all((arr == 0) | (arr == 1)):
        raise ValueError("Bits must contain only 0 and 1.")
    return arr


def save_bits_ascii01(bits, filepath: str | Path) -> None:
    """
    Guarda los bits como texto '0'/'1' (sin espacios).

    Formato típico para algunos tests o para inspección humana.

    Ejemplo de contenido:
        0100110101010110...

    Parámetros
    ----------
    bits : array-like de 0/1
    filepath : ruta del fichero de salida
    """
    arr = _ensure_bits_array(bits)
    filepath = Path(filepath)

    # Convertimos 0/1 a caracteres '0'/'1' y unimos en una sola línea
    chars = arr.astype(str)
    text = "".join(chars) + "\n"

    filepath.write_text(text, encoding="utf-8")


def save_bits_binary_packed(bits, filepath: str | Path) -> None:
    """
    Guarda los bits empaquetados en bytes (binario crudo).

    - Se empaquetan 8 bits por byte usando np.packbits.
    - Si la longitud no es múltiplo de 8, se rellena con ceros al final.

    Este formato es útil como entrada genérica para muchos tests
    (por ejemplo, algunos modos de NIST STS o para procesar con
    herramientas propias).
    """
    arr = _ensure_bits_array(bits)
    filepath = Path(filepath)

    n_bits = arr.size
    pad = (-n_bits) % 8
    if pad:
        arr = np.concatenate([arr, np.zeros(pad, dtype=np.uint8)])

    packed = np.packbits(arr)
    filepath.write_bytes(packed.tobytes())


def save_bits_uint32_binary(bits, filepath: str | Path, endian: str = "little") -> None:
    """
    Guarda los bits como stream de enteros uint32 en binario.

    - Agrupa los bits en bloques de 32.
    - Cada bloque se interpreta como un entero sin signo de 32 bits.
    - Si la longitud no es múltiplo de 32, se rellena con ceros al final.
    - Se escribe el array de uint32 en binario (formato nativo).

    Este formato es típico para herramientas como Dieharder o TestU01,
    que suelen trabajar con secuencias de enteros de 32 bits.

    Parámetros
    ----------
    bits : array-like de 0/1
    filepath : ruta del fichero binario de salida
    endian : 'little' o 'big'
        Orden de los bits dentro del entero:
        - 'little': el bit menos significativo es el último del bloque.
        - 'big'   : el bit más significativo es el primero del bloque.
    """
    arr = _ensure_bits_array(bits)
    filepath = Path(filepath)

    n_bits = arr.size
    pad = (-n_bits) % 32
    if pad:
        arr = np.concatenate([arr, np.zeros(pad, dtype=np.uint8)])

    bits32 = arr.reshape(-1, 32)  # cada fila = 32 bits

    # Calculamos el valor entero de cada bloque de 32 bits
    if endian == "big":
        # b0 b1 ... b31  ->  b0 es bit más significativo
        powers = (1 << np.arange(31, -1, -1, dtype=np.uint32))
    elif endian == "little":
        # b0 b1 ... b31  ->  b0 es bit menos significativo
        powers = (1 << np.arange(0, 32, dtype=np.uint32))
    else:
        raise ValueError("endian must be 'little' or 'big'.")

    # Multiplicamos bits por potencias de 2 y sumamos en cada fila
    values = (bits32.astype(np.uint32) * powers).sum(axis=1, dtype=np.uint64)
    values = values.astype(np.uint32)

    # Guardamos en binario
    with open(filepath, "wb") as f:
        values.tofile(f)
