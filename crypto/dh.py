"""
Diffie–Hellman muy sencillo (de demostración).

IMPORTANTE: esto NO es seguro a nivel producción. Se usa un primo moderado
(2^127-1) y claves privadas de 128 bits solo para ilustrar el protocolo
y poder integrar el PRNG caótico en la derivación de claves.
"""

import secrets

# Primo de Mersenne 2^127 - 1 (es primo conocido)
P = (1 << 127) - 1
# Generador pequeño (válido para la demo)
G = 5


def generate_private_key(bit_size: int = 128) -> int:
    """
    Genera una clave privada aleatoria (entero) para DH.

    bit_size: nº de bits de la clave privada. 128 es suficiente para demo.
    """
    return secrets.randbits(bit_size)


def generate_public_key(private_key: int) -> int:
    """
    Calcula la clave pública DH:

        pub = g^priv mod p
    """
    return pow(G, private_key, P)


def compute_shared_secret(their_public: int, my_private: int) -> int:
    """
    Calcula el secreto compartido DH:

        s = their_public^my_private mod p
    """
    return pow(their_public, my_private, P)


def shared_secret_to_bytes(shared_secret: int) -> bytes:
    """
    Codifica el entero del secreto compartido a bytes (big-endian).
    """
    if shared_secret == 0:
        # caso degenerado muy improbable, pero por si acaso
        return b"\x00"
    length = (shared_secret.bit_length() + 7) // 8
    return shared_secret.to_bytes(length, byteorder="big")
