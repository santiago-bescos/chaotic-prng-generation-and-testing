"""
Cifrado XOR muy simple (de demostración).

NO es seguro en la práctica (one-time pad mal utilizado). El objetivo es
mostrar cómo usar un flujo de bits/bytes como keystream para cifrar
y descifrar un mensaje.
"""

from typing import ByteString


def xor_bytes(data: ByteString, key: ByteString) -> bytes:
    """
    Aplica XOR entre data y key, repitiendo la key si hace falta.
    """
    key_len = len(key)
    if key_len == 0:
        raise ValueError("Key must not be empty.")
    return bytes(b ^ key[i % key_len] for i, b in enumerate(data))


def xor_encrypt(message: str, key: bytes, encoding: str = "utf-8") -> bytes:
    """
    Cifra un mensaje de texto (str) usando XOR con la key (bytes).
    """
    plaintext = message.encode(encoding)
    return xor_bytes(plaintext, key)


def xor_decrypt(ciphertext: ByteString, key: bytes, encoding: str = "utf-8") -> str:
    """
    Descifra un ciphertext (bytes) cifrado con xor_encrypt.
    """
    plaintext = xor_bytes(ciphertext, key)
    return plaintext.decode(encoding)
