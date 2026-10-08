"""
Demo criptográfica parametrizable:

Puedes elegir:
  - MAP_NAME  ∈ {"logistic", "tent", "henon"}
  - PIPELINE ∈ {"raw", "von_neumann", "hash"}

MAPA:
  logistic  : mapa logístico
  tent      : mapa de tent
  henon     : mapa de Hénon

PIPELINE:
  raw         : bits crudos del mapa
  von_neumann : mapa + extractor de Von Neumann
  hash        : mapa + extractor hash (SHA-256)

Flujo:
1) Alice y Bob ejecutan Diffie–Hellman y acuerdan un secreto compartido.
2) A partir de ese secreto generan un keystream según (MAP_NAME, PIPELINE).
3) Se cifra y descifra un mensaje con XOR.
"""

from crypto.dh import (
    P,
    G,
    generate_private_key,
    generate_public_key,
    compute_shared_secret,
)
from crypto.chaos_keygen import generate_keystream_bytes_from_shared_secret
from crypto.xor_cipher import xor_encrypt, xor_decrypt


# ============================================================
# CONFIGURACIÓN A TOCAR POR TI
# ============================================================

MAP_NAME = "henon"       # "logistic", "tent" o "henon"
PIPELINE = "hash"           # "raw", "von_neumann" o "hash"
N_KEY_BITS = 256            # longitud de la clave/keystream en bits


def main() -> None:
    print("=== Demo criptográfica con caos ===")
    print(f"Mapa caótico  : {MAP_NAME}")
    print(f"Pipeline bits : {PIPELINE}")
    print(f"Longitud clave: {N_KEY_BITS} bits")
    print()

    print("=== Parámetros públicos de Diffie–Hellman (demo) ===")
    print(f"p (bits): {P.bit_length()}")
    print(f"g      : {G}")
    print()

    # --------------------------
    # 1) Claves de Alice y Bob
    # --------------------------
    a_priv = generate_private_key()
    A_pub = generate_public_key(a_priv)

    b_priv = generate_private_key()
    B_pub = generate_public_key(b_priv)

    print("=== Intercambio DH (simulado) ===")
    print(f"Alice public A = g^a mod p: {A_pub}")
    print(f"Bob   public B = g^b mod p: {B_pub}")
    print()

    # --------------------------
    # 2) Cálculo del secreto compartido
    # --------------------------
    secret_alice = compute_shared_secret(B_pub, a_priv)
    secret_bob = compute_shared_secret(A_pub, b_priv)

    print("=== Secreto compartido ===")
    print(f"¿secret_alice == secret_bob? {secret_alice == secret_bob}")
    print()

    # --------------------------
    # 3) Derivación de keystream caótico
    # --------------------------
    key_alice = generate_keystream_bytes_from_shared_secret(
        map_name=MAP_NAME,
        shared_secret=secret_alice,
        n_bits=N_KEY_BITS,
        pipeline=PIPELINE,
    )
    key_bob = generate_keystream_bytes_from_shared_secret(
        map_name=MAP_NAME,
        shared_secret=secret_bob,
        n_bits=N_KEY_BITS,
        pipeline=PIPELINE,
    )

    print("=== Keystream derivado ===")
    print(f"Longitud (bytes): {len(key_alice)}")
    print(f"¿key_alice == key_bob? {key_alice == key_bob}")
    print(f"Keystream (hex, truncado): {key_alice.hex()[:64]}...")
    print()

    # --------------------------
    # 4) Cifrado/descifrado con XOR
    # --------------------------
    mensaje = "Hola Caracola con DH + mapa caótico"
    print("=== Cifrado de mensaje ===")
    print(f"Mensaje original: {mensaje}")

    ciphertext = xor_encrypt(mensaje, key_alice)
    print(f"Ciphertext (hex): {ciphertext.hex()}")

    mensaje_recuperado = xor_decrypt(ciphertext, key_bob)
    print(f"Mensaje recuperado por Bob: {mensaje_recuperado}")
    print(f"¿Mensaje OK? {mensaje_recuperado == mensaje}")
    print()

    print("Demostración completada.")


if __name__ == "__main__":
    main()
