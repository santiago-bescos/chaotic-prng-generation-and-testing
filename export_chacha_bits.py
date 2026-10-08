from __future__ import annotations

from pathlib import Path
import secrets
from Crypto.Cipher import ChaCha20

OUTPUT_ROOT = Path("outputs_randomness") / "chacha"
OUTPUT_ROOT.mkdir(parents=True, exist_ok=True)

N_STREAMS = 10
N_BITS = 1_000_000
N_BYTES = N_BITS // 8  # 125000

# Tabla rápida byte->"01010101"
BYTE_TO_BITS = [format(i, "08b") for i in range(256)]


def chacha20_keystream(n_bytes: int, key: bytes, nonce: bytes) -> bytes:
    cipher = ChaCha20.new(key=key, nonce=nonce)
    return cipher.encrypt(b"\x00" * n_bytes)


def bytes_to_ascii01(data: bytes) -> str:
    # 125k bytes -> 1e6 chars: OK en memoria
    return "".join(BYTE_TO_BITS[b] for b in data)


def main() -> None:
    allseq_path = OUTPUT_ROOT / "chacha_raw_allseq_ascii01.txt"
    with allseq_path.open("w", encoding="utf-8", newline="\n") as f_all:
        for i in range(1, N_STREAMS + 1):
            key = secrets.token_bytes(32)     # 256-bit key
            nonce = secrets.token_bytes(8)    # pycryptodome ChaCha20 usa nonce de 8 bytes por defecto

            stream = chacha20_keystream(N_BYTES, key, nonce)
            bits_ascii = bytes_to_ascii01(stream)

            seq_path = OUTPUT_ROOT / f"chacha_raw_seq{i:02d}_ascii01.txt"
            seq_path.write_text(bits_ascii, encoding="utf-8")

            # Para STS: concatenamos y metemos salto de línea (STS suele ignorar whitespace)
            f_all.write(bits_ascii + "\n")

    print("[+] Generado:", allseq_path.resolve())
    print("[+] Carpeta:", OUTPUT_ROOT.resolve())


if __name__ == "__main__":
    main()
