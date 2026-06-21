#!/usr/bin/env python3
"""Generate a camera-station key pair for encrypted Moodle embeddings.

The private key stays only on the camera station. Copy only the printed public key
into the Face Attendance activity setting: "Station public key for encrypted embeddings".
"""

import base64
from pathlib import Path

try:
    from nacl.public import PrivateKey
except Exception as exc:
    raise SystemExit("PyNaCl is required. Install it with: py -m pip install PyNaCl") from exc

out = Path(__file__).resolve().parent
private_key = PrivateKey.generate()
private_b64 = base64.b64encode(bytes(private_key)).decode("ascii")
public_b64 = base64.b64encode(bytes(private_key.public_key)).decode("ascii")

private_path = out / "station_private_key.b64"
public_path = out / "station_public_key.b64"

if private_path.exists():
    raise SystemExit(f"Refusing to overwrite existing private key: {private_path}")

private_path.write_text(private_b64 + "\n", encoding="utf-8")
public_path.write_text(public_b64 + "\n", encoding="utf-8")

print("Station keys generated.")
print(f"Private key file, keep secret on this station only: {private_path}")
print(f"Public key file, safe to copy into Moodle: {public_path}")
print("\nCopy this public key into Moodle Face Attendance -> Station public key for encrypted embeddings:\n")
print(public_b64)
