# Encrypted Moodle embeddings for one camera station

This version supports station-only decryption of Moodle SFace embeddings.

## What changed

1. The camera station generates a public/private key pair.
2. The private key remains only on the camera-station machine.
3. The public key is copied into the Moodle Face Attendance activity settings.
4. New embedding registrations are stored in Moodle as a libsodium sealed-box ciphertext.
5. During bootstrap, Moodle returns only the encrypted envelope.
6. The camera station decrypts the envelope locally and loads the SFace descriptors into memory.

## Generate station keys

```powershell
py -m pip install PyNaCl
py generate_station_crypto_keys.py
```

Copy the printed public key into Moodle:

```text
Face Attendance activity settings -> Station public key for encrypted embeddings
```

Keep this file only on the station:

```text
station_private_key.b64
```

## Configure the station

In `station_config.json`:

```json
{
  "station_id": "camera-lab-1",
  "station_private_key_path": "station_private_key.b64"
}
```

## Important limitation

Only new or re-saved embedding records become encrypted. Existing plaintext records must be re-registered after the public key is configured.
