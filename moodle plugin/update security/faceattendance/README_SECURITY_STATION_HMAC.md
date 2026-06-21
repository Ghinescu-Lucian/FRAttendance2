# Face Attendance station communication security

This version replaces the old raw-secret authentication flow with per-request HMAC-SHA256 signatures.

## What changed

- The camera station no longer sends `secret` in URL parameters, JSON payloads, or the `X-FaceAttendance-Secret` header.
- Each request is signed with these headers:
  - `X-FaceAttendance-Station`
  - `X-FaceAttendance-Timestamp`
  - `X-FaceAttendance-Nonce`
  - `X-FaceAttendance-Signature`
- Moodle verifies the signature using the activity `apisecret`.
- Moodle rejects expired timestamps and reused nonces using the `faceattendance_station_nonces` table.

## Required deployment steps

1. Upgrade the Moodle plugin so the new nonce table is created.
2. Use a long random API secret in the Face Attendance activity settings.
3. Put the same secret in the station config or in `FACEATTENDANCE_API_SECRET`.
4. Use HTTPS with valid certificate verification: `verify_tls: true`.

For local testing with a self-signed certificate, install the local CA certificate instead of disabling TLS verification.
