# Secure Moodle communication

This station now uses HMAC-SHA256 request signing when it communicates with the Moodle Face Attendance plugin.

## What changed

- The raw `api_secret` is no longer sent in URLs, JSON bodies, or `X-FaceAttendance-Secret` headers.
- Every request includes:
  - `X-FaceAttendance-Station`
  - `X-FaceAttendance-Timestamp`
  - `X-FaceAttendance-Nonce`
  - `X-FaceAttendance-Signature`
- Moodle verifies the signature, rejects old timestamps, and rejects reused nonces.

## Required config

Set the same long random secret in the Moodle activity and in the station config:

```json
{
  "moodle_base_url": "https://your-moodle.example.ro",
  "cmid": 20,
  "api_secret": "put-the-long-random-secret-here",
  "station_id": "camera-lab-1",
  "verify_tls": true
}
```

For production, keep `verify_tls` set to `true`. For self-signed certificates, install the local CA certificate on the station machine instead of disabling TLS verification.
