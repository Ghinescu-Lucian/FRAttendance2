<?php
// This file is part of Moodle - http://moodle.org/

/**
 * HMAC request authentication helpers for external Face Attendance camera stations.
 *
 * The station secret is never transmitted. Instead, each request is signed with:
 * HMAC-SHA256(secret, METHOD + "\n" + request-target + "\n" + timestamp + "\n" + nonce + "\n" + stationid + "\n" + SHA256(raw-body)).
 *
 * @package     mod_faceattendance
 * @license     http://www.gnu.org/copyleft/gpl.html GNU GPL v3 or later
 */

defined('MOODLE_INTERNAL') || die();

/** Maximum tolerated clock skew for signed station requests. */
define('FACEATTENDANCE_STATION_MAX_CLOCK_SKEW', 120);

/** How long anti-replay nonces are retained. */
define('FACEATTENDANCE_STATION_NONCE_TTL', 600);

function faceattendance_station_header(string $name): string {
    $serverkey = 'HTTP_' . strtoupper(str_replace('-', '_', $name));
    return trim((string)($_SERVER[$serverkey] ?? ''));
}

function faceattendance_station_request_target(): string {
    $uri = (string)($_SERVER['REQUEST_URI'] ?? '');
    if ($uri === '') {
        return (string)($_SERVER['SCRIPT_NAME'] ?? '');
    }

    $path = parse_url($uri, PHP_URL_PATH);
    $query = parse_url($uri, PHP_URL_QUERY);
    if ($path === null || $path === false || $path === '') {
        $path = (string)($_SERVER['SCRIPT_NAME'] ?? '/');
    }

    return $query !== null && $query !== false && $query !== '' ? $path . '?' . $query : $path;
}

function faceattendance_station_require_signed_request(stdClass $faceattendance, string $rawbody = ''): string {
    global $DB;

    if (empty($faceattendance->apisecret)) {
        faceattendance_station_response(['ok' => false, 'error' => 'Station API secret is not configured.'], 403);
    }

    $stationid = faceattendance_station_header('X-FaceAttendance-Station');
    $timestamp = faceattendance_station_header('X-FaceAttendance-Timestamp');
    $nonce = faceattendance_station_header('X-FaceAttendance-Nonce');
    $signature = faceattendance_station_header('X-FaceAttendance-Signature');

    if ($stationid === '' || $timestamp === '' || $nonce === '' || $signature === '') {
        faceattendance_station_response([
            'ok' => false,
            'error' => 'Missing station authentication headers. Required: X-FaceAttendance-Station, X-FaceAttendance-Timestamp, X-FaceAttendance-Nonce, X-FaceAttendance-Signature.',
        ], 403);
    }

    $timestampint = clean_param($timestamp, PARAM_INT);
    $signature = strtolower($signature);

    if (!preg_match('/^[A-Za-z0-9_.:-]{1,100}$/', $stationid) ||
            !preg_match('/^[A-Za-z0-9_.:-]{1,128}$/', $nonce) ||
            !preg_match('/^[a-f0-9]{64}$/', $signature)) {
        faceattendance_station_response(['ok' => false, 'error' => 'Invalid station authentication header format.'], 403);
    }

    $now = time();
    if ($timestampint <= 0 || abs($now - $timestampint) > FACEATTENDANCE_STATION_MAX_CLOCK_SKEW) {
        faceattendance_station_response(['ok' => false, 'error' => 'Expired station request timestamp. Check station/server clock synchronization.'], 403);
    }

    $method = strtoupper((string)($_SERVER['REQUEST_METHOD'] ?? 'GET'));
    $target = faceattendance_station_request_target();
    $bodyhash = hash('sha256', $rawbody);
    $canonical = implode("\n", [$method, $target, (string)$timestampint, $nonce, $stationid, $bodyhash]);
    $expected = hash_hmac('sha256', $canonical, (string)$faceattendance->apisecret);

    if (!hash_equals($expected, $signature)) {
        faceattendance_station_response(['ok' => false, 'error' => 'Invalid station request signature.'], 403);
    }

    // Anti-replay protection: the same station/nonce cannot be accepted twice.
    $DB->delete_records_select('faceattendance_station_nonces', 'timecreated < :cutoff', [
        'cutoff' => $now - FACEATTENDANCE_STATION_NONCE_TTL,
    ]);

    $noncehash = hash('sha256', $stationid . ':' . $nonce);
    $existing = $DB->get_record('faceattendance_station_nonces', [
        'faceattendanceid' => (int)$faceattendance->id,
        'stationid' => $stationid,
        'noncehash' => $noncehash,
    ]);
    if ($existing) {
        faceattendance_station_response(['ok' => false, 'error' => 'Replay station request rejected.'], 403);
    }

    try {
        $DB->insert_record('faceattendance_station_nonces', (object)[
            'faceattendanceid' => (int)$faceattendance->id,
            'stationid' => $stationid,
            'noncehash' => $noncehash,
            'timecreated' => $now,
        ]);
    } catch (dml_exception $e) {
        faceattendance_station_response(['ok' => false, 'error' => 'Replay station request rejected.'], 403);
    }

    return $stationid;
}
