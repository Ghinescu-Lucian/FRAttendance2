<?php
// This file is part of Moodle - http://moodle.org/

/**
 * Simple JSON API endpoint used by the external face-recognition program.
 *
 * Example POST body:
 * {
 *   "cmid": 12,
 *   "secret": "dev-secret",
 *   "detections": [
 *     {"externalid": "face_student_001", "confidence": 0.92, "source": "camera-lab-1"}
 *   ]
 * }
 *
 * The endpoint can resolve users by externalid, userid, email, or username.
 *
 * @package     mod_faceattendance
 * @copyright   2026
 * @license     http://www.gnu.org/copyleft/gpl.html GNU GPL v3 or later
 */

define('AJAX_SCRIPT', true);
require_once(__DIR__ . '/../../../config.php');

function faceattendance_api_response($payload, int $status = 200): void {
    http_response_code($status);
    header('Content-Type: application/json; charset=utf-8');
    echo json_encode($payload, JSON_UNESCAPED_SLASHES);
    exit;
}

function faceattendance_get_json_payload(): array {
    $raw = file_get_contents('php://input');
    if (empty($raw)) {
        return [];
    }
    $payload = json_decode($raw, true);
    if (!is_array($payload)) {
        faceattendance_api_response(['ok' => false, 'error' => 'Invalid JSON body.'], 400);
    }
    return $payload;
}

function faceattendance_detection_externalid(array $detection): ?string {
    foreach (['externalid', 'external_id', 'faceid', 'face_id', 'label'] as $field) {
        if (!empty($detection[$field])) {
            return clean_param((string)$detection[$field], PARAM_TEXT);
        }
    }
    return null;
}

function faceattendance_resolve_user(array $detection, int $courseid, int $faceattendanceid): ?stdClass {
    global $DB;

    $user = null;
    $externalid = faceattendance_detection_externalid($detection);

    if ($externalid !== null && $externalid !== '') {
        $registration = $DB->get_record('faceattendance_registrations', [
            'faceattendanceid' => $faceattendanceid,
            'externalid' => $externalid,
        ]);

        if ($registration) {
            $user = $DB->get_record('user', ['id' => $registration->userid, 'deleted' => 0]);
        }
    }

    if (!$user && !empty($detection['userid'])) {
        $user = $DB->get_record('user', ['id' => clean_param($detection['userid'], PARAM_INT), 'deleted' => 0]);
    } else if (!$user && !empty($detection['email'])) {
        $user = $DB->get_record('user', ['email' => clean_param($detection['email'], PARAM_EMAIL), 'deleted' => 0]);
    } else if (!$user && !empty($detection['username'])) {
        $user = $DB->get_record('user', ['username' => clean_param($detection['username'], PARAM_USERNAME), 'deleted' => 0]);
    }

    if (!$user) {
        return null;
    }

    $coursecontext = context_course::instance($courseid);
    if (!is_enrolled($coursecontext, $user, '', true)) {
        return null;
    }

    return $user;
}

$payload = faceattendance_get_json_payload();

$cmid = optional_param('cmid', 0, PARAM_INT);
if (!$cmid && isset($payload['cmid'])) {
    $cmid = clean_param($payload['cmid'], PARAM_INT);
}

$secret = $_SERVER['HTTP_X_FACEATTENDANCE_SECRET'] ?? optional_param('secret', '', PARAM_RAW_TRIMMED);
if ($secret === '' && isset($payload['secret'])) {
    $secret = trim((string)$payload['secret']);
}

if (!$cmid) {
    faceattendance_api_response(['ok' => false, 'error' => 'Missing cmid.'], 400);
}

$cm = get_coursemodule_from_id('faceattendance', $cmid, 0, false, MUST_EXIST);
$course = get_course($cm->course);
$faceattendance = $DB->get_record('faceattendance', ['id' => $cm->instance], '*', MUST_EXIST);

if (empty($faceattendance->apisecret) || !hash_equals((string)$faceattendance->apisecret, (string)$secret)) {
    faceattendance_api_response(['ok' => false, 'error' => 'Invalid API secret.'], 403);
}

$detections = [];
if (!empty($payload['detections']) && is_array($payload['detections'])) {
    $detections = $payload['detections'];
} else if (!empty($payload['students']) && is_array($payload['students'])) {
    $detections = $payload['students'];
}

if (empty($detections)) {
    faceattendance_api_response(['ok' => false, 'error' => 'No detections provided.'], 400);
}

$threshold = (float)$faceattendance->confidence;
$now = time();
$created = 0;
$updated = 0;
$ignored = [];

foreach ($detections as $index => $detection) {
    if (!is_array($detection)) {
        $ignored[] = ['index' => $index, 'reason' => 'Invalid detection item.'];
        continue;
    }

    $confidence = isset($detection['confidence']) ? (float)$detection['confidence'] : 0.0;
    $source = isset($detection['source']) ? clean_param($detection['source'], PARAM_TEXT) : 'external-face-program';
    $externalid = faceattendance_detection_externalid($detection);

    $user = faceattendance_resolve_user($detection, (int)$course->id, (int)$faceattendance->id);
    if (!$user) {
        $ignored[] = ['index' => $index, 'reason' => 'User not found, not registered, or not enrolled.'];
        continue;
    }

    $status = $confidence >= $threshold ? 'present' : 'review';

    $identifier = $user->username;
    if ($externalid !== null && $externalid !== '') {
        $identifier = $externalid;
    } else if (!empty($detection['email'])) {
        $identifier = clean_param($detection['email'], PARAM_EMAIL);
    } else if (!empty($detection['username'])) {
        $identifier = clean_param($detection['username'], PARAM_USERNAME);
    }

    $recorddata = (object)[
        'faceattendanceid' => (int)$faceattendance->id,
        'course' => (int)$course->id,
        'userid' => (int)$user->id,
        'identifier' => $identifier,
        'confidence' => $confidence,
        'status' => $status,
        'source' => $source,
        'rawpayload' => json_encode($detection, JSON_UNESCAPED_SLASHES),
        'timemodified' => $now,
    ];

    $existing = $DB->get_record('faceattendance_records', [
        'faceattendanceid' => $faceattendance->id,
        'userid' => $user->id,
    ]);

    if ($existing) {
        $recorddata->id = $existing->id;
        $recorddata->timecreated = $existing->timecreated;

        // Keep the best confidence if the student is detected multiple times.
        if ((float)$existing->confidence > $confidence) {
            $recorddata->confidence = (float)$existing->confidence;
            $recorddata->status = $existing->status;
        }

        $DB->update_record('faceattendance_records', $recorddata);
        $updated++;
    } else {
        $recorddata->timecreated = $now;
        $DB->insert_record('faceattendance_records', $recorddata);
        $created++;
    }
}

faceattendance_api_response([
    'ok' => true,
    'created' => $created,
    'updated' => $updated,
    'ignored' => $ignored,
]);
