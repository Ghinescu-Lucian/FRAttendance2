<?php
// This file is part of Moodle - http://moodle.org/

/**
 * Secret-authenticated mark endpoint for an external Python/OpenCV camera station.
 *
 * @package     mod_faceattendance
 * @license     http://www.gnu.org/copyleft/gpl.html GNU GPL v3 or later
 */

define('AJAX_SCRIPT', true);
require_once(__DIR__ . '/../../../config.php');
require_once(__DIR__ . '/station_security.php');

function faceattendance_station_response($payload, int $status = 200): void {
    http_response_code($status);
    header('Content-Type: application/json; charset=utf-8');
    echo json_encode($payload, JSON_UNESCAPED_SLASHES);
    exit;
}

function faceattendance_station_raw_body(): string {
    return file_get_contents('php://input') ?: '';
}

function faceattendance_station_payload(string $raw): array {
    $payload = json_decode($raw ?: '', true);
    if (json_last_error() !== JSON_ERROR_NONE || !is_array($payload)) {
        faceattendance_station_response(['ok' => false, 'error' => 'Invalid JSON body: ' . json_last_error_msg()], 400);
    }
    return $payload;
}

try {
    global $DB, $PAGE;

    $rawbody = faceattendance_station_raw_body();
    $payload = faceattendance_station_payload($rawbody);
    $cmid = isset($payload['cmid']) ? clean_param($payload['cmid'], PARAM_INT) : 0;
    $sessionid = isset($payload['sessionid']) ? clean_param($payload['sessionid'], PARAM_INT) : 0;
    $userid = isset($payload['userid']) ? clean_param($payload['userid'], PARAM_INT) : 0;
    $confidence = isset($payload['confidence']) ? (float)$payload['confidence'] : 0.0;
    $distance = isset($payload['distance']) ? (float)$payload['distance'] : 0.0;
    $source = isset($payload['source']) ? clean_param((string)$payload['source'], PARAM_TEXT) : 'opencv-python-station';

    if ($cmid <= 0 || $sessionid <= 0 || $userid <= 0) {
        faceattendance_station_response(['ok' => false, 'error' => 'cmid, sessionid and userid are required.'], 400);
    }

    $cm = get_coursemodule_from_id('faceattendance', $cmid, 0, false, MUST_EXIST);
    $course = get_course($cm->course);
    $context = context_module::instance($cm->id);

    $PAGE->set_url(new moodle_url('/mod/faceattendance/api/station_mark.php'));
    $PAGE->set_context($context);
    $PAGE->set_course($course);
    $PAGE->set_cm($cm, $course);

    $faceattendance = $DB->get_record('faceattendance', ['id' => $cm->instance], '*', MUST_EXIST);

    $stationid = faceattendance_station_require_signed_request($faceattendance, $rawbody);

    $session = $DB->get_record('faceattendance_sessions', [
        'id' => $sessionid,
        'faceattendanceid' => $faceattendance->id,
        'course' => $course->id,
    ], '*', MUST_EXIST);

    $coursecontext = context_course::instance($course->id);
    $user = $DB->get_record('user', ['id' => $userid, 'deleted' => 0], '*', MUST_EXIST);
    if (!is_enrolled($coursecontext, $user, '', true)) {
        faceattendance_station_response(['ok' => false, 'error' => 'Detected user is not enrolled in this course.'], 403);
    }

    $now = time();
    $status = ($now > ((int)$session->starttime + (int)$session->latethreshold)) ? 'late' : 'present';

    $existing = $DB->get_record('faceattendance_session_records', [
        'sessionid' => $session->id,
        'userid' => $userid,
    ]);

    if ($existing) {
        $existing->confidence = max((float)$existing->confidence, $confidence);
        $existing->distance = $distance > 0 ? min((float)$existing->distance ?: $distance, $distance) : (float)$existing->distance;
        $existing->detectioncount = (int)$existing->detectioncount + 1;
        $existing->lastseen = $now;
        $existing->source = $source;
        $existing->timemodified = $now;
        $DB->update_record('faceattendance_session_records', $existing);
        $recordid = $existing->id;
        $saved = 'updated';
    } else {
        $record = (object)[
            'sessionid' => (int)$session->id,
            'faceattendanceid' => (int)$faceattendance->id,
            'course' => (int)$course->id,
            'userid' => $userid,
            'status' => $status,
            'confidence' => $confidence,
            'distance' => $distance,
            'detectioncount' => 1,
            'firstseen' => $now,
            'lastseen' => $now,
            'source' => $source,
            'timecreated' => $now,
            'timemodified' => $now,
        ];
        $recordid = $DB->insert_record('faceattendance_session_records', $record);
        $saved = 'created';
    }

    $DB->insert_record('faceattendance_detections', (object)[
        'sessionid' => (int)$session->id,
        'faceattendanceid' => (int)$faceattendance->id,
        'course' => (int)$course->id,
        'userid' => $userid,
        'decision' => 'matched',
        'confidence' => $confidence,
        'distance' => $distance,
        'source' => $source,
        'rawpayload' => json_encode($payload + ['stationid' => $stationid], JSON_UNESCAPED_SLASHES),
        'timecreated' => $now,
    ]);

    faceattendance_station_response([
        'ok' => true,
        'saved' => $saved,
        'recordid' => (int)$recordid,
        'userid' => $userid,
        'name' => fullname($user),
        'status' => $status,
    ]);
} catch (Throwable $e) {
    faceattendance_station_response(['ok' => false, 'error' => $e->getMessage()], 500);
}
