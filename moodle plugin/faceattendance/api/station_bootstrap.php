<?php
// This file is part of Moodle - http://moodle.org/

/**
 * Secret-authenticated bootstrap endpoint for an external Python/OpenCV camera station.
 * Returns the active session and all registered SFace embeddings for the activity.
 *
 * @package     mod_faceattendance
 * @license     http://www.gnu.org/copyleft/gpl.html GNU GPL v3 or later
 */

define('AJAX_SCRIPT', true);
require_once(__DIR__ . '/../../../config.php');

function faceattendance_station_response($payload, int $status = 200): void {
    http_response_code($status);
    header('Content-Type: application/json; charset=utf-8');
    echo json_encode($payload, JSON_UNESCAPED_SLASHES);
    exit;
}

function faceattendance_station_secret(): string {
    $secret = $_SERVER['HTTP_X_FACEATTENDANCE_SECRET'] ?? optional_param('secret', '', PARAM_RAW_TRIMMED);
    return trim((string)$secret);
}

try {
    global $DB;

    $cmid = required_param('cmid', PARAM_INT);
    $secret = faceattendance_station_secret();

    $cm = get_coursemodule_from_id('faceattendance', $cmid, 0, false, MUST_EXIST);
    $course = get_course($cm->course);
    $faceattendance = $DB->get_record('faceattendance', ['id' => $cm->instance], '*', MUST_EXIST);

    $profilelabels = [
        'fast_short' => 'Fast short labels',
        'many_faces_unknown' => 'Many faces + unknown review',
        'fast_clean' => 'Fast clean known-only display',
        'high_recall_many_faces' => 'High recall many faces',
        'multi_attendance_zoom' => 'Multi-attendance auto-zoom',
        'entrance_mode' => 'Entrance/door mode',
    ];
    $modelprofile = !empty($faceattendance->modelprofile) ? (string)$faceattendance->modelprofile : 'fast_short';
    if (!array_key_exists($modelprofile, $profilelabels)) {
        $modelprofile = 'fast_short';
    }

    if (empty($faceattendance->apisecret) || !hash_equals((string)$faceattendance->apisecret, $secret)) {
        faceattendance_station_response(['ok' => false, 'error' => 'Invalid API secret.'], 403);
    }

    $now = time();
    $sessions = $DB->get_records_select('faceattendance_sessions',
        'faceattendanceid = :faceattendanceid AND starttime <= :now1 AND endtime >= :now2 AND status <> :cancelled',
        [
            'faceattendanceid' => $faceattendance->id,
            'now1' => $now,
            'now2' => $now,
            'cancelled' => 'cancelled',
        ],
        'starttime ASC',
        '*',
        0,
        1
    );
    $session = $sessions ? reset($sessions) : false;

    $sql = "SELECT e.*, u.firstname, u.lastname, u.email, u.username
              FROM {faceattendance_embeddings} e
              JOIN {user} u ON u.id = e.userid
             WHERE e.faceattendanceid = :faceattendanceid
               AND e.course = :course
               AND e.status IN ('registered', 'active', 'approved')
          ORDER BY u.lastname ASC, u.firstname ASC";
    $records = $DB->get_records_sql($sql, [
        'faceattendanceid' => $faceattendance->id,
        'course' => $course->id,
    ]);

    $faces = [];
    foreach ($records as $record) {
        $payload = json_decode($record->embeddingjson);
        if (!is_object($payload) || empty($payload->captures) || !is_array($payload->captures)) {
            continue;
        }
        foreach ($payload->captures as $index => $capture) {
            if (!is_object($capture) || empty($capture->descriptor) || !is_array($capture->descriptor)) {
                continue;
            }
            if (count($capture->descriptor) !== 128) {
                continue;
            }
            $faces[] = [
                'userid' => (int)$record->userid,
                'studentId' => (string)$record->userid,
                'name' => fullname($record),
                'username' => $record->username,
                'email' => $record->email,
                'pose' => isset($capture->pose) ? (string)$capture->pose : 'unknown',
                'label' => isset($capture->label) ? (string)$capture->label : 'sample_' . ($index + 1),
                'descriptor' => array_map('floatval', $capture->descriptor),
                'sourceFile' => 'moodle_embedding_' . $record->id,
            ];
        }
    }

    faceattendance_station_response([
        'ok' => true,
        'serverTime' => $now,
        'cmid' => (int)$cm->id,
        'courseid' => (int)$course->id,
        'faceattendanceid' => (int)$faceattendance->id,
        'activityName' => format_string($faceattendance->name),
        'confidenceThreshold' => (float)$faceattendance->confidence,
        'modelProfile' => $modelprofile,
        'modelProfileLabel' => $profilelabels[$modelprofile],
        'active' => $session ? true : false,
        'session' => $session ? [
            'id' => (int)$session->id,
            'name' => format_string($session->name),
            'starttime' => (int)$session->starttime,
            'endtime' => (int)$session->endtime,
            'latethreshold' => (int)$session->latethreshold,
            'mindetections' => (int)$session->mindetections,
            'status' => $session->status,
        ] : null,
        'faces' => $faces,
        'count' => count($faces),
    ]);
} catch (Throwable $e) {
    faceattendance_station_response(['ok' => false, 'error' => $e->getMessage()], 500);
}
