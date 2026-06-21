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
require_once(__DIR__ . '/station_security.php');
require_once(__DIR__ . '/embedding_crypto.php');

function faceattendance_station_response($payload, int $status = 200): void {
    http_response_code($status);
    header('Content-Type: application/json; charset=utf-8');
    echo json_encode($payload, JSON_UNESCAPED_SLASHES);
    exit;
}


try {
    global $DB, $PAGE;

    $cmid = required_param('cmid', PARAM_INT);

    $cm = get_coursemodule_from_id('faceattendance', $cmid, 0, false, MUST_EXIST);
    $course = get_course($cm->course);
    $context = context_module::instance($cm->id);

    // This endpoint is authenticated by signed HMAC station requests, not by a Moodle browser session.
    // Some Moodle helpers used below, for example format_string(), still require
    // PAGE context to be set even for AJAX/API scripts.
    $PAGE->set_url(new moodle_url('/mod/faceattendance/api/station_bootstrap.php', [
        'cmid' => $cmid,
    ]));
    $PAGE->set_context($context);
    $PAGE->set_course($course);
    $PAGE->set_cm($cm, $course);

    $faceattendance = $DB->get_record('faceattendance', ['id' => $cm->instance], '*', MUST_EXIST);

    $profilelabels = [
        'fast_short' => 'Fast short labels',
        'many_faces_unknown' => 'Many faces + unknown review',
        'fast_clean' => 'Fast clean known-only display',
        'high_recall_many_faces' => 'High recall many faces',
        'multi_attendance_zoom' => 'Multi-attendance auto-zoom',
        'entrance_mode' => 'Entrance/door mode',
    ];
    $modelprofile = optional_param('profile', '', PARAM_ALPHANUMEXT);
    if ($modelprofile === '') {
        $modelprofile = optional_param('modelprofile', '', PARAM_ALPHANUMEXT);
    }
    if ($modelprofile === '') {
        $modelprofile = 'fast_short';
    }
    if (!array_key_exists($modelprofile, $profilelabels)) {
        $modelprofile = 'fast_short';
    }

    $stationid = faceattendance_station_require_signed_request($faceattendance, '');

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
              FROM {faceattendance_course_embeddings} e
              JOIN {user} u ON u.id = e.userid
             WHERE e.course = :course
               AND e.status IN ('registered', 'active', 'approved')
          ORDER BY u.lastname ASC, u.firstname ASC";
    $records = $DB->get_records_sql($sql, [
        'course' => $course->id,
    ]);

    $faces = [];
    $encryptedcount = 0;
    foreach ($records as $record) {
        $decoded = json_decode($record->embeddingjson, true);
        if (faceattendance_embedding_is_encrypted_payload($decoded)) {
            // Do not decrypt on the Moodle side. The encrypted envelope is delivered
            // only after the HMAC station request was authenticated; the station then
            // opens it with its private key and expands the captures locally.
            $faces[] = [
                'userid' => (int)$record->userid,
                'studentId' => (string)$record->userid,
                'name' => fullname($record),
                // Minimize personal data sent to the station; username/email are not needed
                // for local biometric comparison.
                'encryptedEmbedding' => $decoded,
                'sourceFile' => 'moodle_encrypted_embedding_' . $record->id,
            ];
            $encryptedcount++;
            continue;
        }

        // Backward compatibility for records saved before encryption was enabled.
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
        'stationId' => $stationid,
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
        'encryptedEmbeddingRecords' => $encryptedcount,
    ]);
} catch (Throwable $e) {
    faceattendance_station_response(['ok' => false, 'error' => $e->getMessage()], 500);
}
