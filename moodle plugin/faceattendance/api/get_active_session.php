<?php
// This file is part of Moodle - http://moodle.org/

/**
 * Returns the currently active scheduled session for a Face Attendance activity.
 *
 * @package     mod_faceattendance
 * @copyright   2026
 * @license     http://www.gnu.org/copyleft/gpl.html GNU GPL v3 or later
 */

require_once(__DIR__ . '/../../../config.php');

header('Content-Type: application/json; charset=utf-8');

function faceattendance_json_response($data, int $status = 200): void {
    http_response_code($status);
    echo json_encode($data, JSON_UNESCAPED_SLASHES);
    exit;
}

try {
    $cmid = required_param('cmid', PARAM_INT);

    $cm = get_coursemodule_from_id('faceattendance', $cmid, 0, false, MUST_EXIST);
    $course = get_course($cm->course);
    $faceattendance = $DB->get_record('faceattendance', ['id' => $cm->instance], '*', MUST_EXIST);

    require_login($course, true, $cm);
    $context = context_module::instance($cm->id);
    require_capability('mod/faceattendance:takeattendance', $context);

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

    if (!$session) {
        faceattendance_json_response([
            'ok' => true,
            'active' => false,
            'message' => 'No active attendance session right now.',
        ]);
    }

    faceattendance_json_response([
        'ok' => true,
        'active' => true,
        'session' => [
            'id' => (int)$session->id,
            'name' => format_string($session->name),
            'starttime' => (int)$session->starttime,
            'endtime' => (int)$session->endtime,
            'latethreshold' => (int)$session->latethreshold,
            'mindetections' => (int)$session->mindetections,
            'status' => $session->status,
        ],
    ]);
} catch (Throwable $e) {
    faceattendance_json_response(['ok' => false, 'error' => $e->getMessage()], 500);
}
