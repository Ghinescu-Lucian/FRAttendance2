<?php
// This file is part of Moodle - http://moodle.org/

/**
 * Returns enrolled SFace embeddings for the classroom camera station.
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
                'email' => $record->email,
                'pose' => isset($capture->pose) ? (string)$capture->pose : 'unknown',
                'label' => isset($capture->label) ? (string)$capture->label : 'sample_' . ($index + 1),
                'descriptor' => array_map('floatval', $capture->descriptor),
                'sourceFile' => 'moodle_embedding_' . $record->id,
            ];
        }
    }

    faceattendance_json_response([
        'ok' => true,
        'cmid' => (int)$cm->id,
        'courseid' => (int)$course->id,
        'faceattendanceid' => (int)$faceattendance->id,
        'faces' => $faces,
        'count' => count($faces),
    ]);
} catch (Throwable $e) {
    faceattendance_json_response(['ok' => false, 'error' => $e->getMessage()], 500);
}
