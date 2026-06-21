<?php
// This file is part of Moodle - http://moodle.org/

/**
 * Attendance report for mod_faceattendance.
 *
 * @package     mod_faceattendance
 * @copyright   2026
 * @license     http://www.gnu.org/copyleft/gpl.html GNU GPL v3 or later
 */

require_once(__DIR__ . '/../../config.php');

$id = required_param('id', PARAM_INT); // Course module id.

$cm = get_coursemodule_from_id('faceattendance', $id, 0, false, MUST_EXIST);
$course = get_course($cm->course);
$faceattendance = $DB->get_record('faceattendance', ['id' => $cm->instance], '*', MUST_EXIST);

require_login($course, true, $cm);

$context = context_module::instance($cm->id);
require_capability('mod/faceattendance:viewreports', $context);

$PAGE->set_url('/mod/faceattendance/report.php', ['id' => $cm->id]);
$PAGE->set_title(get_string('report', 'faceattendance'));
$PAGE->set_heading(format_string($course->fullname));
$PAGE->set_context($context);

$sql = "SELECT sr.*, s.name AS sessionname, s.starttime, s.endtime, u.firstname, u.lastname, u.email, u.username
          FROM {faceattendance_session_records} sr
          JOIN {faceattendance_sessions} s ON s.id = sr.sessionid
          JOIN {user} u ON u.id = sr.userid
         WHERE sr.faceattendanceid = :faceattendanceid
      ORDER BY s.starttime DESC, u.lastname ASC, u.firstname ASC";
$sessionrecords = $DB->get_records_sql($sql, ['faceattendanceid' => $faceattendance->id]);

$sql = "SELECT r.*, u.firstname, u.lastname, u.email, u.username
          FROM {faceattendance_records} r
          JOIN {user} u ON u.id = r.userid
         WHERE r.faceattendanceid = :faceattendanceid
      ORDER BY r.timemodified DESC";
$legacyrecords = $DB->get_records_sql($sql, ['faceattendanceid' => $faceattendance->id]);

echo $OUTPUT->header();
echo $OUTPUT->heading(get_string('reportfor', 'faceattendance', format_string($faceattendance->name)));

if ($sessionrecords) {
    echo $OUTPUT->heading(get_string('scheduledsessionrecords', 'faceattendance'), 3);
    $table = new html_table();
    $table->head = [
        get_string('sessionname', 'faceattendance'),
        get_string('student', 'faceattendance'),
        get_string('email'),
        get_string('status', 'faceattendance'),
        get_string('confidence', 'faceattendance'),
        get_string('detections', 'faceattendance'),
        get_string('firstseen', 'faceattendance'),
        get_string('lastseen', 'faceattendance'),
        get_string('source', 'faceattendance'),
    ];
    $table->data = [];

    foreach ($sessionrecords as $record) {
        $table->data[] = [
            format_string($record->sessionname),
            s(fullname($record)),
            s($record->email),
            s($record->status),
            format_float($record->confidence, 3),
            (int)$record->detectioncount,
            userdate((int)$record->firstseen),
            userdate((int)$record->lastseen),
            s($record->source),
        ];
    }
    echo html_writer::table($table);
}

if ($legacyrecords) {
    echo $OUTPUT->heading(get_string('legacyapirecords', 'faceattendance'), 3);
    $table = new html_table();
    $table->head = [
        get_string('student', 'faceattendance'),
        get_string('email'),
        get_string('status', 'faceattendance'),
        get_string('confidence', 'faceattendance'),
        get_string('source', 'faceattendance'),
        get_string('timecreated', 'faceattendance'),
        get_string('timemodified', 'faceattendance'),
    ];
    $table->data = [];

    foreach ($legacyrecords as $record) {
        $table->data[] = [
            s(fullname($record)),
            s($record->email),
            s($record->status),
            format_float($record->confidence, 3),
            s($record->source),
            userdate($record->timecreated),
            userdate($record->timemodified),
        ];
    }
    echo html_writer::table($table);
}

if (!$sessionrecords && !$legacyrecords) {
    echo $OUTPUT->notification(get_string('norecords', 'faceattendance'), 'info');
}

echo html_writer::div(
    html_writer::link(new moodle_url('/mod/faceattendance/view.php', ['id' => $cm->id]), get_string('backtoactivity', 'faceattendance'), ['class' => 'btn btn-secondary']),
    'mt-3'
);

echo $OUTPUT->footer();
