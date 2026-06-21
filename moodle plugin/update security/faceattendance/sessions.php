<?php
// This file is part of Moodle - http://moodle.org/

/**
 * Teacher page for creating scheduled face attendance sessions.
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
require_capability('mod/faceattendance:manage', $context);

$pageurl = new moodle_url('/mod/faceattendance/sessions.php', ['id' => $cm->id]);
$returnurl = new moodle_url('/mod/faceattendance/view.php', ['id' => $cm->id]);
$stationurl = new moodle_url('/mod/faceattendance/station.php', ['id' => $cm->id]);

$PAGE->set_url($pageurl);
$PAGE->set_title(get_string('managesessions', 'faceattendance'));
$PAGE->set_heading(format_string($course->fullname));
$PAGE->set_context($context);

function faceattendance_parse_datetime_local(string $value): int {
    $value = trim($value);
    if ($value === '') {
        return 0;
    }
    $timestamp = strtotime(str_replace('T', ' ', $value));
    return $timestamp === false ? 0 : $timestamp;
}

$formvalues = [
    'name' => get_string('defaultsessionname', 'faceattendance'),
    'starttime' => '',
    'endtime' => '',
    'latethreshold' => '10',
    'mindetections' => '3',
];
$formerrors = [];

if ($_SERVER['REQUEST_METHOD'] === 'POST') {
    require_sesskey();
    $action = required_param('action', PARAM_ALPHA);

    if ($action === 'create') {
        $name = trim(optional_param('name', '', PARAM_TEXT));
        $starttext = optional_param('starttime', '', PARAM_RAW_TRIMMED);
        $endtext = optional_param('endtime', '', PARAM_RAW_TRIMMED);
        $latethresholdminutes = optional_param('latethreshold', 10, PARAM_INT);
        $mindetectionsraw = optional_param('mindetections', 3, PARAM_INT);

        $formvalues = [
            'name' => $name,
            'starttime' => $starttext,
            'endtime' => $endtext,
            'latethreshold' => (string)$latethresholdminutes,
            'mindetections' => (string)$mindetectionsraw,
        ];

        $starttime = faceattendance_parse_datetime_local($starttext);
        $endtime = faceattendance_parse_datetime_local($endtext);
        $latethresholdminutes = max(0, $latethresholdminutes);
        $latethreshold = $latethresholdminutes * 60;
        $mindetections = max(1, $mindetectionsraw);

        if ($name === '') {
            $formerrors[] = get_string('sessionnamerequired', 'faceattendance');
        }
        if ($starttime <= 0 || $endtime <= 0 || $endtime <= $starttime) {
            $formerrors[] = get_string('invalidsessiontime', 'faceattendance');
        }
        if ($mindetectionsraw < 1) {
            $formerrors[] = get_string('mindetectionsinvalid', 'faceattendance');
        }
        if ($latethresholdminutes < 0) {
            $formerrors[] = get_string('latethresholdinvalid', 'faceattendance');
        }

        if ($formerrors) {
            // Do not redirect. Keep the teacher on this page with all valid fields still filled in.
        } else {
            $now = time();
            $session = (object)[
            'faceattendanceid' => (int)$faceattendance->id,
            'course' => (int)$course->id,
            'name' => $name,
            'starttime' => $starttime,
            'endtime' => $endtime,
            'latethreshold' => $latethreshold,
            'mindetections' => $mindetections,
            'status' => 'scheduled',
            'createdby' => (int)$USER->id,
            'timecreated' => $now,
            'timemodified' => $now,
        ];
            $DB->insert_record('faceattendance_sessions', $session);
            redirect($pageurl, get_string('sessioncreated', 'faceattendance'), null, \core\output\notification::NOTIFY_SUCCESS);
        }
    }

    if ($action === 'cancel') {
        $sessionid = required_param('sessionid', PARAM_INT);
        $session = $DB->get_record('faceattendance_sessions', [
            'id' => $sessionid,
            'faceattendanceid' => $faceattendance->id,
        ], '*', MUST_EXIST);
        $session->status = 'cancelled';
        $session->timemodified = time();
        $DB->update_record('faceattendance_sessions', $session);
        redirect($pageurl, get_string('sessioncancelled', 'faceattendance'), null, \core\output\notification::NOTIFY_SUCCESS);
    }
}

$sessions = $DB->get_records('faceattendance_sessions', ['faceattendanceid' => $faceattendance->id], 'starttime DESC');

echo $OUTPUT->header();
echo $OUTPUT->heading(get_string('managesessions', 'faceattendance'));
echo html_writer::tag('p', get_string('sessionsintro', 'faceattendance'), ['class' => 'alert alert-info']);
if (!empty($formerrors)) {
    echo $OUTPUT->notification(implode(html_writer::empty_tag('br'), array_map('s', $formerrors)), 'error');
}

echo html_writer::start_tag('form', ['method' => 'post', 'class' => 'mb-4']);
echo html_writer::empty_tag('input', ['type' => 'hidden', 'name' => 'id', 'value' => $cm->id]);
echo html_writer::empty_tag('input', ['type' => 'hidden', 'name' => 'sesskey', 'value' => sesskey()]);
echo html_writer::empty_tag('input', ['type' => 'hidden', 'name' => 'action', 'value' => 'create']);
echo html_writer::start_div('card card-body');
echo html_writer::tag('h3', get_string('createsession', 'faceattendance'));
echo html_writer::div('<label>' . get_string('sessionname', 'faceattendance') . '</label>' . html_writer::empty_tag('input', ['type' => 'text', 'name' => 'name', 'required' => 'required', 'class' => 'form-control', 'value' => $formvalues['name']]));
echo html_writer::div('<label>' . get_string('starttime', 'faceattendance') . '</label>' . html_writer::empty_tag('input', ['type' => 'datetime-local', 'name' => 'starttime', 'required' => 'required', 'class' => 'form-control', 'value' => $formvalues['starttime']]), 'mt-2');
echo html_writer::div('<label>' . get_string('endtime', 'faceattendance') . '</label>' . html_writer::empty_tag('input', ['type' => 'datetime-local', 'name' => 'endtime', 'required' => 'required', 'class' => 'form-control', 'value' => $formvalues['endtime']]), 'mt-2');
echo html_writer::div('<label>' . get_string('latethresholdminutes', 'faceattendance') . '</label>' . html_writer::empty_tag('input', ['type' => 'number', 'name' => 'latethreshold', 'min' => '0', 'class' => 'form-control', 'value' => $formvalues['latethreshold']]), 'mt-2');
echo html_writer::div('<label>' . get_string('mindetections', 'faceattendance') . '</label>' . html_writer::empty_tag('input', ['type' => 'number', 'name' => 'mindetections', 'min' => '1', 'class' => 'form-control', 'value' => $formvalues['mindetections']]), 'mt-2');
echo html_writer::empty_tag('input', ['type' => 'submit', 'class' => 'btn btn-primary mt-3', 'value' => get_string('createsession', 'faceattendance')]);
echo html_writer::end_div();
echo html_writer::end_tag('form');

if ($sessions) {
    $table = new html_table();
    $table->head = [get_string('sessionname', 'faceattendance'), get_string('starttime', 'faceattendance'), get_string('endtime', 'faceattendance'), get_string('latethresholdminutes', 'faceattendance'), get_string('mindetections', 'faceattendance'), get_string('status', 'faceattendance'), get_string('actions')];
    $table->data = [];
    foreach ($sessions as $session) {
        $cancel = '';
        if ($session->status !== 'cancelled') {
            $cancel = html_writer::start_tag('form', ['method' => 'post', 'style' => 'display:inline']) .
                html_writer::empty_tag('input', ['type' => 'hidden', 'name' => 'id', 'value' => $cm->id]) .
                html_writer::empty_tag('input', ['type' => 'hidden', 'name' => 'sesskey', 'value' => sesskey()]) .
                html_writer::empty_tag('input', ['type' => 'hidden', 'name' => 'action', 'value' => 'cancel']) .
                html_writer::empty_tag('input', ['type' => 'hidden', 'name' => 'sessionid', 'value' => $session->id]) .
                html_writer::empty_tag('input', ['type' => 'submit', 'class' => 'btn btn-sm btn-outline-danger', 'value' => get_string('cancel')]) .
                html_writer::end_tag('form');
        }
        $table->data[] = [
            format_string($session->name),
            userdate((int)$session->starttime),
            userdate((int)$session->endtime),
            round(((int)$session->latethreshold) / 60),
            (int)$session->mindetections,
            s($session->status),
            $cancel,
        ];
    }
    echo html_writer::table($table);
} else {
    echo $OUTPUT->notification(get_string('nosessions', 'faceattendance'), 'info');
}

echo html_writer::div(
    html_writer::link($stationurl, get_string('openstation', 'faceattendance'), ['class' => 'btn btn-warning mr-2']) .
    html_writer::link($returnurl, get_string('backtoactivity', 'faceattendance'), ['class' => 'btn btn-secondary']),
    'mt-3'
);

echo $OUTPUT->footer();
