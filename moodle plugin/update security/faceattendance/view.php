<?php
// This file is part of Moodle - http://moodle.org/

/**
 * Main view page for mod_faceattendance.
 *
 * @package     mod_faceattendance
 * @copyright   2026
 * @license     http://www.gnu.org/copyleft/gpl.html GNU GPL v3 or later
 */

require_once(__DIR__ . '/../../config.php');
require_once($CFG->libdir . '/tablelib.php');

$id = required_param('id', PARAM_INT); // Course module id.

$cm = get_coursemodule_from_id('faceattendance', $id, 0, false, MUST_EXIST);
$course = get_course($cm->course);
$faceattendance = $DB->get_record('faceattendance', ['id' => $cm->instance], '*', MUST_EXIST);

require_login($course, true, $cm);

$context = context_module::instance($cm->id);
require_capability('mod/faceattendance:view', $context);

$PAGE->set_url('/mod/faceattendance/view.php', ['id' => $cm->id]);
$PAGE->set_title(format_string($faceattendance->name));
$PAGE->set_heading(format_string($course->fullname));
$PAGE->set_context($context);

$reporturl = new moodle_url('/mod/faceattendance/report.php', ['id' => $cm->id]);
$registerurl = new moodle_url('/mod/faceattendance/register.php', ['id' => $cm->id]);
$selfregisterurl = new moodle_url('/mod/faceattendance/selfregister.php', ['id' => $cm->id]);
$sessionsurl = new moodle_url('/mod/faceattendance/sessions.php', ['id' => $cm->id]);
$stationurl = new moodle_url('/mod/faceattendance/station.php', ['id' => $cm->id]);
$captureurl = new moodle_url('/mod/faceattendance/capture.php', ['id' => $cm->id]);
$capturereviewurl = new moodle_url('/mod/faceattendance/capture_review.php', ['id' => $cm->id]);
$unknownsurl = new moodle_url('/mod/faceattendance/unknowns.php', ['id' => $cm->id]);
$markurl = new moodle_url('/mod/faceattendance/api/mark.php');
$rosterurl = new moodle_url('/mod/faceattendance/api/roster.php', ['cmid' => $cm->id]);

$canmanage = has_capability('mod/faceattendance:manage', $context);
$canstudentregister = has_capability('mod/faceattendance:selfregister', $context);
$cantakeattendance = has_capability('mod/faceattendance:takeattendance', $context);
$canreviewunknowns = has_capability('mod/faceattendance:reviewunknowns', $context);


$embedding = $DB->get_record('faceattendance_course_embeddings', [
    'course' => $course->id,
    'userid' => $USER->id,
]);

$activecandidates = $DB->get_records_select('faceattendance_sessions',
    'faceattendanceid = :faceattendanceid AND starttime <= :now1 AND endtime >= :now2 AND status <> :cancelled',
    [
        'faceattendanceid' => $faceattendance->id,
        'now1' => time(),
        'now2' => time(),
        'cancelled' => 'cancelled',
    ],
    'starttime ASC',
    '*',
    0,
    1
);
$activesession = $activecandidates ? reset($activecandidates) : false;

$unknowncount = $DB->count_records('faceattendance_unknowns', [
    'faceattendanceid' => $faceattendance->id,
    'status' => 'unknown',
]);

$capturependingcount = $DB->count_records('faceattendance_capture_groups', [
    'faceattendanceid' => $faceattendance->id,
    'status' => 'pending',
]);

$upcomingsessions = $DB->get_records_select('faceattendance_sessions',
    'faceattendanceid = :faceattendanceid AND endtime >= :now AND status <> :cancelled',
    ['faceattendanceid' => $faceattendance->id, 'now' => time(), 'cancelled' => 'cancelled'],
    'starttime ASC',
    '*',
    0,
    5
);

// Keep Moodle string API simple for generated prototype labels.
function faceattendance_button(moodle_url $url, string $label, string $class = 'btn btn-secondary mr-2 mb-2'): string {
    return html_writer::link($url, $label, ['class' => $class]);
}

echo $OUTPUT->header();
echo $OUTPUT->heading(format_string($faceattendance->name));

echo format_module_intro('faceattendance', $faceattendance, $cm->id);


$buttons = '';
if ($canstudentregister) {
    $buttons .= faceattendance_button($selfregisterurl, get_string('registermyface', 'faceattendance'), 'btn btn-success mr-2 mb-2');
}
if (has_capability('mod/faceattendance:viewreports', $context)) {
    $buttons .= faceattendance_button($reporturl, get_string('viewreport', 'faceattendance'), 'btn btn-primary mr-2 mb-2');
}
if ($canmanage) {
    $buttons .= faceattendance_button($registerurl, get_string('registerstudents', 'faceattendance'));
    $buttons .= faceattendance_button($sessionsurl, get_string('managesessions', 'faceattendance'));
}
if ($cantakeattendance) {
    $buttons .= faceattendance_button($stationurl, get_string('openstation', 'faceattendance'), 'btn btn-warning mr-2 mb-2');
    $buttons .= faceattendance_button($captureurl, get_string('captureenteringfaces', 'faceattendance'), 'btn btn-dark mr-2 mb-2');
}
if ($canreviewunknowns) {
    $capturelabel = get_string('reviewcaptures', 'faceattendance');
    if ($capturependingcount > 0) {
        $capturelabel .= ' (' . $capturependingcount . ')';
    }
    $buttons .= faceattendance_button($capturereviewurl, $capturelabel, 'btn btn-info mr-2 mb-2');

    $unknownlabel = get_string('reviewunknowns', 'faceattendance');
    if ($unknowncount > 0) {
        $unknownlabel .= ' (' . $unknowncount . ')';
    }
    $buttons .= faceattendance_button($unknownsurl, $unknownlabel, 'btn btn-info mr-2 mb-2');
}

echo html_writer::div($buttons, 'mb-3');

if ($canstudentregister && !$canmanage) {
    if ($embedding) {
        $msg = get_string('yourfaceisregistered', 'faceattendance', (object)[
            'samples' => (int)$embedding->samples,
            'time' => userdate((int)$embedding->timemodified),
        ]);
        echo $OUTPUT->notification($msg, 'success');
    } else {
        echo $OUTPUT->notification(get_string('yourfaceisnotregistered', 'faceattendance'), 'info');
    }
}

if ($activesession) {
    echo $OUTPUT->notification(get_string('activesessionnotice', 'faceattendance', (object)[
        'name' => format_string($activesession->name),
        'end' => userdate((int)$activesession->endtime),
    ]), 'info');
}

if ($upcomingsessions) {
    echo $OUTPUT->heading(get_string('upcomingsessions', 'faceattendance'), 3);
    $table = new html_table();
    $table->head = [get_string('sessionname', 'faceattendance'), get_string('starttime', 'faceattendance'), get_string('endtime', 'faceattendance'), get_string('status', 'faceattendance')];
    $table->data = [];
    foreach ($upcomingsessions as $session) {
        $table->data[] = [
            format_string($session->name),
            userdate((int)$session->starttime),
            userdate((int)$session->endtime),
            s($session->status),
        ];
    }
    echo html_writer::table($table);
}

if ($canmanage) {
    echo $OUTPUT->heading(get_string('integrationdetails', 'faceattendance'), 3);

    echo html_writer::tag('p', get_string('integrationintro', 'faceattendance'));

    $sample = [
        'cmid' => (int)$cm->id,
        'secret' => 'your-api-secret-here',
        'detections' => [
            [
                'externalid' => 'face_student_001',
                'confidence' => 0.92,
                'source' => 'camera-lab-1'
            ]
        ]
    ];

    echo html_writer::tag('p', html_writer::tag('strong', 'Mark endpoint: ') . s($markurl->out(false)));
    echo html_writer::tag('p', html_writer::tag('strong', 'Roster endpoint: ') . s($rosterurl->out(false)));
    echo html_writer::tag('p', html_writer::tag('strong', 'Course module id: ') . (int)$cm->id);

    echo html_writer::tag('pre', s(json_encode($sample, JSON_PRETTY_PRINT)));

    echo html_writer::tag('p', get_string('curlhint', 'faceattendance'));
}

echo $OUTPUT->footer();
