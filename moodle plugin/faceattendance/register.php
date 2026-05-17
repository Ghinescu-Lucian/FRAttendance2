<?php
// This file is part of Moodle - http://moodle.org/

/**
 * Student registration and embedding recorder launcher for mod_faceattendance.
 *
 * @package     mod_faceattendance
 * @copyright   2026
 * @license     http://www.gnu.org/copyleft/gpl.html GNU GPL v3 or later
 */

require_once(__DIR__ . '/../../config.php');

$id = required_param('id', PARAM_INT); // Course module id.
$action = optional_param('action', '', PARAM_ALPHA);

$cm = get_coursemodule_from_id('faceattendance', $id, 0, false, MUST_EXIST);
$course = get_course($cm->course);
$faceattendance = $DB->get_record('faceattendance', ['id' => $cm->instance], '*', MUST_EXIST);

require_login($course, true, $cm);

$context = context_module::instance($cm->id);
require_capability('mod/faceattendance:manage', $context);

$pageurl = new moodle_url('/mod/faceattendance/register.php', ['id' => $cm->id]);
$PAGE->set_url($pageurl);
$PAGE->set_title(get_string('registerstudents', 'faceattendance'));
$PAGE->set_heading(format_string($course->fullname));
$PAGE->set_context($context);

$coursecontext = context_course::instance($course->id);

if ($action === 'save') {
    require_sesskey();

    $userid = required_param('userid', PARAM_INT);
    $externalid = trim(optional_param('externalid', '', PARAM_TEXT));
    $notes = optional_param('notes', '', PARAM_TEXT);

    $user = $DB->get_record('user', ['id' => $userid, 'deleted' => 0], '*', MUST_EXIST);
    if (!is_enrolled($coursecontext, $user, '', true)) {
        redirect($pageurl, get_string('usernotenrolled', 'faceattendance'), null, \core\output\notification::NOTIFY_ERROR);
    }

    if ($externalid === '') {
        redirect($pageurl, get_string('externalidrequired', 'faceattendance'), null, \core\output\notification::NOTIFY_ERROR);
    }

    $duplicate = $DB->get_record_select('faceattendance_registrations',
        'faceattendanceid = :faceattendanceid AND externalid = :externalid AND userid <> :userid',
        [
            'faceattendanceid' => $faceattendance->id,
            'externalid' => $externalid,
            'userid' => $userid,
        ]
    );

    if ($duplicate) {
        redirect($pageurl, get_string('externalidalreadyused', 'faceattendance'), null, \core\output\notification::NOTIFY_ERROR);
    }

    $now = time();
    $existing = $DB->get_record('faceattendance_registrations', [
        'faceattendanceid' => $faceattendance->id,
        'userid' => $userid,
    ]);

    $record = (object)[
        'faceattendanceid' => (int)$faceattendance->id,
        'course' => (int)$course->id,
        'userid' => (int)$userid,
        'externalid' => $externalid,
        'status' => 'registered',
        'notes' => $notes,
        'timemodified' => $now,
    ];

    if ($existing) {
        $record->id = $existing->id;
        $record->timecreated = $existing->timecreated;
        $DB->update_record('faceattendance_registrations', $record);
        redirect($pageurl, get_string('registrationupdated', 'faceattendance'), null, \core\output\notification::NOTIFY_SUCCESS);
    }

    $record->timecreated = $now;
    $DB->insert_record('faceattendance_registrations', $record);
    redirect($pageurl, get_string('registrationcreated', 'faceattendance'), null, \core\output\notification::NOTIFY_SUCCESS);
}

if ($action === 'delete') {
    require_sesskey();
    $userid = required_param('userid', PARAM_INT);
    $DB->delete_records('faceattendance_registrations', [
        'faceattendanceid' => $faceattendance->id,
        'userid' => $userid,
    ]);
    $DB->delete_records('faceattendance_embeddings', [
        'faceattendanceid' => $faceattendance->id,
        'userid' => $userid,
    ]);
    redirect($pageurl, get_string('registrationdeleted', 'faceattendance'), null, \core\output\notification::NOTIFY_SUCCESS);
}

$users = get_enrolled_users($coursecontext, '', 0, 'u.id, u.username, u.firstname, u.lastname, u.email', 'u.lastname ASC, u.firstname ASC');
$registrations = $DB->get_records('faceattendance_registrations', ['faceattendanceid' => $faceattendance->id], '', 'userid, externalid, status, notes, timemodified');
$embeddings = $DB->get_records('faceattendance_embeddings', ['faceattendanceid' => $faceattendance->id], '', 'userid, samples, modelname, qualityscore, status, timemodified');

$useroptions = [];
foreach ($users as $user) {
    $useroptions[$user->id] = fullname($user) . ' (' . $user->email . ')';
}

$rosterurl = new moodle_url('/mod/faceattendance/api/roster.php', ['cmid' => $cm->id]);
$markurl = new moodle_url('/mod/faceattendance/api/mark.php');
$viewurl = new moodle_url('/mod/faceattendance/view.php', ['id' => $cm->id]);

$sample = [
    'cmid' => (int)$cm->id,
    'secret' => 'your-api-secret-here',
    'detections' => [
        [
            'externalid' => 'moodle_user_5',
            'confidence' => 0.93,
            'source' => 'camera-lab-1',
        ],
    ],
];

echo $OUTPUT->header();
echo $OUTPUT->heading(get_string('registerstudentsfor', 'faceattendance', format_string($faceattendance->name)));

echo html_writer::div(
    html_writer::link($viewurl, get_string('backtoactivity', 'faceattendance'), ['class' => 'btn btn-secondary']),
    'mb-3'
);

echo $OUTPUT->notification(get_string('registerintro_embeddings', 'faceattendance'), 'info');

echo $OUTPUT->heading(get_string('addregistration', 'faceattendance'), 3);

echo html_writer::start_tag('form', ['method' => 'post', 'action' => $pageurl->out(false), 'class' => 'mb-4']);
echo html_writer::empty_tag('input', ['type' => 'hidden', 'name' => 'id', 'value' => $cm->id]);
echo html_writer::empty_tag('input', ['type' => 'hidden', 'name' => 'sesskey', 'value' => sesskey()]);
echo html_writer::empty_tag('input', ['type' => 'hidden', 'name' => 'action', 'value' => 'save']);

echo html_writer::start_div('form-group');
echo html_writer::tag('label', get_string('student', 'faceattendance'), ['for' => 'faceattendance-userid']);
echo html_writer::select($useroptions, 'userid', '', ['' => get_string('choosestudent', 'faceattendance')], ['id' => 'faceattendance-userid', 'class' => 'form-control']);
echo html_writer::end_div();

echo html_writer::start_div('form-group');
echo html_writer::tag('label', get_string('externalid', 'faceattendance'), ['for' => 'faceattendance-externalid']);
echo html_writer::empty_tag('input', [
    'type' => 'text',
    'name' => 'externalid',
    'id' => 'faceattendance-externalid',
    'class' => 'form-control',
    'placeholder' => 'moodle_user_5',
    'maxlength' => 255,
    'required' => 'required',
]);
echo html_writer::tag('small', get_string('externalid_helptext', 'faceattendance'), ['class' => 'form-text text-muted']);
echo html_writer::end_div();

echo html_writer::start_div('form-group');
echo html_writer::tag('label', get_string('notes', 'faceattendance'), ['for' => 'faceattendance-notes']);
echo html_writer::tag('textarea', '', [
    'name' => 'notes',
    'id' => 'faceattendance-notes',
    'class' => 'form-control',
    'rows' => 2,
]);
echo html_writer::end_div();

echo html_writer::tag('button', get_string('saveregistration', 'faceattendance'), ['type' => 'submit', 'class' => 'btn btn-primary']);
echo html_writer::end_tag('form');

echo $OUTPUT->heading(get_string('registeredstudents', 'faceattendance'), 3);

$table = new html_table();
$table->head = [
    get_string('student', 'faceattendance'),
    get_string('email'),
    get_string('username'),
    get_string('externalid', 'faceattendance'),
    get_string('embeddingstatus', 'faceattendance'),
    get_string('timemodified', 'faceattendance'),
    get_string('actions'),
];
$table->data = [];

foreach ($users as $user) {
    $registration = $registrations[$user->id] ?? null;
    $embedding = $embeddings[$user->id] ?? null;

    $externalid = $registration ? s($registration->externalid) : html_writer::span(get_string('notregistered', 'faceattendance'), 'badge badge-secondary');

    if ($embedding) {
        $embeddingstatus = html_writer::span(get_string('embeddingregistered', 'faceattendance', (object)[
            'samples' => (int)$embedding->samples,
            'quality' => round((float)$embedding->qualityscore, 3),
        ]), 'badge badge-success');
        $modified = !empty($embedding->timemodified) ? userdate($embedding->timemodified) : '-';
    } else {
        $embeddingstatus = html_writer::span(get_string('embeddingmissing', 'faceattendance'), 'badge badge-warning');
        $modified = $registration && !empty($registration->timemodified) ? userdate($registration->timemodified) : '-';
    }

    $recorderurl = new moodle_url('/mod/faceattendance/recorder.php', [
        'id' => $cm->id,
        'userid' => $user->id,
    ]);

    $actions = html_writer::link($recorderurl, get_string('recordembedding', 'faceattendance'), ['class' => 'btn btn-sm btn-primary']);

    if ($registration || $embedding) {
        $deleteurl = new moodle_url('/mod/faceattendance/register.php', [
            'id' => $cm->id,
            'action' => 'delete',
            'userid' => $user->id,
            'sesskey' => sesskey(),
        ]);
        $actions .= ' ' . html_writer::link($deleteurl, get_string('delete'), ['class' => 'btn btn-sm btn-outline-danger']);
    }

    $table->data[] = [
        s(fullname($user)),
        s($user->email),
        s($user->username),
        $externalid,
        $embeddingstatus,
        $modified,
        $actions,
    ];
}

echo html_writer::table($table);

echo $OUTPUT->heading(get_string('recognitionprogramusage', 'faceattendance'), 3);
echo html_writer::tag('p', get_string('registerusageintro_embeddings', 'faceattendance'));
echo html_writer::tag('p', html_writer::tag('strong', 'Roster endpoint: ') . s($rosterurl->out(false)));
echo html_writer::tag('p', html_writer::tag('strong', 'Mark endpoint: ') . s($markurl->out(false)));
echo html_writer::tag('pre', s(json_encode($sample, JSON_PRETTY_PRINT | JSON_UNESCAPED_SLASHES)));

echo $OUTPUT->footer();
