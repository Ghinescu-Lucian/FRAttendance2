<?php
// This file is part of Moodle - http://moodle.org/

/**
 * Capture-first intake station page.
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
require_capability('mod/faceattendance:takeattendance', $context);

$pageurl = new moodle_url('/mod/faceattendance/capture.php', ['id' => $cm->id]);
$returnurl = new moodle_url('/mod/faceattendance/view.php', ['id' => $cm->id]);
$reviewurl = new moodle_url('/mod/faceattendance/capture_review.php', ['id' => $cm->id]);
$modelurl = new moodle_url('/mod/faceattendance/uploader/models/face_recognition_sface_2021dec.onnx');

$profiles = [
    'fast_short' => get_string('modelprofile_fast_short', 'faceattendance'),
    'many_faces_unknown' => get_string('modelprofile_many_faces_unknown', 'faceattendance'),
    'fast_clean' => get_string('modelprofile_fast_clean', 'faceattendance'),
    'high_recall_many_faces' => get_string('modelprofile_high_recall_many_faces', 'faceattendance'),
    'multi_attendance_zoom' => get_string('modelprofile_multi_attendance_zoom', 'faceattendance'),
    'entrance_mode' => get_string('modelprofile_entrance_mode', 'faceattendance'),
];
$stationprofile = optional_param('profile', 'high_recall_many_faces', PARAM_ALPHANUMEXT);
if (!array_key_exists($stationprofile, $profiles)) {
    $stationprofile = 'high_recall_many_faces';
}

function faceattendance_get_or_create_capture_session(stdClass $faceattendance, stdClass $course): stdClass {
    global $DB, $USER;

    $now = time();
    $startofday = usergetmidnight($now);
    $existing = $DB->get_record_select('faceattendance_capture_sessions',
        'faceattendanceid = :faceattendanceid AND course = :course AND status = :status AND starttime >= :startofday',
        [
            'faceattendanceid' => $faceattendance->id,
            'course' => $course->id,
            'status' => 'open',
            'startofday' => $startofday,
        ],
        '*',
        IGNORE_MULTIPLE
    );

    if ($existing) {
        return $existing;
    }

    $id = $DB->insert_record('faceattendance_capture_sessions', (object)[
        'faceattendanceid' => (int)$faceattendance->id,
        'course' => (int)$course->id,
        'name' => get_string('defaultcapturesessionname', 'faceattendance', userdate($now)),
        'source' => 'browser-capture-intake',
        'status' => 'open',
        'createdby' => (int)$USER->id,
        'starttime' => $now,
        'endtime' => $now + 3 * HOURSECS,
        'timecreated' => $now,
        'timemodified' => $now,
    ]);

    return $DB->get_record('faceattendance_capture_sessions', ['id' => $id], '*', MUST_EXIST);
}

$capturesession = faceattendance_get_or_create_capture_session($faceattendance, $course);

$PAGE->set_url($pageurl);
$PAGE->set_title(get_string('captureenteringfaces', 'faceattendance'));
$PAGE->set_heading(format_string($course->fullname));
$PAGE->set_context($context);

$assetsdir = __DIR__ . '/uploader/assets';
$assetbaseurl = $CFG->wwwroot . '/mod/faceattendance/uploader/assets/';
$cssfiles = glob($assetsdir . '/*.css') ?: [];
$jsfiles = glob($assetsdir . '/*.js') ?: [];

$contextdata = [
    'mode' => 'capture',
    'cmid' => (int)$cm->id,
    'userid' => (int)$USER->id,
    'sesskey' => sesskey(),
    'studentId' => (string)$USER->id,
    'studentName' => fullname($USER),
    'sfaceModelUrl' => $modelurl->out(false),
    'stationProfile' => $stationprofile,
    'stationProfileLabel' => $profiles[$stationprofile],
    'knownFacesUrl' => (new moodle_url('/mod/faceattendance/api/get_embeddings.php', ['cmid' => $cm->id]))->out(false),
    'captureFaceUrl' => (new moodle_url('/mod/faceattendance/api/capture_face.php'))->out(false),
    'captureSessionId' => (int)$capturesession->id,
    'returnUrl' => $returnurl->out(false),
];

$pendingcount = $DB->count_records('faceattendance_capture_groups', [
    'capturesessionid' => $capturesession->id,
    'status' => 'pending',
]);

// Keep Moodle string API simple for generated prototype labels.
function faceattendance_capture_button(moodle_url $url, string $label, string $class = 'btn btn-secondary mr-2 mb-2'): string {
    return html_writer::link($url, $label, ['class' => $class]);
}

echo $OUTPUT->header();
echo $OUTPUT->heading(get_string('captureenteringfaces', 'faceattendance'));
echo html_writer::div(
    faceattendance_capture_button($returnurl, get_string('backtoactivity', 'faceattendance')) .
    faceattendance_capture_button($reviewurl, get_string('reviewcaptures', 'faceattendance'), 'btn btn-info mr-2 mb-2'),
    'mb-3'
);
echo html_writer::tag('p', get_string('capturestationintro', 'faceattendance'), ['class' => 'alert alert-info']);
echo $OUTPUT->notification(get_string('capturesessionnotice', 'faceattendance', (object)[
    'name' => format_string($capturesession->name),
    'pending' => $pendingcount,
]), 'info');

echo html_writer::start_tag('form', ['method' => 'get', 'class' => 'form-inline mb-3']);
echo html_writer::empty_tag('input', ['type' => 'hidden', 'name' => 'id', 'value' => $cm->id]);
echo html_writer::tag('label', get_string('modelprofile_station', 'faceattendance'), ['for' => 'faceattendance-capture-profile', 'class' => 'mr-2 font-weight-bold']);
echo html_writer::select($profiles, 'profile', $stationprofile, false, [
    'id' => 'faceattendance-capture-profile',
    'class' => 'custom-select mr-2',
    'onchange' => 'this.form.submit()'
]);
echo html_writer::tag('span', get_string('captureprofilehelp', 'faceattendance'), ['class' => 'text-muted']);
echo html_writer::end_tag('form');

foreach ($cssfiles as $cssfile) {
    echo html_writer::empty_tag('link', [
        'rel' => 'stylesheet',
        'href' => $assetbaseurl . basename($cssfile),
    ]);
}

echo html_writer::script('window.FACEATTENDANCE_CONTEXT = ' . json_encode($contextdata, JSON_UNESCAPED_SLASHES) . ';');
echo html_writer::tag('div', '', ['id' => 'root']);

echo html_writer::script('', 'https://cdn.jsdelivr.net/npm/onnxruntime-web/dist/ort.min.js');
echo html_writer::script('', 'https://cdn.jsdelivr.net/npm/@vladmandic/face-api/dist/face-api.min.js');

foreach ($jsfiles as $jsfile) {
    echo html_writer::tag('script', '', [
        'type' => 'module',
        'src' => $assetbaseurl . basename($jsfile),
    ]);
}

echo $OUTPUT->footer();
