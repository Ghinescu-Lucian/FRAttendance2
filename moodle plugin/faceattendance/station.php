<?php
// This file is part of Moodle - http://moodle.org/

/**
 * Classroom camera station page.
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

$pageurl = new moodle_url('/mod/faceattendance/station.php', ['id' => $cm->id]);
$returnurl = new moodle_url('/mod/faceattendance/view.php', ['id' => $cm->id]);
$modelurl = new moodle_url('/mod/faceattendance/uploader/models/face_recognition_sface_2021dec.onnx');

$PAGE->set_url($pageurl);
$PAGE->set_title(get_string('classroomstation', 'faceattendance'));
$PAGE->set_heading(format_string($course->fullname));
$PAGE->set_context($context);

$assetsdir = __DIR__ . '/uploader/assets';
$assetbaseurl = $CFG->wwwroot . '/mod/faceattendance/uploader/assets/';
$cssfiles = glob($assetsdir . '/*.css') ?: [];
$jsfiles = glob($assetsdir . '/*.js') ?: [];

$contextdata = [
    'mode' => 'station',
    'cmid' => (int)$cm->id,
    'userid' => (int)$USER->id,
    'sesskey' => sesskey(),
    'studentId' => (string)$USER->id,
    'studentName' => fullname($USER),
    'sfaceModelUrl' => $modelurl->out(false),
    'knownFacesUrl' => (new moodle_url('/mod/faceattendance/api/get_embeddings.php', ['cmid' => $cm->id]))->out(false),
    'activeSessionUrl' => (new moodle_url('/mod/faceattendance/api/get_active_session.php', ['cmid' => $cm->id]))->out(false),
    'markDetectionUrl' => (new moodle_url('/mod/faceattendance/api/mark_session_detection.php'))->out(false),
    'saveUnknownUrl' => (new moodle_url('/mod/faceattendance/api/save_unknown.php'))->out(false),
    'returnUrl' => $returnurl->out(false),
];

echo $OUTPUT->header();
echo $OUTPUT->heading(get_string('classroomstation', 'faceattendance'));
echo html_writer::div(
    html_writer::link($returnurl, get_string('backtoactivity', 'faceattendance'), ['class' => 'btn btn-secondary mb-3']),
    'faceattendance-station-back'
);
echo html_writer::tag('p', get_string('stationintro', 'faceattendance'), ['class' => 'alert alert-info']);

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
