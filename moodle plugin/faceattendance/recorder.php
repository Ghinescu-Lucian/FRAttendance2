<?php
// This file is part of Moodle - http://moodle.org/

/**
 * Moodle wrapper page for the browser-side SFace embedding recorder.
 *
 * @package     mod_faceattendance
 * @copyright   2026
 * @license     http://www.gnu.org/copyleft/gpl.html GNU GPL v3 or later
 */

require_once(__DIR__ . '/../../config.php');

$id = required_param('id', PARAM_INT); // Course module id.
$userid = required_param('userid', PARAM_INT);

$cm = get_coursemodule_from_id('faceattendance', $id, 0, false, MUST_EXIST);
$course = get_course($cm->course);
$faceattendance = $DB->get_record('faceattendance', ['id' => $cm->instance], '*', MUST_EXIST);

require_login($course, true, $cm);

$context = context_module::instance($cm->id);
$canmanage = has_capability('mod/faceattendance:manage', $context);
$canselfregister = has_capability('mod/faceattendance:selfregister', $context);

if (!$canmanage) {
    if (!$canselfregister || $userid !== (int)$USER->id) {
        throw new required_capability_exception($context, 'mod/faceattendance:manage', 'nopermissions', '');
    }
}

$coursecontext = context_course::instance($course->id);
$user = $DB->get_record('user', ['id' => $userid, 'deleted' => 0], '*', MUST_EXIST);
if (!is_enrolled($coursecontext, $user, '', true)) {
    throw new moodle_exception('usernotenrolled', 'faceattendance');
}

$pageurl = new moodle_url('/mod/faceattendance/recorder.php', ['id' => $cm->id, 'userid' => $userid]);
$returnurl = $canmanage
    ? new moodle_url('/mod/faceattendance/register.php', ['id' => $cm->id])
    : new moodle_url('/mod/faceattendance/selfregister.php', ['id' => $cm->id]);
$saveurl = new moodle_url('/mod/faceattendance/api/save_embedding_file.php');
$modelurl = new moodle_url('/mod/faceattendance/uploader/models/face_recognition_sface_2021dec.onnx');

$PAGE->set_url($pageurl);
$PAGE->set_title(get_string('recordembeddingfor', 'faceattendance', fullname($user)));
$PAGE->set_heading(format_string($course->fullname));
$PAGE->set_context($context);

$assetsdir = __DIR__ . '/uploader/assets';
$assetbaseurl = $CFG->wwwroot . '/mod/faceattendance/uploader/assets/';
$cssfiles = glob($assetsdir . '/*.css') ?: [];
$jsfiles = glob($assetsdir . '/*.js') ?: [];

$contextdata = [
    'cmid' => (int)$cm->id,
    'userid' => (int)$userid,
    'sesskey' => sesskey(),
    'studentId' => (string)$userid,
    'studentName' => fullname($user),
    'saveEmbeddingUrl' => $saveurl->out(false),
    'sfaceModelUrl' => $modelurl->out(false),
    'returnUrl' => $returnurl->out(false),
];

echo $OUTPUT->header();
echo $OUTPUT->heading(get_string('recordembeddingfor', 'faceattendance', fullname($user)));
echo html_writer::div(
    html_writer::link($returnurl, get_string('backtoactivity', 'faceattendance'), ['class' => 'btn btn-secondary mb-3']),
    'faceattendance-recorder-back'
);

echo html_writer::tag('p', get_string('recordembeddingintro', 'faceattendance'), ['class' => 'alert alert-info']);

foreach ($cssfiles as $cssfile) {
    $name = basename($cssfile);
    echo html_writer::empty_tag('link', [
        'rel' => 'stylesheet',
        'href' => $assetbaseurl . $name,
    ]);
}

echo html_writer::script('window.FACEATTENDANCE_CONTEXT = ' . json_encode($contextdata, JSON_UNESCAPED_SLASHES) . ';');
echo html_writer::tag('div', '', ['id' => 'root']);

echo html_writer::script('', 'https://cdn.jsdelivr.net/npm/onnxruntime-web/dist/ort.min.js');
echo html_writer::script('', 'https://cdn.jsdelivr.net/npm/@vladmandic/face-api/dist/face-api.min.js');

foreach ($jsfiles as $jsfile) {
    $name = basename($jsfile);
    echo html_writer::tag('script', '', [
        'type' => 'module',
        'src' => $assetbaseurl . $name,
    ]);
}

echo $OUTPUT->footer();
