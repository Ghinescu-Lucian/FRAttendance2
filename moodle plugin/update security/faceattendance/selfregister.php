<?php
// This file is part of Moodle - http://moodle.org/

/**
 * Student self-registration page for browser-generated face embeddings.
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
require_capability('mod/faceattendance:selfregister', $context);

$PAGE->set_url('/mod/faceattendance/selfregister.php', ['id' => $cm->id]);
$PAGE->set_title(get_string('registermyface', 'faceattendance'));
$PAGE->set_heading(format_string($course->fullname));
$PAGE->set_context($context);

$recordurl = new moodle_url('/mod/faceattendance/recorder.php', ['id' => $cm->id, 'userid' => $USER->id]);
$returnurl = new moodle_url('/mod/faceattendance/view.php', ['id' => $cm->id]);

$embedding = $DB->get_record('faceattendance_course_embeddings', [
    'course' => $course->id,
    'userid' => $USER->id,
]);

echo $OUTPUT->header();
echo $OUTPUT->heading(get_string('registermyface', 'faceattendance'));

echo html_writer::tag('p', get_string('selfregisterintro', 'faceattendance'), ['class' => 'alert alert-info']);
echo $OUTPUT->notification(get_string('courseembeddingstore', 'faceattendance'), 'success');

if ($embedding) {
    echo $OUTPUT->notification(get_string('yourfaceisregistered', 'faceattendance', (object)[
        'samples' => (int)$embedding->samples,
        'time' => userdate((int)$embedding->timemodified),
    ]), 'success');
    echo html_writer::tag('p', get_string('reregisterhint', 'faceattendance'));
} else {
    echo $OUTPUT->notification(get_string('yourfaceisnotregistered', 'faceattendance'), 'info');
}

echo html_writer::div(
    html_writer::link($recordurl, get_string('startfacerecording', 'faceattendance'), ['class' => 'btn btn-success mr-2']) .
    html_writer::link($returnurl, get_string('backtoactivity', 'faceattendance'), ['class' => 'btn btn-secondary']),
    'mt-3'
);

echo $OUTPUT->footer();
