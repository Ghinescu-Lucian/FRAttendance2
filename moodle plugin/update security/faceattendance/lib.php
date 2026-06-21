<?php
// This file is part of Moodle - http://moodle.org/

/**
 * Library callbacks for mod_faceattendance.
 *
 * @package     mod_faceattendance
 * @copyright   2026
 * @license     http://www.gnu.org/copyleft/gpl.html GNU GPL v3 or later
 */

defined('MOODLE_INTERNAL') || die();

/**
 * Feature support declaration.
 *
 * @param string $feature
 * @return mixed
 */
function faceattendance_supports($feature) {
    switch ($feature) {
        case FEATURE_MOD_INTRO:
            return true;
        case FEATURE_SHOW_DESCRIPTION:
            return true;
        case FEATURE_BACKUP_MOODLE2:
            return true;
        default:
            return null;
    }
}

/**
 * Add a new Face Attendance activity instance.
 *
 * @param stdClass $data
 * @param mod_faceattendance_mod_form|null $mform
 * @return int
 */
function faceattendance_add_instance($data, $mform = null) {
    global $DB;

    $data->timecreated = time();
    $data->timemodified = time();

    if (empty($data->confidence)) {
        $data->confidence = 0.85;
    }

    if (empty($data->modelprofile)) {
        $data->modelprofile = 'fast_short';
    }

    return $DB->insert_record('faceattendance', $data);
}

/**
 * Update a Face Attendance activity instance.
 *
 * @param stdClass $data
 * @param mod_faceattendance_mod_form|null $mform
 * @return bool
 */
function faceattendance_update_instance($data, $mform = null) {
    global $DB;

    $data->id = $data->instance;
    $data->timemodified = time();

    if (empty($data->confidence)) {
        $data->confidence = 0.85;
    }

    if (empty($data->modelprofile)) {
        $data->modelprofile = 'fast_short';
    }

    return $DB->update_record('faceattendance', $data);
}

/**
 * Delete a Face Attendance activity instance.
 *
 * @param int $id
 * @return bool
 */
function faceattendance_delete_instance($id) {
    global $DB;

    if (!$faceattendance = $DB->get_record('faceattendance', ['id' => $id])) {
        return false;
    }

    $DB->delete_records('faceattendance_records', ['faceattendanceid' => $faceattendance->id]);
    $DB->delete_records('faceattendance_registrations', ['faceattendanceid' => $faceattendance->id]);
    // Course-level embeddings are intentionally preserved when one Face Attendance
    // activity is deleted. They are shared by all Face Attendance sessions/activities
    // in the same course.
    $DB->delete_records('faceattendance_session_records', ['faceattendanceid' => $faceattendance->id]);
    $DB->delete_records('faceattendance_detections', ['faceattendanceid' => $faceattendance->id]);
    $DB->delete_records('faceattendance_unknowns', ['faceattendanceid' => $faceattendance->id]);
    $DB->delete_records('faceattendance_sessions', ['faceattendanceid' => $faceattendance->id]);
    $DB->delete_records('faceattendance', ['id' => $faceattendance->id]);

    return true;
}


/**
 * Serves private unknown-face review thumbnails.
 *
 * The image is visible only to users who can review unknown faces in this activity.
 *
 * @param stdClass $course
 * @param cm_info|stdClass $cm
 * @param context $context
 * @param string $filearea
 * @param array $args
 * @param bool $forcedownload
 * @param array $options
 * @return bool
 */
function faceattendance_pluginfile($course, $cm, $context, $filearea, $args, $forcedownload, array $options = []) {
    if ($context->contextlevel !== CONTEXT_MODULE || !in_array($filearea, ['unknownface', 'captureface'], true)) {
        return false;
    }

    require_login($course, true, $cm);
    require_capability('mod/faceattendance:reviewunknowns', $context);

    if (empty($args)) {
        return false;
    }

    $itemid = (int)array_shift($args);
    $filename = array_pop($args);
    if (!$filename) {
        return false;
    }

    $filepath = '/';
    if (!empty($args)) {
        $filepath = '/' . implode('/', $args) . '/';
    }

    $fs = get_file_storage();
    $file = $fs->get_file($context->id, 'mod_faceattendance', $filearea, $itemid, $filepath, $filename);
    if (!$file || $file->is_directory()) {
        return false;
    }

    send_stored_file($file, 0, 0, false, $options);
    return true;
}

/**
 * Deletes the private thumbnail associated with an unknown face record.
 *
 * @param int $contextid
 * @param int $unknownid
 * @return void
 */
function faceattendance_delete_unknown_thumbnail(int $contextid, int $unknownid): void {
    $fs = get_file_storage();
    $fs->delete_area_files($contextid, 'mod_faceattendance', 'unknownface', $unknownid);
}


/**
 * Deletes the private thumbnails associated with an intake capture group.
 *
 * @param int $contextid
 * @param int $groupid
 * @return void
 */
function faceattendance_delete_capture_group_thumbnails(int $contextid, int $groupid): void {
    $fs = get_file_storage();
    $fs->delete_area_files($contextid, 'mod_faceattendance', 'captureface', $groupid);
}
