<?php
// This file is part of Moodle - http://moodle.org/

/**
 * Returns the course roster to the external face-recognition program.
 *
 * @package     mod_faceattendance
 * @copyright   2026
 * @license     http://www.gnu.org/copyleft/gpl.html GNU GPL v3 or later
 */

define('AJAX_SCRIPT', true);
require_once(__DIR__ . '/../../../config.php');

function faceattendance_roster_response($payload, int $status = 200): void {
    http_response_code($status);
    header('Content-Type: application/json; charset=utf-8');
    echo json_encode($payload, JSON_UNESCAPED_SLASHES);
    exit;
}

$cmid = required_param('cmid', PARAM_INT);
$secret = $_SERVER['HTTP_X_FACEATTENDANCE_SECRET'] ?? optional_param('secret', '', PARAM_RAW_TRIMMED);

$cm = get_coursemodule_from_id('faceattendance', $cmid, 0, false, MUST_EXIST);
$course = get_course($cm->course);
$faceattendance = $DB->get_record('faceattendance', ['id' => $cm->instance], '*', MUST_EXIST);

if (empty($faceattendance->apisecret) || !hash_equals((string)$faceattendance->apisecret, (string)$secret)) {
    faceattendance_roster_response(['ok' => false, 'error' => 'Invalid API secret.'], 403);
}

$coursecontext = context_course::instance($course->id);
$users = get_enrolled_users($coursecontext, '', 0, 'u.id, u.username, u.firstname, u.lastname, u.email');
$registrations = $DB->get_records('faceattendance_registrations', ['faceattendanceid' => $faceattendance->id], '', 'userid, externalid, status');
$embeddings = $DB->get_records('faceattendance_course_embeddings', ['course' => $course->id], '', 'userid, samples, modelname, qualityscore, status, timemodified');

$result = [];
foreach ($users as $user) {
    $registration = $registrations[$user->id] ?? null;
    $embedding = $embeddings[$user->id] ?? null;
    $result[] = [
        'userid' => (int)$user->id,
        'username' => $user->username,
        'firstname' => $user->firstname,
        'lastname' => $user->lastname,
        'email' => $user->email,
        'registered' => $registration !== null,
        'externalid' => $registration ? $registration->externalid : null,
        'registrationstatus' => $registration ? $registration->status : 'notregistered',
        'embeddingregistered' => $embedding !== null,
        'embeddingsamples' => $embedding ? (int)$embedding->samples : 0,
        'embeddingmodel' => $embedding ? $embedding->modelname : null,
        'embeddingquality' => $embedding ? (float)$embedding->qualityscore : null,
    ];
}

faceattendance_roster_response([
    'ok' => true,
    'courseid' => (int)$course->id,
    'cmid' => (int)$cm->id,
    'users' => $result,
]);
