<?php
// This file is part of Moodle - http://moodle.org/

/**
 * Receives browser-generated SFace embedding JSON files and stores them in Moodle.
 *
 * @package     mod_faceattendance
 * @copyright   2026
 * @license     http://www.gnu.org/copyleft/gpl.html GNU GPL v3 or later
 */

require_once(__DIR__ . '/../../../config.php');
require_once(__DIR__ . '/embedding_crypto.php');

header('Content-Type: application/json; charset=utf-8');

function faceattendance_json_response($data, int $status = 200): void {
    http_response_code($status);
    echo json_encode($data, JSON_UNESCAPED_SLASHES);
    exit;
}

function faceattendance_descriptor_norm(array $values): float {
    $sum = 0.0;
    foreach ($values as $value) {
        $number = (float)$value;
        $sum += $number * $number;
    }
    return sqrt($sum);
}

try {
    require_login();
    require_sesskey();

    $cmid = required_param('cmid', PARAM_INT);
    $userid = required_param('userid', PARAM_INT);

    $cm = get_coursemodule_from_id('faceattendance', $cmid, 0, false, MUST_EXIST);
    $course = get_course($cm->course);
    $faceattendance = $DB->get_record('faceattendance', ['id' => $cm->instance], '*', MUST_EXIST);

    require_login($course, true, $cm);
    $context = context_module::instance($cm->id);
    $canmanage = has_capability('mod/faceattendance:manage', $context);
    $canselfregister = has_capability('mod/faceattendance:selfregister', $context);
    if (!$canmanage) {
        if (!$canselfregister || $userid !== (int)$USER->id) {
            faceattendance_json_response(['ok' => false, 'error' => 'You can only register your own face embedding.'], 403);
        }
    }

    $coursecontext = context_course::instance($course->id);
    $user = $DB->get_record('user', ['id' => $userid, 'deleted' => 0], '*', MUST_EXIST);
    if (!is_enrolled($coursecontext, $user, '', true)) {
        faceattendance_json_response(['ok' => false, 'error' => 'The selected user is not enrolled in this course.'], 403);
    }

    if (empty($_FILES['embeddingFile']) || !is_uploaded_file($_FILES['embeddingFile']['tmp_name'])) {
        faceattendance_json_response(['ok' => false, 'error' => 'Missing embedding file. Expected multipart field: embeddingFile.'], 400);
    }

    $rawjson = file_get_contents($_FILES['embeddingFile']['tmp_name']);
    if ($rawjson === false || trim($rawjson) === '') {
        faceattendance_json_response(['ok' => false, 'error' => 'The uploaded embedding file is empty.'], 400);
    }

    $payload = json_decode($rawjson, false);
    if (json_last_error() !== JSON_ERROR_NONE || !is_object($payload)) {
        faceattendance_json_response(['ok' => false, 'error' => 'Invalid JSON embedding file: ' . json_last_error_msg()], 400);
    }

    $errors = [];
    $model = $payload->model ?? null;
    $family = is_object($model) && isset($model->family) ? strtolower((string)$model->family) : '';
    $recognizer = is_object($model) && isset($model->recognizer) ? strtolower((string)$model->recognizer) : '';
    $descriptorlength = is_object($model) && isset($model->descriptorLength) ? (int)$model->descriptorLength : 0;

    if ($family !== 'opencv') {
        $errors[] = "model.family must be 'opencv'.";
    }
    if ($recognizer !== 'sface') {
        $errors[] = "model.recognizer must be 'sface'.";
    }
    if ($descriptorlength !== 128) {
        $errors[] = 'model.descriptorLength must be 128.';
    }
    if (empty($payload->captures) || !is_array($payload->captures)) {
        $errors[] = 'captures must be a non-empty array.';
    } else {
        foreach ($payload->captures as $index => $capture) {
            $label = is_object($capture) && !empty($capture->label) ? (string)$capture->label : 'capture ' . ($index + 1);
            if (!is_object($capture) || empty($capture->descriptor) || !is_array($capture->descriptor)) {
                $errors[] = $label . ': descriptor must be an array.';
                continue;
            }
            if (count($capture->descriptor) !== 128) {
                $errors[] = $label . ': descriptor must have 128 numbers.';
                continue;
            }
            foreach ($capture->descriptor as $value) {
                if (!is_numeric($value) || !is_finite((float)$value)) {
                    $errors[] = $label . ': descriptor contains non-numeric values.';
                    continue 2;
                }
            }
            $norm = faceattendance_descriptor_norm($capture->descriptor);
            if (abs($norm - 1.0) > 0.08) {
                $errors[] = $label . ': descriptor is not L2-normalized enough. Norm=' . round($norm, 4);
            }
        }
    }

    if ($errors) {
        faceattendance_json_response([
            'ok' => false,
            'error' => 'Rejected embedding file because it is not a valid OpenCV SFace embedding payload.',
            'details' => $errors,
        ], 400);
    }

    $now = time();
    $storedjson = faceattendance_embedding_encrypt_json_for_station($rawjson, $faceattendance, (int)$course->id, (int)$userid);
    $samples = count($payload->captures);
    $qualitysum = 0.0;
    $qualitycount = 0;
    foreach ($payload->captures as $capture) {
        if (isset($capture->quality) && is_object($capture->quality) && isset($capture->quality->score) && is_numeric($capture->quality->score)) {
            $qualitysum += (float)$capture->quality->score;
            $qualitycount++;
        }
    }
    $qualityscore = $qualitycount > 0 ? $qualitysum / $qualitycount : 0.0;

    $existing = $DB->get_record('faceattendance_course_embeddings', [
        'course' => $course->id,
        'userid' => $userid,
    ]);

    $record = (object)[
        'course' => (int)$course->id,
        'userid' => (int)$userid,
        'embeddingjson' => $storedjson,
        'modelname' => 'opencv-sface-2021dec',
        'embeddingdim' => 128,
        'samples' => $samples,
        'qualityscore' => $qualityscore,
        'status' => 'registered',
        'timemodified' => $now,
    ];

    if ($existing) {
        $record->id = $existing->id;
        $record->timecreated = $existing->timecreated;
        $DB->update_record('faceattendance_course_embeddings', $record);
        $embeddingid = $existing->id;
        $saved = 'updated';
    } else {
        $record->timecreated = $now;
        $embeddingid = $DB->insert_record('faceattendance_course_embeddings', $record);
        $saved = 'created';
    }

    $registration = $DB->get_record('faceattendance_registrations', [
        'faceattendanceid' => $faceattendance->id,
        'userid' => $userid,
    ]);

    $externalid = 'moodle_user_' . $userid;
    $regrecord = (object)[
        'faceattendanceid' => (int)$faceattendance->id,
        'course' => (int)$course->id,
        'userid' => (int)$userid,
        'externalid' => $registration && !empty($registration->externalid) ? $registration->externalid : $externalid,
        'status' => 'registered',
        'notes' => $registration->notes ?? '',
        'timemodified' => $now,
    ];
    if ($registration) {
        $regrecord->id = $registration->id;
        $regrecord->timecreated = $registration->timecreated;
        $DB->update_record('faceattendance_registrations', $regrecord);
    } else {
        $regrecord->timecreated = $now;
        $DB->insert_record('faceattendance_registrations', $regrecord);
    }

    faceattendance_json_response([
        'ok' => true,
        'saved' => $saved,
        'embeddingid' => (int)$embeddingid,
        'userid' => (int)$userid,
        'name' => fullname($user),
        'samples' => (int)$samples,
        'qualityscore' => round($qualityscore, 5),
        'savedImages' => 0,
        'savedEmbeddingFiles' => 1,
        'uploadDir' => 'Moodle database table mdl_faceattendance_course_embeddings',
        'encryptedAtRest' => faceattendance_embedding_crypto_enabled($faceattendance),
    ]);
} catch (Throwable $e) {
    faceattendance_json_response(['ok' => false, 'error' => $e->getMessage()], 500);
}
