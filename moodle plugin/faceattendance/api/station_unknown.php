<?php
// This file is part of Moodle - http://moodle.org/

/**
 * Secret-authenticated unknown-face endpoint for an external Python/OpenCV camera station.
 *
 * @package     mod_faceattendance
 * @license     http://www.gnu.org/copyleft/gpl.html GNU GPL v3 or later
 */

define('AJAX_SCRIPT', true);
require_once(__DIR__ . '/../../../config.php');

function faceattendance_station_response($payload, int $status = 200): void {
    http_response_code($status);
    header('Content-Type: application/json; charset=utf-8');
    echo json_encode($payload, JSON_UNESCAPED_SLASHES);
    exit;
}

function faceattendance_station_payload(): array {
    $raw = file_get_contents('php://input');
    $payload = json_decode($raw ?: '', true);
    if (json_last_error() !== JSON_ERROR_NONE || !is_array($payload)) {
        faceattendance_station_response(['ok' => false, 'error' => 'Invalid JSON body: ' . json_last_error_msg()], 400);
    }
    return $payload;
}


function faceattendance_extract_thumbnail_binary($thumbnail): ?array {
    if (empty($thumbnail) || !is_string($thumbnail)) {
        return null;
    }

    $thumbnail = trim($thumbnail);
    if (preg_match('/^data:image\/(jpeg|jpg|png);base64,/i', $thumbnail, $matches)) {
        $thumbnail = preg_replace('/^data:image\/(jpeg|jpg|png);base64,/i', '', $thumbnail);
    }

    $binary = base64_decode($thumbnail, true);
    if ($binary === false || strlen($binary) === 0) {
        throw new coding_exception('Invalid thumbnail base64 payload.');
    }

    // Keep unknown-face review images small. The Python station sends a small JPEG crop.
    if (strlen($binary) > 2 * 1024 * 1024) {
        throw new coding_exception('Thumbnail is too large. Maximum accepted size is 2 MB.');
    }

    $info = @getimagesizefromstring($binary);
    if (!$info || empty($info[2])) {
        throw new coding_exception('Thumbnail payload is not a valid image.');
    }

    if ((int)$info[2] === IMAGETYPE_JPEG) {
        return ['binary' => $binary, 'filename' => 'unknown.jpg'];
    }
    if ((int)$info[2] === IMAGETYPE_PNG) {
        return ['binary' => $binary, 'filename' => 'unknown.png'];
    }

    throw new coding_exception('Unsupported thumbnail image type. Use JPEG or PNG.');
}


function faceattendance_descriptor_from_embeddingjson(?string $embeddingjson): array {
    if (empty($embeddingjson)) {
        return [];
    }
    $data = json_decode($embeddingjson, true);
    if (!is_array($data) || empty($data['descriptor']) || !is_array($data['descriptor'])) {
        return [];
    }
    return array_map('floatval', $data['descriptor']);
}

function faceattendance_vector_distance(array $a, array $b): float {
    $n = min(count($a), count($b));
    if ($n === 0) {
        return INF;
    }
    $sum = 0.0;
    for ($i = 0; $i < $n; $i++) {
        $d = (float)$a[$i] - (float)$b[$i];
        $sum += $d * $d;
    }
    return sqrt($sum);
}

function faceattendance_vector_cosine(array $a, array $b): float {
    $n = min(count($a), count($b));
    if ($n === 0) {
        return -1.0;
    }
    $dot = 0.0;
    $na = 0.0;
    $nb = 0.0;
    for ($i = 0; $i < $n; $i++) {
        $av = (float)$a[$i];
        $bv = (float)$b[$i];
        $dot += $av * $bv;
        $na += $av * $av;
        $nb += $bv * $bv;
    }
    if ($na <= 0 || $nb <= 0) {
        return -1.0;
    }
    return $dot / (sqrt($na) * sqrt($nb));
}

function faceattendance_find_similar_unknown_record(int $sessionid, int $faceattendanceid, array $descriptor): ?stdClass {
    global $DB;

    // SFace descriptors from the same unknown person are usually very close.
    // Use a strict threshold so two different students are not merged accidentally.
    $similaritythreshold = 0.78;
    $distancethreshold = 0.75;

    $existingrecords = $DB->get_records('faceattendance_unknowns', [
        'sessionid' => $sessionid,
        'faceattendanceid' => $faceattendanceid,
        'status' => 'unknown',
    ], 'timemodified DESC', '*', 0, 40);

    $best = null;
    $bestscore = -INF;
    foreach ($existingrecords as $record) {
        $existingdescriptor = faceattendance_descriptor_from_embeddingjson($record->embeddingjson ?? null);
        if (count($existingdescriptor) !== count($descriptor)) {
            continue;
        }

        $similarity = faceattendance_vector_cosine($descriptor, $existingdescriptor);
        $distance = faceattendance_vector_distance($descriptor, $existingdescriptor);
        if ($similarity >= $similaritythreshold || $distance <= $distancethreshold) {
            $score = $similarity - ($distance * 0.05);
            if ($score > $bestscore) {
                $bestscore = $score;
                $best = $record;
            }
        }
    }

    return $best;
}

function faceattendance_unknown_has_thumbnail(context_module $context, int $unknownid): bool {
    $fs = get_file_storage();
    $files = $fs->get_area_files($context->id, 'mod_faceattendance', 'unknownface', $unknownid, 'id DESC', false);
    return !empty($files);
}

function faceattendance_store_unknown_thumbnail(context_module $context, int $unknownid, ?array $thumbnail): bool {
    if (!$thumbnail) {
        return false;
    }

    $fs = get_file_storage();
    $fs->delete_area_files($context->id, 'mod_faceattendance', 'unknownface', $unknownid);

    $fileinfo = [
        'contextid' => $context->id,
        'component' => 'mod_faceattendance',
        'filearea' => 'unknownface',
        'itemid' => $unknownid,
        'filepath' => '/',
        'filename' => $thumbnail['filename'],
    ];

    $fs->create_file_from_string($fileinfo, $thumbnail['binary']);
    return true;
}

try {
    global $DB;

    $payload = faceattendance_station_payload();
    $cmid = isset($payload['cmid']) ? clean_param($payload['cmid'], PARAM_INT) : 0;
    $sessionid = isset($payload['sessionid']) ? clean_param($payload['sessionid'], PARAM_INT) : 0;
    $secret = trim((string)($_SERVER['HTTP_X_FACEATTENDANCE_SECRET'] ?? ($payload['secret'] ?? '')));
    $source = isset($payload['source']) ? clean_param((string)$payload['source'], PARAM_TEXT) : 'opencv-python-station';

    if ($cmid <= 0 || $sessionid <= 0 || empty($payload['descriptor']) || !is_array($payload['descriptor'])) {
        faceattendance_station_response(['ok' => false, 'error' => 'cmid, sessionid and descriptor are required.'], 400);
    }
    if (count($payload['descriptor']) !== 128) {
        faceattendance_station_response(['ok' => false, 'error' => 'Unknown descriptor must have 128 numbers.'], 400);
    }

    $cm = get_coursemodule_from_id('faceattendance', $cmid, 0, false, MUST_EXIST);
    $course = get_course($cm->course);
    $context = context_module::instance($cm->id);
    $faceattendance = $DB->get_record('faceattendance', ['id' => $cm->instance], '*', MUST_EXIST);

    if (empty($faceattendance->apisecret) || !hash_equals((string)$faceattendance->apisecret, $secret)) {
        faceattendance_station_response(['ok' => false, 'error' => 'Invalid API secret.'], 403);
    }

    $session = $DB->get_record('faceattendance_sessions', [
        'id' => $sessionid,
        'faceattendanceid' => $faceattendance->id,
        'course' => $course->id,
    ], '*', MUST_EXIST);

    $now = time();
    $embeddingjson = json_encode([
        'version' => 1,
        'model' => 'opencv-sface-2021dec',
        'descriptorLength' => 128,
        'descriptor' => array_map('floatval', $payload['descriptor']),
        'capturedAt' => gmdate('c', $now),
    ], JSON_UNESCAPED_SLASHES);

    $descriptor = array_map('floatval', $payload['descriptor']);
    $thumbnail = null;
    if (!empty($payload['thumbnail']) && is_string($payload['thumbnail'])) {
        $thumbnail = faceattendance_extract_thumbnail_binary($payload['thumbnail']);
    }

    $similarunknown = faceattendance_find_similar_unknown_record((int)$session->id, (int)$faceattendance->id, $descriptor);
    if ($similarunknown) {
        $similarunknown->lastseen = $now;
        $similarunknown->detectioncount = max(1, (int)$similarunknown->detectioncount) + 1;
        $similarunknown->timemodified = $now;
        if (isset($payload['candidate'])) {
            $similarunknown->candidatejson = json_encode($payload['candidate'], JSON_UNESCAPED_SLASHES);
        }
        $DB->update_record('faceattendance_unknowns', $similarunknown);

        $hasthumbnail = faceattendance_unknown_has_thumbnail($context, (int)$similarunknown->id);
        if (!$hasthumbnail && $thumbnail) {
            $hasthumbnail = faceattendance_store_unknown_thumbnail($context, (int)$similarunknown->id, $thumbnail);
        }
        $unknownid = (int)$similarunknown->id;
    } else {
        $unknownid = $DB->insert_record('faceattendance_unknowns', (object)[
            'sessionid' => (int)$session->id,
            'faceattendanceid' => (int)$faceattendance->id,
            'course' => (int)$course->id,
            'embeddingjson' => $embeddingjson,
            'candidatejson' => isset($payload['candidate']) ? json_encode($payload['candidate'], JSON_UNESCAPED_SLASHES) : null,
            'source' => $source,
            'status' => 'unknown',
            'resolveduserid' => null,
            'resolvedby' => null,
            'firstseen' => $now,
            'lastseen' => $now,
            'detectioncount' => 1,
            'timecreated' => $now,
            'timemodified' => $now,
        ]);
        $hasthumbnail = faceattendance_store_unknown_thumbnail($context, (int)$unknownid, $thumbnail);
    }

    $DB->insert_record('faceattendance_detections', (object)[
        'sessionid' => (int)$session->id,
        'faceattendanceid' => (int)$faceattendance->id,
        'course' => (int)$course->id,
        'userid' => null,
        'decision' => 'unknown',
        'confidence' => 0,
        'distance' => 0,
        'source' => $source,
        'rawpayload' => json_encode(array_diff_key($payload, ['thumbnail' => true]), JSON_UNESCAPED_SLASHES),
        'timecreated' => $now,
    ]);

    faceattendance_station_response([
        'ok' => true,
        'unknownid' => (int)$unknownid,
        'thumbnail' => !empty($hasthumbnail),
        'deduplicated' => !empty($similarunknown),
    ]);
} catch (Throwable $e) {
    faceattendance_station_response(['ok' => false, 'error' => $e->getMessage()], 500);
}
