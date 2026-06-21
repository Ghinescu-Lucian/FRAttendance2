<?php
// This file is part of Moodle - http://moodle.org/

/**
 * Stores an unlabeled entering face into a temporary capture group.
 *
 * @package     mod_faceattendance
 * @copyright   2026
 * @license     http://www.gnu.org/copyleft/gpl.html GNU GPL v3 or later
 */

require_once(__DIR__ . '/../../../config.php');
require_once(__DIR__ . '/../lib.php');

header('Content-Type: application/json; charset=utf-8');

function faceattendance_capture_json_response($data, int $status = 200): void {
    http_response_code($status);
    echo json_encode($data, JSON_UNESCAPED_SLASHES);
    exit;
}

function faceattendance_capture_read_json_payload(): stdClass {
    $raw = file_get_contents('php://input');
    $payload = json_decode($raw ?: '', false);
    if (json_last_error() !== JSON_ERROR_NONE || !is_object($payload)) {
        faceattendance_capture_json_response(['ok' => false, 'error' => 'Invalid JSON body: ' . json_last_error_msg()], 400);
    }
    return $payload;
}

function faceattendance_capture_extract_thumbnail_binary($thumbnail): ?array {
    if (empty($thumbnail) || !is_string($thumbnail)) {
        return null;
    }

    $thumbnail = trim($thumbnail);
    if (preg_match('/^data:image\/(jpeg|jpg|png);base64,/i', $thumbnail)) {
        $thumbnail = preg_replace('/^data:image\/(jpeg|jpg|png);base64,/i', '', $thumbnail);
    }

    $binary = base64_decode($thumbnail, true);
    if ($binary === false || strlen($binary) === 0) {
        throw new coding_exception('Invalid thumbnail base64 payload.');
    }
    if (strlen($binary) > 2 * 1024 * 1024) {
        throw new coding_exception('Thumbnail is too large. Maximum accepted size is 2 MB.');
    }

    $info = @getimagesizefromstring($binary);
    if (!$info || empty($info[2])) {
        throw new coding_exception('Thumbnail payload is not a valid image.');
    }

    if ((int)$info[2] === IMAGETYPE_JPEG) {
        return ['binary' => $binary, 'filename' => 'capture_' . time() . '_' . random_int(1000, 9999) . '.jpg'];
    }
    if ((int)$info[2] === IMAGETYPE_PNG) {
        return ['binary' => $binary, 'filename' => 'capture_' . time() . '_' . random_int(1000, 9999) . '.png'];
    }

    throw new coding_exception('Unsupported thumbnail image type. Use JPEG or PNG.');
}

function faceattendance_capture_descriptor_from_payload(?string $embeddingjson): array {
    if (empty($embeddingjson)) {
        return [];
    }
    $data = json_decode($embeddingjson, true);
    if (!is_array($data) || empty($data['descriptor']) || !is_array($data['descriptor'])) {
        return [];
    }
    return array_map('floatval', $data['descriptor']);
}

function faceattendance_capture_vector_distance(array $a, array $b): float {
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

function faceattendance_capture_vector_cosine(array $a, array $b): float {
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

function faceattendance_capture_find_similar_group(int $capturesessionid, int $faceattendanceid, array $descriptor): ?stdClass {
    global $DB;

    // Strict grouping: it is safer to create two groups for the same person than to merge two students.
    $similaritythreshold = 0.80;
    $distancethreshold = 0.70;

    $groups = $DB->get_records('faceattendance_capture_groups', [
        'capturesessionid' => $capturesessionid,
        'faceattendanceid' => $faceattendanceid,
        
    ], 'timemodified DESC', '*', 0, 120);

    $groups = array_filter($groups, static function($group) {
        return in_array((string)$group->status, ['pending', 'autoassigned'], true);
    });

    $best = null;
    $bestscore = -INF;
    foreach ($groups as $group) {
        $existing = faceattendance_capture_descriptor_from_payload($group->prototypeembeddingjson ?? null);
        if (count($existing) !== count($descriptor)) {
            continue;
        }
        $similarity = faceattendance_capture_vector_cosine($descriptor, $existing);
        $distance = faceattendance_capture_vector_distance($descriptor, $existing);
        if ($similarity >= $similaritythreshold || $distance <= $distancethreshold) {
            $score = $similarity - ($distance * 0.05);
            if ($score > $bestscore) {
                $bestscore = $score;
                $best = $group;
            }
        }
    }
    return $best;
}


function faceattendance_capture_extract_descriptors_from_course_embedding(?string $embeddingjson): array {
    if (empty($embeddingjson)) {
        return [];
    }

    $payload = json_decode($embeddingjson, true);
    if (!is_array($payload)) {
        return [];
    }

    $descriptors = [];
    if (!empty($payload['descriptor']) && is_array($payload['descriptor'])) {
        $descriptors[] = array_map('floatval', $payload['descriptor']);
    }
    if (!empty($payload['embedding']) && is_array($payload['embedding'])) {
        $descriptors[] = array_map('floatval', $payload['embedding']);
    }
    if (!empty($payload['captures']) && is_array($payload['captures'])) {
        foreach ($payload['captures'] as $capture) {
            if (is_array($capture) && !empty($capture['descriptor']) && is_array($capture['descriptor'])) {
                $descriptors[] = array_map('floatval', $capture['descriptor']);
            }
        }
    }

    return array_values(array_filter($descriptors, static function($descriptor) {
        return count($descriptor) === 128;
    }));
}

function faceattendance_capture_find_registered_student_match(int $courseid, array $descriptor): ?array {
    global $DB;

    // This is deliberately stricter than the live recognition threshold.
    // Automatic learning must prefer "no auto-label" over poisoning the student's biometric template.
    $autosimilaritythreshold = 0.52;
    $requiredmargin = 0.04;

    $records = $DB->get_records_select('faceattendance_course_embeddings',
        'course = :course AND status IN (\'registered\', \'active\', \'approved\')',
        ['course' => $courseid]
    );

    $bestbyuser = [];
    foreach ($records as $record) {
        foreach (faceattendance_capture_extract_descriptors_from_course_embedding($record->embeddingjson ?? '') as $known) {
            $similarity = faceattendance_capture_vector_cosine($descriptor, $known);
            if (!isset($bestbyuser[$record->userid]) || $similarity > $bestbyuser[$record->userid]['similarity']) {
                $bestbyuser[$record->userid] = [
                    'userid' => (int)$record->userid,
                    'embeddingid' => (int)$record->id,
                    'similarity' => (float)$similarity,
                ];
            }
        }
    }

    if (!$bestbyuser) {
        return null;
    }

    usort($bestbyuser, static function($a, $b) {
        return $b['similarity'] <=> $a['similarity'];
    });

    $best = $bestbyuser[0];
    $second = $bestbyuser[1]['similarity'] ?? -1.0;
    $margin = $best['similarity'] - $second;

    if ($best['similarity'] >= $autosimilaritythreshold && $margin >= $requiredmargin) {
        $best['secondSimilarity'] = (float)$second;
        $best['margin'] = (float)$margin;
        $best['threshold'] = (float)$autosimilaritythreshold;
        return $best;
    }

    return null;
}

function faceattendance_capture_make_autolearn_capture(array $descriptor, int $groupid, array $match, int $now, string $source): array {
    return [
        'pose' => 'auto_intake_capture',
        'label' => 'Auto intake group #' . $groupid,
        'descriptor' => array_values($descriptor),
        'quality' => [
            'score' => 0,
            'autoSimilarity' => (float)$match['similarity'],
            'autoMargin' => (float)$match['margin'],
        ],
        'source' => $source,
        'capturegroupid' => $groupid,
        'autoAssigned' => true,
        'capturedAt' => gmdate('c', $now),
    ];
}

function faceattendance_capture_append_auto_descriptor(int $courseid, int $userid, array $descriptor, int $groupid, array $match, string $source): bool {
    global $DB;

    $now = time();
    $maxsamplesperstudent = 16;
    $duplicatethreshold = 0.995;

    $existing = $DB->get_record('faceattendance_course_embeddings', [
        'course' => $courseid,
        'userid' => $userid,
    ]);

    $user = $DB->get_record('user', ['id' => $userid, 'deleted' => 0], 'id, firstname, lastname, email, username', IGNORE_MISSING);
    if (!$user) {
        return false;
    }

    if ($existing) {
        $payload = json_decode($existing->embeddingjson, true);
        if (!is_array($payload)) {
            $payload = [
                'version' => 1,
                'name' => fullname($user),
                'studentId' => (string)$userid,
                'model' => [
                    'family' => 'opencv',
                    'detector' => 'capture-intake-station',
                    'recognizer' => 'sface',
                    'descriptorLength' => 128,
                ],
                'captures' => [],
            ];
        }
        if (empty($payload['captures']) || !is_array($payload['captures'])) {
            $payload['captures'] = [];
        }

        foreach ($payload['captures'] as $capture) {
            if (is_array($capture) && isset($capture['capturegroupid']) && (int)$capture['capturegroupid'] === $groupid) {
                return true;
            }
            if (is_array($capture) && !empty($capture['descriptor']) && is_array($capture['descriptor'])) {
                if (count($capture['descriptor']) === 128 && faceattendance_capture_vector_cosine($descriptor, array_map('floatval', $capture['descriptor'])) >= $duplicatethreshold) {
                    return true;
                }
            }
        }

        if (count($payload['captures']) >= $maxsamplesperstudent) {
            return true;
        }

        $payload['captures'][] = faceattendance_capture_make_autolearn_capture($descriptor, $groupid, $match, $now, $source);
        $existing->embeddingjson = json_encode($payload, JSON_UNESCAPED_SLASHES);
        $existing->samples = count($payload['captures']);
        $existing->modelname = 'opencv-sface-2021dec';
        $existing->embeddingdim = 128;
        $existing->status = 'registered';
        $existing->timemodified = $now;
        $DB->update_record('faceattendance_course_embeddings', $existing);
        return true;
    }

    // This should rarely happen because auto-assignment requires an existing match,
    // but keep it safe in case the store was changed concurrently.
    $payload = [
        'version' => 1,
        'name' => fullname($user),
        'studentId' => (string)$userid,
        'model' => [
            'family' => 'opencv',
            'detector' => 'capture-intake-station',
            'recognizer' => 'sface',
            'descriptorLength' => 128,
        ],
        'captures' => [faceattendance_capture_make_autolearn_capture($descriptor, $groupid, $match, $now, $source)],
    ];

    $DB->insert_record('faceattendance_course_embeddings', (object)[
        'course' => $courseid,
        'userid' => $userid,
        'embeddingjson' => json_encode($payload, JSON_UNESCAPED_SLASHES),
        'modelname' => 'opencv-sface-2021dec',
        'embeddingdim' => 128,
        'samples' => 1,
        'qualityscore' => 0,
        'status' => 'registered',
        'timecreated' => $now,
        'timemodified' => $now,
    ]);
    return true;
}

function faceattendance_capture_thumbnail_count(context_module $context, int $groupid): int {
    $fs = get_file_storage();
    $files = $fs->get_area_files($context->id, 'mod_faceattendance', 'captureface', $groupid, 'id ASC', false);
    return count($files);
}

function faceattendance_capture_store_thumbnail(context_module $context, int $groupid, ?array $thumbnail): bool {
    if (!$thumbnail) {
        return false;
    }

    $fs = get_file_storage();
    $files = $fs->get_area_files($context->id, 'mod_faceattendance', 'captureface', $groupid, 'id ASC', false);
    if (count($files) >= 2) {
        return false;
    }

    $fileinfo = [
        'contextid' => $context->id,
        'component' => 'mod_faceattendance',
        'filearea' => 'captureface',
        'itemid' => $groupid,
        'filepath' => '/',
        'filename' => $thumbnail['filename'],
    ];

    $fs->create_file_from_string($fileinfo, $thumbnail['binary']);
    return true;
}

try {
    require_login();
    $payload = faceattendance_capture_read_json_payload();

    if (empty($payload->sesskey) || !confirm_sesskey((string)$payload->sesskey)) {
        faceattendance_capture_json_response(['ok' => false, 'error' => 'Invalid Moodle sesskey.'], 403);
    }

    $cmid = isset($payload->cmid) ? (int)$payload->cmid : 0;
    $capturesessionid = isset($payload->capturesessionid) ? (int)$payload->capturesessionid : 0;
    $source = isset($payload->source) ? clean_param((string)$payload->source, PARAM_TEXT) : 'browser-capture-intake';

    if ($cmid <= 0 || $capturesessionid <= 0 || empty($payload->descriptor) || !is_array($payload->descriptor)) {
        faceattendance_capture_json_response(['ok' => false, 'error' => 'cmid, capturesessionid and descriptor are required.'], 400);
    }
    if (count($payload->descriptor) !== 128) {
        faceattendance_capture_json_response(['ok' => false, 'error' => 'Capture descriptor must have 128 numbers.'], 400);
    }

    $cm = get_coursemodule_from_id('faceattendance', $cmid, 0, false, MUST_EXIST);
    $course = get_course($cm->course);
    $faceattendance = $DB->get_record('faceattendance', ['id' => $cm->instance], '*', MUST_EXIST);
    require_login($course, true, $cm);
    $context = context_module::instance($cm->id);
    require_capability('mod/faceattendance:takeattendance', $context);

    $capturesession = $DB->get_record('faceattendance_capture_sessions', [
        'id' => $capturesessionid,
        'faceattendanceid' => $faceattendance->id,
        'course' => $course->id,
    ], '*', MUST_EXIST);

    if ($capturesession->status !== 'open') {
        faceattendance_capture_json_response(['ok' => false, 'error' => 'Capture session is not open.'], 400);
    }

    $now = time();
    $descriptor = array_map('floatval', $payload->descriptor);
    $embeddingjson = json_encode([
        'version' => 1,
        'model' => 'opencv-sface-2021dec',
        'descriptorLength' => 128,
        'descriptor' => $descriptor,
        'capturedAt' => gmdate('c', $now),
    ], JSON_UNESCAPED_SLASHES);

    $thumbnail = null;
    if (!empty($payload->thumbnail) && is_string($payload->thumbnail)) {
        $thumbnail = faceattendance_capture_extract_thumbnail_binary($payload->thumbnail);
    }

    $group = faceattendance_capture_find_similar_group((int)$capturesession->id, (int)$faceattendance->id, $descriptor);
    $deduplicated = !empty($group);
    $automatch = faceattendance_capture_find_registered_student_match((int)$course->id, $descriptor);
    $autoassigned = false;
    $autoembeddingadded = false;

    if ($group) {
        $group->lastseen = $now;
        $group->detectioncount = max(1, (int)$group->detectioncount) + 1;
        $group->sampleembeddingjson = $embeddingjson;
        if (isset($payload->candidate)) {
            $group->candidatejson = json_encode($payload->candidate, JSON_UNESCAPED_SLASHES);
        }

        if ($automatch && (empty($group->assigneduserid) || (int)$group->assigneduserid === (int)$automatch['userid'])) {
            $group->status = 'autoassigned';
            $group->assigneduserid = (int)$automatch['userid'];
            $group->assignedby = null;
            $autoassigned = true;
        }

        $group->timemodified = $now;
        $DB->update_record('faceattendance_capture_groups', $group);
        $groupid = (int)$group->id;
    } else {
        $initialstatus = $automatch ? 'autoassigned' : 'pending';
        $groupid = $DB->insert_record('faceattendance_capture_groups', (object)[
            'capturesessionid' => (int)$capturesession->id,
            'faceattendanceid' => (int)$faceattendance->id,
            'course' => (int)$course->id,
            'prototypeembeddingjson' => $embeddingjson,
            'sampleembeddingjson' => $embeddingjson,
            'candidatejson' => isset($payload->candidate) ? json_encode($payload->candidate, JSON_UNESCAPED_SLASHES) : null,
            'source' => $source,
            'status' => $initialstatus,
            'assigneduserid' => $automatch ? (int)$automatch['userid'] : null,
            'assignedby' => null,
            'firstseen' => $now,
            'lastseen' => $now,
            'detectioncount' => 1,
            'thumbnailcount' => 0,
            'qualityscore' => 0,
            'timecreated' => $now,
            'timemodified' => $now,
        ]);
        $autoassigned = !empty($automatch);
    }

    if ($autoassigned && $automatch) {
        $autoembeddingadded = faceattendance_capture_append_auto_descriptor((int)$course->id, (int)$automatch['userid'], $descriptor, (int)$groupid, $automatch, $source);
        // Privacy rule: if the system is confident enough to auto-assign, there is no reason to keep temporary face images.
        faceattendance_delete_capture_group_thumbnails($context->id, (int)$groupid);
        $storedthumbnail = false;
        $thumbnailcount = 0;
        $DB->set_field('faceattendance_capture_groups', 'thumbnailcount', 0, ['id' => $groupid]);
    } else {
        $storedthumbnail = faceattendance_capture_store_thumbnail($context, (int)$groupid, $thumbnail);
        $thumbnailcount = faceattendance_capture_thumbnail_count($context, (int)$groupid);
        $DB->set_field('faceattendance_capture_groups', 'thumbnailcount', $thumbnailcount, ['id' => $groupid]);
    }

    faceattendance_capture_json_response([
        'ok' => true,
        'groupid' => (int)$groupid,
        'deduplicated' => $deduplicated,
        'autoAssigned' => $autoassigned,
        'autoAssignedUserId' => $autoassigned && $automatch ? (int)$automatch['userid'] : null,
        'autoSimilarity' => $autoassigned && $automatch ? (float)$automatch['similarity'] : null,
        'autoMargin' => $autoassigned && $automatch ? (float)$automatch['margin'] : null,
        'autoEmbeddingAdded' => $autoembeddingadded,
        'thumbnailStored' => $storedthumbnail,
        'thumbnailCount' => $thumbnailcount,
    ]);
} catch (Throwable $e) {
    faceattendance_capture_json_response(['ok' => false, 'error' => $e->getMessage()], 500);
}
