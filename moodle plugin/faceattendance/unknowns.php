<?php
// This file is part of Moodle - http://moodle.org/

/**
 * Teacher review page for unknown faces.
 *
 * @package     mod_faceattendance
 * @copyright   2026
 * @license     http://www.gnu.org/copyleft/gpl.html GNU GPL v3 or later
 */

require_once(__DIR__ . '/../../config.php');
require_once(__DIR__ . '/lib.php');

$id = required_param('id', PARAM_INT); // Course module id.

$cm = get_coursemodule_from_id('faceattendance', $id, 0, false, MUST_EXIST);
$course = get_course($cm->course);
$faceattendance = $DB->get_record('faceattendance', ['id' => $cm->instance], '*', MUST_EXIST);

require_login($course, true, $cm);

$context = context_module::instance($cm->id);
require_capability('mod/faceattendance:reviewunknowns', $context);

$pageurl = new moodle_url('/mod/faceattendance/unknowns.php', ['id' => $cm->id]);
$returnurl = new moodle_url('/mod/faceattendance/view.php', ['id' => $cm->id]);

$PAGE->set_url($pageurl);
$PAGE->set_title(get_string('reviewunknowns', 'faceattendance'));
$PAGE->set_heading(format_string($course->fullname));
$PAGE->set_context($context);

$coursecontext = context_course::instance($course->id);
$users = get_enrolled_users($coursecontext, '', 0, 'u.id, u.firstname, u.lastname, u.email, u.username', 'u.lastname ASC, u.firstname ASC');
$options = [];
foreach ($users as $user) {
    if (isguestuser($user)) {
        continue;
    }
    $options[$user->id] = fullname($user) . ' (' . $user->email . ')';
}

$assigneduserid = optional_param('assigneduserid', 0, PARAM_INT);
if ($assigneduserid > 0 && !isset($options[$assigneduserid])) {
    $assigneduserid = 0;
}

$assigneduseroptions = [0 => get_string('allassignedusers', 'faceattendance')];
$assigneduserssql = "SELECT DISTINCT u.id, u.firstname, u.lastname, u.email
                       FROM {faceattendance_unknowns} ukn
                       JOIN {user} u ON u.id = ukn.resolveduserid
                      WHERE ukn.faceattendanceid = :faceattendanceid
                        AND ukn.status = :resolved
                        AND ukn.resolveduserid IS NOT NULL
                   ORDER BY u.lastname ASC, u.firstname ASC";
$assignedusers = $DB->get_records_sql($assigneduserssql, [
    'faceattendanceid' => $faceattendance->id,
    'resolved' => 'resolved',
]);
foreach ($assignedusers as $assigneduser) {
    $assigneduseroptions[(int)$assigneduser->id] = fullname($assigneduser) . ' (' . $assigneduser->email . ')';
}


function faceattendance_unknown_descriptor_from_payload(?string $embeddingjson): array {
    if (empty($embeddingjson)) {
        return [];
    }
    $data = json_decode($embeddingjson, true);
    if (!is_array($data) || empty($data['descriptor']) || !is_array($data['descriptor'])) {
        return [];
    }
    return array_map('floatval', $data['descriptor']);
}

function faceattendance_create_embedding_payload_for_unknown(stdClass $user, stdClass $unknown, array $descriptor, int $now): string {
    return json_encode([
        'version' => 1,
        'name' => fullname($user),
        'studentId' => (string)$user->id,
        'model' => [
            'family' => 'opencv',
            'detector' => 'yunet',
            'recognizer' => 'sface',
            'descriptorLength' => 128,
        ],
        'captures' => [[
            'pose' => 'teacher_resolved_unknown',
            'label' => 'Teacher assigned unknown #' . (int)$unknown->id,
            'descriptor' => array_values($descriptor),
            'quality' => ['score' => 0],
            'source' => 'teacher-resolved-unknown',
            'unknownid' => (int)$unknown->id,
            'capturedAt' => gmdate('c', (int)$unknown->lastseen ?: $now),
        ]],
    ], JSON_UNESCAPED_SLASHES);
}

function faceattendance_append_unknown_to_user_embedding(stdClass $faceattendance, stdClass $course, stdClass $user, stdClass $unknown): bool {
    global $DB;

    $descriptor = faceattendance_unknown_descriptor_from_payload($unknown->embeddingjson ?? null);
    if (count($descriptor) !== 128) {
        return false;
    }

    $now = time();
    $existing = $DB->get_record('faceattendance_embeddings', [
        'faceattendanceid' => $faceattendance->id,
        'userid' => $user->id,
    ]);

    if ($existing) {
        $payload = json_decode($existing->embeddingjson, true);
        if (!is_array($payload)) {
            $payload = json_decode(faceattendance_create_embedding_payload_for_unknown($user, $unknown, [], $now), true);
            $payload['captures'] = [];
        }
        if (empty($payload['captures']) || !is_array($payload['captures'])) {
            $payload['captures'] = [];
        }

        // Avoid appending the same resolved unknown twice if the form is resubmitted.
        foreach ($payload['captures'] as $capture) {
            if (is_array($capture) && isset($capture['unknownid']) && (int)$capture['unknownid'] === (int)$unknown->id) {
                return true;
            }
        }

        $payload['captures'][] = [
            'pose' => 'teacher_resolved_unknown',
            'label' => 'Teacher assigned unknown #' . (int)$unknown->id,
            'descriptor' => array_values($descriptor),
            'quality' => ['score' => 0],
            'source' => 'teacher-resolved-unknown',
            'unknownid' => (int)$unknown->id,
            'capturedAt' => gmdate('c', (int)$unknown->lastseen ?: $now),
        ];

        $existing->embeddingjson = json_encode($payload, JSON_UNESCAPED_SLASHES);
        $existing->samples = count($payload['captures']);
        $existing->modelname = 'opencv-sface-2021dec';
        $existing->embeddingdim = 128;
        $existing->status = 'registered';
        $existing->timemodified = $now;
        $DB->update_record('faceattendance_embeddings', $existing);
        return true;
    }

    $embeddingjson = faceattendance_create_embedding_payload_for_unknown($user, $unknown, $descriptor, $now);
    $DB->insert_record('faceattendance_embeddings', (object)[
        'faceattendanceid' => (int)$faceattendance->id,
        'course' => (int)$course->id,
        'userid' => (int)$user->id,
        'embeddingjson' => $embeddingjson,
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

function faceattendance_ensure_registration_for_resolved_user(stdClass $faceattendance, stdClass $course, stdClass $user): void {
    global $DB;

    $now = time();
    $registration = $DB->get_record('faceattendance_registrations', [
        'faceattendanceid' => $faceattendance->id,
        'userid' => $user->id,
    ]);

    $externalid = 'moodle_user_' . (int)$user->id;
    if ($registration) {
        $registration->status = 'registered';
        if (empty($registration->externalid)) {
            $registration->externalid = $externalid;
        }
        $registration->timemodified = $now;
        $DB->update_record('faceattendance_registrations', $registration);
        return;
    }

    $DB->insert_record('faceattendance_registrations', (object)[
        'faceattendanceid' => (int)$faceattendance->id,
        'course' => (int)$course->id,
        'userid' => (int)$user->id,
        'externalid' => $externalid,
        'status' => 'registered',
        'notes' => 'Created after teacher assigned an unknown face.',
        'timecreated' => $now,
        'timemodified' => $now,
    ]);
}

if ($_SERVER['REQUEST_METHOD'] === 'POST') {
    require_sesskey();
    $action = required_param('action', PARAM_ALPHA);
    $unknownid = required_param('unknownid', PARAM_INT);
    $unknown = $DB->get_record('faceattendance_unknowns', [
        'id' => $unknownid,
        'faceattendanceid' => $faceattendance->id,
    ], '*', MUST_EXIST);

    if ($action === 'ignore') {
        faceattendance_delete_unknown_thumbnail($context->id, (int)$unknown->id);
        $unknown->status = 'ignored';
        $unknown->resolvedby = $USER->id;
        $unknown->timemodified = time();
        $DB->update_record('faceattendance_unknowns', $unknown);
        redirect($pageurl, get_string('unknownignored', 'faceattendance'), null, \core\output\notification::NOTIFY_SUCCESS);
    }

    if ($action === 'assign') {
        $userid = required_param('userid', PARAM_INT);
        if (!isset($options[$userid])) {
            redirect($pageurl, get_string('usernotenrolled', 'faceattendance'), null, \core\output\notification::NOTIFY_ERROR);
        }

        $assigneduser = $DB->get_record('user', ['id' => $userid, 'deleted' => 0], '*', MUST_EXIST);

        $now = time();
        $session = $DB->get_record('faceattendance_sessions', [
            'id' => $unknown->sessionid,
            'faceattendanceid' => $faceattendance->id,
        ], '*', MUST_EXIST);

        $existing = $DB->get_record('faceattendance_session_records', [
            'sessionid' => $session->id,
            'userid' => $userid,
        ]);

        $status = ($unknown->firstseen > ((int)$session->starttime + (int)$session->latethreshold)) ? 'late' : 'present';
        if ($existing) {
            $existing->status = $status;
            $existing->detectioncount = max((int)$existing->detectioncount, 1);
            $existing->lastseen = max((int)$existing->lastseen, (int)$unknown->lastseen);
            $existing->source = 'teacher-resolved-unknown';
            $existing->timemodified = $now;
            $DB->update_record('faceattendance_session_records', $existing);
        } else {
            $DB->insert_record('faceattendance_session_records', (object)[
                'sessionid' => (int)$session->id,
                'faceattendanceid' => (int)$faceattendance->id,
                'course' => (int)$course->id,
                'userid' => $userid,
                'status' => $status,
                'confidence' => 0,
                'distance' => 0,
                'detectioncount' => 1,
                'firstseen' => (int)$unknown->firstseen,
                'lastseen' => (int)$unknown->lastseen,
                'source' => 'teacher-resolved-unknown',
                'timecreated' => $now,
                'timemodified' => $now,
            ]);
        }

        $embeddingadded = faceattendance_append_unknown_to_user_embedding($faceattendance, $course, $assigneduser, $unknown);
        faceattendance_ensure_registration_for_resolved_user($faceattendance, $course, $assigneduser);

        // Privacy rule: once the teacher labels the unknown face correctly,
        // remove the temporary face thumbnail and keep only the audit/attendance metadata.
        faceattendance_delete_unknown_thumbnail($context->id, (int)$unknown->id);

        $unknown->status = 'resolved';
        $unknown->resolveduserid = $userid;
        $unknown->resolvedby = $USER->id;
        $unknown->timemodified = $now;
        $DB->update_record('faceattendance_unknowns', $unknown);

        $message = get_string($embeddingadded ? 'unknownassigned_embeddingadded' : 'unknownassigned_embeddingnotadded', 'faceattendance');
        redirect($pageurl, $message, null, \core\output\notification::NOTIFY_SUCCESS);
    }
}


function faceattendance_unknowns_descriptor_from_embeddingjson(?string $embeddingjson): array {
    if (empty($embeddingjson)) {
        return [];
    }
    $data = json_decode($embeddingjson, true);
    if (!is_array($data) || empty($data['descriptor']) || !is_array($data['descriptor'])) {
        return [];
    }
    return array_map('floatval', $data['descriptor']);
}

function faceattendance_unknowns_distance(array $a, array $b): float {
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

function faceattendance_unknowns_cosine(array $a, array $b): float {
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

function faceattendance_unknowns_are_similar(array $a, array $b): bool {
    return faceattendance_unknowns_cosine($a, $b) >= 0.78 || faceattendance_unknowns_distance($a, $b) <= 0.75;
}

function faceattendance_limit_similar_unknown_rows(array $unknowns, int $maxpercluster = 2): array {
    $visible = [];
    $clusters = [];

    foreach ($unknowns as $unknown) {
        if (($unknown->status ?? '') !== 'unknown') {
            $visible[] = $unknown;
            continue;
        }

        $descriptor = faceattendance_unknowns_descriptor_from_embeddingjson($unknown->embeddingjson ?? null);
        if (!$descriptor) {
            $visible[] = $unknown;
            continue;
        }

        $matched = false;
        foreach ($clusters as &$cluster) {
            if ((int)$cluster['sessionid'] !== (int)$unknown->sessionid) {
                continue;
            }
            if (!faceattendance_unknowns_are_similar($descriptor, $cluster['descriptor'])) {
                continue;
            }

            $matched = true;
            $cluster['count']++;
            if ($cluster['shown'] < $maxpercluster) {
                $visible[] = $unknown;
                $cluster['shown']++;
            }
            break;
        }
        unset($cluster);

        if (!$matched) {
            $clusters[] = [
                'sessionid' => (int)$unknown->sessionid,
                'descriptor' => $descriptor,
                'count' => 1,
                'shown' => 1,
            ];
            $visible[] = $unknown;
        }
    }

    return $visible;
}

$params = ['faceattendanceid' => $faceattendance->id];
$assignedfilter = '';
if ($assigneduserid > 0) {
    $assignedfilter = ' AND ukn.resolveduserid = :assigneduserid';
    $params['assigneduserid'] = $assigneduserid;
}
$sql = "SELECT ukn.*, s.name AS sessionname
          FROM {faceattendance_unknowns} ukn
          JOIN {faceattendance_sessions} s ON s.id = ukn.sessionid
         WHERE ukn.faceattendanceid = :faceattendanceid
               $assignedfilter
      ORDER BY ukn.status ASC, ukn.timemodified DESC";
$unknowns = $DB->get_records_sql($sql, $params);
$unknowns = faceattendance_limit_similar_unknown_rows(array_values($unknowns), 2);

echo $OUTPUT->header();
echo $OUTPUT->heading(get_string('reviewunknowns', 'faceattendance'));
echo html_writer::tag('p', get_string('unknownsintro', 'faceattendance'), ['class' => 'alert alert-info']);

if (count($assigneduseroptions) > 1) {
    echo html_writer::start_tag('form', ['method' => 'get', 'class' => 'form-inline mb-3']);
    echo html_writer::empty_tag('input', ['type' => 'hidden', 'name' => 'id', 'value' => $cm->id]);
    echo html_writer::tag('label', get_string('filterassigneduser', 'faceattendance'), ['for' => 'assigneduserid', 'class' => 'mr-2']);
    echo html_writer::select($assigneduseroptions, 'assigneduserid', $assigneduserid, false, [
        'id' => 'assigneduserid',
        'class' => 'custom-select mr-2',
        'onchange' => 'this.form.submit()'
    ]);
    echo html_writer::empty_tag('input', ['type' => 'submit', 'class' => 'btn btn-secondary', 'value' => get_string('filter')]);
    echo html_writer::end_tag('form');
}

if (!$unknowns) {
    echo $OUTPUT->notification(get_string('nounknowns', 'faceattendance'), 'info');
} else {
    $table = new html_table();
    $table->head = [
        get_string('id'),
        get_string('unknownphoto', 'faceattendance'),
        get_string('sessionname', 'faceattendance'),
        get_string('status', 'faceattendance'),
        get_string('source', 'faceattendance'),
        get_string('firstseen', 'faceattendance'),
        get_string('lastseen', 'faceattendance'),
        get_string('assignedstudent', 'faceattendance'),
        get_string('actions'),
    ];
    $table->data = [];

    foreach ($unknowns as $unknown) {
        $thumbhtml = get_string('nothumbnail', 'faceattendance');
        $fs = get_file_storage();
        $files = $fs->get_area_files($context->id, 'mod_faceattendance', 'unknownface', (int)$unknown->id, 'id DESC', false);
        if (!empty($files)) {
            $file = reset($files);
            $thumbnailurl = moodle_url::make_pluginfile_url(
                $context->id,
                'mod_faceattendance',
                'unknownface',
                (int)$unknown->id,
                '/',
                $file->get_filename()
            );
            $thumbhtml = html_writer::empty_tag('img', [
                'src' => $thumbnailurl,
                'alt' => get_string('unknownphoto', 'faceattendance'),
                'style' => 'max-width:140px; max-height:140px; object-fit:cover; border-radius:8px; border:1px solid #ddd;',
            ]);
        }

        $assignedstudent = '-';
        if (!empty($unknown->resolveduserid)) {
            $assignedrecord = $DB->get_record('user', ['id' => $unknown->resolveduserid], 'id, firstname, lastname, email', IGNORE_MISSING);
            if ($assignedrecord) {
                $assignedstudent = fullname($assignedrecord) . ' (' . s($assignedrecord->email) . ')';
            }
        }

        $actions = '';
        if ($unknown->status === 'unknown') {
            $select = html_writer::select($options, 'userid', '', ['' => get_string('choosestudent', 'faceattendance')], ['class' => 'form-control d-inline-block', 'style' => 'width:260px']);
            $actions = html_writer::start_tag('form', ['method' => 'post', 'style' => 'display:inline-block; margin-right: 8px;']) .
                html_writer::empty_tag('input', ['type' => 'hidden', 'name' => 'id', 'value' => $cm->id]) .
                html_writer::empty_tag('input', ['type' => 'hidden', 'name' => 'sesskey', 'value' => sesskey()]) .
                html_writer::empty_tag('input', ['type' => 'hidden', 'name' => 'action', 'value' => 'assign']) .
                html_writer::empty_tag('input', ['type' => 'hidden', 'name' => 'unknownid', 'value' => $unknown->id]) .
                $select . ' ' .
                html_writer::empty_tag('input', ['type' => 'submit', 'class' => 'btn btn-sm btn-primary', 'value' => get_string('assign', 'faceattendance')]) .
                html_writer::end_tag('form');
            $actions .= html_writer::start_tag('form', ['method' => 'post', 'style' => 'display:inline-block']) .
                html_writer::empty_tag('input', ['type' => 'hidden', 'name' => 'id', 'value' => $cm->id]) .
                html_writer::empty_tag('input', ['type' => 'hidden', 'name' => 'sesskey', 'value' => sesskey()]) .
                html_writer::empty_tag('input', ['type' => 'hidden', 'name' => 'action', 'value' => 'ignore']) .
                html_writer::empty_tag('input', ['type' => 'hidden', 'name' => 'unknownid', 'value' => $unknown->id]) .
                html_writer::empty_tag('input', ['type' => 'submit', 'class' => 'btn btn-sm btn-outline-secondary', 'value' => get_string('ignore', 'faceattendance')]) .
                html_writer::end_tag('form');
        } else if ($unknown->status === 'resolved' && $unknown->resolveduserid) {
            $resolveduser = $DB->get_record('user', ['id' => $unknown->resolveduserid], 'id, firstname, lastname, email', IGNORE_MISSING);
            $actions = $resolveduser ? get_string('resolvedto', 'faceattendance', fullname($resolveduser)) : get_string('resolved', 'faceattendance');
        }

        $table->data[] = [
            (int)$unknown->id,
            $thumbhtml,
            format_string($unknown->sessionname),
            s($unknown->status),
            s($unknown->source),
            userdate((int)$unknown->firstseen),
            userdate((int)$unknown->lastseen),
            $assignedstudent,
            $actions,
        ];
    }

    echo html_writer::table($table);
}

echo html_writer::div(
    html_writer::link($returnurl, get_string('backtoactivity', 'faceattendance'), ['class' => 'btn btn-secondary']),
    'mt-3'
);

echo $OUTPUT->footer();
