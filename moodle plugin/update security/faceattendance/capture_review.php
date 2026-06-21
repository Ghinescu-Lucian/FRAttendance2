<?php
// This file is part of Moodle - http://moodle.org/

/**
 * Teacher review page for capture-first intake groups.
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

$pageurl = new moodle_url('/mod/faceattendance/capture_review.php', ['id' => $cm->id]);
$returnurl = new moodle_url('/mod/faceattendance/view.php', ['id' => $cm->id]);
$captureurl = new moodle_url('/mod/faceattendance/capture.php', ['id' => $cm->id]);

$PAGE->set_url($pageurl);
$PAGE->set_title(get_string('reviewcaptures', 'faceattendance'));
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

function faceattendance_create_embedding_payload_for_capture(stdClass $user, stdClass $group, array $descriptor, int $now): string {
    return json_encode([
        'version' => 1,
        'name' => fullname($user),
        'studentId' => (string)$user->id,
        'model' => [
            'family' => 'opencv',
            'detector' => 'browser-face-api-or-yunet-station',
            'recognizer' => 'sface',
            'descriptorLength' => 128,
        ],
        'captures' => [[
            'pose' => 'teacher_labeled_intake_capture',
            'label' => 'Teacher labeled intake group #' . (int)$group->id,
            'descriptor' => array_values($descriptor),
            'quality' => ['score' => (float)($group->qualityscore ?? 0)],
            'source' => 'teacher-labeled-intake-capture',
            'capturegroupid' => (int)$group->id,
            'capturedAt' => gmdate('c', (int)$group->lastseen ?: $now),
        ]],
    ], JSON_UNESCAPED_SLASHES);
}

function faceattendance_append_capture_group_to_user_embedding(stdClass $course, stdClass $user, stdClass $group): bool {
    global $DB;

    $descriptor = faceattendance_capture_descriptor_from_payload($group->prototypeembeddingjson ?? null);
    if (count($descriptor) !== 128) {
        return false;
    }

    $now = time();
    $existing = $DB->get_record('faceattendance_course_embeddings', [
        'course' => $course->id,
        'userid' => $user->id,
    ]);

    if ($existing) {
        $payload = json_decode($existing->embeddingjson, true);
        if (!is_array($payload)) {
            $payload = json_decode(faceattendance_create_embedding_payload_for_capture($user, $group, [], $now), true);
            $payload['captures'] = [];
        }
        if (empty($payload['captures']) || !is_array($payload['captures'])) {
            $payload['captures'] = [];
        }

        foreach ($payload['captures'] as $capture) {
            if (is_array($capture) && isset($capture['capturegroupid']) && (int)$capture['capturegroupid'] === (int)$group->id) {
                return true;
            }
        }

        $payload['captures'][] = [
            'pose' => 'teacher_labeled_intake_capture',
            'label' => 'Teacher labeled intake group #' . (int)$group->id,
            'descriptor' => array_values($descriptor),
            'quality' => ['score' => (float)($group->qualityscore ?? 0)],
            'source' => 'teacher-labeled-intake-capture',
            'capturegroupid' => (int)$group->id,
            'capturedAt' => gmdate('c', (int)$group->lastseen ?: $now),
        ];

        $existing->embeddingjson = json_encode($payload, JSON_UNESCAPED_SLASHES);
        $existing->samples = count($payload['captures']);
        $existing->modelname = 'opencv-sface-2021dec';
        $existing->embeddingdim = 128;
        $existing->status = 'registered';
        $existing->timemodified = $now;
        $DB->update_record('faceattendance_course_embeddings', $existing);
        return true;
    }

    $DB->insert_record('faceattendance_course_embeddings', (object)[
        'course' => (int)$course->id,
        'userid' => (int)$user->id,
        'embeddingjson' => faceattendance_create_embedding_payload_for_capture($user, $group, $descriptor, $now),
        'modelname' => 'opencv-sface-2021dec',
        'embeddingdim' => 128,
        'samples' => 1,
        'qualityscore' => (float)($group->qualityscore ?? 0),
        'status' => 'registered',
        'timecreated' => $now,
        'timemodified' => $now,
    ]);
    return true;
}

function faceattendance_ensure_registration_after_capture(stdClass $faceattendance, stdClass $course, stdClass $user): void {
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
        'notes' => 'Created after teacher labeled an intake capture group.',
        'timecreated' => $now,
        'timemodified' => $now,
    ]);
}

if ($_SERVER['REQUEST_METHOD'] === 'POST') {
    require_sesskey();
    $action = required_param('action', PARAM_ALPHA);
    $groupid = required_param('groupid', PARAM_INT);
    $group = $DB->get_record('faceattendance_capture_groups', [
        'id' => $groupid,
        'faceattendanceid' => $faceattendance->id,
    ], '*', MUST_EXIST);

    if ($action === 'ignore') {
        faceattendance_delete_capture_group_thumbnails($context->id, (int)$group->id);
        $group->status = 'ignored';
        $group->assignedby = $USER->id;
        $group->timemodified = time();
        $DB->update_record('faceattendance_capture_groups', $group);
        redirect($pageurl, get_string('captureignored', 'faceattendance'), null, \core\output\notification::NOTIFY_SUCCESS);
    }

    if ($action === 'assign') {
        $userid = required_param('userid', PARAM_INT);
        if (!isset($options[$userid])) {
            redirect($pageurl, get_string('usernotenrolled', 'faceattendance'), null, \core\output\notification::NOTIFY_ERROR);
        }

        $assigneduser = $DB->get_record('user', ['id' => $userid, 'deleted' => 0], '*', MUST_EXIST);
        $embeddingadded = faceattendance_append_capture_group_to_user_embedding($course, $assigneduser, $group);
        faceattendance_ensure_registration_after_capture($faceattendance, $course, $assigneduser);

        // Privacy rule: after teacher labels the group, delete temporary images and keep only the descriptor/audit data.
        faceattendance_delete_capture_group_thumbnails($context->id, (int)$group->id);

        $group->status = 'assigned';
        $group->assigneduserid = $userid;
        $group->assignedby = $USER->id;
        $group->timemodified = time();
        $DB->update_record('faceattendance_capture_groups', $group);

        $message = get_string($embeddingadded ? 'captureassigned_embeddingadded' : 'captureassigned_embeddingnotadded', 'faceattendance');
        redirect($pageurl, $message, null, \core\output\notification::NOTIFY_SUCCESS);
    }
}

$assigneduseroptions = [0 => get_string('allassignedusers', 'faceattendance')];
$assigneduserssql = "SELECT DISTINCT u.id, u.firstname, u.lastname, u.email
                       FROM {faceattendance_capture_groups} cg
                       JOIN {user} u ON u.id = cg.assigneduserid
                      WHERE cg.faceattendanceid = :faceattendanceid
                        AND cg.status IN ('assigned', 'autoassigned')
                        AND cg.assigneduserid IS NOT NULL
                   ORDER BY u.lastname ASC, u.firstname ASC";
$assignedusers = $DB->get_records_sql($assigneduserssql, [
    'faceattendanceid' => $faceattendance->id,
    ]);
foreach ($assignedusers as $assigneduser) {
    $assigneduseroptions[(int)$assigneduser->id] = fullname($assigneduser) . ' (' . $assigneduser->email . ')';
}

$params = ['faceattendanceid' => $faceattendance->id];
$assignedfilter = '';
if ($assigneduserid > 0) {
    $assignedfilter = ' AND cg.assigneduserid = :assigneduserid';
    $params['assigneduserid'] = $assigneduserid;
}

$sql = "SELECT cg.*, cs.name AS capturesessionname
          FROM {faceattendance_capture_groups} cg
          JOIN {faceattendance_capture_sessions} cs ON cs.id = cg.capturesessionid
         WHERE cg.faceattendanceid = :faceattendanceid
               $assignedfilter
      ORDER BY cg.status ASC, cg.timemodified DESC";
$groups = $DB->get_records_sql($sql, $params);

echo $OUTPUT->header();
echo $OUTPUT->heading(get_string('reviewcaptures', 'faceattendance'));
echo html_writer::tag('p', get_string('capturesreviewintro', 'faceattendance'), ['class' => 'alert alert-info']);
echo html_writer::tag('p', get_string('captureautolearnnote', 'faceattendance'), ['class' => 'alert alert-secondary']);

echo html_writer::div(
    html_writer::link($returnurl, get_string('backtoactivity', 'faceattendance'), ['class' => 'btn btn-secondary mr-2']) .
    html_writer::link($captureurl, get_string('captureenteringfaces', 'faceattendance'), ['class' => 'btn btn-dark mr-2']),
    'mb-3'
);

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

if (!$groups) {
    echo $OUTPUT->notification(get_string('nocapturegroups', 'faceattendance'), 'info');
} else {
    $table = new html_table();
    $table->head = [
        get_string('id'),
        get_string('capturephotos', 'faceattendance'),
        get_string('capturesession', 'faceattendance'),
        get_string('status', 'faceattendance'),
        get_string('source', 'faceattendance'),
        get_string('firstseen', 'faceattendance'),
        get_string('lastseen', 'faceattendance'),
        get_string('detections', 'faceattendance'),
        get_string('assignedstudent', 'faceattendance'),
        get_string('actions'),
    ];
    $table->data = [];

    foreach ($groups as $group) {
        $thumbhtml = get_string('nothumbnail', 'faceattendance');
        $fs = get_file_storage();
        $files = $fs->get_area_files($context->id, 'mod_faceattendance', 'captureface', (int)$group->id, 'id ASC', false);
        if (!empty($files)) {
            $imgs = [];
            foreach ($files as $file) {
                $thumbnailurl = moodle_url::make_pluginfile_url(
                    $context->id,
                    'mod_faceattendance',
                    'captureface',
                    (int)$group->id,
                    '/',
                    $file->get_filename()
                );
                $imgs[] = html_writer::empty_tag('img', [
                    'src' => $thumbnailurl,
                    'alt' => get_string('capturephotos', 'faceattendance'),
                    'style' => 'max-width:120px; max-height:120px; object-fit:cover; border-radius:8px; border:1px solid #ddd; margin-right:6px;',
                ]);
            }
            $thumbhtml = implode('', $imgs);
        }

        $assignedstudent = '-';
        if (!empty($group->assigneduserid)) {
            $assignedrecord = $DB->get_record('user', ['id' => $group->assigneduserid], 'id, firstname, lastname, email', IGNORE_MISSING);
            if ($assignedrecord) {
                $assignedstudent = fullname($assignedrecord) . ' (' . s($assignedrecord->email) . ')';
            }
        }

        $actions = '';
        if ($group->status === 'pending') {
            $select = html_writer::select($options, 'userid', '', ['' => get_string('choosestudent', 'faceattendance')], ['class' => 'form-control d-inline-block', 'style' => 'width:260px']);
            $actions = html_writer::start_tag('form', ['method' => 'post', 'style' => 'display:inline-block; margin-right: 8px;']) .
                html_writer::empty_tag('input', ['type' => 'hidden', 'name' => 'id', 'value' => $cm->id]) .
                html_writer::empty_tag('input', ['type' => 'hidden', 'name' => 'sesskey', 'value' => sesskey()]) .
                html_writer::empty_tag('input', ['type' => 'hidden', 'name' => 'action', 'value' => 'assign']) .
                html_writer::empty_tag('input', ['type' => 'hidden', 'name' => 'groupid', 'value' => $group->id]) .
                $select . ' ' .
                html_writer::empty_tag('input', ['type' => 'submit', 'class' => 'btn btn-sm btn-primary', 'value' => get_string('assign', 'faceattendance')]) .
                html_writer::end_tag('form');
            $actions .= html_writer::start_tag('form', ['method' => 'post', 'style' => 'display:inline-block']) .
                html_writer::empty_tag('input', ['type' => 'hidden', 'name' => 'id', 'value' => $cm->id]) .
                html_writer::empty_tag('input', ['type' => 'hidden', 'name' => 'sesskey', 'value' => sesskey()]) .
                html_writer::empty_tag('input', ['type' => 'hidden', 'name' => 'action', 'value' => 'ignore']) .
                html_writer::empty_tag('input', ['type' => 'hidden', 'name' => 'groupid', 'value' => $group->id]) .
                html_writer::empty_tag('input', ['type' => 'submit', 'class' => 'btn btn-sm btn-outline-secondary', 'value' => get_string('ignore', 'faceattendance')]) .
                html_writer::end_tag('form');
        } else if (in_array($group->status, ['assigned', 'autoassigned'], true) && $group->assigneduserid) {
            $actions = $group->status === 'autoassigned' ? get_string('autoassigned', 'faceattendance') : get_string('resolved', 'faceattendance');
        }

        $table->data[] = [
            (int)$group->id,
            $thumbhtml,
            format_string($group->capturesessionname),
            s($group->status),
            s($group->source),
            userdate((int)$group->firstseen),
            userdate((int)$group->lastseen),
            (int)$group->detectioncount,
            $assignedstudent,
            $actions,
        ];
    }

    echo html_writer::table($table);
}

echo $OUTPUT->footer();
