<?php
// This file is part of Moodle - http://moodle.org/

/**
 * Upgrade steps for mod_faceattendance.
 *
 * @package     mod_faceattendance
 * @copyright   2026
 * @license     http://www.gnu.org/copyleft/gpl.html GNU GPL v3 or later
 */

defined('MOODLE_INTERNAL') || die();

/**
 * Execute mod_faceattendance upgrade steps.
 *
 * @param int $oldversion
 * @return bool
 */
function xmldb_faceattendance_upgrade($oldversion) {
    global $DB;

    $dbman = $DB->get_manager();

    if ($oldversion < 2026051701) {
        $table = new xmldb_table('faceattendance_registrations');

        $table->add_field('id', XMLDB_TYPE_INTEGER, '10', null, XMLDB_NOTNULL, XMLDB_SEQUENCE, null);
        $table->add_field('faceattendanceid', XMLDB_TYPE_INTEGER, '10', null, XMLDB_NOTNULL, null, '0');
        $table->add_field('course', XMLDB_TYPE_INTEGER, '10', null, XMLDB_NOTNULL, null, '0');
        $table->add_field('userid', XMLDB_TYPE_INTEGER, '10', null, XMLDB_NOTNULL, null, '0');
        $table->add_field('externalid', XMLDB_TYPE_CHAR, '255', null, null, null, null);
        $table->add_field('status', XMLDB_TYPE_CHAR, '20', null, XMLDB_NOTNULL, null, 'registered');
        $table->add_field('notes', XMLDB_TYPE_TEXT, null, null, null, null, null);
        $table->add_field('timecreated', XMLDB_TYPE_INTEGER, '10', null, XMLDB_NOTNULL, null, '0');
        $table->add_field('timemodified', XMLDB_TYPE_INTEGER, '10', null, XMLDB_NOTNULL, null, '0');

        $table->add_key('primary', XMLDB_KEY_PRIMARY, ['id']);
        $table->add_key('faceattendanceid', XMLDB_KEY_FOREIGN, ['faceattendanceid'], 'faceattendance', ['id']);
        $table->add_key('userid', XMLDB_KEY_FOREIGN, ['userid'], 'user', ['id']);
        $table->add_key('face_user_uq', XMLDB_KEY_UNIQUE, ['faceattendanceid', 'userid']);

        $table->add_index('course', XMLDB_INDEX_NOTUNIQUE, ['course']);
        $table->add_index('externalid', XMLDB_INDEX_NOTUNIQUE, ['externalid']);

        if (!$dbman->table_exists($table)) {
            $dbman->create_table($table);
        }

        upgrade_mod_savepoint(true, 2026051701, 'faceattendance');
    }

    if ($oldversion < 2026051702) {
        $table = new xmldb_table('faceattendance_embeddings');

        $table->add_field('id', XMLDB_TYPE_INTEGER, '10', null, XMLDB_NOTNULL, XMLDB_SEQUENCE, null);
        $table->add_field('faceattendanceid', XMLDB_TYPE_INTEGER, '10', null, XMLDB_NOTNULL, null, '0');
        $table->add_field('course', XMLDB_TYPE_INTEGER, '10', null, XMLDB_NOTNULL, null, '0');
        $table->add_field('userid', XMLDB_TYPE_INTEGER, '10', null, XMLDB_NOTNULL, null, '0');
        $table->add_field('embeddingjson', XMLDB_TYPE_TEXT, null, null, XMLDB_NOTNULL, null, null);
        $table->add_field('modelname', XMLDB_TYPE_CHAR, '100', null, XMLDB_NOTNULL, null, 'opencv-sface-2021dec');
        $table->add_field('embeddingdim', XMLDB_TYPE_INTEGER, '10', null, XMLDB_NOTNULL, null, '128');
        $table->add_field('samples', XMLDB_TYPE_INTEGER, '10', null, XMLDB_NOTNULL, null, '0');
        $table->add_field('qualityscore', XMLDB_TYPE_NUMBER, '10, 5', null, XMLDB_NOTNULL, null, '0');
        $table->add_field('status', XMLDB_TYPE_CHAR, '20', null, XMLDB_NOTNULL, null, 'registered');
        $table->add_field('timecreated', XMLDB_TYPE_INTEGER, '10', null, XMLDB_NOTNULL, null, '0');
        $table->add_field('timemodified', XMLDB_TYPE_INTEGER, '10', null, XMLDB_NOTNULL, null, '0');

        $table->add_key('primary', XMLDB_KEY_PRIMARY, ['id']);
        $table->add_key('faceattendanceid', XMLDB_KEY_FOREIGN, ['faceattendanceid'], 'faceattendance', ['id']);
        $table->add_key('userid', XMLDB_KEY_FOREIGN, ['userid'], 'user', ['id']);
        $table->add_key('face_embed_user_uq', XMLDB_KEY_UNIQUE, ['faceattendanceid', 'userid']);

        $table->add_index('course', XMLDB_INDEX_NOTUNIQUE, ['course']);
        $table->add_index('status', XMLDB_INDEX_NOTUNIQUE, ['status']);

        if (!$dbman->table_exists($table)) {
            $dbman->create_table($table);
        }

        upgrade_mod_savepoint(true, 2026051702, 'faceattendance');
    }


    if ($oldversion < 2026051706) {
        $table = new xmldb_table('faceattendance_sessions');
        $table->add_field('id', XMLDB_TYPE_INTEGER, '10', null, XMLDB_NOTNULL, XMLDB_SEQUENCE, null);
        $table->add_field('faceattendanceid', XMLDB_TYPE_INTEGER, '10', null, XMLDB_NOTNULL, null, '0');
        $table->add_field('course', XMLDB_TYPE_INTEGER, '10', null, XMLDB_NOTNULL, null, '0');
        $table->add_field('name', XMLDB_TYPE_CHAR, '255', null, XMLDB_NOTNULL, null, '');
        $table->add_field('starttime', XMLDB_TYPE_INTEGER, '10', null, XMLDB_NOTNULL, null, '0');
        $table->add_field('endtime', XMLDB_TYPE_INTEGER, '10', null, XMLDB_NOTNULL, null, '0');
        $table->add_field('latethreshold', XMLDB_TYPE_INTEGER, '10', null, XMLDB_NOTNULL, null, '600');
        $table->add_field('mindetections', XMLDB_TYPE_INTEGER, '10', null, XMLDB_NOTNULL, null, '3');
        $table->add_field('status', XMLDB_TYPE_CHAR, '20', null, XMLDB_NOTNULL, null, 'scheduled');
        $table->add_field('createdby', XMLDB_TYPE_INTEGER, '10', null, XMLDB_NOTNULL, null, '0');
        $table->add_field('timecreated', XMLDB_TYPE_INTEGER, '10', null, XMLDB_NOTNULL, null, '0');
        $table->add_field('timemodified', XMLDB_TYPE_INTEGER, '10', null, XMLDB_NOTNULL, null, '0');
        $table->add_key('primary', XMLDB_KEY_PRIMARY, ['id']);
        $table->add_key('faceattendanceid', XMLDB_KEY_FOREIGN, ['faceattendanceid'], 'faceattendance', ['id']);
        $table->add_key('createdby', XMLDB_KEY_FOREIGN, ['createdby'], 'user', ['id']);
        $table->add_index('course', XMLDB_INDEX_NOTUNIQUE, ['course']);
        $table->add_index('starttime', XMLDB_INDEX_NOTUNIQUE, ['starttime']);
        $table->add_index('status', XMLDB_INDEX_NOTUNIQUE, ['status']);
        if (!$dbman->table_exists($table)) {
            $dbman->create_table($table);
        }

        $table = new xmldb_table('faceattendance_session_records');
        $table->add_field('id', XMLDB_TYPE_INTEGER, '10', null, XMLDB_NOTNULL, XMLDB_SEQUENCE, null);
        $table->add_field('sessionid', XMLDB_TYPE_INTEGER, '10', null, XMLDB_NOTNULL, null, '0');
        $table->add_field('faceattendanceid', XMLDB_TYPE_INTEGER, '10', null, XMLDB_NOTNULL, null, '0');
        $table->add_field('course', XMLDB_TYPE_INTEGER, '10', null, XMLDB_NOTNULL, null, '0');
        $table->add_field('userid', XMLDB_TYPE_INTEGER, '10', null, XMLDB_NOTNULL, null, '0');
        $table->add_field('status', XMLDB_TYPE_CHAR, '20', null, XMLDB_NOTNULL, null, 'present');
        $table->add_field('confidence', XMLDB_TYPE_NUMBER, '10, 5', null, XMLDB_NOTNULL, null, '0');
        $table->add_field('distance', XMLDB_TYPE_NUMBER, '10, 5', null, XMLDB_NOTNULL, null, '0');
        $table->add_field('detectioncount', XMLDB_TYPE_INTEGER, '10', null, XMLDB_NOTNULL, null, '0');
        $table->add_field('firstseen', XMLDB_TYPE_INTEGER, '10', null, XMLDB_NOTNULL, null, '0');
        $table->add_field('lastseen', XMLDB_TYPE_INTEGER, '10', null, XMLDB_NOTNULL, null, '0');
        $table->add_field('source', XMLDB_TYPE_CHAR, '255', null, null, null, null);
        $table->add_field('timecreated', XMLDB_TYPE_INTEGER, '10', null, XMLDB_NOTNULL, null, '0');
        $table->add_field('timemodified', XMLDB_TYPE_INTEGER, '10', null, XMLDB_NOTNULL, null, '0');
        $table->add_key('primary', XMLDB_KEY_PRIMARY, ['id']);
        $table->add_key('sessionid', XMLDB_KEY_FOREIGN, ['sessionid'], 'faceattendance_sessions', ['id']);
        $table->add_key('faceattendanceid', XMLDB_KEY_FOREIGN, ['faceattendanceid'], 'faceattendance', ['id']);
        $table->add_key('userid', XMLDB_KEY_FOREIGN, ['userid'], 'user', ['id']);
        $table->add_key('session_user_uq', XMLDB_KEY_UNIQUE, ['sessionid', 'userid']);
        $table->add_index('course', XMLDB_INDEX_NOTUNIQUE, ['course']);
        $table->add_index('status', XMLDB_INDEX_NOTUNIQUE, ['status']);
        if (!$dbman->table_exists($table)) {
            $dbman->create_table($table);
        }

        $table = new xmldb_table('faceattendance_detections');
        $table->add_field('id', XMLDB_TYPE_INTEGER, '10', null, XMLDB_NOTNULL, XMLDB_SEQUENCE, null);
        $table->add_field('sessionid', XMLDB_TYPE_INTEGER, '10', null, XMLDB_NOTNULL, null, '0');
        $table->add_field('faceattendanceid', XMLDB_TYPE_INTEGER, '10', null, XMLDB_NOTNULL, null, '0');
        $table->add_field('course', XMLDB_TYPE_INTEGER, '10', null, XMLDB_NOTNULL, null, '0');
        $table->add_field('userid', XMLDB_TYPE_INTEGER, '10', null, null, null, null);
        $table->add_field('decision', XMLDB_TYPE_CHAR, '20', null, XMLDB_NOTNULL, null, 'matched');
        $table->add_field('confidence', XMLDB_TYPE_NUMBER, '10, 5', null, XMLDB_NOTNULL, null, '0');
        $table->add_field('distance', XMLDB_TYPE_NUMBER, '10, 5', null, XMLDB_NOTNULL, null, '0');
        $table->add_field('source', XMLDB_TYPE_CHAR, '255', null, null, null, null);
        $table->add_field('rawpayload', XMLDB_TYPE_TEXT, null, null, null, null, null);
        $table->add_field('timecreated', XMLDB_TYPE_INTEGER, '10', null, XMLDB_NOTNULL, null, '0');
        $table->add_key('primary', XMLDB_KEY_PRIMARY, ['id']);
        $table->add_key('sessionid', XMLDB_KEY_FOREIGN, ['sessionid'], 'faceattendance_sessions', ['id']);
        $table->add_key('faceattendanceid', XMLDB_KEY_FOREIGN, ['faceattendanceid'], 'faceattendance', ['id']);
        $table->add_index('course', XMLDB_INDEX_NOTUNIQUE, ['course']);
        $table->add_index('decision', XMLDB_INDEX_NOTUNIQUE, ['decision']);
        if (!$dbman->table_exists($table)) {
            $dbman->create_table($table);
        }

        $table = new xmldb_table('faceattendance_unknowns');
        $table->add_field('id', XMLDB_TYPE_INTEGER, '10', null, XMLDB_NOTNULL, XMLDB_SEQUENCE, null);
        $table->add_field('sessionid', XMLDB_TYPE_INTEGER, '10', null, XMLDB_NOTNULL, null, '0');
        $table->add_field('faceattendanceid', XMLDB_TYPE_INTEGER, '10', null, XMLDB_NOTNULL, null, '0');
        $table->add_field('course', XMLDB_TYPE_INTEGER, '10', null, XMLDB_NOTNULL, null, '0');
        $table->add_field('embeddingjson', XMLDB_TYPE_TEXT, null, null, XMLDB_NOTNULL, null, null);
        $table->add_field('candidatejson', XMLDB_TYPE_TEXT, null, null, null, null, null);
        $table->add_field('source', XMLDB_TYPE_CHAR, '255', null, null, null, null);
        $table->add_field('status', XMLDB_TYPE_CHAR, '20', null, XMLDB_NOTNULL, null, 'unknown');
        $table->add_field('resolveduserid', XMLDB_TYPE_INTEGER, '10', null, null, null, null);
        $table->add_field('resolvedby', XMLDB_TYPE_INTEGER, '10', null, null, null, null);
        $table->add_field('firstseen', XMLDB_TYPE_INTEGER, '10', null, XMLDB_NOTNULL, null, '0');
        $table->add_field('lastseen', XMLDB_TYPE_INTEGER, '10', null, XMLDB_NOTNULL, null, '0');
        $table->add_field('detectioncount', XMLDB_TYPE_INTEGER, '10', null, XMLDB_NOTNULL, null, '1');
        $table->add_field('timecreated', XMLDB_TYPE_INTEGER, '10', null, XMLDB_NOTNULL, null, '0');
        $table->add_field('timemodified', XMLDB_TYPE_INTEGER, '10', null, XMLDB_NOTNULL, null, '0');
        $table->add_key('primary', XMLDB_KEY_PRIMARY, ['id']);
        $table->add_key('sessionid', XMLDB_KEY_FOREIGN, ['sessionid'], 'faceattendance_sessions', ['id']);
        $table->add_key('faceattendanceid', XMLDB_KEY_FOREIGN, ['faceattendanceid'], 'faceattendance', ['id']);
        $table->add_index('course', XMLDB_INDEX_NOTUNIQUE, ['course']);
        $table->add_index('status', XMLDB_INDEX_NOTUNIQUE, ['status']);
        if (!$dbman->table_exists($table)) {
            $dbman->create_table($table);
        }

        upgrade_mod_savepoint(true, 2026051706, 'faceattendance');
    }


    if ($oldversion < 2026051707) {
        // No database schema change. This version updates the browser camera startup code
        // and adds Moodle activity icons.
        upgrade_mod_savepoint(true, 2026051707, 'faceattendance');
    }


    if ($oldversion < 2026051710) {
        // No database schema change. Unknown-face thumbnails are stored in Moodle's file API
        // under filearea 'unknownface' and are deleted after teacher resolution.
        upgrade_mod_savepoint(true, 2026051710, 'faceattendance');
    }


    if ($oldversion < 2026051711) {
        $table = new xmldb_table('faceattendance');
        $field = new xmldb_field('modelprofile', XMLDB_TYPE_CHAR, '60', null, XMLDB_NOTNULL, null, 'fast_short', 'confidence');

        if (!$dbman->field_exists($table, $field)) {
            $dbman->add_field($table, $field);
        }

        upgrade_mod_savepoint(true, 2026051711, 'faceattendance');
    }


    if ($oldversion < 2026051712) {
        // No database schema change. This version adds browser unknown-face thumbnails
        // and de-duplicates repeated unknown detections from the same physical person.
        upgrade_mod_savepoint(true, 2026051712, 'faceattendance');
    }


    if ($oldversion < 2026051713) {
        // No database schema change. This version appends teacher-resolved unknown descriptors
        // to the selected student's embeddings, adds assigned-student filtering, and keeps
        // session form values after validation errors.
        upgrade_mod_savepoint(true, 2026051713, 'faceattendance');
    }


    if ($oldversion < 2026051714) {
        $table = new xmldb_table('faceattendance_course_embeddings');
        $table->add_field('id', XMLDB_TYPE_INTEGER, '10', null, XMLDB_NOTNULL, XMLDB_SEQUENCE, null);
        $table->add_field('course', XMLDB_TYPE_INTEGER, '10', null, XMLDB_NOTNULL, null, '0');
        $table->add_field('userid', XMLDB_TYPE_INTEGER, '10', null, XMLDB_NOTNULL, null, '0');
        $table->add_field('embeddingjson', XMLDB_TYPE_TEXT, null, null, XMLDB_NOTNULL, null, null);
        $table->add_field('modelname', XMLDB_TYPE_CHAR, '100', null, XMLDB_NOTNULL, null, 'opencv-sface-2021dec');
        $table->add_field('embeddingdim', XMLDB_TYPE_INTEGER, '10', null, XMLDB_NOTNULL, null, '128');
        $table->add_field('samples', XMLDB_TYPE_INTEGER, '10', null, XMLDB_NOTNULL, null, '0');
        $table->add_field('qualityscore', XMLDB_TYPE_NUMBER, '10, 5', null, XMLDB_NOTNULL, null, '0');
        $table->add_field('status', XMLDB_TYPE_CHAR, '20', null, XMLDB_NOTNULL, null, 'registered');
        $table->add_field('timecreated', XMLDB_TYPE_INTEGER, '10', null, XMLDB_NOTNULL, null, '0');
        $table->add_field('timemodified', XMLDB_TYPE_INTEGER, '10', null, XMLDB_NOTNULL, null, '0');
        $table->add_key('primary', XMLDB_KEY_PRIMARY, ['id']);
        $table->add_key('userid', XMLDB_KEY_FOREIGN, ['userid'], 'user', ['id']);
        $table->add_key('course_user_uq', XMLDB_KEY_UNIQUE, ['course', 'userid']);
        $table->add_index('course', XMLDB_INDEX_NOTUNIQUE, ['course']);
        $table->add_index('status', XMLDB_INDEX_NOTUNIQUE, ['status']);
        if (!$dbman->table_exists($table)) {
            $dbman->create_table($table);
        }

        // Migrate the most recently modified legacy activity-level embedding into
        // the new course-level store. Existing data is not deleted.
        if ($DB->get_manager()->table_exists(new xmldb_table('faceattendance_embeddings'))) {
            $legacy = $DB->get_records_sql("SELECT *
                                             FROM {faceattendance_embeddings}
                                            WHERE course <> 0
                                         ORDER BY course ASC, userid ASC, timemodified ASC, id ASC");
            foreach ($legacy as $old) {
                $existing = $DB->get_record('faceattendance_course_embeddings', [
                    'course' => $old->course,
                    'userid' => $old->userid,
                ]);
                $record = (object)[
                    'course' => (int)$old->course,
                    'userid' => (int)$old->userid,
                    'embeddingjson' => $old->embeddingjson,
                    'modelname' => $old->modelname,
                    'embeddingdim' => (int)$old->embeddingdim,
                    'samples' => (int)$old->samples,
                    'qualityscore' => (float)$old->qualityscore,
                    'status' => $old->status,
                    'timecreated' => (int)$old->timecreated,
                    'timemodified' => (int)$old->timemodified,
                ];
                if ($existing) {
                    if ((int)$old->timemodified >= (int)$existing->timemodified) {
                        $record->id = $existing->id;
                        $record->timecreated = $existing->timecreated;
                        $DB->update_record('faceattendance_course_embeddings', $record);
                    }
                } else {
                    $DB->insert_record('faceattendance_course_embeddings', $record);
                }
            }
        }

        upgrade_mod_savepoint(true, 2026051714, 'faceattendance');
    }


    if ($oldversion < 2026051715) {
        $table = new xmldb_table('faceattendance_capture_sessions');
        $table->add_field('id', XMLDB_TYPE_INTEGER, '10', null, XMLDB_NOTNULL, XMLDB_SEQUENCE, null);
        $table->add_field('faceattendanceid', XMLDB_TYPE_INTEGER, '10', null, XMLDB_NOTNULL, null, '0');
        $table->add_field('course', XMLDB_TYPE_INTEGER, '10', null, XMLDB_NOTNULL, null, '0');
        $table->add_field('name', XMLDB_TYPE_CHAR, '255', null, XMLDB_NOTNULL, null, '');
        $table->add_field('source', XMLDB_TYPE_CHAR, '255', null, null, null, null);
        $table->add_field('status', XMLDB_TYPE_CHAR, '20', null, XMLDB_NOTNULL, null, 'open');
        $table->add_field('createdby', XMLDB_TYPE_INTEGER, '10', null, XMLDB_NOTNULL, null, '0');
        $table->add_field('starttime', XMLDB_TYPE_INTEGER, '10', null, XMLDB_NOTNULL, null, '0');
        $table->add_field('endtime', XMLDB_TYPE_INTEGER, '10', null, XMLDB_NOTNULL, null, '0');
        $table->add_field('timecreated', XMLDB_TYPE_INTEGER, '10', null, XMLDB_NOTNULL, null, '0');
        $table->add_field('timemodified', XMLDB_TYPE_INTEGER, '10', null, XMLDB_NOTNULL, null, '0');
        $table->add_key('primary', XMLDB_KEY_PRIMARY, ['id']);
        $table->add_key('faceattendanceid', XMLDB_KEY_FOREIGN, ['faceattendanceid'], 'faceattendance', ['id']);
        $table->add_key('createdby', XMLDB_KEY_FOREIGN, ['createdby'], 'user', ['id']);
        $table->add_index('course', XMLDB_INDEX_NOTUNIQUE, ['course']);
        $table->add_index('status', XMLDB_INDEX_NOTUNIQUE, ['status']);
        $table->add_index('starttime', XMLDB_INDEX_NOTUNIQUE, ['starttime']);
        if (!$dbman->table_exists($table)) {
            $dbman->create_table($table);
        }

        $table = new xmldb_table('faceattendance_capture_groups');
        $table->add_field('id', XMLDB_TYPE_INTEGER, '10', null, XMLDB_NOTNULL, XMLDB_SEQUENCE, null);
        $table->add_field('capturesessionid', XMLDB_TYPE_INTEGER, '10', null, XMLDB_NOTNULL, null, '0');
        $table->add_field('faceattendanceid', XMLDB_TYPE_INTEGER, '10', null, XMLDB_NOTNULL, null, '0');
        $table->add_field('course', XMLDB_TYPE_INTEGER, '10', null, XMLDB_NOTNULL, null, '0');
        $table->add_field('prototypeembeddingjson', XMLDB_TYPE_TEXT, null, null, XMLDB_NOTNULL, null, null);
        $table->add_field('sampleembeddingjson', XMLDB_TYPE_TEXT, null, null, null, null, null);
        $table->add_field('candidatejson', XMLDB_TYPE_TEXT, null, null, null, null, null);
        $table->add_field('source', XMLDB_TYPE_CHAR, '255', null, null, null, null);
        $table->add_field('status', XMLDB_TYPE_CHAR, '20', null, XMLDB_NOTNULL, null, 'pending');
        $table->add_field('assigneduserid', XMLDB_TYPE_INTEGER, '10', null, null, null, null);
        $table->add_field('assignedby', XMLDB_TYPE_INTEGER, '10', null, null, null, null);
        $table->add_field('firstseen', XMLDB_TYPE_INTEGER, '10', null, XMLDB_NOTNULL, null, '0');
        $table->add_field('lastseen', XMLDB_TYPE_INTEGER, '10', null, XMLDB_NOTNULL, null, '0');
        $table->add_field('detectioncount', XMLDB_TYPE_INTEGER, '10', null, XMLDB_NOTNULL, null, '1');
        $table->add_field('thumbnailcount', XMLDB_TYPE_INTEGER, '10', null, XMLDB_NOTNULL, null, '0');
        $table->add_field('qualityscore', XMLDB_TYPE_NUMBER, '10, 5', null, XMLDB_NOTNULL, null, '0');
        $table->add_field('timecreated', XMLDB_TYPE_INTEGER, '10', null, XMLDB_NOTNULL, null, '0');
        $table->add_field('timemodified', XMLDB_TYPE_INTEGER, '10', null, XMLDB_NOTNULL, null, '0');
        $table->add_key('primary', XMLDB_KEY_PRIMARY, ['id']);
        $table->add_key('capturesessionid', XMLDB_KEY_FOREIGN, ['capturesessionid'], 'faceattendance_capture_sessions', ['id']);
        $table->add_key('faceattendanceid', XMLDB_KEY_FOREIGN, ['faceattendanceid'], 'faceattendance', ['id']);
        $table->add_index('course', XMLDB_INDEX_NOTUNIQUE, ['course']);
        $table->add_index('status', XMLDB_INDEX_NOTUNIQUE, ['status']);
        $table->add_index('lastseen', XMLDB_INDEX_NOTUNIQUE, ['lastseen']);
        if (!$dbman->table_exists($table)) {
            $dbman->create_table($table);
        }

        upgrade_mod_savepoint(true, 2026051715, 'faceattendance');
    }


    if ($oldversion < 2026051716) {
        // No database structure changes. This version adds auto-learning behavior for capture-intake mode.
        upgrade_mod_savepoint(true, 2026051716, 'faceattendance');
    }


    if ($oldversion < 2026051718) {
        $table = new xmldb_table('faceattendance_station_nonces');
        $table->add_field('id', XMLDB_TYPE_INTEGER, '10', null, XMLDB_NOTNULL, XMLDB_SEQUENCE, null);
        $table->add_field('faceattendanceid', XMLDB_TYPE_INTEGER, '10', null, XMLDB_NOTNULL, null, '0');
        $table->add_field('stationid', XMLDB_TYPE_CHAR, '100', null, XMLDB_NOTNULL, null, '');
        $table->add_field('noncehash', XMLDB_TYPE_CHAR, '64', null, XMLDB_NOTNULL, null, '');
        $table->add_field('timecreated', XMLDB_TYPE_INTEGER, '10', null, XMLDB_NOTNULL, null, '0');
        $table->add_key('primary', XMLDB_KEY_PRIMARY, ['id']);
        $table->add_key('faceattendanceid', XMLDB_KEY_FOREIGN, ['faceattendanceid'], 'faceattendance', ['id']);
        $table->add_key('station_nonce_uq', XMLDB_KEY_UNIQUE, ['faceattendanceid', 'stationid', 'noncehash']);
        $table->add_index('timecreated', XMLDB_INDEX_NOTUNIQUE, ['timecreated']);

        if (!$dbman->table_exists($table)) {
            $dbman->create_table($table);
        }

        upgrade_mod_savepoint(true, 2026051718, 'faceattendance');
    }


    if ($oldversion < 2026051719) {
        $table = new xmldb_table('faceattendance');
        $field = new xmldb_field('stationpublickey', XMLDB_TYPE_TEXT, null, null, null, null, null, 'apisecret');

        if (!$dbman->field_exists($table, $field)) {
            $dbman->add_field($table, $field);
        }

        upgrade_mod_savepoint(true, 2026051719, 'faceattendance');
    }

    return true;
}
