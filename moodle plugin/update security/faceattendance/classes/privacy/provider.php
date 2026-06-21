<?php
// This file is part of Moodle - http://moodle.org/

namespace mod_faceattendance\privacy;

use core_privacy\local\metadata\collection;

/**
 * Privacy metadata provider for mod_faceattendance.
 *
 * @package     mod_faceattendance
 * @copyright   2026
 * @license     http://www.gnu.org/copyleft/gpl.html GNU GPL v3 or later
 */
class provider implements \core_privacy\local\metadata\provider {

    /**
     * Describe personal data stored by this plugin.
     *
     * @param collection $collection
     * @return collection
     */
    public static function get_metadata(collection $collection): collection {
        $collection->add_database_table('faceattendance_records', [
            'userid' => 'privacy:metadata:faceattendance_records:userid',
            'identifier' => 'privacy:metadata:faceattendance_records:identifier',
            'confidence' => 'privacy:metadata:faceattendance_records:confidence',
            'status' => 'privacy:metadata:faceattendance_records:status',
            'source' => 'privacy:metadata:faceattendance_records:source',
            'rawpayload' => 'privacy:metadata:faceattendance_records:rawpayload',
            'timecreated' => 'privacy:metadata:faceattendance_records:timecreated',
            'timemodified' => 'privacy:metadata:faceattendance_records:timemodified',
        ], 'privacy:metadata:faceattendance_records');


        $collection->add_database_table('faceattendance_registrations', [
            'userid' => 'privacy:metadata:faceattendance_registrations:userid',
            'externalid' => 'privacy:metadata:faceattendance_registrations:externalid',
            'status' => 'privacy:metadata:faceattendance_registrations:status',
            'notes' => 'privacy:metadata:faceattendance_registrations:notes',
            'timecreated' => 'privacy:metadata:faceattendance_registrations:timecreated',
            'timemodified' => 'privacy:metadata:faceattendance_registrations:timemodified',
        ], 'privacy:metadata:faceattendance_registrations');


        $collection->add_database_table('faceattendance_course_embeddings', [
            'course' => 'privacy:metadata:faceattendance_course_embeddings:course',
            'userid' => 'privacy:metadata:faceattendance_embeddings:userid',
            'embeddingjson' => 'privacy:metadata:faceattendance_embeddings:embeddingjson',
            'modelname' => 'privacy:metadata:faceattendance_embeddings:modelname',
            'embeddingdim' => 'privacy:metadata:faceattendance_embeddings:embeddingdim',
            'samples' => 'privacy:metadata:faceattendance_embeddings:samples',
            'qualityscore' => 'privacy:metadata:faceattendance_embeddings:qualityscore',
            'status' => 'privacy:metadata:faceattendance_embeddings:status',
            'timecreated' => 'privacy:metadata:faceattendance_embeddings:timecreated',
            'timemodified' => 'privacy:metadata:faceattendance_embeddings:timemodified',
        ], 'privacy:metadata:faceattendance_course_embeddings');

        $collection->add_database_table('faceattendance_embeddings', [
            'userid' => 'privacy:metadata:faceattendance_embeddings:userid',
            'embeddingjson' => 'privacy:metadata:faceattendance_embeddings:embeddingjson',
            'modelname' => 'privacy:metadata:faceattendance_embeddings:modelname',
            'embeddingdim' => 'privacy:metadata:faceattendance_embeddings:embeddingdim',
            'samples' => 'privacy:metadata:faceattendance_embeddings:samples',
            'qualityscore' => 'privacy:metadata:faceattendance_embeddings:qualityscore',
            'status' => 'privacy:metadata:faceattendance_embeddings:status',
            'timecreated' => 'privacy:metadata:faceattendance_embeddings:timecreated',
            'timemodified' => 'privacy:metadata:faceattendance_embeddings:timemodified',
        ], 'privacy:metadata:faceattendance_embeddings');


        $collection->add_database_table('faceattendance_session_records', [
            'userid' => 'privacy:metadata:faceattendance_session_records:userid',
            'status' => 'privacy:metadata:faceattendance_session_records:status',
            'confidence' => 'privacy:metadata:faceattendance_session_records:confidence',
            'distance' => 'privacy:metadata:faceattendance_session_records:distance',
            'detectioncount' => 'privacy:metadata:faceattendance_session_records:detectioncount',
            'firstseen' => 'privacy:metadata:faceattendance_session_records:firstseen',
            'lastseen' => 'privacy:metadata:faceattendance_session_records:lastseen',
            'source' => 'privacy:metadata:faceattendance_session_records:source',
        ], 'privacy:metadata:faceattendance_session_records');

        $collection->add_database_table('faceattendance_detections', [
            'userid' => 'privacy:metadata:faceattendance_detections:userid',
            'decision' => 'privacy:metadata:faceattendance_detections:decision',
            'confidence' => 'privacy:metadata:faceattendance_detections:confidence',
            'distance' => 'privacy:metadata:faceattendance_detections:distance',
            'source' => 'privacy:metadata:faceattendance_detections:source',
            'rawpayload' => 'privacy:metadata:faceattendance_detections:rawpayload',
            'timecreated' => 'privacy:metadata:faceattendance_detections:timecreated',
        ], 'privacy:metadata:faceattendance_detections');

        $collection->add_database_table('faceattendance_unknowns', [
            'embeddingjson' => 'privacy:metadata:faceattendance_unknowns:embeddingjson',
            'candidatejson' => 'privacy:metadata:faceattendance_unknowns:candidatejson',
            'source' => 'privacy:metadata:faceattendance_unknowns:source',
            'status' => 'privacy:metadata:faceattendance_unknowns:status',
            'resolveduserid' => 'privacy:metadata:faceattendance_unknowns:resolveduserid',
            'resolvedby' => 'privacy:metadata:faceattendance_unknowns:resolvedby',
            'firstseen' => 'privacy:metadata:faceattendance_unknowns:firstseen',
            'lastseen' => 'privacy:metadata:faceattendance_unknowns:lastseen',
        ], 'privacy:metadata:faceattendance_unknowns');


        $collection->add_database_table('faceattendance_capture_sessions', [
            'course' => 'privacy:metadata:faceattendance_capture_sessions',
            'name' => 'privacy:metadata:faceattendance_capture_sessions',
            'createdby' => 'privacy:metadata:faceattendance_capture_sessions',
            'starttime' => 'privacy:metadata:faceattendance_capture_sessions',
            'endtime' => 'privacy:metadata:faceattendance_capture_sessions',
        ], 'privacy:metadata:faceattendance_capture_sessions');

        $collection->add_database_table('faceattendance_capture_groups', [
            'prototypeembeddingjson' => 'privacy:metadata:faceattendance_capture_groups:prototypeembeddingjson',
            'sampleembeddingjson' => 'privacy:metadata:faceattendance_capture_groups:sampleembeddingjson',
            'status' => 'privacy:metadata:faceattendance_capture_groups:status',
            'assigneduserid' => 'privacy:metadata:faceattendance_capture_groups:assigneduserid',
            'assignedby' => 'privacy:metadata:faceattendance_capture_groups:assignedby',
            'firstseen' => 'privacy:metadata:faceattendance_capture_groups:firstseen',
            'lastseen' => 'privacy:metadata:faceattendance_capture_groups:lastseen',
        ], 'privacy:metadata:faceattendance_capture_groups');

        return $collection;
    }
}
