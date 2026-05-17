<?php
// This file is part of Moodle - http://moodle.org/

/**
 * Activity creation/editing form for mod_faceattendance.
 *
 * @package     mod_faceattendance
 * @copyright   2026
 * @license     http://www.gnu.org/copyleft/gpl.html GNU GPL v3 or later
 */

defined('MOODLE_INTERNAL') || die();

require_once($CFG->dirroot . '/course/moodleform_mod.php');

/**
 * Face Attendance module form.
 */
class mod_faceattendance_mod_form extends moodleform_mod {

    /**
     * Define the form.
     */
    public function definition() {
        $mform = $this->_form;

        $mform->addElement('header', 'general', get_string('general', 'form'));

        $mform->addElement('text', 'name', get_string('faceattendancename', 'faceattendance'), ['size' => '64']);
        $mform->setType('name', PARAM_TEXT);
        $mform->addRule('name', null, 'required', null, 'client');

        $this->standard_intro_elements();

        $mform->addElement('passwordunmask', 'apisecret', get_string('apisecret', 'faceattendance'));
        $mform->setType('apisecret', PARAM_RAW_TRIMMED);
        $mform->addRule('apisecret', null, 'required', null, 'client');
        $mform->addHelpButton('apisecret', 'apisecret', 'faceattendance');

        $mform->addElement('text', 'confidence', get_string('confidence', 'faceattendance'), ['size' => '6']);
        $mform->setDefault('confidence', '0.85');
        $mform->setType('confidence', PARAM_FLOAT);
        $mform->addHelpButton('confidence', 'confidence', 'faceattendance');


        $profiles = [
            'fast_short' => get_string('modelprofile_fast_short', 'faceattendance'),
            'many_faces_unknown' => get_string('modelprofile_many_faces_unknown', 'faceattendance'),
            'fast_clean' => get_string('modelprofile_fast_clean', 'faceattendance'),
            'high_recall_many_faces' => get_string('modelprofile_high_recall_many_faces', 'faceattendance'),
            'multi_attendance_zoom' => get_string('modelprofile_multi_attendance_zoom', 'faceattendance'),
            'entrance_mode' => get_string('modelprofile_entrance_mode', 'faceattendance'),
        ];
        $mform->addElement('select', 'modelprofile', get_string('modelprofile', 'faceattendance'), $profiles);
        $mform->setDefault('modelprofile', 'fast_short');
        $mform->addHelpButton('modelprofile', 'modelprofile', 'faceattendance');

        $this->standard_coursemodule_elements();
        $this->add_action_buttons();
    }
}
