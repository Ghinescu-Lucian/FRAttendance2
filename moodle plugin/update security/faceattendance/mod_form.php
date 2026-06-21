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

        $mform->addElement('textarea', 'stationpublickey', get_string('stationpublickey', 'faceattendance'), ['rows' => 3, 'cols' => 80]);
        $mform->setType('stationpublickey', PARAM_RAW_TRIMMED);
        $mform->addHelpButton('stationpublickey', 'stationpublickey', 'faceattendance');

        $mform->addElement('text', 'confidence', get_string('confidence', 'faceattendance'), ['size' => '6']);
        $mform->setDefault('confidence', '0.85');
        $mform->setType('confidence', PARAM_FLOAT);
        $mform->addHelpButton('confidence', 'confidence', 'faceattendance');

        $this->standard_coursemodule_elements();
        $this->add_action_buttons();
    }
}
