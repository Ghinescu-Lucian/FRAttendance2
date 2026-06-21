<?php
// This file is part of Moodle - http://moodle.org/
//
// Moodle is free software: you can redistribute it and/or modify
// it under the terms of the GNU General Public License as published by
// the Free Software Foundation, either version 3 of the License, or
// (at your option) any later version.

/**
 * Version information for the Face Attendance activity module.
 *
 * @package     mod_faceattendance
 * @copyright   2026
 * @license     http://www.gnu.org/copyleft/gpl.html GNU GPL v3 or later
 */

defined('MOODLE_INTERNAL') || die();

$plugin->component = 'mod_faceattendance';
$plugin->version   = 2026051719;
$plugin->requires  = 2022112800; // Moodle 4.1+.
$plugin->maturity  = MATURITY_ALPHA;
$plugin->release   = '0.4.4-encrypted-embeddings';
