#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# This file has been created (totally or partially) with the assistance of
# artificial intelligence tools. All content has been generated under the
# direct supervision of a named individual and the AI.Backbone Orchestrator
# Compliance framework.

"""
Unit tests for SopraGP4U Clock In/Out automation
===============================================

These tests validate the core functionality without actually interacting
with the website or performing real clock-in/out actions.

Run with: python -m pytest src/test_sopra_clockin.py -v
"""

import sys
import unittest
from datetime import datetime, timedelta
from unittest.mock import Mock, patch, MagicMock
from pathlib import Path

from selenium.webdriver.common.by import By

# Add src directory to path
sys.path.insert(0, str(Path(__file__).parent))

from sopra_clockin import SopraClockInAutomation, get_selector_tuple
from config.config import (
    CLOCK_IN_THRESHOLD, CLOCK_OUT_THRESHOLD,
    CLOCK_IN_SELECTOR, CLOCK_OUT_SELECTOR
)


class TestSelectorConversion(unittest.TestCase):
    """Test selector conversion functionality."""
    
    def test_get_selector_tuple_id(self):
        """Test ID selector conversion."""
        selector = {'method': 'id', 'value': 'test-id'}
        result = get_selector_tuple(selector)
        self.assertEqual(result, (By.ID, 'test-id'))
    
    def test_get_selector_tuple_xpath(self):
        """Test XPath selector conversion."""
        selector = {'method': 'xpath', 'value': '//button[@id="test"]'}
        result = get_selector_tuple(selector)
        self.assertEqual(result, (By.XPATH, '//button[@id="test"]'))
    
    def test_get_selector_tuple_css(self):
        """Test CSS selector conversion."""
        selector = {'method': 'css', 'value': '.btn-primary'}
        result = get_selector_tuple(selector)
        self.assertEqual(result, (By.CSS_SELECTOR, '.btn-primary'))
    
    def test_get_selector_tuple_class(self):
        """Test class selector conversion (treated as CSS)."""
        selector = {'method': 'class', 'value': 'btn-primary'}
        result = get_selector_tuple(selector)
        self.assertEqual(result, (By.CSS_SELECTOR, 'btn-primary'))
    
    def test_get_selector_tuple_name(self):
        """Test name selector conversion."""
        selector = {'method': 'name', 'value': 'submit'}
        result = get_selector_tuple(selector)
        self.assertEqual(result, (By.NAME, 'submit'))
    
    def test_get_selector_tuple_link_text(self):
        """Test link text selector conversion."""
        selector = {'method': 'link_text', 'value': 'Click here'}
        result = get_selector_tuple(selector)
        self.assertEqual(result, (By.LINK_TEXT, 'Click here'))
    
    def test_get_selector_tuple_partial_link_text(self):
        """Test partial link text selector conversion."""
        selector = {'method': 'partial_link_text', 'value': 'Click'}
        result = get_selector_tuple(selector)
        self.assertEqual(result, (By.PARTIAL_LINK_TEXT, 'Click'))
    
    def test_get_selector_tuple_default(self):
        """Test default selector conversion (invalid method defaults to ID)."""
        selector = {'method': 'invalid', 'value': 'test'}
        result = get_selector_tuple(selector)
        self.assertEqual(result, ('id', 'test'))


class TestActionDetermination(unittest.TestCase):
    """Test action determination logic."""
    
    def test_clock_in_before_threshold(self):
        """Test clock-in determination inside the preferred morning window."""
        automation = SopraClockInAutomation()
        automation.current_hour = 9
        automation.current_minute = 0
        self.assertEqual(automation._determine_action(), 'CLOCK_IN')
    
    def test_clock_out_after_threshold(self):
        """Test clock-out after the minimum nine-hour duration."""
        automation = SopraClockInAutomation()
        automation.current_weekday = 0
        automation.current_hour = 18
        automation.current_minute = 0
        automation.state['clock_in_at'] = (datetime.now() - timedelta(hours=10)).isoformat()
        self.assertEqual(automation._determine_action(), 'CLOCK_OUT')

    def test_friday_clock_out_starts_at_15(self):
        automation = SopraClockInAutomation()
        automation.current_weekday = 4
        automation.current_hour = 15
        automation.current_minute = 0
        automation.state['clock_in_at'] = (datetime.now() - timedelta(hours=9)).isoformat()
        self.assertEqual(automation._determine_action(), 'CLOCK_OUT')

    def test_friday_clock_out_after_seven_hours(self):
        automation = SopraClockInAutomation()
        automation.current_weekday = 4
        automation.current_hour = 15
        automation.current_minute = 0
        automation.state['clock_in_at'] = (datetime.now() - timedelta(hours=7)).isoformat()
        self.assertEqual(automation._determine_action(), 'CLOCK_OUT')

    def test_friday_clock_out_waits_for_seven_hours(self):
        automation = SopraClockInAutomation()
        automation.current_weekday = 4
        automation.current_hour = 15
        automation.current_minute = 0
        automation.state['clock_in_at'] = (datetime.now() - timedelta(hours=6)).isoformat()
        self.assertEqual(automation._determine_action(), 'NONE')

    def test_friday_clock_out_waits_until_15(self):
        automation = SopraClockInAutomation()
        automation.current_weekday = 4
        automation.current_hour = 14
        automation.current_minute = 59
        automation.state['clock_in_at'] = (datetime.now() - timedelta(hours=9)).isoformat()
        self.assertEqual(automation._determine_action(), 'NONE')
    
    def test_no_action_during_work_hours(self):
        """Test that checkout waits until nine hours have elapsed."""
        automation = SopraClockInAutomation()
        automation.current_weekday = 0
        automation.current_hour = 17
        automation.current_minute = 30
        automation.state['clock_in_at'] = (datetime.now() - timedelta(hours=8)).isoformat()
        self.assertEqual(automation._determine_action(), 'NONE')

    def test_late_login_uses_nearest_available_check(self):
        """Test fallback clock-in after the preferred window is missed."""
        automation = SopraClockInAutomation()
        automation.current_weekday = 0
        automation.current_hour = 16
        automation.current_minute = 30
        self.assertEqual(automation._determine_action(), 'CLOCK_IN')

    def test_none_action_does_not_save_state(self):
        """A NONE run must never write clock_out_at (regression for false clock-out block)."""
        automation = SopraClockInAutomation()
        automation.state['clock_in_at'] = (datetime.now() - timedelta(hours=1)).isoformat()
        automation._save_state(datetime.now(), 'NONE')
        self.assertIsNone(automation.state['clock_out_at'])

    def test_missing_state_after_clock_out_start_uses_clock_out(self):
        """Test that a late run after checkout time clicks clock-out, not clock-in."""
        automation = SopraClockInAutomation()
        automation.current_hour = 17
        automation.current_minute = 31
        automation.state['clock_in_at'] = None
        self.assertEqual(automation._determine_action(), 'CLOCK_OUT')


class TestDriverSetup(unittest.TestCase):
    """Test driver setup functionality."""
    
    @patch('sopra_clockin.webdriver.Chrome')
    @patch('sopra_clockin.Options')
    def test_chrome_driver_setup(self, mock_options, mock_chrome):
        """Test Chrome driver setup."""
        mock_driver = Mock()
        mock_chrome.return_value = mock_driver
        mock_options.return_value = Mock()
        
        automation = SopraClockInAutomation()
        
        # Mock the config to use Chrome
        with patch('sopra_clockin.BROWSER', 'chrome'):
            result = automation._setup_chrome_driver()
            
            self.assertTrue(result)
            mock_chrome.assert_called_once()
            mock_driver.set_page_load_timeout.assert_called_once()
    
    @patch('sopra_clockin.webdriver.Edge')
    @patch('sopra_clockin.EdgeOptions')
    def test_edge_driver_setup(self, mock_options, mock_edge):
        """Test Edge driver setup."""
        mock_driver = Mock()
        mock_edge.return_value = mock_driver
        mock_options.return_value = Mock()
        
        automation = SopraClockInAutomation()
        
        result = automation._setup_edge_driver()
        
        self.assertTrue(result)
        mock_edge.assert_called_once()
        mock_driver.set_page_load_timeout.assert_called_once()


class TestNavigation(unittest.TestCase):
    """Test navigation functionality."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.automation = SopraClockInAutomation()
        self.automation.driver = Mock()
        self.automation.wait = Mock()
    
    @patch('sopra_clockin.time.sleep')
    def test_navigate_to_portal_success(self, mock_sleep):
        """Test successful portal navigation."""
        self.automation.driver.get = Mock()
        self.automation.driver.current_url = "https://example.test/SopraGP4U/WAW05B02"
        
        result = self.automation.navigate_to_portal()
        
        self.assertTrue(result)
        self.automation.driver.get.assert_called_once()
        mock_sleep.assert_called_once_with(3)
    
    @patch('sopra_clockin.time.sleep')
    def test_navigate_to_portal_failure(self, mock_sleep):
        """Test failed portal navigation."""
        self.automation.driver.get = Mock(side_effect=Exception("Network error"))
        
        result = self.automation.navigate_to_portal()
        
        self.assertFalse(result)


class TestLoginDetection(unittest.TestCase):
    """Test login requirement detection."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.automation = SopraClockInAutomation()
        self.automation.driver = Mock()
    
    def test_login_required_found(self):
        """Test login form detection when found."""
        mock_element = Mock()
        mock_element.is_displayed.return_value = True
        self.automation.driver.find_element.return_value = mock_element
        
        result = self.automation.check_login_required()
        
        self.assertTrue(result)
    
    def test_login_not_required(self):
        """Test login form detection when not found."""
        self.automation.driver.find_element.side_effect = Exception("Not found")
        
        result = self.automation.check_login_required()
        
        self.assertFalse(result)


class TestButtonClicking(unittest.TestCase):
    """Test button clicking functionality."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.automation = SopraClockInAutomation()
        self.automation.driver = Mock()
        self.automation.wait = Mock()
        self.automation.action_type = 'CLOCK_IN'
    
    @patch('sopra_clockin.time.sleep')
    def test_click_clock_button_success(self, mock_sleep):
        """Test successful button click."""
        mock_button = Mock()
        self.automation._find_element_in_frames = Mock(return_value=mock_button)
        self.automation._click_element = Mock()
        
        with patch('sopra_clockin.DRY_RUN', False):
            result = self.automation.click_clock_button()
            
            self.assertTrue(result)
            self.automation._click_element.assert_called_once_with(mock_button, 'CLOCK_IN button')

    def test_disabled_clock_in_infers_eight_am_when_clock_out_is_enabled(self):
        clock_in_button = Mock()
        clock_in_button.is_enabled.return_value = False
        clock_out_button = Mock()
        clock_out_button.is_enabled.return_value = True
        self.automation._find_element_in_frames = Mock(side_effect=[clock_in_button, clock_out_button])
        self.automation._click_element = Mock()
        self.automation._save_state = Mock()

        with patch('sopra_clockin.DRY_RUN', False):
            result = self.automation.click_clock_button()

        self.assertTrue(result)
        self.assertEqual(self.automation.action_type, 'NONE')
        self.assertTrue(self.automation.state['clock_in_at'].endswith('T08:00:00'))
        self.automation._click_element.assert_not_called()
        self.automation._save_state.assert_called_once()
    
    @patch('sopra_clockin.time.sleep')
    def test_click_clock_button_dry_run(self, mock_sleep):
        """Test button click in dry run mode."""
        mock_button = Mock()
        self.automation._find_element_in_frames = Mock(return_value=mock_button)
        self.automation._click_element = Mock()
        
        with patch('sopra_clockin.DRY_RUN', True):
            result = self.automation.click_clock_button()
            
            self.assertTrue(result)
            self.automation._click_element.assert_called_once_with(mock_button, 'CLOCK_IN button')
            mock_sleep.assert_not_called()
    
    def test_click_clock_button_no_action(self):
        """Test button click when no action needed."""
        self.automation.action_type = 'NONE'
        
        result = self.automation.click_clock_button()
        
        self.assertTrue(result)
        self.automation.wait.until.assert_not_called()

    def test_click_clock_button_returns_false_when_missing(self):
        self.automation._find_element_in_frames = Mock(return_value=None)
        self.automation._save_debug_snapshot = Mock()

        result = self.automation.click_clock_button()

        self.assertFalse(result)
        self.automation._save_debug_snapshot.assert_called_once_with(
            'CLOCK_IN button was not found'
        )

    def test_menu_returns_false_when_menu_and_clock_control_missing(self):
        self.automation._find_element_in_frames = Mock(return_value=None)
        self.automation._save_debug_snapshot = Mock()

        result = self.automation.click_menu_link()

        self.assertFalse(result)
        self.automation._save_debug_snapshot.assert_called_once()

    def test_menu_clicks_when_found_in_frames(self):
        menu_link = Mock()
        self.automation._find_element_in_frames = Mock(return_value=menu_link)
        self.automation._click_element = Mock()

        result = self.automation.click_menu_link()

        self.assertTrue(result)
        self.automation._click_element.assert_called_once_with(menu_link, 'menu link Registro de entrada / salida')

    @patch('sopra_clockin.WebDriverWait')
    def test_alternative_button_uses_valid_short_wait(self, mock_wait_class):
        fallback_wait = Mock()
        fallback_wait.until.return_value = Mock()
        mock_wait_class.return_value = fallback_wait
        self.automation.wait.until.side_effect = Exception("primary selector missing")

        result = self.automation._try_alternative_clock_button()

        self.assertTrue(result)
        mock_wait_class.assert_called_once_with(self.automation.driver, 5)
        fallback_wait.until.assert_called_once()


class TestConfiguration(unittest.TestCase):
    """Test configuration loading."""
    
    def test_default_selectors(self):
        """Test default selector values."""
        self.assertIn('method', CLOCK_IN_SELECTOR)
        self.assertIn('value', CLOCK_IN_SELECTOR)
        self.assertIn('method', CLOCK_OUT_SELECTOR)
        self.assertIn('value', CLOCK_OUT_SELECTOR)


if __name__ == '__main__':
    # Run tests
    unittest.main(verbosity=2)