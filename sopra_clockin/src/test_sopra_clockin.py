#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Unit tests for SopraGP4U Clock In/Out automation
===============================================

These tests validate the core functionality without actually interacting
with the website or performing real clock-in/out actions.

Run with: python -m pytest src/test_sopra_clockin.py -v
"""

import sys
import unittest
from unittest.mock import Mock, patch, MagicMock
from pathlib import Path

# Add config directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

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
        self.assertEqual(result, ('id', 'test-id'))
    
    def test_get_selector_tuple_xpath(self):
        """Test XPath selector conversion."""
        selector = {'method': 'xpath', 'value': '//button[@id="test"]'}
        result = get_selector_tuple(selector)
        self.assertEqual(result, ('xpath', '//button[@id="test"]'))
    
    def test_get_selector_tuple_css(self):
        """Test CSS selector conversion."""
        selector = {'method': 'css', 'value': '.btn-primary'}
        result = get_selector_tuple(selector)
        self.assertEqual(result, ('css_selector', '.btn-primary'))
    
    def test_get_selector_tuple_class(self):
        """Test class selector conversion (treated as CSS)."""
        selector = {'method': 'class', 'value': 'btn-primary'}
        result = get_selector_tuple(selector)
        self.assertEqual(result, ('css_selector', 'btn-primary'))
    
    def test_get_selector_tuple_name(self):
        """Test name selector conversion."""
        selector = {'method': 'name', 'value': 'submit'}
        result = get_selector_tuple(selector)
        self.assertEqual(result, ('name', 'submit'))
    
    def test_get_selector_tuple_link_text(self):
        """Test link text selector conversion."""
        selector = {'method': 'link_text', 'value': 'Click here'}
        result = get_selector_tuple(selector)
        self.assertEqual(result, ('link_text', 'Click here'))
    
    def test_get_selector_tuple_partial_link_text(self):
        """Test partial link text selector conversion."""
        selector = {'method': 'partial_link_text', 'value': 'Click'}
        result = get_selector_tuple(selector)
        self.assertEqual(result, ('partial_link_text', 'Click'))
    
    def test_get_selector_tuple_default(self):
        """Test default selector conversion (invalid method defaults to ID)."""
        selector = {'method': 'invalid', 'value': 'test'}
        result = get_selector_tuple(selector)
        self.assertEqual(result, ('id', 'test'))


class TestActionDetermination(unittest.TestCase):
    """Test action determination logic."""
    
    def test_clock_in_before_threshold(self):
        """Test clock-in determination before threshold."""
        automation = SopraClockInAutomation()
        automation.current_hour = CLOCK_IN_THRESHOLD - 1
        self.assertEqual(automation._determine_action(), 'CLOCK_IN')
    
    def test_clock_out_after_threshold(self):
        """Test clock-out determination after threshold."""
        automation = SopraClockInAutomation()
        automation.current_hour = CLOCK_OUT_THRESHOLD + 1
        self.assertEqual(automation._determine_action(), 'CLOCK_OUT')
    
    def test_no_action_during_work_hours(self):
        """Test no action during work hours."""
        automation = SopraClockInAutomation()
        automation.current_hour = (CLOCK_IN_THRESHOLD + CLOCK_OUT_THRESHOLD) // 2
        self.assertEqual(automation._determine_action(), 'NONE')


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
        
        result = self.automation.navigate_to_portal()
        
        self.assertTrue(result)
        self.automation.driver.get.assert_called_once()
        mock_sleep.assert_called_once_with(2)
    
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
        self.automation.wait.until.return_value = mock_button
        
        with patch('sopra_clockin.DRY_RUN', False):
            result = self.automation.click_clock_button()
            
            self.assertTrue(result)
            mock_button.click.assert_called_once()
            mock_sleep.assert_called_once_with(2)
    
    @patch('sopra_clockin.time.sleep')
    def test_click_clock_button_dry_run(self, mock_sleep):
        """Test button click in dry run mode."""
        mock_button = Mock()
        self.automation.wait.until.return_value = mock_button
        
        with patch('sopra_clockin.DRY_RUN', True):
            result = self.automation.click_clock_button()
            
            self.assertTrue(result)
            mock_button.click.assert_not_called()
            mock_sleep.assert_not_called()
    
    def test_click_clock_button_no_action(self):
        """Test button click when no action needed."""
        self.automation.action_type = 'NONE'
        
        result = self.automation.click_clock_button()
        
        self.assertTrue(result)
        self.automation.wait.until.assert_not_called()


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