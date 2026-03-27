#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
SopraGP4U Automatic Clock In/Out automation script
=====================================================

This script automates the clock in/out process for SopraGP4U portal.

Features:
- Automatic clock-in before 10:00 AM
- Automatic clock-out after 5:00 PM
- Comprehensive logging to file and console
- Retry logic for failed attempts
- Compatible with Windows Task Scheduler

Author: Emilio Guillem Simón
Date: 2026
"""

import sys
import time
from datetime import datetime
from pathlib import Path

# Add config directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from logger_config import setup_logger
from config.config import (
    SOPRA_URL, MENU_LINK_TEXT, CLOCK_IN_THRESHOLD, CLOCK_OUT_THRESHOLD,
    CLOCK_IN_BUTTON_ID, CLOCK_OUT_BUTTON_ID, SOPRA_USERNAME, SOPRA_PASSWORD,
    WAIT_TIMEOUT, PAGE_LOAD_TIMEOUT, MAX_RETRIES, RETRY_DELAY, DRY_RUN,
    HEADLESS_MODE, CHROME_OPTIONS, EDGE_OPTIONS, BROWSER,
    CLOCK_IN_SELECTOR, CLOCK_OUT_SELECTOR
)

try:
    from selenium import webdriver
    from selenium.webdriver.common.by import By
    from selenium.webdriver.support.ui import WebDriverWait
    from selenium.webdriver.support import expected_conditions as EC
    from selenium.webdriver.chrome.service import Service
    from selenium.webdriver.chrome.options import Options
    from selenium.webdriver.edge.options import Options as EdgeOptions
except ImportError:
    print("ERROR: Selenium is not installed. Install it with: pip install selenium")
    sys.exit(1)

# Initialize logger
logger = setup_logger(__name__)


def get_selector_tuple(selector_dict):
    """
    Convert selector dictionary to Selenium By tuple.
    
    Args:
        selector_dict (dict): {'method': 'id', 'value': 'CLOCK-IN'}
    
    Returns:
        tuple: (By.ID, 'CLOCK-IN')
    """
    method = selector_dict.get('method', 'id').lower()
    value = selector_dict.get('value', '')
    
    if method == 'id':
        return (By.ID, value)
    elif method == 'xpath':
        return (By.XPATH, value)
    elif method == 'css' or method == 'class':
        return (By.CSS_SELECTOR, value)
    elif method == 'name':
        return (By.NAME, value)
    elif method == 'link_text':
        return (By.LINK_TEXT, value)
    elif method == 'partial_link_text':
        return (By.PARTIAL_LINK_TEXT, value)
    else:
        # Default to ID
        return (By.ID, value)


class SopraClockInAutomation:
    """Main automation class for SopraGP4U clock in/out."""
    
    def __init__(self):
        """Initialize the automation class."""
        self.driver = None
        self.wait = None
        self.current_hour = datetime.now().hour
        self.action_type = self._determine_action()
        
    def _determine_action(self):
        """
        Determine whether to clock in or clock out based on current time.
        
        Returns:
            str: 'CLOCK_IN', 'CLOCK_OUT', or 'NONE'
        """
        if self.current_hour < CLOCK_IN_THRESHOLD:
            return 'CLOCK_IN'
        elif self.current_hour >= CLOCK_OUT_THRESHOLD:
            return 'CLOCK_OUT'
        else:
            return 'NONE'
    
    def setup_driver(self):
        """
        Initialize and configure the WebDriver (Chrome or Edge).
        
        Returns:
            bool: True if successful, False otherwise
        """
        try:
            if BROWSER == "edge":
                return self._setup_edge_driver()
            else:
                return self._setup_chrome_driver()
            
        except Exception as e:
            logger.error(f"Failed to setup WebDriver: {str(e)}", exc_info=True)
            return False
    
    def _setup_chrome_driver(self):
        """Setup Chrome WebDriver."""
        try:
            logger.info("Setting up Chrome WebDriver...")
            
            chrome_options = Options()
            
            # Apply configuration options
            if HEADLESS_MODE:
                chrome_options.add_argument("--headless")
                logger.info("Running in headless mode")
            
            chrome_options.add_argument("--window-size=1920,1080")
            chrome_options.add_argument("--no-sandbox")
            chrome_options.add_argument("--disable-dev-shm-usage")
            chrome_options.add_argument("--disable-blink-features=AutomationControlled")
            chrome_options.add_experimental_option("excludeSwitches", ["enable-automation"])
            chrome_options.add_experimental_option('useAutomationExtension', False)
            
            # Initialize WebDriver
            self.driver = webdriver.Chrome(options=chrome_options)
            self.driver.set_page_load_timeout(PAGE_LOAD_TIMEOUT)
            self.wait = WebDriverWait(self.driver, WAIT_TIMEOUT)
            
            logger.info("Chrome WebDriver initialized successfully")
            return True
            
        except Exception as e:
            logger.error(f"Failed to setup Chrome WebDriver: {str(e)}", exc_info=True)
            return False
    
    def _setup_edge_driver(self):
        """Setup Edge WebDriver."""
        try:
            logger.info("Setting up Edge WebDriver...")
            
            edge_options = EdgeOptions()
            
            # Apply configuration options
            if HEADLESS_MODE:
                edge_options.add_argument("--headless")
                logger.info("Running in headless mode")
            
            edge_options.add_argument("--window-size=1920,1080")
            edge_options.add_argument("--disable-blink-features=AutomationControlled")
            edge_options.add_experimental_option("excludeSwitches", ["enable-automation"])
            edge_options.add_experimental_option('useAutomationExtension', False)
            
            # Initialize WebDriver
            self.driver = webdriver.Edge(options=edge_options)
            self.driver.set_page_load_timeout(PAGE_LOAD_TIMEOUT)
            self.wait = WebDriverWait(self.driver, WAIT_TIMEOUT)
            
            logger.info("Edge WebDriver initialized successfully")
            return True
            
        except Exception as e:
            logger.error(f"Failed to setup Edge WebDriver: {str(e)}", exc_info=True)
            return False
    
    def close_driver(self):
        """Close the WebDriver and cleanup resources."""
        if self.driver:
            try:
                self.driver.quit()
                logger.info("WebDriver closed successfully")
            except Exception as e:
                logger.error(f"Error closing WebDriver: {str(e)}")
    
    def navigate_to_portal(self):
        """
        Navigate to the SopraGP4U portal.
        
        Returns:
            bool: True if successful, False otherwise
        """
        try:
            logger.info(f"Navigating to {SOPRA_URL}")
            self.driver.get(SOPRA_URL)
            time.sleep(2)  # Wait for page to load
            logger.info("Successfully navigated to portal")
            return True
            
        except Exception as e:
            logger.error(f"Failed to navigate to portal: {str(e)}", exc_info=True)
            return False
    
    def check_login_required(self):
        """
        Check if portal requires authentication.
        
        Returns:
            bool: True if login form detected, False otherwise
        """
        try:
            # More specific login indicators - check for actual login forms
            login_indicators = [
                (By.ID, "username"),
                (By.ID, "user"),
                (By.ID, "login"),
                (By.NAME, "username"),
                (By.NAME, "user"),
                (By.CLASS_NAME, "login-form"),
                (By.CLASS_NAME, "login"),
            ]
            
            for locator in login_indicators:
                try:
                    element = self.driver.find_element(*locator)
                    # Check if element is actually visible and not just in the DOM
                    if element.is_displayed():
                        logger.info(f"Login form detected using {locator}")
                        return True
                except:
                    continue
            
            logger.info("No login form detected")
            return False
        except Exception as e:
            logger.warning(f"Error checking login requirement: {str(e)}")
            return False
    
    def perform_login(self):
        """
        Handle portal authentication if required.
        Only attempts login if credentials are available in environment.
        
        Returns:
            bool: True if login successful or skipped, False only on actual failure
        """
        try:
            # Get credentials from environment only
            username = SOPRA_USERNAME
            password = SOPRA_PASSWORD
            
            if not username or not password:
                logger.warning("No credentials found in environment. Attempting to continue without login...")
                return True  # Continue without login
            
            logger.info("Credentials found. Attempting authentication...")
            
            # Find and fill username field
            username_field = None
            username_locators = [
                (By.ID, "username"),
                (By.ID, "user"),
                (By.NAME, "username"),
                (By.NAME, "user"),
            ]
            
            for locator in username_locators:
                try:
                    username_field = self.wait.until(
                        EC.presence_of_element_located(locator),
                        message=f"Username field {locator} not found"
                    )
                    break
                except:
                    continue
            
            if not username_field:
                logger.warning("Username field not found. Assuming portal doesn't require login.")
                return True  # Continue anyway - maybe we're already logged in or no login needed
            
            # Find and fill password field
            password_field = None
            password_locators = [
                (By.ID, "password"),
                (By.NAME, "password"),
                (By.CSS_SELECTOR, "input[type='password']"),
            ]
            
            for locator in password_locators:
                try:
                    password_field = self.driver.find_element(*locator)
                    break
                except:
                    continue
            
            if not password_field:
                logger.error("Password field not found in portal")
                return False
            
            # Enter credentials
            logger.info("Entering credentials...")
            username_field.clear()
            username_field.send_keys(username)
            time.sleep(0.5)
            
            password_field.clear()
            password_field.send_keys(password)
            time.sleep(0.5)
            
            # Find and click login button
            login_button = None
            login_locators = [
                (By.ID, "login"),
                (By.ID, "login-button"),
                (By.ID, "submit"),
                (By.NAME, "login"),
                (By.XPATH, "//button[contains(text(), 'Login')]"),
                (By.XPATH, "//button[contains(text(), 'Sign in')]"),
                (By.XPATH, "//button[@type='submit']"),
            ]
            
            for locator in login_locators:
                try:
                    login_button = self.driver.find_element(*locator)
                    break
                except:
                    continue
            
            if not login_button:
                logger.error("Login button not found")
                return False
            
            if DRY_RUN:
                logger.info("[DRY RUN] Would click login button")
            else:
                logger.info("Clicking login button...")
                login_button.click()
                time.sleep(3)  # Wait for authentication
            
            logger.info("Login completed successfully")
            return True
            
        except Exception as e:
            logger.error(f"Login failed: {str(e)}", exc_info=True)
            return False
    
    def click_menu_link(self):
        """
        Click on the 'Registro de entrada/salida' menu link.
        
        Returns:
            bool: True if successful or not needed, False otherwise
        """
        try:
            logger.info(f"Looking for menu link: '{MENU_LINK_TEXT}'")
            
            # Try to find the link by text
            menu_link = self.wait.until(
                EC.element_to_be_clickable((By.LINK_TEXT, MENU_LINK_TEXT)),
                message=f"Menu link '{MENU_LINK_TEXT}' not found"
            )
            
            if DRY_RUN:
                logger.info(f"[DRY RUN] Would click on menu link: {MENU_LINK_TEXT}")
            else:
                menu_link.click()
                time.sleep(2)  # Wait for page transition
                logger.info(f"Successfully clicked on menu link: {MENU_LINK_TEXT}")
            
            return True
            
        except Exception as e:
            logger.info(f"Menu link not found ({str(e)}) - assuming we're already on the clock page")
            # Try alternative selectors
            return self._try_alternative_menu_links()
    
    def _try_alternative_menu_links(self):
        """
        Try alternative methods to locate and click the menu link.
        
        Returns:
            bool: True if successful or not needed, False otherwise
        """
        try:
            logger.info("Trying alternative selectors for menu link...")
            
            # Try partial link text
            menu_link = self.wait.until(
                EC.element_to_be_clickable((By.PARTIAL_LINK_TEXT, "Registro")),
                timeout=2
            )
            
            if DRY_RUN:
                logger.info("[DRY RUN] Would click on alternative menu link")
            else:
                menu_link.click()
                time.sleep(2)
                logger.info("Successfully clicked on alternative menu link")
            
            return True
            
        except Exception as e:
            logger.info(f"Menu link not found with alternative selectors - assuming already on clock page")
            return True  # Assume we're already on the correct page
    
    def click_clock_button(self):
        """
        Click the appropriate clock in/out button based on current time.
        
        Returns:
            bool: True if successful, False otherwise
        """
        if self.action_type == 'NONE':
            logger.info(f"Current time is {self.current_hour}:00 - outside clock-in/out windows")
            return True
        
        # Get the appropriate selector
        selector_dict = CLOCK_IN_SELECTOR if self.action_type == 'CLOCK_IN' else CLOCK_OUT_SELECTOR
        selector_tuple = get_selector_tuple(selector_dict)
        
        try:
            logger.info(f"Looking for {self.action_type} button using {selector_tuple}")
            
            button = self.wait.until(
                EC.element_to_be_clickable(selector_tuple),
                timeout=5
            )
            
            if DRY_RUN:
                logger.info(f"[DRY RUN] Would click {self.action_type} button")
            else:
                button.click()
                time.sleep(2)  # Wait for action to process
                
                # Log button state change
                self._log_button_state_change()
                
                logger.info(f"Successfully clicked {self.action_type} button")
            
            return True
            
        except Exception as e:
            logger.info(f"Button not found with configured selector {selector_tuple}: {str(e)}")
            # Try alternative selectors
            return self._try_alternative_clock_button()
    
    def _log_button_state_change(self):
        """
        Log the expected state change after clicking a button.
        Clock-in disables clock-in button and enables clock-out button.
        Clock-out disables clock-out button and enables clock-in button.
        """
        if self.action_type == 'CLOCK_IN':
            logger.info("Clock-in button clicked - expecting clock-out button to be enabled now")
        elif self.action_type == 'CLOCK_OUT':
            logger.info("Clock-out button clicked - expecting clock-in button to be enabled now")
    
    def _try_alternative_clock_button(self):
        """
        Try alternative methods to locate and click the clock button.
        
        Returns:
            bool: True if successful, False otherwise
        """
        try:
            logger.info("Trying alternative selectors for clock button...")
            
            # Use lowercase button text as specified
            button_text = "clock-in" if self.action_type == 'CLOCK_IN' else "clock-out"
            
            # Try button by text (case-insensitive)
            button = self.wait.until(
                EC.element_to_be_clickable((By.XPATH, f"//button[contains(translate(text(), 'ABCDEFGHIJKLMNOPQRSTUVWXYZ', 'abcdefghijklmnopqrstuvwxyz'), '{button_text}')]")),
                message=f"Button with text '{button_text}' not found"
            )
            
            if DRY_RUN:
                logger.info(f"[DRY RUN] Would click alternative clock button")
            else:
                button.click()
                time.sleep(2)
                self._log_button_state_change()
                logger.info(f"Successfully clicked alternative clock button")
            
            return True
            
        except Exception as e:
            logger.error(f"Failed with alternative selectors: {str(e)}", exc_info=True)
            return False
    
    def run(self):
        """
        Execute the main automation workflow with retry logic.
        
        Returns:
            bool: True if successful, False otherwise
        """
        logger.info("="*80)
        logger.info("Starting SopraGP4U Clock In/Out Automation")
        logger.info(f"Current time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        logger.info(f"Browser: {BROWSER.upper()}")
        logger.info(f"Action type: {self.action_type}")
        logger.info(f"Dry run mode: {DRY_RUN}")
        logger.info("="*80)
        
        for attempt in range(1, MAX_RETRIES + 1):
            try:
                logger.info(f"\nAttempt {attempt}/{MAX_RETRIES}")
                
                # Setup driver
                if not self.setup_driver():
                    raise Exception("Failed to setup WebDriver")
                
                # Navigate to portal
                if not self.navigate_to_portal():
                    raise Exception("Failed to navigate to portal")
                
                # Check if authentication is required
                if self.check_login_required():
                    logger.info("Portal requires authentication")
                    if not self.perform_login():
                        raise Exception("Authentication failed")
                else:
                    logger.info("Portal is accessible without authentication")
                
                # Click menu link
                if not self.click_menu_link():
                    raise Exception("Failed to click menu link")
                
                # Click clock button
                if not self.click_clock_button():
                    raise Exception("Failed to click clock button")
                
                logger.info("="*80)
                logger.info("Automation completed successfully!")
                logger.info("="*80)
                return True
                
            except Exception as e:
                logger.error(f"Attempt {attempt} failed: {str(e)}")
                
                if attempt < MAX_RETRIES:
                    logger.info(f"Retrying in {RETRY_DELAY} seconds...")
                    time.sleep(RETRY_DELAY)
                else:
                    logger.error("All retry attempts failed!")
                    logger.info("="*80)
                    logger.info("Automation failed after all retries")
                    logger.info("="*80)
                    return False
            
            finally:
                self.close_driver()
        
        return False


def main():
    """Main entry point."""
    try:
        automation = SopraClockInAutomation()
        success = automation.run()
        sys.exit(0 if success else 1)
        
    except Exception as e:
        logger.critical(f"Critical error: {str(e)}", exc_info=True)
        sys.exit(2)


if __name__ == "__main__":
    main()
