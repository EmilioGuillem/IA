#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# This file has been created (totally or partially) with the assistance of
# artificial intelligence tools. All content has been generated under the
# direct supervision of a named individual and the AI.Backbone Orchestrator
# Compliance framework.

"""
SopraGP4U Automatic Clock In/Out automation script
=====================================================

This script automates the clock in/out process for SopraGP4U portal.

Features:
- Automatic clock-in from 08:00 to 09:30, with a late-login fallback
- Automatic clock-out from 17:30 after more than nine hours worked
- Comprehensive logging to file and console
- Retry logic for failed attempts
- Compatible with Windows Task Scheduler

Author: Emilio Guillem Simón
Date: 2026
"""

import sys
import time
import json
from datetime import datetime
from pathlib import Path

# Add config directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from logger_config import setup_logger
from config.config import (
    SOPRA_URL, MENU_LINK_TEXT, CLOCK_IN_THRESHOLD, CLOCK_OUT_THRESHOLD,
    CLOCK_IN_START_HOUR, CLOCK_IN_START_MINUTE,
    CLOCK_IN_END_HOUR, CLOCK_IN_END_MINUTE,
    CLOCK_OUT_START_HOUR, CLOCK_OUT_START_MINUTE,
    CLOCK_OUT_FRIDAY_START_HOUR, CLOCK_OUT_FRIDAY_START_MINUTE,
    MIN_WORK_HOURS, MIN_WORK_HOURS_FRIDAY,
    STATE_FILE,
    CLOCK_IN_BUTTON_ID, CLOCK_OUT_BUTTON_ID, SOPRA_USERNAME, SOPRA_PASSWORD,
    WAIT_TIMEOUT, PAGE_LOAD_TIMEOUT, MAX_RETRIES, RETRY_DELAY, DRY_RUN,
    HEADLESS_MODE, CHROME_OPTIONS, EDGE_OPTIONS, BROWSER,
    CLOCK_IN_SELECTOR, CLOCK_OUT_SELECTOR, MENU_LINK_SELECTOR
)

try:
    from selenium import webdriver
    from selenium.common.exceptions import NoSuchElementException, StaleElementReferenceException
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
        now = datetime.now()
        self.current_hour = now.hour
        self.current_minute = now.minute
        self.current_weekday = now.weekday()
        self.state = self._load_state(now)
        self.action_type = self._determine_action()

    def _load_state(self, now):
        """Load today's successful actions and discard stale daily state."""
        empty_state = {"date": now.date().isoformat(), "clock_in_at": None, "clock_out_at": None}
        if not STATE_FILE.exists():
            return empty_state

        try:
            with open(STATE_FILE, 'r', encoding='utf-8') as state_file:
                state = json.load(state_file)
        except (OSError, json.JSONDecodeError):
            logger.warning("Could not read clock state; starting with empty state")
            return empty_state

        if state.get("date") != now.date().isoformat():
            return empty_state

        return {
            "date": state.get("date"),
            "clock_in_at": state.get("clock_in_at"),
            "clock_out_at": state.get("clock_out_at"),
        }

    def _save_state(self, timestamp, action):
        """Persist a real CLOCK_IN/CLOCK_OUT action after its browser flow succeeds."""
        if action not in ("CLOCK_IN", "CLOCK_OUT"):
            return
        key = "clock_in_at" if action == "CLOCK_IN" else "clock_out_at"
        self.state[key] = timestamp.isoformat(timespec='seconds')
        with open(STATE_FILE, 'w', encoding='utf-8') as state_file:
            json.dump(self.state, state_file, indent=2)

    def _clock_in_timestamp(self):
        value = self.state.get("clock_in_at")
        if not value:
            return None
        try:
            return datetime.fromisoformat(value)
        except ValueError:
            logger.warning("Ignoring invalid clock-in timestamp in state")
            return None
        
    def _determine_action(self):
        """
        Determine whether to clock in or clock out based on current time.
        
        Returns:
            str: 'CLOCK_IN', 'CLOCK_OUT', or 'NONE'
        """
        current_minutes = self.current_hour * 60 + self.current_minute
        clock_in_start = CLOCK_IN_START_HOUR * 60 + CLOCK_IN_START_MINUTE
        if self.current_weekday == 4:
            clock_out_start = CLOCK_OUT_FRIDAY_START_HOUR * 60 + CLOCK_OUT_FRIDAY_START_MINUTE
        else:
            clock_out_start = CLOCK_OUT_START_HOUR * 60 + CLOCK_OUT_START_MINUTE

        if self.state.get("clock_out_at"):
            return 'NONE'

        clock_in_at = self._clock_in_timestamp()
        if clock_in_at is None:
            if current_minutes >= clock_out_start:
                logger.warning("No local clock-in state found; current time is past clock-out start, using CLOCK_OUT")
                return 'CLOCK_OUT'
            if current_minutes >= clock_in_start:
                # Preferred window, or immediate late-login fallback.
                if current_minutes > CLOCK_IN_END_HOUR * 60 + CLOCK_IN_END_MINUTE:
                    logger.warning("Clock-in window missed; using the nearest available check")
                return 'CLOCK_IN'
            return 'NONE'

        elapsed_hours = (datetime.now() - clock_in_at).total_seconds() / 3600
        minimum_work_hours = MIN_WORK_HOURS_FRIDAY if self.current_weekday == 4 else MIN_WORK_HOURS
        if current_minutes >= clock_out_start and elapsed_hours >= minimum_work_hours:
            return 'CLOCK_OUT'
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
            time.sleep(3)  # Wait for page to load and React routing
            if "WAW05B02" not in self.driver.current_url:
                logger.warning(
                    "Portal redirected to %s; retrying direct clock page",
                    self.driver.current_url,
                )
                self.driver.get(SOPRA_URL)
                time.sleep(3)
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

    def _save_debug_snapshot(self, reason):
        """Persist the current browser view for diagnosing selector failures."""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        html_path = STATE_FILE.parent / f"debug_{timestamp}.html"
        screenshot_path = STATE_FILE.parent / f"debug_{timestamp}.png"

        try:
            self.driver.switch_to.default_content()
            frame_count = len(self.driver.find_elements(By.CSS_SELECTOR, "iframe, frame"))
            logger.error(
                "%s. URL=%r title=%r top-level frames=%s",
                reason,
                self.driver.current_url,
                self.driver.title,
                frame_count,
            )
            html_path.write_text(self.driver.page_source, encoding="utf-8")
            self.driver.save_screenshot(str(screenshot_path))
            logger.error("Saved debug HTML to %s and screenshot to %s", html_path, screenshot_path)
        except Exception as e:
            logger.error("Could not save debug snapshot: %s", str(e))

    def _find_element_in_frames(self, locator, depth=0, max_depth=6):
        """Find an element in the current document or recursively in frames."""
        try:
            return self.driver.find_element(*locator)
        except NoSuchElementException:
            pass

        if depth >= max_depth:
            return None

        frames = self.driver.find_elements(By.CSS_SELECTOR, "iframe, frame")
        for index in range(len(frames)):
            try:
                frames = self.driver.find_elements(By.CSS_SELECTOR, "iframe, frame")
                self.driver.switch_to.frame(frames[index])
                element = self._find_element_in_frames(locator, depth + 1, max_depth)
                if element is not None:
                    return element
                self.driver.switch_to.parent_frame()
            except (NoSuchElementException, StaleElementReferenceException):
                self.driver.switch_to.default_content()

        return None

    def _click_element(self, element, description):
        """Click using JavaScript after scrolling the element into view."""
        if DRY_RUN:
            logger.info(f"[DRY RUN] Would click {description}")
            return

        logger.info("Clicking %s", description)
        self.driver.execute_script("arguments[0].scrollIntoView({block: 'center'});", element)
        self.driver.execute_script("arguments[0].click();", element)
        time.sleep(2)
    
    def click_menu_link(self):
        """
        Click on the 'Registro de entrada / salida' menu link.
        
        Returns:
            bool: True if successful or not needed, False otherwise
        """
        logger.info(f"Looking for menu link: '{MENU_LINK_TEXT}'")

        def portal_control_available(_driver):
            self.driver.switch_to.default_content()
            menu_locator = get_selector_tuple(MENU_LINK_SELECTOR)
            if self._find_element_in_frames(menu_locator) is not None:
                return True

            selector_dict = CLOCK_IN_SELECTOR if self.action_type == 'CLOCK_IN' else CLOCK_OUT_SELECTOR
            return self._find_element_in_frames(get_selector_tuple(selector_dict)) is not None

        try:
            self.wait.until(portal_control_available)
        except Exception:
            logger.warning("Portal loaded without the menu or clock control after %s seconds", WAIT_TIMEOUT)

        self.driver.switch_to.default_content()
        menu_link = self._find_element_in_frames(get_selector_tuple(MENU_LINK_SELECTOR))

        if menu_link is not None:
            logger.info("Menu link HTML: %s", menu_link.get_attribute("outerHTML"))
            self._click_element(menu_link, f"menu link {MENU_LINK_TEXT}")
            return True

        selector_dict = CLOCK_IN_SELECTOR if self.action_type == 'CLOCK_IN' else CLOCK_OUT_SELECTOR
        clock_element = self._find_element_in_frames(get_selector_tuple(selector_dict))
        if clock_element is not None:
            logger.info("Menu link not found, but target clock control is already available")
            return True

        self._save_debug_snapshot("Neither menu link nor target clock control was found")
        return False
    
    def _try_alternative_menu_links(self):
        """
        Try alternative methods to locate and click the menu link.
        
        Returns:
            bool: True if successful or not needed, False otherwise
        """
        logger.info("Trying alternative selectors for menu link...")
        self.driver.switch_to.default_content()
        menu_link = self._find_element_in_frames(get_selector_tuple(MENU_LINK_SELECTOR))
        if menu_link is None:
            return False

        self._click_element(menu_link, "alternative menu link")
        return True
    
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
        
        logger.info(f"Looking for {self.action_type} button using {selector_tuple}")
        self.driver.switch_to.default_content()
        button = self._find_element_in_frames(selector_tuple)

        if button is None:
            logger.info("Button not found with configured selector %s", selector_tuple)
            self._save_debug_snapshot(f"{self.action_type} button was not found")
            return False

        if self.action_type == 'CLOCK_IN' and not button.is_enabled():
            self.driver.switch_to.default_content()
            clock_out_button = self._find_element_in_frames(get_selector_tuple(CLOCK_OUT_SELECTOR))
            if clock_out_button is not None and clock_out_button.is_enabled():
                inferred_clock_in = datetime.now().replace(hour=8, minute=0, second=0, microsecond=0)
                self.state['clock_in_at'] = inferred_clock_in.isoformat(timespec='seconds')
                self.action_type = 'NONE'
                if not DRY_RUN:
                    self._save_state(inferred_clock_in, 'CLOCK_IN')
                logger.warning(
                    "Clock-in button is disabled and clock-out is enabled; assuming clock-in at %s",
                    inferred_clock_in.strftime('%H:%M'),
                )
                return True

        logger.info("%s button HTML: %s", self.action_type, button.get_attribute("outerHTML"))
        self._click_element(button, f"{self.action_type} button")
        self._log_button_state_change()
        logger.info(f"Successfully clicked {self.action_type} button")
        return True
    
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
            button = WebDriverWait(self.driver, 5).until(
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
        for attempt in range(1, MAX_RETRIES + 1):
            try:
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

                if not DRY_RUN and self.action_type in ("CLOCK_IN", "CLOCK_OUT"):
                    self._save_state(datetime.now(), self.action_type)
                    logger.info(f"Saved successful {self.action_type} state")
                
                logger.info("[OK] Automation completed successfully")
                return True
                
            except Exception as e:
                if attempt < MAX_RETRIES:
                    time.sleep(RETRY_DELAY)
                else:
                    logger.error("[KO] Automation failed after all retries: %s", str(e))
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
