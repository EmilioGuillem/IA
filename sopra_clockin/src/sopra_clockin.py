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
    HEADLESS_MODE, CHROME_OPTIONS
)

try:
    from selenium import webdriver
    from selenium.webdriver.common.by import By
    from selenium.webdriver.support.ui import WebDriverWait
    from selenium.webdriver.support import expected_conditions as EC
    from selenium.webdriver.chrome.service import Service
    from selenium.webdriver.chrome.options import Options
except ImportError:
    print("ERROR: Selenium is not installed. Install it with: pip install selenium")
    sys.exit(1)

# Initialize logger
logger = setup_logger(__name__)


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
        Initialize and configure the Chrome WebDriver.
        
        Returns:
            bool: True if successful, False otherwise
        """
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
            logger.error(f"Failed to setup WebDriver: {str(e)}", exc_info=True)
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
    
    def login(self):
        """
        Authenticate to the SopraGP4U portal.
        
        Returns:
            bool: True if successful, False otherwise
        """
        try:
            if not SOPRA_USERNAME or not SOPRA_PASSWORD:
                logger.warning("Credentials not provided via environment variables")
                logger.info("Credentials should be set in environment variables: SOPRA_USERNAME, SOPRA_PASSWORD")
                return False
            
            logger.info("Attempting to login...")
            
            # Look for login form elements
            # NOTE: Update selectors based on actual portal HTML structure
            username_field = self.wait.until(
                EC.presence_of_element_located((By.ID, "username")),
                message="Username field not found"
            )
            
            password_field = self.driver.find_element(By.ID, "password")
            login_button = self.driver.find_element(By.ID, "login-button")
            
            # Enter credentials
            username_field.clear()
            username_field.send_keys(SOPRA_USERNAME)
            password_field.clear()
            password_field.send_keys(SOPRA_PASSWORD)
            
            # Click login
            login_button.click()
            time.sleep(3)  # Wait for authentication
            
            logger.info("Login successful")
            return True
            
        except Exception as e:
            logger.error(f"Login failed: {str(e)}", exc_info=True)
            return False
    
    def click_menu_link(self):
        """
        Click on the 'Registro de entrada/salida' menu link.
        
        Returns:
            bool: True if successful, False otherwise
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
            logger.error(f"Failed to click menu link: {str(e)}", exc_info=True)
            # Try alternative selectors
            return self._try_alternative_menu_links()
    
    def _try_alternative_menu_links(self):
        """
        Try alternative methods to locate and click the menu link.
        
        Returns:
            bool: True if successful, False otherwise
        """
        try:
            logger.info("Trying alternative selectors for menu link...")
            
            # Try partial link text
            menu_link = self.wait.until(
                EC.element_to_be_clickable((By.PARTIAL_LINK_TEXT, "Registro")),
                message="Menu link with partial text 'Registro' not found"
            )
            
            if DRY_RUN:
                logger.info("[DRY RUN] Would click on alternative menu link")
            else:
                menu_link.click()
                time.sleep(2)
                logger.info("Successfully clicked on alternative menu link")
            
            return True
            
        except Exception as e:
            logger.error(f"Failed to find menu link with alternative selectors: {str(e)}", exc_info=True)
            return False
    
    def click_clock_button(self):
        """
        Click the appropriate clock in/out button based on current time.
        
        Returns:
            bool: True if successful, False otherwise
        """
        if self.action_type == 'NONE':
            logger.info(f"Current time is {self.current_hour}:00 - outside clock-in/out windows")
            return True
        
        button_id = CLOCK_IN_BUTTON_ID if self.action_type == 'CLOCK_IN' else CLOCK_OUT_BUTTON_ID
        
        try:
            logger.info(f"Looking for {self.action_type} button...")
            
            button = self.wait.until(
                EC.element_to_be_clickable((By.ID, button_id)),
                message=f"Button '{button_id}' not found"
            )
            
            if DRY_RUN:
                logger.info(f"[DRY RUN] Would click {self.action_type} button")
            else:
                button.click()
                time.sleep(2)  # Wait for action to process
                logger.info(f"Successfully clicked {self.action_type} button")
            
            return True
            
        except Exception as e:
            logger.error(f"Failed to click {self.action_type} button: {str(e)}", exc_info=True)
            # Try alternative selectors
            return self._try_alternative_clock_button()
    
    def _try_alternative_clock_button(self):
        """
        Try alternative methods to locate and click the clock button.
        
        Returns:
            bool: True if successful, False otherwise
        """
        try:
            logger.info("Trying alternative selectors for clock button...")
            
            button_text = "CLOCK-IN" if self.action_type == 'CLOCK_IN' else "CLOCK-OUT"
            
            # Try button by text
            button = self.wait.until(
                EC.element_to_be_clickable((By.XPATH, f"//button[contains(text(), '{button_text}')]")),
                message=f"Button with text '{button_text}' not found"
            )
            
            if DRY_RUN:
                logger.info(f"[DRY RUN] Would click alternative clock button")
            else:
                button.click()
                time.sleep(2)
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
                
                # Login (if credentials provided)
                if SOPRA_USERNAME and SOPRA_PASSWORD:
                    if not self.login():
                        raise Exception("Login failed")
                else:
                    logger.warning("Skipping login - credentials not provided")
                
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
