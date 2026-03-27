#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Portal Element Inspector Tool
=====================================================

This tool helps identify HTML element selectors in the SopraGP4U portal.

It opens the portal in a VISIBLE browser window so you can inspect elements
and helps you find the correct CSS selectors/IDs to use in config.py

Usage:
    python src/inspect_portal_elements.py
"""

import sys
import time
import json
from pathlib import Path
from datetime import datetime

sys.path.insert(0, str(Path(__file__).parent.parent))

from logger_config import setup_logger
from config.config import SOPRA_URL, BROWSER

logger = setup_logger(__name__)

try:
    from selenium import webdriver
    from selenium.webdriver.chrome.options import Options
    from selenium.webdriver.edge.options import Options as EdgeOptions
    from selenium.webdriver.common.by import By
    from selenium.webdriver.support.ui import WebDriverWait
    from selenium.webdriver.support import expected_conditions as EC
except ImportError:
    logger.error("Selenium is not installed. Run: pip install -r requirements.txt")
    sys.exit(1)


def inspect_portal():
    """Open portal in visible browser for inspection."""
    
    logger.info("="*80)
    logger.info("SopraGP4U Portal Element Inspector")
    logger.info("="*80)
    logger.info("\nThis tool will open the portal in a visible browser window.")
    logger.info("You can inspect elements and save element information.\n")
    
    driver = None
    try:
        # Ask user to select browser if not already set
        browser = BROWSER
        if not browser or browser not in ["chrome", "edge"]:
            logger.info("\n" + "="*80)
            logger.info("SELECT BROWSER")
            logger.info("="*80)
            print("\nWhich browser would you like to use?")
            print("  1. Chrome (default)")
            print("  2. Edge")
            choice = input("\nEnter choice (1 or 2) [1]: ").strip() or "1"
            browser = "edge" if choice == "2" else "chrome"
        
        # Setup browser
        options = None
        driver_class = None
        
        if browser == "edge":
            logger.info("Opening Edge browser...")
            options = EdgeOptions()
            driver_class = webdriver.Edge
        else:
            logger.info("Opening Chrome browser...")
            options = Options()
            driver_class = webdriver.Chrome
        
        # Configure options
        options.add_argument("--window-size=1920,1080")
        if browser == "chrome":
            options.add_argument("--disable-blink-features=AutomationControlled")
        options.add_experimental_option("excludeSwitches", ["enable-automation"])
        options.add_experimental_option('useAutomationExtension', False)
        
        # Initialize driver
        driver = driver_class(options=options)
        driver.set_page_load_timeout(30)
        
        logger.info(f"Navigating to {SOPRA_URL}")
        driver.get(SOPRA_URL)
        time.sleep(2)
        
        # Check if login is required
        logger.info("Checking if authentication is required...")
        needs_login = check_if_login_required(driver)
        
        if needs_login:
            logger.info("Login form detected. Requesting credentials...")
            if not handle_login(driver):
                logger.error("Login failed. Cannot continue.")
                sys.exit(1)
            logger.info("Login successful!")
        else:
            logger.info("Portal accessible without authentication")
        
        logger.info("\n" + "="*80)
        logger.info("INSTRUCTIONS:")
        logger.info("="*80)
        logger.info("""
1. [OK] Browser is open with SopraGP4U portal

2. [OK] Authentication completed (if required)

3. [OK] You're now at the clock in/out page (no menu navigation needed)

4. IDENTIFY THE BUTTONS:
   
   a) Right-click on the CLOCK-IN button -> "Inspect" (or press F12)
   b) Note the element information:
      - ID (e.g., id="CLOCK-IN")
      - Class (e.g., class="btn btn-primary")
      - Data attributes (e.g., data-action="clock-in")
   
   c) Do the same for CLOCK-OUT button

5. When you've identified all elements, press Enter in this console
   to continue to the next step

6. You'll be asked to enter the information you found

7. The configuration will be saved to config/config.py
        """)
        
        logger.info("\nPress Enter when you've found all the element information...")
        input()
        
        # Collect information from user
        logger.info("\n" + "="*80)
        logger.info("ELEMENT INFORMATION COLLECTION")
        logger.info("="*80)
        
        elements_info = {}
        
        # Clock-In button
        logger.info("\n--- CLOCK-IN Button ---")
        logger.info("How to identify this button? (examples below)")
        logger.info("  • by ID: id='CLOCK-IN'")
        logger.info("  • by text: //button[contains(text(), 'CLOCK-IN')]")
        logger.info("  • by class: .btn-clock-in")
        
        clock_in_method = input("Identification method (id/xpath/class/css): ").strip().lower()
        clock_in_value = input("Value: ").strip()
        
        if not clock_in_method or not clock_in_value:
            logger.error("CLOCK-IN method and value are required")
            return False
            
        elements_info['clock_in'] = {'method': clock_in_method, 'value': clock_in_value}
        
        # Clock-Out button
        logger.info("\n--- CLOCK-OUT Button ---")
        clock_out_method = input("Identification method (id/xpath/class/css): ").strip().lower()
        clock_out_value = input("Value: ").strip()
        
        if not clock_out_method or not clock_out_value:
            logger.error("CLOCK-OUT method and value are required")
            return False
            
        elements_info['clock_out'] = {'method': clock_out_method, 'value': clock_out_value}
        
        # Save results to config.json
        logger.info("\n" + "="*80)
        logger.info("SAVING CONFIGURATION")
        logger.info("="*80)
        
        config_file = Path(__file__).parent.parent / "config" / "config.json"
        config_data = {}
        if config_file.exists():
            try:
                with open(config_file, 'r', encoding='utf-8') as f:
                    config_data = json.load(f)
            except Exception as e:
                logger.warning(f"Could not load existing config: {e}")
        
        # Update with new selectors
        config_data['clock_in'] = elements_info['clock_in']
        config_data['clock_out'] = elements_info['clock_out']
        
        try:
            config_file.parent.mkdir(exist_ok=True)
            with open(config_file, 'w', encoding='utf-8') as f:
                json.dump(config_data, f, indent=2, ensure_ascii=False)
            logger.info(f"\n[OK] Configuration saved to: {config_file}")
        except Exception as e:
            logger.error(f"Failed to save configuration: {str(e)}")
            return False
        
        # Display saved configuration
        logger.info("\n" + "="*80)
        logger.info("CONFIGURATION SUMMARY:")
        logger.info("="*80)
        
        for element, info in elements_info.items():
            method = info['method']
            value = info['value']
            logger.info(f"{element.upper()}: {method} = '{value}'")
        
        logger.info("\n" + "="*80)
        logger.info("NEXT STEPS:")
        logger.info("="*80)
        logger.info("""
1. Configuration automatically saved to config/config.json

2. Run: python src/test_setup.py (to verify)

3. Run: python src/sopra_clockin.py (to test)
        """)
        
        logger.info("Press Enter to close browser and exit...")
        input()
        
        return True
        
    except Exception as e:
        logger.error(f"Error: {str(e)}", exc_info=True)
        return False
    finally:
        if driver:
            try:
                driver.quit()
            except:
                pass


def check_if_login_required(driver):
    """
    Check if the portal requires authentication.
    
    Returns:
        bool: True if login form is found, False otherwise
    """
    try:
        # Look for common login form indicators
        login_indicators = [
            (By.ID, "username"),
            (By.ID, "user"),
            (By.ID, "login"),
            (By.NAME, "username"),
            (By.NAME, "user"),
            (By.CSS_SELECTOR, "input[type='password']"),
            (By.XPATH, "//input[@type='password']"),
        ]
        
        for locator in login_indicators:
            try:
                driver.find_element(*locator)
                logger.info(f"Login form detected using {locator}")
                return True
            except:
                continue
        
        return False
    except Exception as e:
        logger.warning(f"Error checking login requirement: {str(e)}")
        return False


def handle_login(driver):
    """
    Handle portal authentication if required.
    
    Returns:
        bool: True if login successful, False otherwise
    """
    try:
        # Ask user for credentials
        logger.info("\n" + "="*80)
        logger.info("AUTHENTICATION REQUIRED")
        logger.info("="*80)
        
        username = input("\nEnter username: ").strip()
        if not username:
            logger.error("Username cannot be empty")
            return False
        
        password = input("Enter password: ").strip()
        if not password:
            logger.error("Password cannot be empty")
            return False
        
        # Try to find and fill username field
        username_field = None
        username_locators = [
            (By.ID, "username"),
            (By.ID, "user"),
            (By.NAME, "username"),
            (By.NAME, "user"),
        ]
        
        for locator in username_locators:
            try:
                username_field = driver.find_element(*locator)
                break
            except:
                continue
        
        if not username_field:
            logger.error("Username field not found")
            return False
        
        # Try to find and fill password field
        password_field = None
        password_locators = [
            (By.ID, "password"),
            (By.NAME, "password"),
            (By.CSS_SELECTOR, "input[type='password']"),
        ]
        
        for locator in password_locators:
            try:
                password_field = driver.find_element(*locator)
                break
            except:
                continue
        
        if not password_field:
            logger.error("Password field not found")
            return False
        
        # Fill in credentials
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
            (By.NAME, "login"),
            (By.XPATH, "//button[contains(text(), 'Login')]"),
            (By.XPATH, "//button[contains(text(), 'Sign in')]"),
            (By.XPATH, "//button[@type='submit']"),
        ]
        
        for locator in login_locators:
            try:
                login_button = driver.find_element(*locator)
                break
            except:
                continue
        
        if not login_button:
            logger.error("Login button not found")
            return False
        
        # Click login and wait
        login_button.click()
        time.sleep(3)  # Wait for authentication to complete
        
        logger.info("Login credentials submitted successfully")
        return True
        
    except Exception as e:
        logger.error(f"Login failed: {str(e)}", exc_info=True)
        return False


if __name__ == "__main__":
    inspect_portal()
