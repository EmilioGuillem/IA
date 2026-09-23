#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# This file has been created (totally or partially) with the assistance of
# artificial intelligence tools. All content has been generated under the
# direct supervision of a named individual and the AI.Backbone Orchestrator
# Compliance framework.

"""
Test script to verify Selenium setup and portal connectivity
=====================================================

This script tests:
1. Selenium WebDriver installation
2. Configured browser availability
3. Network connectivity to SopraGP4U
4. Portal page loading

Run this before using the main automation script.
"""

import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
sys.path.insert(0, str(Path(__file__).parent.parent))

from logger_config import setup_logger
from config.config import SOPRA_URL, WAIT_TIMEOUT, BROWSER

logger = setup_logger(__name__)

def test_imports():
    """Test if required Python packages are installed."""
    logger.info("Testing Python imports...")
    
    try:
        import selenium
        logger.info(f"[OK] Selenium {selenium.__version__} is installed")
    except ImportError:
        logger.error("[ERROR] Selenium is not installed")
        return False
    
    try:
        from selenium import webdriver
        logger.info("[OK] Selenium WebDriver module found")
    except ImportError:
        logger.error("[ERROR] Selenium WebDriver module not found")
        return False
    
    try:
        import webdriver_manager
        logger.info(f"[OK] WebDriver Manager is installed")
    except ImportError:
        logger.warning("[WARNING] WebDriver Manager is not installed (optional)")
    
    return True


def test_browser_driver():
    """Test if the configured browser driver can be initialized."""
    logger.info(f"\nTesting {BROWSER.upper()} WebDriver initialization...")
    
    try:
        from selenium import webdriver
        from selenium.webdriver.chrome.options import Options
        from selenium.webdriver.edge.options import Options as EdgeOptions
        
        if BROWSER == "edge":
            options = EdgeOptions()
            driver_factory = webdriver.Edge
        else:
            options = Options()
            driver_factory = webdriver.Chrome

        options.add_argument("--headless")
        options.add_argument("--no-sandbox")
        
        driver = driver_factory(options=options)
        user_agent = driver.execute_script("return navigator.userAgent")
        logger.info(f"[OK] {BROWSER.upper()} WebDriver initialized successfully")
        logger.info(f"[OK] Browser user agent: {user_agent}")
        
        driver.quit()
        return True
        
    except Exception as e:
        logger.error(f"[ERROR] {BROWSER.upper()} WebDriver initialization failed: {str(e)}")
        logger.info(f"  Make sure {BROWSER} is installed")
        return False


def test_portal_connectivity():
    """Test connection to SopraGP4U portal."""
    logger.info(f"\nTesting connectivity to {SOPRA_URL}")
    
    try:
        import urllib.request
        
        response = urllib.request.urlopen(SOPRA_URL, timeout=10)
        logger.info(f"[OK] Portal is reachable (HTTP {response.status})")
        return True
        
    except Exception as e:
        logger.error(f"[ERROR] Portal connection failed: {str(e)}")
        logger.info("  Check your internet connection or VPN")
        return False


def test_full_navigation():
    """Test full navigation with Selenium."""
    logger.info(f"\nTesting full portal navigation...")
    
    try:
        from selenium import webdriver
        from selenium.webdriver.chrome.options import Options
        from selenium.webdriver.edge.options import Options as EdgeOptions
        from selenium.webdriver.common.by import By
        from selenium.webdriver.support.ui import WebDriverWait
        from selenium.webdriver.support import expected_conditions as EC
        
        if BROWSER == "edge":
            options = EdgeOptions()
            driver_factory = webdriver.Edge
        else:
            options = Options()
            driver_factory = webdriver.Chrome

        options.add_argument("--headless")
        options.add_argument("--no-sandbox")
        options.add_argument("--disable-dev-shm-usage")
        
        driver = driver_factory(options=options)
        driver.set_page_load_timeout(15)
        
        logger.info(f"  -> Navigating to {SOPRA_URL}")
        
        try:
            driver.get(SOPRA_URL)
            
            logger.info("  -> Waiting for page to load...")
            wait = WebDriverWait(driver, 10)
            
            # Check if page title suggests login or main portal
            title = driver.title
            logger.info(f"  -> Page title: '{title}'")
            
            # Try to find common portal elements
            try:
                # Look for body element to confirm page loaded
                body = wait.until(EC.presence_of_element_located((By.TAG_NAME, "body")))
                logger.info("[OK] Portal page loaded successfully")
                
                # Check page source for key indicators
                page_source = driver.page_source.lower()
                
                if "login" in page_source or "autentication" in page_source:
                    logger.info("  -> Login page detected (expected)")
                
                if "sopra" in page_source:
                    logger.info("  -> SopraGP4U content detected")
                
            except Exception as e:
                logger.warning(f"  [WARNING] Could not verify page elements: {str(e)}")
            
            driver.quit()
            return True
            
        except Exception as e:
            logger.warning(f"  [WARNING] Portal navigation failed: {str(e)}")
            logger.info("  -> Check VPN, network access, and portal availability")
            driver.quit()
            return False
        
    except Exception as e:
        logger.error(f"[ERROR] Full navigation test failed: {str(e)}")
        return False


def main():
    """Run all tests."""
    logger.info("="*80)
    logger.info("SopraGP4U Automation - Connectivity Test")
    logger.info("="*80)
    
    results = {
        "Imports": test_imports(),
        f"{BROWSER.upper()} Driver": test_browser_driver(),
        "Portal Connectivity": test_portal_connectivity(),
        "Full Navigation": test_full_navigation(),
    }
    
    logger.info("\n" + "="*80)
    logger.info("Test Results Summary")
    logger.info("="*80)
    
    passed = sum(1 for v in results.values() if v)
    total = len(results)
    
    for test_name, result in results.items():
        status = "[OK] PASS" if result else "[ERROR] FAIL"
        logger.info(f"{test_name}: {status}")
    
    logger.info(f"\nTotal: {passed}/{total} tests passed")
    
    if passed == total:
        logger.info("\n[OK] All tests passed! You can now run the main automation script.")
        return 0
    else:
        logger.warning("\n[WARNING] Some tests failed. Fix the issues above before running automation.")
        return 1


if __name__ == "__main__":
    sys.exit(main())
