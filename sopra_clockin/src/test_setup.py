#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Test script to verify Selenium setup and portal connectivity
=====================================================

This script tests:
1. Selenium WebDriver installation
2. Chrome browser availability
3. Network connectivity to SopraGP4U
4. Portal page loading

Run this before using the main automation script.
"""

import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from logger_config import setup_logger
from config.config import SOPRA_URL, WAIT_TIMEOUT

logger = setup_logger(__name__)

def test_imports():
    """Test if required Python packages are installed."""
    logger.info("Testing Python imports...")
    
    try:
        import selenium
        logger.info(f"✓ Selenium {selenium.__version__} is installed")
    except ImportError:
        logger.error("✗ Selenium is not installed")
        return False
    
    try:
        from selenium import webdriver
        logger.info("✓ Selenium WebDriver module found")
    except ImportError:
        logger.error("✗ Selenium WebDriver module not found")
        return False
    
    try:
        import webdriver_manager
        logger.info(f"✓ WebDriver Manager is installed")
    except ImportError:
        logger.warning("⚠ WebDriver Manager is not installed (optional)")
    
    return True


def test_chrome_driver():
    """Test if Chrome driver can be initialized."""
    logger.info("\nTesting Chrome WebDriver initialization...")
    
    try:
        from selenium import webdriver
        from selenium.webdriver.chrome.options import Options
        
        options = Options()
        options.add_argument("--headless")
        options.add_argument("--no-sandbox")
        
        driver = webdriver.Chrome(options=options)
        version = driver.execute_script("return navigator.chromeVersion")
        logger.info(f"✓ Chrome WebDriver initialized successfully")
        logger.info(f"✓ Chrome version: {version}")
        
        driver.quit()
        return True
        
    except Exception as e:
        logger.error(f"✗ Chrome WebDriver initialization failed: {str(e)}")
        logger.info("  Make sure Google Chrome is installed")
        return False


def test_portal_connectivity():
    """Test connection to SopraGP4U portal."""
    logger.info(f"\nTesting connectivity to {SOPRA_URL}")
    
    try:
        import urllib.request
        
        response = urllib.request.urlopen(SOPRA_URL, timeout=10)
        logger.info(f"✓ Portal is reachable (HTTP {response.status})")
        return True
        
    except Exception as e:
        logger.error(f"✗ Portal connection failed: {str(e)}")
        logger.info("  Check your internet connection or VPN")
        return False


def test_full_navigation():
    """Test full navigation with Selenium."""
    logger.info(f"\nTesting full portal navigation...")
    
    try:
        from selenium import webdriver
        from selenium.webdriver.chrome.options import Options
        from selenium.webdriver.common.by import By
        from selenium.webdriver.support.ui import WebDriverWait
        from selenium.webdriver.support import expected_conditions as EC
        
        options = Options()
        options.add_argument("--headless")
        options.add_argument("--no-sandbox")
        options.add_argument("--disable-dev-shm-usage")
        
        driver = webdriver.Chrome(options=options)
        driver.set_page_load_timeout(30)
        
        logger.info(f"  → Navigating to {SOPRA_URL}")
        driver.get(SOPRA_URL)
        
        logger.info("  → Waiting for page to load...")
        wait = WebDriverWait(driver, WAIT_TIMEOUT)
        
        # Check if page title suggests login or main portal
        title = driver.title
        logger.info(f"  → Page title: '{title}'")
        
        # Try to find common portal elements
        try:
            # Look for body element to confirm page loaded
            body = wait.until(EC.presence_of_element_located((By.TAG_NAME, "body")))
            logger.info("✓ Portal page loaded successfully")
            
            # Check page source for key indicators
            page_source = driver.page_source.lower()
            
            if "login" in page_source or "autentication" in page_source:
                logger.info("  → Login page detected (expected)")
            
            if "sopra" in page_source:
                logger.info("  → SopraGP4U content detected")
            
        except Exception as e:
            logger.warning(f"  ⚠ Could not verify page elements: {str(e)}")
        
        driver.quit()
        return True
        
    except Exception as e:
        logger.error(f"✗ Full navigation test failed: {str(e)}")
        return False


def main():
    """Run all tests."""
    logger.info("="*80)
    logger.info("SopraGP4U Automation - Connectivity Test")
    logger.info("="*80)
    
    results = {
        "Imports": test_imports(),
        "Chrome Driver": test_chrome_driver(),
        "Portal Connectivity": test_portal_connectivity(),
        "Full Navigation": test_full_navigation(),
    }
    
    logger.info("\n" + "="*80)
    logger.info("Test Results Summary")
    logger.info("="*80)
    
    passed = sum(1 for v in results.values() if v)
    total = len(results)
    
    for test_name, result in results.items():
        status = "✓ PASS" if result else "✗ FAIL"
        logger.info(f"{test_name}: {status}")
    
    logger.info(f"\nTotal: {passed}/{total} tests passed")
    
    if passed == total:
        logger.info("\n✓ All tests passed! You can now run the main automation script.")
        return 0
    else:
        logger.warning("\n⚠ Some tests failed. Fix the issues above before running automation.")
        return 1


if __name__ == "__main__":
    sys.exit(main())
