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
from config.config import SOPRA_URL

logger = setup_logger(__name__)

try:
    from selenium import webdriver
    from selenium.webdriver.chrome.options import Options
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
    
    try:
        # Setup Chrome with NO headless mode (visible window)
        options = Options()
        options.add_argument("--window-size=1920,1080")
        # options.add_argument("--start-maximized")  # Uncomment for full screen
        
        logger.info("Opening Chrome browser...")
        driver = webdriver.Chrome(options=options)
        
        logger.info(f"Navigating to {SOPRA_URL}")
        driver.get(SOPRA_URL)
        
        logger.info("\n" + "="*80)
        logger.info("INSTRUCTIONS:")
        logger.info("="*80)
        logger.info("""
1. The browser window is now open with the SopraGP4U portal

2. LOGIN if needed with your credentials

3. NAVIGATE to "Registro de entrada/salida" menu

4. Once you see the buttons, DO THE FOLLOWING:
   
   a) Right-click on the CLOCK-IN button → "Inspect" (or press F12)
   b) Note the element information:
      - ID (e.g., id="CLOCK-IN")
      - Class (e.g., class="btn btn-primary")
      - Data attributes (e.g., data-action="clock-in")
   
   c) Do the same for CLOCK-OUT button
   
   d) Right-click on the menu link "Registro de entrada/salida" → "Inspect"
      Note how to identify it (text, class, id, etc.)

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
        elements_info['clock_in'] = {'method': clock_in_method, 'value': clock_in_value}
        
        # Clock-Out button
        logger.info("\n--- CLOCK-OUT Button ---")
        clock_out_method = input("Identification method (id/xpath/class/css): ").strip().lower()
        clock_out_value = input("Value: ").strip()
        elements_info['clock_out'] = {'method': clock_out_method, 'value': clock_out_value}
        
        # Menu link
        logger.info("\n--- Menu Link: 'Registro de entrada/salida' ---")
        menu_method = input("Identification method (link_text/partial_link_text/xpath/class/css): ").strip().lower()
        menu_value = input("Value: ").strip()
        elements_info['menu'] = {'method': menu_method, 'value': menu_value}
        
        # Login form (if provided)
        logger.info("\n--- Login Form Elements (Optional) ---")
        use_login = input("Did you need to login? (y/n): ").strip().lower()
        
        if use_login == 'y':
            logger.info("Username field:")
            username_method = input("  Identification method (id/name/xpath/css): ").strip().lower()
            username_value = input("  Value: ").strip()
            elements_info['username'] = {'method': username_method, 'value': username_value}
            
            logger.info("Password field:")
            password_method = input("  Identification method (id/name/xpath/css): ").strip().lower()
            password_value = input("  Value: ").strip()
            elements_info['password'] = {'method': password_method, 'value': password_value}
            
            logger.info("Login button:")
            login_method = input("  Identification method (id/xpath/css): ").strip().lower()
            login_value = input("  Value: ").strip()
            elements_info['login_button'] = {'method': login_method, 'value': login_value}
        
        # Save results
        logger.info("\n" + "="*80)
        logger.info("SAVING RESULTS")
        logger.info("="*80)
        
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        report_file = Path(__file__).parent.parent / "logs" / f"portal_inspection_{timestamp}.json"
        
        with open(report_file, 'w', encoding='utf-8') as f:
            json.dump(elements_info, f, indent=2, ensure_ascii=False)
        
        logger.info(f"\n✓ Inspection report saved to:")
        logger.info(f"  {report_file}")
        
        # Display configuration suggestions
        logger.info("\n" + "="*80)
        logger.info("UPDATE config/config.py WITH THESE SETTINGS:")
        logger.info("="*80)
        
        for element, info in elements_info.items():
            method = info['method']
            value = info['value']
            
            if method == 'id':
                logger.info(f"\n# {element.upper()}")
                logger.info(f"Element selector: (By.ID, \"{value}\")")
            elif method == 'xpath':
                logger.info(f"\n# {element.upper()}")
                logger.info(f"Element selector: (By.XPATH, \"{value}\")")
            elif method == 'class' or method == 'css':
                logger.info(f"\n# {element.upper()}")
                logger.info(f"Element selector: (By.CSS_SELECTOR, \"{value}\")")
            elif method == 'link_text':
                logger.info(f"\n# {element.upper()}")
                logger.info(f"Element selector: (By.LINK_TEXT, \"{value}\")")
            elif method == 'partial_link_text':
                logger.info(f"\n# {element.upper()}")
                logger.info(f"Element selector: (By.PARTIAL_LINK_TEXT, \"{value}\")")
            elif method == 'name':
                logger.info(f"\n# {element.upper()}")
                logger.info(f"Element selector: (By.NAME, \"{value}\")")
        
        logger.info("\n" + "="*80)
        logger.info("NEXT STEPS:")
        logger.info("="*80)
        logger.info("""
1. Edit: config/config.py

2. Find the lines with element selectors (e.g., By.ID, By.XPATH, etc.)

3. Update them with the values you found

4. Save the file

5. Run: python src/test_setup.py (to verify)

6. Run: python src/sopra_clockin.py (to test)
        """)
        
        logger.info("Press Enter to close browser and exit...")
        input()
        
        driver.quit()
        
    except Exception as e:
        logger.error(f"Error: {str(e)}", exc_info=True)
        sys.exit(1)


if __name__ == "__main__":
    inspect_portal()
