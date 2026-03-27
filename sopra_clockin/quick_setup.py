#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Quick Setup Wizard for SopraGP4U Clock In/Out Automation
=====================================================

This script provides an interactive setup wizard to configure everything
at once without needing separate commands.

Usage:
    python quick_setup.py
"""

import sys
import os
import json
import subprocess
from pathlib import Path

def print_header(text):
    """Print formatted header."""
    print("\n" + "="*60)
    print(f"  {text}")
    print("="*60 + "\n")


def check_python():
    """Verify Python version."""
    print_header("Checking Python Installation")
    
    version = sys.version_info
    print(f"Python version: {version.major}.{version.minor}.{version.micro}")
    
    if version.major >= 3 and version.minor >= 8:
        print("[OK] Python version is compatible\n")
        return True
    else:
        print("[ERROR] Python 3.8 or higher is required")
        print("  Download from: https://www.python.org/\n")
        return False


def install_dependencies():
    """Install Python dependencies."""
    print_header("Installing Dependencies")
    
    try:
        subprocess.check_call([sys.executable, "-m", "pip", "install", "-r", "requirements.txt"])
        print("\n[OK] Dependencies installed successfully\n")
        return True
    except subprocess.CalledProcessError as e:
        print(f"\n[ERROR] Failed to install dependencies: {e}\n")
        return False


def setup_credentials():
    """Setup credentials as environment variables."""
    print_header("Setting Up Credentials")
    
    print("Your credentials will be stored in the configuration file.")
    print("They will be available for this script to use.\n")
    
    username = input("Enter your SopraGP4U username: ").strip()
    
    if not username:
        print("[ERROR] Username cannot be empty\n")
        return False
    
    password = input("Enter your SopraGP4U password: ").strip()
    
    if not password:
        print("[ERROR] Password cannot be empty\n")
        return False
    
    # Save to config file
    config_file = Path("config/config.json")
    config_data = {}
    if config_file.exists():
        try:
            with open(config_file, 'r', encoding='utf-8') as f:
                config_data = json.load(f)
        except:
            pass
    
    config_data['username'] = username
    config_data['password'] = password
    
    try:
        config_file.parent.mkdir(exist_ok=True)
        with open(config_file, 'w', encoding='utf-8') as f:
            json.dump(config_data, f, indent=2)
        print("\n[OK] Credentials configured and saved")
    except Exception as e:
        print(f"\n[WARNING] Could not save config: {e}")
    
    return True


def setup_browser():
    """Setup browser preference."""
    print_header("Select Browser")
    
    print("Which browser would you like to use for automation?\n")
    print("  1. Chrome (default)")
    print("  2. Edge\n")
    
    choice = input("Enter your choice (1 or 2): ").strip()
    
    if choice == "2":
        browser = "edge"
        print("\n[OK] Edge browser selected")
    else:
        browser = "chrome"
        print("\n[OK] Chrome browser selected")
    
    # Save to config file
    config_file = Path("config/config.json")
    config_data = {}
    if config_file.exists():
        try:
            with open(config_file, 'r', encoding='utf-8') as f:
                config_data = json.load(f)
        except:
            pass
    
    config_data['browser'] = browser
    
    try:
        config_file.parent.mkdir(exist_ok=True)
        with open(config_file, 'w', encoding='utf-8') as f:
            json.dump(config_data, f, indent=2)
        print(f"[OK] Configuration saved to {config_file}")
    except Exception as e:
        print(f"[WARNING] Could not save config: {e}")
    
    print(f"  Browser set to '{browser}'\n")
    return True


def test_setup():
    """Run setup tests."""
    print_header("Running Setup Tests")
    
    try:
        result = subprocess.run(
            [sys.executable, "src/test_setup.py"],
            capture_output=False,
            text=True
        )
        return result.returncode == 0
    except Exception as e:
        print(f"[ERROR] Test failed: {e}\n")
        return False


def configure_portal_elements():
    """Guide user to configure portal elements."""
    print_header("Configuring Portal Elements")
    
    print("Next, we'll inspect the SopraGP4U portal to find element selectors.")
    print("This is required for the automation to work correctly.\n")
    
    response = input("Do you want to run the element inspector now? (y/n): ").strip().lower()
    
    if response == 'y':
        try:
            subprocess.check_call([sys.executable, "src/inspect_portal_elements.py"])
            return True
        except Exception as e:
            print(f"[ERROR] Inspector failed: {e}\n")
            return False
    else:
        print("\nYou can run it later with:")
        print("  python src/inspect_portal_elements.py\n")
        return True


def test_automation():
    """Test the automation script."""
    print_header("Testing Automation Script")
    
    response = input("Do you want to test the automation script now? (y/n): ").strip().lower()
    
    if response == 'y':
        try:
            result = subprocess.run(
                [sys.executable, "src/sopra_clockin.py"],
                capture_output=False,
                text=True
            )
            return result.returncode == 0
        except Exception as e:
            print(f"[ERROR] Test failed: {e}\n")
            return False
    else:
        print("\nYou can run it later with:")
        print("  python src/sopra_clockin.py\n")
        return True


def main():
    """Run setup wizard."""
    
    print("\n")
    print(" "*15 + "SopraGP4U Automation - Quick Setup Wizard")
    print()
    
    steps = [
        ("Check Python", check_python),
        ("Install Dependencies", install_dependencies),
        ("Setup Credentials", setup_credentials),
        ("Select Browser", setup_browser),
        ("Run Setup Tests", test_setup),
        ("Configure Portal Elements", configure_portal_elements),
        ("Test Automation", test_automation),
    ]
    
    completed = 0
    
    for step_name, step_func in steps:
        try:
            if step_func():
                completed += 1
            else:
                print(f"[WARNING] {step_name} did not complete successfully")
                response = input("Continue anyway? (y/n): ").strip().lower()
                if response != 'y':
                    break
        except KeyboardInterrupt:
            print("\n\n[ERROR] Setup interrupted by user")
            sys.exit(1)
    
    # Summary
    print_header("Setup Complete!")
    
    print(f"Completed {completed}/{len(steps)} setup steps\n")
    
    if completed == len(steps):
        print("[OK] All setup steps completed successfully!")
        print("\nYour automation is ready to use.")
        print("\nNext steps:")
        print("1. Review config/config.py for any adjustments")
        print("2. Set up Windows Task Scheduler (see README.md for instructions)")
        print("3. Test before deploying to production")
    else:
        print("[WARNING] Some setup steps were skipped or failed.")
        print("\nReview the output above and fix any issues.")
    
    print("\nFor help, see: README.md\n")


if __name__ == "__main__":
    main()
