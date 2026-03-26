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
        print("✓ Python version is compatible\n")
        return True
    else:
        print("✗ Python 3.8 or higher is required")
        print("  Download from: https://www.python.org/\n")
        return False


def install_dependencies():
    """Install Python dependencies."""
    print_header("Installing Dependencies")
    
    try:
        subprocess.check_call([sys.executable, "-m", "pip", "install", "-r", "requirements.txt"])
        print("\n✓ Dependencies installed successfully\n")
        return True
    except subprocess.CalledProcessError as e:
        print(f"\n✗ Failed to install dependencies: {e}\n")
        return False


def setup_credentials():
    """Setup credentials as environment variables."""
    print_header("Setting Up Credentials")
    
    print("Your credentials will be stored as Windows environment variables.")
    print("They will be available for this script to use.\n")
    
    username = input("Enter your SopraGP4U username: ").strip()
    
    if not username:
        print("✗ Username cannot be empty\n")
        return False
    
    password = input("Enter your SopraGP4U password: ").strip()
    
    if not password:
        print("✗ Password cannot be empty\n")
        return False
    
    # Set environment variables
    os.environ['SOPRA_USERNAME'] = username
    os.environ['SOPRA_PASSWORD'] = password
    
    print("\n✓ Credentials configured\n")
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
        print(f"✗ Test failed: {e}\n")
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
            print(f"✗ Inspector failed: {e}\n")
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
            print(f"✗ Test failed: {e}\n")
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
                print(f"⚠ {step_name} did not complete successfully")
                response = input("Continue anyway? (y/n): ").strip().lower()
                if response != 'y':
                    break
        except KeyboardInterrupt:
            print("\n\n✗ Setup interrupted by user")
            sys.exit(1)
    
    # Summary
    print_header("Setup Complete!")
    
    print(f"Completed {completed}/{len(steps)} setup steps\n")
    
    if completed == len(steps):
        print("✓ All setup steps completed successfully!")
        print("\nYour automation is ready to use.")
        print("\nNext steps:")
        print("1. Review config/config.py for any adjustments")
        print("2. Set up Windows Task Scheduler (see README.md for instructions)")
        print("3. Test before deploying to production")
    else:
        print("⚠ Some setup steps were skipped or failed.")
        print("\nReview the output above and fix any issues.")
    
    print("\nFor help, see: README.md\n")


if __name__ == "__main__":
    main()
