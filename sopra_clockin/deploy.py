#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Automated Deployment Script for SopraGP4U Clock In/Out
=====================================================

This script fully automates the deployment process with minimal user input.

Requirements:
- Python 3.8+
- Admin PowerShell for Task Scheduler setup
- Valid SopraGP4U credentials

Usage:
    python deploy.py --headless
    python deploy.py --setup-tasks
    python deploy.py --verify
"""

import argparse
import sys
import subprocess
import os
import json
from datetime import datetime
from pathlib import Path

def run_command(cmd, description="", capture_output=False):
    """Run a shell command and return success status."""
    print(f"\n▶ {description}...")
    
    try:
        if capture_output:
            result = subprocess.run(cmd, capture_output=True, text=True, shell=True)
            if result.returncode == 0:
                print(f"  ✓ {description} completed")
                return True, result.stdout
            else:
                print(f"  ✗ {description} failed: {result.stderr}")
                return False, result.stderr
        else:
            result = subprocess.run(cmd, shell=True)
            if result.returncode == 0:
                print(f"  ✓ {description} completed")
                return True, ""
            else:
                print(f"  ✗ {description} failed")
                return False, ""
    except Exception as e:
        print(f"  ✗ Error: {str(e)}")
        return False, str(e)


def verify_environment():
    """Verify all required components are installed."""
    print("\n" + "="*60)
    print("VERIFYING ENVIRONMENT")
    print("="*60)
    
    checks = {
        "Python 3.8+": ("python --version", True),
        "Selenium": ("python -c 'import selenium'", True),
        "Chrome Browser": ("where chrome", False),
    }
    
    all_ok = True
    
    for check_name, (cmd, required) in checks.items():
        success, output = run_command(cmd, f"Checking {check_name}", capture_output=True)
        
        if success:
            print(f"  ✓ {check_name}: OK")
        else:
            if required:
                print(f"  ✗ {check_name}: MISSING (REQUIRED)")
                all_ok = False
            else:
                print(f"  ⚠ {check_name}: NOT FOUND (optional)")
    
    return all_ok


def install_dependencies():
    """Install Python dependencies."""
    print("\n" + "="*60)
    print("INSTALLING DEPENDENCIES")
    print("="*60)
    
    success, _ = run_command(
        f"{sys.executable} -m pip install -r requirements.txt",
        "Installing Python packages"
    )
    
    return success


def setup_credentials_interactive():
    """Setup credentials interactively."""
    print("\n" + "="*60)
    print("SETTING UP CREDENTIALS")
    print("="*60)
    
    print("\nEnter your SopraGP4U credentials:")
    print("(These will be stored as Windows environment variables)\n")
    
    username = input("Username: ").strip()
    password = input("Password: ").strip()
    
    if not username or not password:
        print("✗ Username and password are required")
        return False
    
    try:
        os.environ['SOPRA_USERNAME'] = username
        os.environ['SOPRA_PASSWORD'] = password
        
        # Also set system variables
        subprocess.run(
            f'setx SOPRA_USERNAME {username}',
            shell=True,
            capture_output=True
        )
        subprocess.run(
            f'setx SOPRA_PASSWORD {password}',
            shell=True,
            capture_output=True
        )
        
        print("✓ Credentials saved")
        return True
        
    except Exception as e:
        print(f"✗ Failed to save credentials: {e}")
        return False


def test_setup():
    """Run setup tests."""
    print("\n" + "="*60)
    print("RUNNING TESTS")
    print("="*60)
    
    success, output = run_command(
        f"{sys.executable} src/test_setup.py",
        "Running connectivity tests"
    )
    
    return success


def setup_scheduled_tasks():
    """Setup Windows Task Scheduler tasks."""
    print("\n" + "="*60)
    print("SETTING UP TASK SCHEDULER")
    print("="*60)
    
    print("\nNOTE: This requires administrator privileges")
    print("Continue with Task Scheduler setup? (y/n): ", end="")
    
    if input().lower() != 'y':
        print("Skipping Task Scheduler setup")
        print("You can set it up manually using:")
        print("  .\setup.ps1 -CreateScheduledTasks")
        return True
    
    project_dir = Path(__file__).parent.absolute()
    bat_file = project_dir / "scheduled_clockin.bat"
    
    # Create CLOCK-IN task
    powershell_cmd = f"""
    $action = New-ScheduledTaskAction -Execute '{bat_file}'
    $trigger = New-ScheduledTaskTrigger -Daily -At 08:00
    Register-ScheduledTask -TaskName 'SopraGP4U Clock In' -Action $action -Trigger $trigger -RunLevel Highest -Force
    """
    
    success, _ = run_command(
        f'powershell -Command "{powershell_cmd}"',
        "Creating CLOCK-IN task"
    )
    
    if success:
        # Create CLOCK-OUT task  
        powershell_cmd = f"""
        $action = New-ScheduledTaskAction -Execute '{bat_file}'
        $trigger = New-ScheduledTaskTrigger -Daily -At 17:15
        Register-ScheduledTask -TaskName 'SopraGP4U Clock Out' -Action $action -Trigger $trigger -RunLevel Highest -Force
        """
        
        success, _ = run_command(
            f'powershell -Command "{powershell_cmd}"',
            "Creating CLOCK-OUT task"
        )
    
    return success


def full_deployment():
    """Full deployment process."""
    print("\n")
    print(" "*15 + "SopraGP4U Automation - Full Deployment")
    print()
    
    steps = [
        ("VERIFY ENVIRONMENT", verify_environment),
        ("INSTALL DEPENDENCIES", install_dependencies),
        ("SETUP CREDENTIALS", setup_credentials_interactive),
        ("RUN TESTS", test_setup),
        ("SETUP TASK SCHEDULER", setup_scheduled_tasks),
    ]
    
    completed = 0
    
    for step_name, step_func in steps:
        try:
            if step_func():
                completed += 1
            else:
                response = input(f"\n{step_name} failed. Continue anyway? (y/n): ").lower()
                if response != 'y':
                    print("Deployment cancelled")
                    return False
        except KeyboardInterrupt:
            print("\n\nDeployment cancelled by user")
            return False
    
    return completed == len(steps)


def main():
    """Main entry point."""
    
    parser = argparse.ArgumentParser(
        description="Deploy SopraGP4U Clock In/Out Automation"
    )
    parser.add_argument(
        "--headless",
        action="store_true",
        help="Run deployment in headless mode (non-interactive)"
    )
    parser.add_argument(
        "--setup-tasks",
        action="store_true",
        help="Setup Windows Task Scheduler tasks"
    )
    parser.add_argument(
        "--verify",
        action="store_true",
        help="Verify environment only"
    )
    
    args = parser.parse_args()
    
    if args.verify:
        success = verify_environment()
    elif args.setup_tasks:
        success = setup_scheduled_tasks()
    elif args.headless:
        print("Headless deployment not yet implemented")
        success = False
    else:
        success = full_deployment()
    
    # Final summary
    print("\n" + "="*60)
    if success:
        print("✓ DEPLOYMENT SUCCESSFUL")
        print("="*60)
        print("\nNext steps:")
        print("1. Review and customize config/config.py if needed")
        print("2. Run element inspector: python src/inspect_portal_elements.py")
        print("3. Test manually: python src/sopra_clockin.py")
        print("4. Monitor via: Get-Content logs/sopra_clockin.log -Wait")
    else:
        print("✗ DEPLOYMENT FAILED")
        print("="*60)
        print("\nReview errors above and fix issues")
    
    print()
    
    return 0 if success else 1


if __name__ == "__main__":
    sys.exit(main())
