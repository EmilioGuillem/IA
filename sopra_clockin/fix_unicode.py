#!/usr/bin/env python3
"""Script to fix Unicode encoding issues in all Python files"""

import os
from pathlib import Path

# Files to fix
files_to_fix = [
    "quick_setup.py",
    "deploy.py",
    "src/test_setup.py",
    "src/inspect_portal_elements.py",
]

replacements = [
    ("✓", "[OK]"),
    ("✗", "[ERROR]"),
    ("⚠", "[WARNING]"),
    ("▶", ">>"),
    ("→", "->"),
]

base_path = Path(__file__).parent

for filename in files_to_fix:
    filepath = base_path / filename
    
    if not filepath.exists():
        print(f"Skip: {filename} not found")
        continue
    
    # Read file
    with open(filepath, 'r', encoding='utf-8') as f:
        content = f.read()
    
    original_content = content
    
    # Apply replacements
    for old, new in replacements:
        content = content.replace(old, new)
    
    # Write back
    if content != original_content:
        with open(filepath, 'w', encoding='utf-8') as f:
            f.write(content)
        print(f"Fixed: {filename}")
    else:
        print(f"Already clean: {filename}")

print("\nDone!")
