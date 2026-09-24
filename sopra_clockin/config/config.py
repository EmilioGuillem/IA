# This file has been created (totally or partially) with the assistance of
# artificial intelligence tools. All content has been generated under the
# direct supervision of a named individual and the AI.Backbone Orchestrator
# Compliance framework.

# Configuration file for SopraGP4U Clock In/Out automation
# ============================================================================

import os
import json
from pathlib import Path

# Get the project root directory
PROJECT_ROOT = Path(__file__).parent.parent

# Configuration file path
CONFIG_FILE = PROJECT_ROOT / "config" / "config.json"

def load_config():
    """Load configuration from JSON file if it exists."""
    if CONFIG_FILE.exists():
        try:
            with open(CONFIG_FILE, 'r', encoding='utf-8') as f:
                return json.load(f)
        except Exception as e:
            print(f"Warning: Could not load config.json: {e}")
    return {}

# Load config
config_data = load_config()

# URLs
SOPRA_URL = "https://sprportal-mcp.soprahronline.com/SopraGP4U/WAW05B02"
MENU_LINK_TEXT = "Registro de entrada / salida"
MENU_LINK_SELECTOR = {
    'method': 'xpath',
    'value': """//*[contains(
        translate(normalize-space(string(.)),
        'ABCDEFGHIJKLMNOPQRSTUVWXYZ',
        'abcdefghijklmnopqrstuvwxyz'),
        'registro de entrada'
    )]/ancestor-or-self::*[self::a or self::button or @role='link' or @role='button'][1]""",
}

# Time windows and minimum working duration.
CLOCK_IN_START_HOUR = 7
CLOCK_IN_START_MINUTE = 0
CLOCK_IN_END_HOUR = 9
CLOCK_IN_END_MINUTE = 30
CLOCK_OUT_START_HOUR = 17
CLOCK_OUT_START_MINUTE = 30
MIN_WORK_HOURS = 9
# Kept for compatibility with older imports.
CLOCK_IN_THRESHOLD = 10
CLOCK_OUT_THRESHOLD = 17

# Element selectors - flexible configuration.
CLOCK_IN_SELECTOR = config_data.get('clock_in', {
    'method': 'css',
    'value': 'button.button-success.register-button',
})
CLOCK_OUT_SELECTOR = config_data.get('clock_out', {
    'method': 'css',
    'value': 'button.button-danger.register-button',
})

# Legacy IDs for backward compatibility
CLOCK_IN_BUTTON_ID = CLOCK_IN_SELECTOR.get('value', 'CLOCK-IN') if CLOCK_IN_SELECTOR.get('method') == 'id' else 'CLOCK-IN'
CLOCK_OUT_BUTTON_ID = CLOCK_OUT_SELECTOR.get('value', 'CLOCK-OUT') if CLOCK_OUT_SELECTOR.get('method') == 'id' else 'CLOCK-OUT'

# Browser selection. Edge is the default browser for this environment.
BROWSER = os.getenv("SOPRA_BROWSER", config_data.get('browser', "edge")).lower()

# Chrome driver options
CHROME_OPTIONS = {
    "headless": True,  # Run in background
    "window_size": "1920,1080",
    "no_sandbox": True,
    "disable_dev_shm_usage": True,
}

# Edge driver options
EDGE_OPTIONS = {
    "headless": True,  # Run in background
    "window_size": "1920,1080",
}

# Logs configuration
LOGS_DIR = PROJECT_ROOT / "logs"
LOGS_DIR.mkdir(exist_ok=True)

LOG_FILE = LOGS_DIR / "sopra_clockin.log"
STATE_FILE = LOGS_DIR / "clock_state.json"
LOG_LEVEL = "INFO"
LOG_FORMAT = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
LOG_DATE_FORMAT = "%Y-%m-%d %H:%M:%S"

# Credentials must remain outside configuration files.
SOPRA_USERNAME = os.getenv("SOPRA_USERNAME", "")
SOPRA_PASSWORD = os.getenv("SOPRA_PASSWORD", "")

# Timeouts (in seconds)
WAIT_TIMEOUT = 30
PAGE_LOAD_TIMEOUT = 60

# Retry settings
MAX_RETRIES = 3
RETRY_DELAY = 5  # seconds

# Enable/Disable features
dry_run_env = os.getenv("SOPRA_DRY_RUN")
DRY_RUN = (
    dry_run_env.lower() == "true"
    if dry_run_env is not None
    else bool(config_data.get('dry_run', False))
)
HEADLESS_MODE = config_data.get('headless', True)
