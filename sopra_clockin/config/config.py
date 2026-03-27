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
MENU_LINK_TEXT = "Registro de entrada/salida"  # No longer needed - direct portal access

# Time thresholds (in 24-hour format)
CLOCK_IN_THRESHOLD = 10  # Before 10:00 AM
CLOCK_OUT_THRESHOLD = 17  # After 5:00 PM (17:00)

# Button selectors - flexible configuration
CLOCK_IN_SELECTOR = config_data.get('clock_in', {'method': 'id', 'value': 'CLOCK-IN'})
CLOCK_OUT_SELECTOR = config_data.get('clock_out', {'method': 'id', 'value': 'CLOCK-OUT'})

# Legacy IDs for backward compatibility
CLOCK_IN_BUTTON_ID = CLOCK_IN_SELECTOR.get('value', 'CLOCK-IN') if CLOCK_IN_SELECTOR.get('method') == 'id' else 'CLOCK-IN'
CLOCK_OUT_BUTTON_ID = CLOCK_OUT_SELECTOR.get('value', 'CLOCK-OUT') if CLOCK_OUT_SELECTOR.get('method') == 'id' else 'CLOCK-OUT'

# Browser selection
BROWSER = config_data.get('browser', os.getenv("SOPRA_BROWSER", "chrome")).lower()

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
LOG_LEVEL = "INFO"
LOG_FORMAT = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
LOG_DATE_FORMAT = "%Y-%m-%d %H:%M:%S"

# Credentials (Optional - NOT required for this portal)
SOPRA_USERNAME = config_data.get('username', os.getenv("SOPRA_USERNAME", ""))
SOPRA_PASSWORD = config_data.get('password', os.getenv("SOPRA_PASSWORD", ""))

# Timeouts (in seconds)
WAIT_TIMEOUT = 30
PAGE_LOAD_TIMEOUT = 60

# Retry settings
MAX_RETRIES = 3
RETRY_DELAY = 5  # seconds

# Enable/Disable features
DRY_RUN = config_data.get('dry_run', os.getenv("SOPRA_DRY_RUN", "false").lower() == "true")
HEADLESS_MODE = config_data.get('headless', True)
