# Configuration file for SopraGP4U Clock In/Out automation
# ============================================================================

import os
from pathlib import Path

# Get the project root directory
PROJECT_ROOT = Path(__file__).parent.parent

# URLs
SOPRA_URL = "https://sprportal-mcp.soprahronline.com/SopraGP4U/"
MENU_LINK_TEXT = "Registro de entrada/salida"

# Time thresholds (in 24-hour format)
CLOCK_IN_THRESHOLD = 10  # Before 10:00 AM
CLOCK_OUT_THRESHOLD = 17  # After 5:00 PM (17:00)

# Button identifiers - Update these based on actual DOM inspection
CLOCK_IN_BUTTON_ID = "CLOCK-IN"
CLOCK_OUT_BUTTON_ID = "CLOCK-OUT"

# Chrome driver options
CHROME_OPTIONS = {
    "headless": True,  # Run in background
    "window_size": "1920,1080",
    "no_sandbox": True,
    "disable_dev_shm_usage": True,
}

# Logs configuration
LOGS_DIR = PROJECT_ROOT / "logs"
LOGS_DIR.mkdir(exist_ok=True)

LOG_FILE = LOGS_DIR / "sopra_clockin.log"
LOG_LEVEL = "INFO"
LOG_FORMAT = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
LOG_DATE_FORMAT = "%Y-%m-%d %H:%M:%S"

# Credentials (These should be stored securely in production)
# DO NOT commit these to version control
# Use environment variables instead
SOPRA_USERNAME = os.getenv("SOPRA_USERNAME", "")
SOPRA_PASSWORD = os.getenv("SOPRA_PASSWORD", "")

# Timeouts (in seconds)
WAIT_TIMEOUT = 30
PAGE_LOAD_TIMEOUT = 60

# Retry settings
MAX_RETRIES = 3
RETRY_DELAY = 5  # seconds

# Enable/Disable features
DRY_RUN = False  # Set to True to simulate without actual clicks
HEADLESS_MODE = True
