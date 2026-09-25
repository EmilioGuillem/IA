# This file has been created (totally or partially) with the assistance of
# artificial intelligence tools. All content has been generated under the
# direct supervision of a named individual and the AI.Backbone Orchestrator
# Compliance framework.
# Logger configuration module
# ============================================================================

import logging
import logging.handlers
import os
import sys
from pathlib import Path
from config.config import LOG_FILE, LOG_LEVEL, LOG_FORMAT, LOG_DATE_FORMAT


class QuietAutomationFilter(logging.Filter):
    """Keep only hourly checks and final status messages in quiet mode."""

    def filter(self, record):
        if os.getenv("SOPRA_QUIET_LOGS", "false").lower() != "true":
            return True
        return record.getMessage().startswith(("[CHECK]", "[OK]", "[KO]"))

def setup_logger(name):
    """
    Setup and return a configured logger instance.
    
    Args:
        name (str): Logger name, typically __name__
        
    Returns:
        logging.Logger: Configured logger instance
    """
    
    logger = logging.getLogger(name)
    
    # Avoid duplicate handlers if logger already exists
    if logger.handlers:
        return logger
    
    logger.setLevel(getattr(logging, LOG_LEVEL))
    
    # Create formatters
    formatter = logging.Formatter(
        fmt=LOG_FORMAT,
        datefmt=LOG_DATE_FORMAT
    )
    
    # File handler - rotates every day or when it reaches 10MB
    file_handler = logging.handlers.RotatingFileHandler(
        LOG_FILE,
        maxBytes=10*1024*1024,  # 10MB
        backupCount=10,  # Keep 10 backup files
        encoding='utf-8'  # Explicit UTF-8 encoding
    )
    file_handler.setLevel(getattr(logging, LOG_LEVEL))
    file_handler.setFormatter(formatter)
    file_handler.addFilter(QuietAutomationFilter())
    
    # Console handler with UTF-8 encoding
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setLevel(logging.INFO)
    console_handler.setFormatter(formatter)
    console_handler.addFilter(QuietAutomationFilter())
    
    # Add handlers to logger
    logger.addHandler(file_handler)
    logger.addHandler(console_handler)
    
    return logger
