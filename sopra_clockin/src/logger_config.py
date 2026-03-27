# Logger configuration module
# ============================================================================

import logging
import logging.handlers
import sys
from pathlib import Path
from config.config import LOG_FILE, LOG_LEVEL, LOG_FORMAT, LOG_DATE_FORMAT

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
    
    # Console handler with UTF-8 encoding
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setLevel(logging.INFO)
    console_handler.setFormatter(formatter)
    
    # Add handlers to logger
    logger.addHandler(file_handler)
    logger.addHandler(console_handler)
    
    return logger
