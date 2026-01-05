"""
Application-wide logging configuration.
Uses rich for pretty console logging and standard file logging for persistence.
"""
import logging
import sys
from logging.handlers import RotatingFileHandler
from rich.logging import RichHandler
from .config import settings

def setup_logging():
    """Configure logging for the application."""
    # Ensure logs directory exists
    settings.ensure_directories()
    
    # Create logger
    logger = logging.getLogger("tpn_rag")
    logger.setLevel(logging.INFO)
    
    # Remove existing handlers
    logger.handlers = []
    
    # Console Handler (Rich)
    console_handler = RichHandler(
        rich_tracebacks=True,
        show_time=True,
        show_path=False
    )
    console_handler.setLevel(logging.INFO)
    console_format = logging.Formatter("%(message)s")
    console_handler.setFormatter(console_format)
    
    # File Handler
    log_file = settings.logs_dir / "app.log"
    file_handler = RotatingFileHandler(
        log_file,
        maxBytes=10*1024*1024,  # 10MB
        backupCount=5,
        encoding="utf-8"
    )
    file_handler.setLevel(logging.DEBUG)
    file_format = logging.Formatter(
        "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    )
    file_handler.setFormatter(file_format)
    
    # Add handlers
    logger.addHandler(console_handler)
    logger.addHandler(file_handler)
    
    return logger

logger = setup_logging()
