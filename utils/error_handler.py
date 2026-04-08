"""
Error Handler for HoloLearn Extractor
Handles logging, error tracking, and graceful failure management.
"""

import os
import logging
import traceback
from datetime import datetime
from pathlib import Path
from typing import Optional, Dict, Any
from functools import wraps

# Import config to get log directory
import sys
sys.path.append(str(Path(__file__).parent.parent))
from utils.configs import LOGS_DIR as LOG_DIR


class ErrorHandler:
    """Centralized error handling and logging"""
    
    def __init__(self, log_name: str = "extractor"):
        """
        Initialize error handler with custom log file
        
        Args:
            log_name: Name for the log file (e.g., "pdf_extractor", "video_processor")
        """
        self.log_name = log_name
        self.log_file = LOG_DIR / f"{log_name}_{datetime.now().strftime('%Y%m%d')}.log"
        
        # Create logger
        self.logger = self._setup_logger()
    
    def _setup_logger(self) -> logging.Logger:
        """Configure and return a logger instance"""
        logger = logging.getLogger(self.log_name)
        logger.setLevel(logging.DEBUG)
        
        # Avoid duplicate handlers
        if logger.handlers:
            return logger
        
        # File handler - writes to log file
        file_handler = logging.FileHandler(self.log_file, encoding='utf-8')
        file_handler.setLevel(logging.DEBUG)
        
        # Console handler - prints to terminal
        console_handler = logging.StreamHandler()
        console_handler.setLevel(logging.INFO)
        
        # Format: [2025-02-02 10:30:45] [ERROR] Message here
        formatter = logging.Formatter(
            '[%(asctime)s] [%(levelname)s] %(message)s',
            datefmt='%Y-%m-%d %H:%M:%S'
        )
        
        file_handler.setFormatter(formatter)
        console_handler.setFormatter(formatter)
        
        logger.addHandler(file_handler)
        logger.addHandler(console_handler)
        
        return logger
    
    def log_error(self, 
                  error: Exception, 
                  context: str = "", 
                  metadata: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """
        Log an error with context and metadata
        
        Args:
            error: The exception that occurred
            context: Description of what was happening when error occurred
            metadata: Additional info (filename, resource_id, etc.)
        
        Returns:
            Error report dictionary
        """
        error_report = {
            "timestamp": datetime.now().isoformat(),
            "error_type": type(error).__name__,
            "error_message": str(error),
            "context": context,
            "traceback": traceback.format_exc(),
            "metadata": metadata or {}
        }
        
        # Log to file
        self.logger.error(f"{context}: {error}")
        self.logger.debug(f"Full traceback: {error_report['traceback']}")
        
        if metadata:
            self.logger.debug(f"Metadata: {metadata}")
        
        return error_report
    
    def log_info(self, message: str, metadata: Optional[Dict[str, Any]] = None):
        """Log informational message"""
        self.logger.info(message)
        if metadata:
            self.logger.debug(f"Metadata: {metadata}")
    
    def log_warning(self, message: str, metadata: Optional[Dict[str, Any]] = None):
        """Log warning message"""
        self.logger.warning(message)
        if metadata:
            self.logger.debug(f"Metadata: {metadata}")
    
    def log_success(self, message: str, metadata: Optional[Dict[str, Any]] = None):
        """Log success message"""
        self.logger.info(f"✓ {message}")
        if metadata:
            self.logger.debug(f"Metadata: {metadata}")


# Decorator for automatic error handling
def handle_errors(error_handler: ErrorHandler, context: str = ""):
    """
    Decorator to automatically catch and log errors in functions
    
    Usage:
        @handle_errors(error_handler, "Processing PDF")
        def extract_pdf(file_path):
            # your code here
    """
    def decorator(func):
        @wraps(func)
        def wrapper(*args, **kwargs):
            try:
                return func(*args, **kwargs)
            except Exception as e:
                error_handler.log_error(
                    error=e,
                    context=context or f"Function: {func.__name__}",
                    metadata={"args": str(args), "kwargs": str(kwargs)}
                )
                raise  # Re-raise after logging
        return wrapper
    return decorator


# Example usage
if __name__ == "__main__":
    # Test the error handler
    handler = ErrorHandler("test")
    
    # Test success log
    handler.log_success("Test successful!", {"file": "test.pdf"})
    
    # Test error log
    try:
        raise ValueError("This is a test error")
    except Exception as e:
        handler.log_error(
            error=e,
            context="Testing error handler",
            metadata={"resource_id": "123", "filename": "test.pdf"}
        )
    
    print(f"\n✓ Log file created at: {handler.log_file}")