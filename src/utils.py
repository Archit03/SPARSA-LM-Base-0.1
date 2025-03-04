import os
import random
import numpy as np
import torch 
import logging
from pathlib import Path

def setup_logging(log_dir: str, name: str = 'LuminaLM') -> logging.Logger:
    """
    Set up logging configuration.
    
    Args:
        log_dir: Directory to store log files
        name: Name of the logger (default: 'LuminaLM')
    
    Returns:
        logging.Logger: Configured logger instance
    """
    # Create log directory if it doesn't exist
    os.makedirs(log_dir, exist_ok=True)
    
    # Create logger with the specified name
    logger = logging.getLogger(name)
    logger.setLevel(logging.INFO)
    
    # Create handlers
    console_handler = logging.StreamHandler()
    file_handler = logging.FileHandler(os.path.join(log_dir, f'{name}.log'))
    
    # Create formatter
    formatter = logging.Formatter(
        '%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )
    
    # Set formatter for handlers
    console_handler.setFormatter(formatter)
    file_handler.setFormatter(formatter)
    
    # Add handlers to logger
    logger.addHandler(console_handler)
    logger.addHandler(file_handler)
    
    return logger

def set_seed(seed):
    """ Set seed for reproducibility"""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

def save_checkpoint(model, optimizer, epoch, loss, path="checkpoint.pth"):
    """Save model checkpoint"""
    checkpoint = {
        "epoch": epoch,
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "loss": loss,
    }
    torch.save(checkpoint, path)
    logging.info(f"Model saved at {path}")

def load_checkpoint(path, model, optimizer=None, scheduler=None):
    """Load model checkpoint"""
    if not os.path.exists(path):
        raise FileNotFoundError(f"Checkpoint not found at {path}")
    
    checkpoint = torch.load(
        path,
        map_location=torch.device("cuda" if torch.cuda.is_available() else "cpu"),
        weights_only=True
    )
    
    model.load_state_dict(checkpoint["model_state_dict"])
    logging.info(f"Model loaded from {path}")

    if optimizer and "optimizer_state_dict" in checkpoint:
        optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        logging.info("Optimizer loaded")
    
    if scheduler and "scheduler_state_dict" in checkpoint:
        scheduler.load_state_dict(checkpoint["scheduler_state_dict"])
        logging.info("Scheduler loaded")
    
    # Just return the epoch (an integer). Everything else is already loaded in-place.
    epoch = checkpoint.get("epoch", 0)
    return epoch


def count_parameters(model):
    """Count number of trainable parameters in a model"""
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

def ensure_dir(directory):
    """Ensure directory exists"""
    if not os.path.exists(directory):
        os.makedirs(directory)
        logging.info(f"Created directory {directory}")

class MemoryMonitor:
    """Monitor GPU memory usage during training."""
    def __init__(self):
        self.enabled = torch.cuda.is_available()
        if self.enabled:
            self.logger = logging.getLogger('LuminaLM_memory')
    
    def log_memory(self, step: int = None, prefix: str = ""):
        """Log current GPU memory usage."""
        if not self.enabled:
            return
        
        try:
            allocated = torch.cuda.memory_allocated() / (1024 * 1024)  # Convert to MB
            reserved = torch.cuda.memory_reserved() / (1024 * 1024)    # Convert to MB
            max_allocated = torch.cuda.max_memory_allocated() / (1024 * 1024)
            
            message = (
                f"{prefix} Memory Usage - "
                f"Allocated: {allocated:.2f}MB, "
                f"Reserved: {reserved:.2f}MB, "
                f"Peak: {max_allocated:.2f}MB"
            )
            if step is not None:
                message = f"Step {step}: {message}"
                
            self.logger.info(message)
            
        except Exception as e:
            self.logger.warning(f"Failed to log memory usage: {str(e)}")
    
    def reset_peak_memory(self):
        """Reset peak memory statistics."""
        if self.enabled:
            torch.cuda.reset_peak_memory_stats()
        