import os
import random
import numpy as np
import torch 
import logging

def setup_logging(log_dir="logs", log_file = "training.log"):
    """Set up for training logs"""
    os.makedirs(log_dir, exist_ok=True)
    log_path = os.path.join(log_dir, log_file)
    logging.basicConfig(
        filename=log_path,
        filemode="a",
        format="%(asctime)s - %(levelname)s - %(message)s",
        datefmt="%d-%b-%y %H:%M:%S",
        level=logging.INFO,
    )
    logging.info("Logging started")

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
        raise FileNotFoundError(f"checkpoint not found at {path}")
    
    checkpoint = torch.load(path, map_location=torch.device("cuda" if torch.cuda.is_available() else "cpu"))
    model.load_state_dict(checkpoint["model_state_dict"])
    logging.info(f"Model loaded from {path}")

    if optimizer and "optimizer_state_dict" in checkpoint:
        optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        logging.info("Optimizer loaded")
    
    if scheduler and "scheduler_state_dict" in checkpoint:
        scheduler.load_state_dict(checkpoint["scheduler_state_dict"])
        logging.info("Scheduler loaded")
    
    epoch = checkpoint.get("epoch", 0)
    return model, optimizer, scheduler, epoch

def count_parameters(model):
    """Count number of trainable parameters in a model"""
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

def ensure_dir(directory):
    """Ensure directory exists"""
    if not os.path.exists(directory):
        os.makedirs(directory)
        logging.info(f"Created directory {directory}")
        