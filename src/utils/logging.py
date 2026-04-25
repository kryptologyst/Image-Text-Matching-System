"""Logging utilities for the image-text matching system."""

import logging
import sys
from pathlib import Path
from typing import Optional

import torch
from omegaconf import DictConfig


def setup_logging(
    config: DictConfig,
    log_level: Optional[str] = None,
    log_file: Optional[str] = None,
) -> logging.Logger:
    """
    Setup logging configuration.
    
    Args:
        config: Configuration object
        log_level: Logging level (overrides config)
        log_file: Log file path (overrides config)
        
    Returns:
        logging.Logger: Configured logger
    """
    # Get logging configuration
    level = log_level or config.get("log_level", "INFO")
    log_to_file = config.get("log_to_file", True)
    file_path = log_file or config.get("log_file", "logs/training.log")
    
    # Create logger
    logger = logging.getLogger("image_text_matching")
    logger.setLevel(getattr(logging, level.upper()))
    
    # Clear existing handlers
    logger.handlers.clear()
    
    # Create formatter
    formatter = logging.Formatter(
        "%(asctime)s - %(name)s - %(levelname)s - %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )
    
    # Console handler
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setLevel(getattr(logging, level.upper()))
    console_handler.setFormatter(formatter)
    logger.addHandler(console_handler)
    
    # File handler
    if log_to_file:
        log_path = Path(file_path)
        log_path.parent.mkdir(parents=True, exist_ok=True)
        
        file_handler = logging.FileHandler(log_path)
        file_handler.setLevel(logging.DEBUG)  # Always log everything to file
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)
    
    return logger


def log_system_info(logger: logging.Logger) -> None:
    """
    Log system information including PyTorch and device details.
    
    Args:
        logger: Logger instance
    """
    logger.info("System Information:")
    logger.info(f"Python version: {sys.version}")
    logger.info(f"PyTorch version: {torch.__version__}")
    logger.info(f"CUDA available: {torch.cuda.is_available()}")
    
    if torch.cuda.is_available():
        logger.info(f"CUDA version: {torch.version.cuda}")
        logger.info(f"GPU count: {torch.cuda.device_count()}")
        for i in range(torch.cuda.device_count()):
            logger.info(f"GPU {i}: {torch.cuda.get_device_name(i)}")
    
    if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        logger.info("MPS (Apple Silicon) available: True")


def log_model_info(logger: logging.Logger, model: torch.nn.Module) -> None:
    """
    Log model information including parameter count and size.
    
    Args:
        logger: Logger instance
        model: PyTorch model
    """
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    
    logger.info("Model Information:")
    logger.info(f"Total parameters: {total_params:,}")
    logger.info(f"Trainable parameters: {trainable_params:,}")
    logger.info(f"Non-trainable parameters: {total_params - trainable_params:,}")
    
    # Estimate memory usage
    param_size = sum(p.numel() * p.element_size() for p in model.parameters())
    buffer_size = sum(b.numel() * b.element_size() for b in model.buffers())
    total_size = (param_size + buffer_size) / (1024 * 1024)
    
    logger.info(f"Model size: {total_size:.2f} MB")


def log_training_start(logger: logging.Logger, config: DictConfig) -> None:
    """
    Log training start information.
    
    Args:
        logger: Logger instance
        config: Training configuration
    """
    logger.info("Starting training...")
    logger.info(f"Epochs: {config.training.get('epochs', 'N/A')}")
    logger.info(f"Batch size: {config.data.get('batch_size', 'N/A')}")
    logger.info(f"Learning rate: {config.training.get('learning_rate', 'N/A')}")
    logger.info(f"Device: {config.get('device', 'auto')}")


def log_epoch_summary(
    logger: logging.Logger,
    epoch: int,
    train_loss: float,
    val_loss: Optional[float] = None,
    metrics: Optional[dict] = None,
) -> None:
    """
    Log epoch summary.
    
    Args:
        logger: Logger instance
        epoch: Current epoch number
        train_loss: Training loss
        val_loss: Validation loss (optional)
        metrics: Validation metrics (optional)
    """
    logger.info(f"Epoch {epoch} Summary:")
    logger.info(f"  Train Loss: {train_loss:.4f}")
    
    if val_loss is not None:
        logger.info(f"  Val Loss: {val_loss:.4f}")
    
    if metrics:
        logger.info("  Validation Metrics:")
        for metric_name, metric_value in metrics.items():
            logger.info(f"    {metric_name}: {metric_value:.4f}")


class TrainingLogger:
    """Custom logger for training progress."""
    
    def __init__(self, logger: logging.Logger, log_every_n_steps: int = 100):
        self.logger = logger
        self.log_every_n_steps = log_every_n_steps
        self.step_count = 0
    
    def log_step(self, loss: float, lr: float, step: Optional[int] = None) -> None:
        """
        Log training step.
        
        Args:
            loss: Current loss value
            lr: Current learning rate
            step: Step number (if None, uses internal counter)
        """
        if step is None:
            step = self.step_count
            self.step_count += 1
        
        if step % self.log_every_n_steps == 0:
            self.logger.info(f"Step {step}: Loss={loss:.4f}, LR={lr:.2e}")
    
    def reset(self) -> None:
        """Reset step counter."""
        self.step_count = 0
