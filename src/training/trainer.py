"""Training utilities for image-text matching models."""

import os
import time
from pathlib import Path
from typing import Dict, Optional, Union

import torch
import torch.nn as nn
import torch.nn.functional as F
from omegaconf import DictConfig
from torch.utils.data import DataLoader
from tqdm import tqdm

from ..losses.contrastive_loss import ContrastiveLoss, CombinedLoss
from ..utils.device import get_device, clear_gpu_memory
from ..utils.logging import setup_logging, TrainingLogger


class ImageTextTrainer:
    """
    Trainer for image-text matching models.
    
    Handles training loop, validation, checkpointing, and logging.
    """
    
    def __init__(
        self,
        model: nn.Module,
        config: DictConfig,
        train_dataloader: DataLoader,
        val_dataloader: Optional[DataLoader] = None,
        logger: Optional[object] = None,
    ):
        """
        Initialize trainer.
        
        Args:
            model: Model to train
            config: Training configuration
            train_dataloader: Training data loader
            val_dataloader: Validation data loader (optional)
            logger: Logger instance (optional)
        """
        self.model = model
        self.config = config
        self.train_dataloader = train_dataloader
        self.val_dataloader = val_dataloader
        self.logger = logger or setup_logging(config)
        
        # Setup device
        self.device = get_device(config.get("device", "auto"))
        self.model.to(self.device)
        
        # Setup loss function
        self.loss_fn = self._setup_loss_function()
        
        # Setup optimizer and scheduler
        self.optimizer = self._setup_optimizer()
        self.scheduler = self._setup_scheduler()
        
        # Setup training logger
        self.training_logger = TrainingLogger(
            self.logger,
            config.training.get("log_every_n_steps", 100),
        )
        
        # Training state
        self.current_epoch = 0
        self.global_step = 0
        self.best_metric = float("inf")
        self.early_stopping_counter = 0
        
        # Mixed precision
        self.use_amp = config.training.get("use_amp", True)
        self.scaler = torch.cuda.amp.GradScaler() if self.use_amp else None
        
        # Gradient accumulation
        self.gradient_accumulation_steps = config.training.get(
            "gradient_accumulation_steps", 1
        )
    
    def _setup_loss_function(self) -> nn.Module:
        """Setup loss function based on configuration."""
        loss_type = self.config.training.get("loss_type", "contrastive")
        
        if loss_type == "contrastive":
            return ContrastiveLoss(
                temperature=self.config.training.get("temperature", 0.07),
                hard_negative_mining=self.config.training.get(
                    "hard_negative_mining", True
                ),
                hard_negative_ratio=self.config.training.get(
                    "hard_negative_ratio", 0.5
                ),
            )
        elif loss_type == "combined":
            return CombinedLoss(
                contrastive_weight=self.config.training.get(
                    "contrastive_weight", 1.0
                ),
                regularization_weight=self.config.training.get(
                    "regularization_weight", 0.01
                ),
                temperature=self.config.training.get("temperature", 0.07),
            )
        else:
            raise ValueError(f"Unknown loss type: {loss_type}")
    
    def _setup_optimizer(self) -> torch.optim.Optimizer:
        """Setup optimizer based on configuration."""
        optimizer_name = self.config.training.get("optimizer", "AdamW")
        learning_rate = self.config.training.get("learning_rate", 1e-4)
        weight_decay = self.config.training.get("weight_decay", 0.01)
        
        if optimizer_name == "AdamW":
            return torch.optim.AdamW(
                self.model.parameters(),
                lr=learning_rate,
                weight_decay=weight_decay,
                betas=self.config.training.get("adam_betas", [0.9, 0.999]),
                eps=self.config.training.get("adam_epsilon", 1e-8),
            )
        elif optimizer_name == "Adam":
            return torch.optim.Adam(
                self.model.parameters(),
                lr=learning_rate,
                weight_decay=weight_decay,
            )
        elif optimizer_name == "SGD":
            return torch.optim.SGD(
                self.model.parameters(),
                lr=learning_rate,
                weight_decay=weight_decay,
                momentum=0.9,
            )
        else:
            raise ValueError(f"Unknown optimizer: {optimizer_name}")
    
    def _setup_scheduler(self) -> Optional[torch.optim.lr_scheduler._LRScheduler]:
        """Setup learning rate scheduler."""
        scheduler_name = self.config.training.get("scheduler", "cosine")
        
        if scheduler_name == "cosine":
            return torch.optim.lr_scheduler.CosineAnnealingLR(
                self.optimizer,
                T_max=self.config.training.get("epochs", 10),
                eta_min=self.config.training.get("learning_rate", 1e-4) * 
                        self.config.training.get("min_lr_ratio", 0.01),
            )
        elif scheduler_name == "step":
            return torch.optim.lr_scheduler.StepLR(
                self.optimizer,
                step_size=self.config.training.get("step_size", 5),
                gamma=self.config.training.get("gamma", 0.1),
            )
        elif scheduler_name == "plateau":
            return torch.optim.lr_scheduler.ReduceLROnPlateau(
                self.optimizer,
                mode="min",
                factor=0.5,
                patience=2,
                verbose=True,
            )
        else:
            return None
    
    def train(self) -> Dict[str, float]:
        """
        Train the model.
        
        Returns:
            Dictionary containing training results
        """
        self.logger.info("Starting training...")
        self.logger.info(f"Device: {self.device}")
        self.logger.info(f"Epochs: {self.config.training.get('epochs', 10)}")
        self.logger.info(f"Batch size: {self.config.data.get('batch_size', 32)}")
        self.logger.info(f"Learning rate: {self.config.training.get('learning_rate', 1e-4)}")
        
        start_time = time.time()
        
        for epoch in range(self.config.training.get("epochs", 10)):
            self.current_epoch = epoch
            
            # Training
            train_loss = self._train_epoch()
            
            # Validation
            val_loss = None
            val_metrics = None
            if self.val_dataloader is not None:
                val_loss, val_metrics = self._validate_epoch()
            
            # Log epoch summary
            self.logger.info(f"Epoch {epoch + 1} Summary:")
            self.logger.info(f"  Train Loss: {train_loss:.4f}")
            if val_loss is not None:
                self.logger.info(f"  Val Loss: {val_loss:.4f}")
            if val_metrics:
                self.logger.info("  Validation Metrics:")
                for metric_name, metric_value in val_metrics.items():
                    self.logger.info(f"    {metric_name}: {metric_value:.4f}")
            
            # Learning rate scheduling
            if self.scheduler is not None:
                if isinstance(self.scheduler, torch.optim.lr_scheduler.ReduceLROnPlateau):
                    self.scheduler.step(val_loss or train_loss)
                else:
                    self.scheduler.step()
            
            # Checkpointing
            self._save_checkpoint(val_loss, val_metrics)
            
            # Early stopping
            if self._check_early_stopping(val_loss, val_metrics):
                self.logger.info("Early stopping triggered!")
                break
        
        training_time = time.time() - start_time
        self.logger.info(f"Training completed in {training_time:.2f} seconds")
        
        return {
            "training_time": training_time,
            "final_epoch": self.current_epoch + 1,
            "best_metric": self.best_metric,
        }
    
    def _train_epoch(self) -> float:
        """Train for one epoch."""
        self.model.train()
        total_loss = 0.0
        num_batches = 0
        
        progress_bar = tqdm(
            self.train_dataloader,
            desc=f"Epoch {self.current_epoch + 1}",
            leave=False,
        )
        
        for batch_idx, batch in enumerate(progress_bar):
            # Move batch to device
            pixel_values = batch["image"].to(self.device)
            input_ids = batch["input_ids"].to(self.device)
            attention_mask = batch["attention_mask"].to(self.device)
            
            # Forward pass with mixed precision
            if self.use_amp:
                with torch.cuda.amp.autocast():
                    outputs = self.model(pixel_values, input_ids, attention_mask)
                    loss = self._compute_loss(outputs)
                    loss = loss / self.gradient_accumulation_steps
            else:
                outputs = self.model(pixel_values, input_ids, attention_mask)
                loss = self._compute_loss(outputs)
                loss = loss / self.gradient_accumulation_steps
            
            # Backward pass
            if self.use_amp:
                self.scaler.scale(loss).backward()
            else:
                loss.backward()
            
            # Gradient accumulation
            if (batch_idx + 1) % self.gradient_accumulation_steps == 0:
                # Gradient clipping
                if self.config.training.get("gradient_clip_norm", 0) > 0:
                    if self.use_amp:
                        self.scaler.unscale_(self.optimizer)
                    
                    torch.nn.utils.clip_grad_norm_(
                        self.model.parameters(),
                        self.config.training.get("gradient_clip_norm", 1.0),
                    )
                
                # Optimizer step
                if self.use_amp:
                    self.scaler.step(self.optimizer)
                    self.scaler.update()
                else:
                    self.optimizer.step()
                
                self.optimizer.zero_grad()
                self.global_step += 1
            
            # Update metrics
            total_loss += loss.item() * self.gradient_accumulation_steps
            num_batches += 1
            
            # Log training progress
            current_lr = self.optimizer.param_groups[0]["lr"]
            self.training_logger.log_step(loss.item(), current_lr, self.global_step)
            
            # Update progress bar
            progress_bar.set_postfix({
                "loss": f"{loss.item():.4f}",
                "lr": f"{current_lr:.2e}",
            })
        
        return total_loss / num_batches
    
    def _validate_epoch(self) -> tuple[Optional[float], Optional[Dict[str, float]]]:
        """Validate for one epoch."""
        if self.val_dataloader is None:
            return None, None
        
        self.model.eval()
        total_loss = 0.0
        num_batches = 0
        
        with torch.no_grad():
            for batch in self.val_dataloader:
                # Move batch to device
                pixel_values = batch["image"].to(self.device)
                input_ids = batch["input_ids"].to(self.device)
                attention_mask = batch["attention_mask"].to(self.device)
                
                # Forward pass
                if self.use_amp:
                    with torch.cuda.amp.autocast():
                        outputs = self.model(pixel_values, input_ids, attention_mask)
                        loss = self._compute_loss(outputs)
                else:
                    outputs = self.model(pixel_values, input_ids, attention_mask)
                    loss = self._compute_loss(outputs)
                
                total_loss += loss.item()
                num_batches += 1
        
        avg_loss = total_loss / num_batches
        
        # Compute validation metrics (simplified for now)
        val_metrics = {"val_loss": avg_loss}
        
        return avg_loss, val_metrics
    
    def _compute_loss(self, outputs: Dict[str, torch.Tensor]) -> torch.Tensor:
        """Compute loss from model outputs."""
        if isinstance(self.loss_fn, ContrastiveLoss):
            return self.loss_fn(
                outputs["logits_per_image"],
                outputs["logits_per_text"],
            )
        elif isinstance(self.loss_fn, CombinedLoss):
            loss_dict = self.loss_fn(
                outputs["logits_per_image"],
                outputs["logits_per_text"],
                outputs["image_embeds"],
                outputs["text_embeds"],
            )
            return loss_dict["total_loss"]
        else:
            raise ValueError("Unknown loss function type")
    
    def _save_checkpoint(
        self,
        val_loss: Optional[float],
        val_metrics: Optional[Dict[str, float]],
    ) -> None:
        """Save model checkpoint."""
        checkpoint_dir = Path(self.config.get("checkpoint_dir", "checkpoints"))
        checkpoint_dir.mkdir(parents=True, exist_ok=True)
        
        # Determine if this is the best model
        best_metric = self.config.training.get("best_metric", "recall_at_1")
        is_best = False
        
        if val_metrics and best_metric in val_metrics:
            current_metric = val_metrics[best_metric]
            if current_metric > self.best_metric:
                self.best_metric = current_metric
                is_best = True
        
        # Save checkpoint
        checkpoint = {
            "epoch": self.current_epoch,
            "global_step": self.global_step,
            "model_state_dict": self.model.state_dict(),
            "optimizer_state_dict": self.optimizer.state_dict(),
            "best_metric": self.best_metric,
            "config": self.config,
        }
        
        if self.scheduler is not None:
            checkpoint["scheduler_state_dict"] = self.scheduler.state_dict()
        
        # Save latest checkpoint
        torch.save(checkpoint, checkpoint_dir / "latest.pt")
        
        # Save best checkpoint
        if is_best:
            torch.save(checkpoint, checkpoint_dir / "best.pt")
            self.logger.info(f"New best model saved! {best_metric}: {self.best_metric:.4f}")
    
    def _check_early_stopping(
        self,
        val_loss: Optional[float],
        val_metrics: Optional[Dict[str, float]],
    ) -> bool:
        """Check if early stopping should be triggered."""
        patience = self.config.training.get("early_stopping_patience", 5)
        
        if patience <= 0:
            return False
        
        best_metric = self.config.training.get("best_metric", "recall_at_1")
        
        if val_metrics and best_metric in val_metrics:
            current_metric = val_metrics[best_metric]
            if current_metric > self.best_metric:
                self.early_stopping_counter = 0
            else:
                self.early_stopping_counter += 1
        else:
            self.early_stopping_counter += 1
        
        return self.early_stopping_counter >= patience
    
    def load_checkpoint(self, checkpoint_path: Union[str, Path]) -> None:
        """Load model from checkpoint."""
        checkpoint = torch.load(checkpoint_path, map_location=self.device)
        
        self.model.load_state_dict(checkpoint["model_state_dict"])
        self.optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        
        if "scheduler_state_dict" in checkpoint and self.scheduler is not None:
            self.scheduler.load_state_dict(checkpoint["scheduler_state_dict"])
        
        self.current_epoch = checkpoint.get("epoch", 0)
        self.global_step = checkpoint.get("global_step", 0)
        self.best_metric = checkpoint.get("best_metric", float("inf"))
        
        self.logger.info(f"Loaded checkpoint from {checkpoint_path}")
        self.logger.info(f"Epoch: {self.current_epoch}, Global step: {self.global_step}")
        self.logger.info(f"Best metric: {self.best_metric}")
