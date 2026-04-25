"""Main training script for image-text matching."""

import argparse
import sys
from pathlib import Path

import hydra
import torch
from omegaconf import DictConfig, OmegaConf

import sys
from pathlib import Path

# Add src to path
sys.path.append(str(Path(__file__).parent.parent / "src"))

from data.dataset import ImageTextDataModule
from models.clip_model import CLIPImageTextMatcher
from training.trainer import ImageTextTrainer
from eval.evaluator import ImageTextEvaluator
from utils.config import create_output_dirs, print_config
from utils.device import setup_device_and_seed
from utils.logging import setup_logging, log_system_info, log_model_info


@hydra.main(version_base=None, config_path="../configs", config_name="config")
def main(config: DictConfig) -> None:
    """
    Main training function.
    
    Args:
        config: Hydra configuration object
    """
    # Setup logging
    logger = setup_logging(config)
    logger.info("Starting image-text matching training...")
    
    # Print configuration
    print_config(config, "Training Configuration")
    
    # Setup device and seed
    device = setup_device_and_seed(config.get("device", "auto"), config.get("seed", 42))
    logger.info(f"Using device: {device}")
    
    # Log system information
    log_system_info(logger)
    
    # Create output directories
    create_output_dirs(config)
    
    # Setup data module
    logger.info("Setting up data module...")
    data_module = ImageTextDataModule(
        data_dir=config.data.data_dir,
        batch_size=config.data.batch_size,
        num_workers=config.data.num_workers,
        pin_memory=config.data.pin_memory,
        image_size=config.data.image_size,
        text_max_length=config.data.text_max_length,
        model_name=config.model.model_name,
        use_augmentation=config.data.use_augmentation,
        augmentation_prob=config.data.augmentation_prob,
    )
    
    # Setup datasets
    data_module.setup("fit")
    train_dataloader = data_module.train_dataloader()
    val_dataloader = data_module.val_dataloader()
    
    logger.info(f"Training samples: {len(data_module.train_dataset)}")
    logger.info(f"Validation samples: {len(data_module.val_dataset)}")
    
    # Setup model
    logger.info("Setting up model...")
    model = hydra.utils.instantiate(config.model)
    log_model_info(logger, model)
    
    # Setup trainer
    logger.info("Setting up trainer...")
    trainer = ImageTextTrainer(
        model=model,
        config=config,
        train_dataloader=train_dataloader,
        val_dataloader=val_dataloader,
        logger=logger,
    )
    
    # Train model
    logger.info("Starting training...")
    training_results = trainer.train()
    
    # Evaluate model
    logger.info("Evaluating model...")
    evaluator = hydra.utils.instantiate(config.evaluation)
    
    # Load best model
    checkpoint_path = Path(config.checkpoint_dir) / "best.pt"
    if checkpoint_path.exists():
        trainer.load_checkpoint(checkpoint_path)
        logger.info("Loaded best model for evaluation")
    
    # Run evaluation
    eval_results = evaluator.evaluate(model, val_dataloader, device)
    
    # Print evaluation results
    leaderboard = evaluator.create_leaderboard(eval_results, "CLIP Model")
    logger.info(leaderboard)
    
    # Save final results
    results_path = Path(config.output_dir) / "final_results.yaml"
    OmegaConf.save(
        {
            "training_results": training_results,
            "evaluation_results": eval_results,
            "config": config,
        },
        results_path,
    )
    
    logger.info(f"Results saved to {results_path}")
    logger.info("Training completed successfully!")


if __name__ == "__main__":
    main()
