"""Data loading and preprocessing utilities for image-text matching."""

import json
import random
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np
import torch
from PIL import Image
from torch.utils.data import Dataset
from transformers import CLIPProcessor


class ImageTextDataset(Dataset):
    """
    Dataset for image-text matching tasks.
    
    Supports various data formats and provides comprehensive preprocessing
    for both images and text data.
    """
    
    def __init__(
        self,
        data_dir: Union[str, Path],
        split: str = "train",
        image_size: int = 224,
        text_max_length: int = 77,
        model_name: str = "openai/clip-vit-base-patch32",
        use_augmentation: bool = True,
        augmentation_prob: float = 0.5,
        **kwargs,
    ):
        """
        Initialize the dataset.
        
        Args:
            data_dir: Directory containing the dataset
            split: Dataset split ('train', 'val', 'test')
            image_size: Target image size for resizing
            text_max_length: Maximum text length
            model_name: CLIP model name for processor
            use_augmentation: Whether to use data augmentation
            augmentation_prob: Probability of applying augmentation
        """
        self.data_dir = Path(data_dir)
        self.split = split
        self.image_size = image_size
        self.text_max_length = text_max_length
        self.use_augmentation = use_augmentation and split == "train"
        self.augmentation_prob = augmentation_prob
        
        # Load processor
        self.processor = CLIPProcessor.from_pretrained(model_name)
        
        # Load data
        self.data = self._load_data()
        
        # Setup augmentation
        if self.use_augmentation:
            self._setup_augmentation()
    
    def _load_data(self) -> List[Dict[str, Any]]:
        """Load dataset from files."""
        data_file = self.data_dir / f"{self.split}.json"
        
        if data_file.exists():
            with open(data_file, 'r') as f:
                return json.load(f)
        
        # If no data file exists, create sample data
        return self._create_sample_data()
    
    def _create_sample_data(self) -> List[Dict[str, Any]]:
        """Create sample data for demonstration."""
        sample_data = [
            {
                "image_path": "sample_images/cat.jpg",
                "text": "A cute cat sitting on a windowsill",
                "image_id": "cat_001",
            },
            {
                "image_path": "sample_images/dog.jpg", 
                "text": "A happy dog playing in the park",
                "image_id": "dog_001",
            },
            {
                "image_path": "sample_images/landscape.jpg",
                "text": "A beautiful mountain landscape at sunset",
                "image_id": "landscape_001",
            },
            {
                "image_path": "sample_images/car.jpg",
                "text": "A red sports car on a highway",
                "image_id": "car_001",
            },
            {
                "image_path": "sample_images/food.jpg",
                "text": "Delicious pasta with tomato sauce",
                "image_id": "food_001",
            },
        ]
        
        # Create sample images directory and placeholder images
        sample_dir = self.data_dir / "sample_images"
        sample_dir.mkdir(parents=True, exist_ok=True)
        
        for item in sample_data:
            image_path = self.data_dir / item["image_path"]
            if not image_path.exists():
                # Create a placeholder image
                self._create_placeholder_image(image_path, item["text"])
        
        return sample_data
    
    def _create_placeholder_image(self, image_path: Path, text: str) -> None:
        """Create a placeholder image for demonstration."""
        # Create a simple colored image based on text content
        colors = {
            "cat": (255, 200, 200),  # Light red
            "dog": (200, 255, 200),  # Light green
            "landscape": (200, 200, 255),  # Light blue
            "car": (255, 255, 200),  # Light yellow
            "food": (255, 200, 255),  # Light magenta
        }
        
        # Find matching color
        color = (128, 128, 128)  # Default gray
        for keyword, col in colors.items():
            if keyword in text.lower():
                color = col
                break
        
        # Create image
        image = Image.new('RGB', (self.image_size, self.image_size), color)
        image.save(image_path)
    
    def _setup_augmentation(self) -> None:
        """Setup data augmentation transforms."""
        from torchvision import transforms
        
        self.augmentation_transforms = transforms.Compose([
            transforms.RandomHorizontalFlip(p=self.augmentation_prob),
            transforms.RandomRotation(degrees=5),
            transforms.ColorJitter(
                brightness=0.2,
                contrast=0.2,
                saturation=0.2,
                hue=0.1,
            ),
        ])
    
    def __len__(self) -> int:
        """Return dataset length."""
        return len(self.data)
    
    def __getitem__(self, idx: int) -> Dict[str, Any]:
        """
        Get dataset item.
        
        Args:
            idx: Item index
            
        Returns:
            Dictionary containing image, text, and metadata
        """
        item = self.data[idx]
        
        # Load image
        image_path = self.data_dir / item["image_path"]
        try:
            image = Image.open(image_path).convert("RGB")
        except Exception as e:
            # Create placeholder if image loading fails
            image = self._create_placeholder_image(image_path, item["text"])
            image = Image.open(image_path).convert("RGB")
        
        # Apply augmentation if training
        if self.use_augmentation and random.random() < self.augmentation_prob:
            image = self.augmentation_transforms(image)
        
        # Process with CLIP processor
        inputs = self.processor(
            text=item["text"],
            images=image,
            return_tensors="pt",
            padding=True,
            truncation=True,
        )
        
        return {
            "image": inputs["pixel_values"].squeeze(0),
            "input_ids": inputs["input_ids"].squeeze(0),
            "attention_mask": inputs["attention_mask"].squeeze(0),
            "text": item["text"],
            "image_id": item.get("image_id", str(idx)),
            "image_path": str(image_path),
        }
    
    def get_text_only(self, idx: int) -> str:
        """Get text only for a given index."""
        return self.data[idx]["text"]
    
    def get_image_only(self, idx: int) -> Image.Image:
        """Get image only for a given index."""
        item = self.data[idx]
        image_path = self.data_dir / item["image_path"]
        return Image.open(image_path).convert("RGB")


class ImageTextDataModule:
    """Data module for managing train/val/test splits."""
    
    def __init__(
        self,
        data_dir: Union[str, Path],
        batch_size: int = 32,
        num_workers: int = 4,
        pin_memory: bool = True,
        **dataset_kwargs,
    ):
        """
        Initialize data module.
        
        Args:
            data_dir: Directory containing the dataset
            batch_size: Batch size for data loaders
            num_workers: Number of worker processes
            pin_memory: Whether to pin memory
            **dataset_kwargs: Additional arguments for dataset
        """
        self.data_dir = Path(data_dir)
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.pin_memory = pin_memory
        self.dataset_kwargs = dataset_kwargs
        
        # Initialize datasets
        self.train_dataset = None
        self.val_dataset = None
        self.test_dataset = None
    
    def setup(self, stage: Optional[str] = None) -> None:
        """
        Setup datasets for given stage.
        
        Args:
            stage: Training stage ('fit', 'test', or None for all)
        """
        if stage == "fit" or stage is None:
            self.train_dataset = ImageTextDataset(
                self.data_dir, split="train", **self.dataset_kwargs
            )
            self.val_dataset = ImageTextDataset(
                self.data_dir, split="val", **self.dataset_kwargs
            )
        
        if stage == "test" or stage is None:
            self.test_dataset = ImageTextDataset(
                self.data_dir, split="test", **self.dataset_kwargs
            )
    
    def train_dataloader(self) -> torch.utils.data.DataLoader:
        """Get training data loader."""
        if self.train_dataset is None:
            self.setup("fit")
        
        return torch.utils.data.DataLoader(
            self.train_dataset,
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=self.num_workers,
            pin_memory=self.pin_memory,
            persistent_workers=self.num_workers > 0,
        )
    
    def val_dataloader(self) -> torch.utils.data.DataLoader:
        """Get validation data loader."""
        if self.val_dataset is None:
            self.setup("fit")
        
        return torch.utils.data.DataLoader(
            self.val_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            pin_memory=self.pin_memory,
            persistent_workers=self.num_workers > 0,
        )
    
    def test_dataloader(self) -> torch.utils.data.DataLoader:
        """Get test data loader."""
        if self.test_dataset is None:
            self.setup("test")
        
        return torch.utils.data.DataLoader(
            self.test_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            pin_memory=self.pin_memory,
            persistent_workers=self.num_workers > 0,
        )
