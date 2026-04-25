"""Unit tests for the image-text matching system."""

import pytest
import torch
from PIL import Image
import numpy as np
from pathlib import Path
import tempfile
import json

from src.models.clip_model import CLIPImageTextMatcher
from src.data.dataset import ImageTextDataset, ImageTextDataModule
from src.losses.contrastive_loss import ContrastiveLoss, TripletLoss, MarginLoss
from src.eval.metrics import compute_retrieval_metrics
from src.utils.device import get_device, set_seed
from src.utils.config import load_config, save_config


class TestCLIPModel:
    """Test CLIP model functionality."""
    
    def test_model_initialization(self):
        """Test model initialization."""
        model = CLIPImageTextMatcher()
        assert model is not None
        assert hasattr(model, 'clip_model')
        assert hasattr(model, 'vision_projection')
        assert hasattr(model, 'text_projection')
    
    def test_model_forward(self):
        """Test model forward pass."""
        model = CLIPImageTextMatcher()
        device = get_device("cpu")
        model.to(device)
        
        # Create dummy inputs
        batch_size = 2
        pixel_values = torch.randn(batch_size, 3, 224, 224)
        input_ids = torch.randint(0, 1000, (batch_size, 77))
        attention_mask = torch.ones(batch_size, 77)
        
        # Forward pass
        with torch.no_grad():
            outputs = model(pixel_values, input_ids, attention_mask)
        
        # Check outputs
        assert "image_embeds" in outputs
        assert "text_embeds" in outputs
        assert "logits_per_image" in outputs
        assert "logits_per_text" in outputs
        
        # Check shapes
        assert outputs["image_embeds"].shape[0] == batch_size
        assert outputs["text_embeds"].shape[0] == batch_size
        assert outputs["logits_per_image"].shape == (batch_size, batch_size)
        assert outputs["logits_per_text"].shape == (batch_size, batch_size)
    
    def test_embedding_normalization(self):
        """Test that embeddings are normalized."""
        model = CLIPImageTextMatcher()
        device = get_device("cpu")
        model.to(device)
        
        # Create dummy inputs
        pixel_values = torch.randn(1, 3, 224, 224)
        input_ids = torch.randint(0, 1000, (1, 77))
        attention_mask = torch.ones(1, 77)
        
        with torch.no_grad():
            outputs = model(pixel_values, input_ids, attention_mask)
        
        # Check normalization
        image_norm = torch.norm(outputs["image_embeds"], p=2, dim=-1)
        text_norm = torch.norm(outputs["text_embeds"], p=2, dim=-1)
        
        assert torch.allclose(image_norm, torch.ones_like(image_norm), atol=1e-6)
        assert torch.allclose(text_norm, torch.ones_like(text_norm), atol=1e-6)


class TestDataset:
    """Test dataset functionality."""
    
    def test_dataset_creation(self):
        """Test dataset creation with sample data."""
        with tempfile.TemporaryDirectory() as temp_dir:
            # Create sample data
            data_dir = Path(temp_dir)
            images_dir = data_dir / "sample_images"
            images_dir.mkdir()
            
            # Create sample images
            for i in range(3):
                image = Image.new('RGB', (224, 224), color=(i*80, i*80, i*80))
                image.save(images_dir / f"image_{i}.jpg")
            
            # Create sample data
            sample_data = [
                {
                    "image_path": f"sample_images/image_{i}.jpg",
                    "text": f"Sample text {i}",
                    "image_id": f"img_{i}"
                }
                for i in range(3)
            ]
            
            with open(data_dir / "train.json", 'w') as f:
                json.dump(sample_data, f)
            
            # Test dataset
            dataset = ImageTextDataset(data_dir, split="train")
            assert len(dataset) == 3
            
            # Test item retrieval
            item = dataset[0]
            assert "image" in item
            assert "text" in item
            assert "image_id" in item
            assert item["text"] == "Sample text 0"
    
    def test_data_module(self):
        """Test data module functionality."""
        with tempfile.TemporaryDirectory() as temp_dir:
            data_dir = Path(temp_dir)
            
            # Create sample data for all splits
            for split in ["train", "val", "test"]:
                images_dir = data_dir / "sample_images"
                images_dir.mkdir(exist_ok=True)
                
                # Create sample images
                for i in range(2):
                    image = Image.new('RGB', (224, 224), color=(i*80, i*80, i*80))
                    image.save(images_dir / f"{split}_image_{i}.jpg")
                
                # Create sample data
                sample_data = [
                    {
                        "image_path": f"sample_images/{split}_image_{i}.jpg",
                        "text": f"{split} text {i}",
                        "image_id": f"{split}_img_{i}"
                    }
                    for i in range(2)
                ]
                
                with open(data_dir / f"{split}.json", 'w') as f:
                    json.dump(sample_data, f)
            
            # Test data module
            data_module = ImageTextDataModule(data_dir, batch_size=2)
            data_module.setup("fit")
            
            # Test dataloaders
            train_loader = data_module.train_dataloader()
            val_loader = data_module.val_dataloader()
            
            assert len(train_loader) == 1  # 2 samples / batch_size 2 = 1 batch
            assert len(val_loader) == 1
            
            # Test batch
            batch = next(iter(train_loader))
            assert batch["image"].shape[0] == 2
            assert batch["input_ids"].shape[0] == 2


class TestLossFunctions:
    """Test loss functions."""
    
    def test_contrastive_loss(self):
        """Test contrastive loss computation."""
        loss_fn = ContrastiveLoss(temperature=0.07)
        
        # Create dummy logits
        batch_size = 4
        logits_per_image = torch.randn(batch_size, batch_size)
        logits_per_text = logits_per_image.t()
        
        # Compute loss
        loss = loss_fn(logits_per_image, logits_per_text)
        
        assert isinstance(loss, torch.Tensor)
        assert loss.item() > 0
        assert loss.requires_grad
    
    def test_triplet_loss(self):
        """Test triplet loss computation."""
        loss_fn = TripletLoss(margin=0.2)
        
        # Create dummy embeddings
        batch_size = 4
        embedding_dim = 512
        image_embeds = torch.randn(batch_size, embedding_dim)
        text_embeds = torch.randn(batch_size, embedding_dim)
        
        # Normalize embeddings
        image_embeds = torch.nn.functional.normalize(image_embeds, p=2, dim=-1)
        text_embeds = torch.nn.functional.normalize(text_embeds, p=2, dim=-1)
        
        # Compute loss
        loss = loss_fn(image_embeds, text_embeds)
        
        assert isinstance(loss, torch.Tensor)
        assert loss.item() >= 0
        assert loss.requires_grad
    
    def test_margin_loss(self):
        """Test margin loss computation."""
        loss_fn = MarginLoss(margin=0.2)
        
        # Create dummy embeddings
        batch_size = 4
        embedding_dim = 512
        image_embeds = torch.randn(batch_size, embedding_dim)
        text_embeds = torch.randn(batch_size, embedding_dim)
        
        # Normalize embeddings
        image_embeds = torch.nn.functional.normalize(image_embeds, p=2, dim=-1)
        text_embeds = torch.nn.functional.normalize(text_embeds, p=2, dim=-1)
        
        # Compute loss
        loss = loss_fn(image_embeds, text_embeds)
        
        assert isinstance(loss, torch.Tensor)
        assert loss.item() >= 0
        assert loss.requires_grad


class TestMetrics:
    """Test evaluation metrics."""
    
    def test_retrieval_metrics(self):
        """Test retrieval metrics computation."""
        # Create dummy similarity matrix
        batch_size = 5
        similarity_matrix = torch.randn(batch_size, batch_size)
        
        # Make diagonal elements highest (perfect retrieval)
        for i in range(batch_size):
            similarity_matrix[i, i] = similarity_matrix[i].max() + 1
        
        # Compute metrics
        metrics = compute_retrieval_metrics(
            similarity_matrix,
            top_k_values=[1, 5],
            metrics=["recall_at_1", "recall_at_5", "median_rank"]
        )
        
        # Check metrics
        assert "recall_at_1" in metrics
        assert "recall_at_5" in metrics
        assert "median_rank" in metrics
        
        # Perfect retrieval should give recall@1 = 1.0
        assert metrics["recall_at_1"] == 1.0
        assert metrics["recall_at_5"] == 1.0
        assert metrics["median_rank"] == 1.0


class TestUtils:
    """Test utility functions."""
    
    def test_device_management(self):
        """Test device management."""
        device = get_device("cpu")
        assert device.type == "cpu"
        
        # Test device info
        device_info = get_device_info()
        assert "cuda_available" in device_info
        assert "cpu_count" in device_info
    
    def test_seed_setting(self):
        """Test seed setting."""
        set_seed(42)
        
        # Generate random numbers
        torch_rand = torch.rand(1).item()
        np_rand = np.random.rand()
        
        # Reset seed and generate again
        set_seed(42)
        torch_rand2 = torch.rand(1).item()
        np_rand2 = np.random.rand()
        
        # Should be the same
        assert torch_rand == torch_rand2
        assert np_rand == np_rand2
    
    def test_config_management(self):
        """Test configuration management."""
        with tempfile.NamedTemporaryFile(mode='w', suffix='.yaml', delete=False) as f:
            config_data = {
                "model": {"name": "test_model"},
                "training": {"epochs": 10}
            }
            import yaml
            yaml.dump(config_data, f)
            config_path = f.name
        
        try:
            # Test loading
            config = load_config(config_path)
            assert config["model"]["name"] == "test_model"
            assert config["training"]["epochs"] == 10
            
            # Test saving
            output_path = config_path.replace('.yaml', '_output.yaml')
            save_config(config, output_path)
            
            # Verify saved config
            loaded_config = load_config(output_path)
            assert loaded_config["model"]["name"] == "test_model"
            
        finally:
            # Cleanup
            Path(config_path).unlink()
            if Path(output_path).exists():
                Path(output_path).unlink()


if __name__ == "__main__":
    pytest.main([__file__])
