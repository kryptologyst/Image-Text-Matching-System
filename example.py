#!/usr/bin/env python3
"""
Simple example script demonstrating the image-text matching system.

This script shows how to use the system for basic image-text matching tasks.
"""

import sys
from pathlib import Path

# Add src to path
sys.path.append(str(Path(__file__).parent / "src"))

import torch
from PIL import Image
import numpy as np

from src.models.clip_model import CLIPImageTextMatcher
from src.utils.device import get_device, set_seed


def create_sample_image(text: str, size: tuple = (224, 224)) -> Image.Image:
    """Create a simple sample image based on text content."""
    # Create a colored image based on text content
    colors = {
        "cat": (255, 200, 200),  # Light red
        "dog": (200, 255, 200),  # Light green
        "car": (200, 200, 255),  # Light blue
        "house": (255, 255, 200),  # Light yellow
        "tree": (200, 255, 255),  # Light cyan
    }
    
    # Find matching color
    color = (128, 128, 128)  # Default gray
    for keyword, col in colors.items():
        if keyword in text.lower():
            color = col
            break
    
    # Create image
    image = Image.new('RGB', size, color)
    return image


def main():
    """Main example function."""
    print("🖼️ Image-Text Matching System Demo")
    print("=" * 50)
    
    # Set seed for reproducibility
    set_seed(42)
    
    # Setup device
    device = get_device("auto")
    print(f"Using device: {device}")
    
    # Initialize model
    print("Loading CLIP model...")
    model = CLIPImageTextMatcher()
    model.to(device)
    model.eval()
    
    # Get processor
    processor = model.get_processor()
    
    # Sample texts
    texts = [
        "A cute cat sitting on a windowsill",
        "A happy dog playing in the park",
        "A red sports car on a highway",
        "A beautiful house with a garden",
        "A tall tree in the forest",
    ]
    
    print(f"\nSample texts:")
    for i, text in enumerate(texts):
        print(f"  {i+1}. {text}")
    
    # Create sample images
    print(f"\nCreating sample images...")
    images = []
    for i, text in enumerate(texts):
        image = create_sample_image(text)
        images.append(image)
        print(f"  Created image {i+1}: {text}")
    
    # Test image-text matching
    print(f"\nTesting image-text matching...")
    
    with torch.no_grad():
        # Get embeddings for all images and texts
        image_embeds = []
        text_embeds = []
        
        # Process images
        for image in images:
            inputs = processor(images=image, return_tensors="pt")
            pixel_values = inputs["pixel_values"].to(device)
            embed = model.encode_image(pixel_values)
            image_embeds.append(embed.cpu())
        
        # Process texts
        for text in texts:
            inputs = processor(text=[text], return_tensors="pt")
            input_ids = inputs["input_ids"].to(device)
            attention_mask = inputs["attention_mask"].to(device)
            embed = model.encode_text(input_ids, attention_mask)
            text_embeds.append(embed.cpu())
        
        # Concatenate embeddings
        image_embeds = torch.cat(image_embeds, dim=0)
        text_embeds = torch.cat(text_embeds, dim=0)
        
        # Compute similarity matrix
        similarity_matrix = torch.matmul(image_embeds, text_embeds.t())
        
        # Convert to probabilities
        probabilities = torch.softmax(similarity_matrix, dim=1)
    
    # Display results
    print(f"\nSimilarity Matrix (Image → Text):")
    print("=" * 60)
    
    for i, image_text in enumerate(texts):
        print(f"\nImage {i+1}: {image_text}")
        print("-" * 40)
        
        # Get top matches for this image
        image_probs = probabilities[i]
        top_indices = torch.topk(image_probs, len(texts)).indices
        
        for rank, text_idx in enumerate(top_indices):
            prob = image_probs[text_idx].item()
            text = texts[text_idx]
            
            # Color coding based on probability
            if prob > 0.7:
                color = "🟢"
            elif prob > 0.5:
                color = "🟡"
            else:
                color = "🔴"
            
            print(f"  Rank {rank+1}: {color} {prob:.3f} - {text}")
    
    # Test single image-text matching
    print(f"\n" + "=" * 60)
    print("Single Image-Text Matching Test")
    print("=" * 60)
    
    # Test with first image
    test_image = images[0]
    test_texts = [
        "A cute cat sitting on a windowsill",  # Should match
        "A happy dog playing in the park",       # Should not match
        "A red sports car on a highway",        # Should not match
    ]
    
    print(f"Test Image: {texts[0]}")
    print(f"Test Texts:")
    for i, text in enumerate(test_texts):
        print(f"  {i+1}. {text}")
    
    with torch.no_grad():
        # Process test image
        inputs = processor(images=test_image, return_tensors="pt")
        pixel_values = inputs["pixel_values"].to(device)
        image_embed = model.encode_image(pixel_values)
        
        # Process test texts
        text_embeds = []
        for text in test_texts:
            inputs = processor(text=[text], return_tensors="pt")
            input_ids = inputs["input_ids"].to(device)
            attention_mask = inputs["attention_mask"].to(device)
            embed = model.encode_text(input_ids, attention_mask)
            text_embeds.append(embed.cpu())
        
        text_embeds = torch.cat(text_embeds, dim=0)
        
        # Compute similarities
        similarities = torch.matmul(image_embed, text_embeds.t()).squeeze()
        probabilities = torch.softmax(similarities, dim=0)
    
    print(f"\nResults:")
    print("-" * 30)
    
    for i, (text, prob) in enumerate(zip(test_texts, probabilities)):
        if prob > 0.7:
            color = "🟢"
        elif prob > 0.5:
            color = "🟡"
        else:
            color = "🔴"
        
        print(f"  {color} {prob:.3f} - {text}")
    
    print(f"\n" + "=" * 60)
    print("Demo completed successfully!")
    print("=" * 60)


if __name__ == "__main__":
    main()
