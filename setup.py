#!/usr/bin/env python3
"""
Setup script for the image-text matching system.

This script helps set up the environment and run basic tests.
"""

import subprocess
import sys
from pathlib import Path


def run_command(command: str, description: str) -> bool:
    """Run a command and return success status."""
    print(f"🔄 {description}...")
    try:
        result = subprocess.run(command, shell=True, check=True, capture_output=True, text=True)
        print(f"✅ {description} completed successfully")
        return True
    except subprocess.CalledProcessError as e:
        print(f"❌ {description} failed:")
        print(f"   Error: {e.stderr}")
        return False


def main():
    """Main setup function."""
    print("🚀 Setting up Image-Text Matching System")
    print("=" * 50)
    
    # Check Python version
    if sys.version_info < (3, 10):
        print("❌ Python 3.10 or higher is required")
        sys.exit(1)
    
    print(f"✅ Python {sys.version_info.major}.{sys.version_info.minor} detected")
    
    # Install dependencies
    print("\n📦 Installing dependencies...")
    
    # Try pip install first
    if not run_command("pip install -r requirements.txt", "Installing requirements"):
        print("⚠️  pip install failed, trying pip install --user")
        if not run_command("pip install --user -r requirements.txt", "Installing requirements (user)"):
            print("❌ Failed to install dependencies")
            print("   Please install manually: pip install -r requirements.txt")
            sys.exit(1)
    
    # Create necessary directories
    print("\n📁 Creating directories...")
    directories = [
        "data/images",
        "data/text", 
        "data/annotations",
        "checkpoints",
        "outputs",
        "assets",
        "logs",
    ]
    
    for directory in directories:
        Path(directory).mkdir(parents=True, exist_ok=True)
        print(f"✅ Created directory: {directory}")
    
    # Run basic tests
    print("\n🧪 Running basic tests...")
    
    # Test imports
    try:
        import torch
        import transformers
        import PIL
        import numpy as np
        print("✅ Core dependencies imported successfully")
    except ImportError as e:
        print(f"❌ Import error: {e}")
        sys.exit(1)
    
    # Test model loading
    try:
        sys.path.append(str(Path(__file__).parent / "src"))
        from src.models.clip_model import CLIPImageTextMatcher
        from src.utils.device import get_device
        
        device = get_device("cpu")
        model = CLIPImageTextMatcher()
        print("✅ Model loaded successfully")
    except Exception as e:
        print(f"❌ Model loading error: {e}")
        sys.exit(1)
    
    # Run unit tests if available
    if Path("tests").exists():
        print("\n🧪 Running unit tests...")
        if run_command("python -m pytest tests/ -v", "Running unit tests"):
            print("✅ All tests passed")
        else:
            print("⚠️  Some tests failed, but setup can continue")
    
    # Create sample data
    print("\n📝 Creating sample data...")
    
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
            "image_path": "sample_images/car.jpg",
            "text": "A red sports car on a highway",
            "image_id": "car_001",
        },
    ]
    
    import json
    with open("data/annotations/train.json", "w") as f:
        json.dump(sample_data, f, indent=2)
    
    print("✅ Sample data created")
    
    # Final message
    print("\n" + "=" * 50)
    print("🎉 Setup completed successfully!")
    print("=" * 50)
    print("\nNext steps:")
    print("1. Run the example: python example.py")
    print("2. Start the demo: streamlit run demo/app.py")
    print("3. Train a model: python scripts/train.py")
    print("\nFor more information, see README.md")


if __name__ == "__main__":
    main()
