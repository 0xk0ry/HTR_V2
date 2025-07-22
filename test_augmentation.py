#!/usr/bin/env python3
"""
Test script for data augmentation
"""

import torch
import numpy as np
from PIL import Image, ImageDraw, ImageFont
import matplotlib.pyplot as plt
from train import HTRDataset, DEFAULT_VOCAB
import os

def create_sample_image(text="Hello World", width=800, height=64):
    """Create a sample handwritten-style image"""
    # Create white image
    img = Image.new('L', (width, height), color=255)
    draw = ImageDraw.Draw(img)
    
    # Try to use a simple font
    try:
        # You might need to adjust the font path for your system
        font = ImageFont.load_default()
    except:
        font = None
    
    # Draw text
    bbox = draw.textbbox((0, 0), text, font=font)
    text_width = bbox[2] - bbox[0]
    text_height = bbox[3] - bbox[1]
    
    x = (width - text_width) // 2
    y = (height - text_height) // 2
    
    draw.text((x, y), text, fill=0, font=font)
    
    return img

def test_augmentation():
    """Test data augmentation pipeline"""
    
    # Create a temporary directory with test data
    test_dir = "temp_test_data"
    os.makedirs(test_dir, exist_ok=True)
    
    # Create sample images
    sample_texts = ["Hello World", "Test Sample", "Data Augmentation"]
    
    for i, text in enumerate(sample_texts):
        img = create_sample_image(text)
        img.save(f"{test_dir}/sample_{i}.png")
        
        # Create corresponding text file
        with open(f"{test_dir}/sample_{i}.txt", 'w') as f:
            f.write(text)
    
    print(f"Created {len(sample_texts)} test samples")
    
    # Test dataset without augmentation
    print("\n=== Testing without augmentation ===")
    dataset_no_aug = HTRDataset(test_dir, DEFAULT_VOCAB, augment=False)
    
    # Test dataset with augmentation  
    print("\n=== Testing with augmentation ===")
    dataset_aug = HTRDataset(test_dir, DEFAULT_VOCAB, augment=True)
    
    # Get sample without augmentation
    sample_orig = dataset_no_aug[0]
    print(f"Original image shape: {sample_orig['image'].shape}")
    print(f"Original text: '{sample_orig['original_text']}'")
    
    # Get multiple augmented versions of the same sample
    print(f"\nGenerating 5 augmented versions...")
    for i in range(5):
        sample_aug = dataset_aug[0]
        print(f"Augmented {i+1} image shape: {sample_aug['image'].shape}")
        
        # Check if image values are reasonable
        img_tensor = sample_aug['image']
        print(f"  - Min/Max values: {img_tensor.min():.3f}/{img_tensor.max():.3f}")
        print(f"  - Mean/Std: {img_tensor.mean():.3f}/{img_tensor.std():.3f}")
    
    # Clean up
    import shutil
    shutil.rmtree(test_dir)
    print(f"\nCleaned up test directory: {test_dir}")
    
    print("\n✅ Data augmentation test completed successfully!")

if __name__ == "__main__":
    test_augmentation()
