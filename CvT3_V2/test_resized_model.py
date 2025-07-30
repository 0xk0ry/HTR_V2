#!/usr/bin/env python3
"""
Test script for the updated HTR model with 64x512 image resizing
"""

import torch
import numpy as np
from PIL import Image
from model.HTR_3Stage import HTRModel, DEFAULT_VOCAB


def calculate_model_parameters(model):
    """Calculate and display model parameters"""
    total_params = 0
    trainable_params = 0
    
    print("\nModel Parameter Analysis:")
    print("=" * 60)
    
    for name, param in model.named_parameters():
        param_count = param.numel()
        total_params += param_count
        
        if param.requires_grad:
            trainable_params += param_count
            
        # Show details for major components
        if any(component in name for component in ['cvt', 'decoder', 'classifier', 'embedding']):
            print(f"{name:50} {param_count:>10,} {'(trainable)' if param.requires_grad else '(frozen)'}")
    
    print("=" * 60)
    print(f"{'Total Parameters:':<50} {total_params:>10,}")
    print(f"{'Trainable Parameters:':<50} {trainable_params:>10,}")
    print(f"{'Non-trainable Parameters:':<50} {total_params - trainable_params:>10,}")
    
    # Calculate memory usage (assuming float32)
    memory_mb = (total_params * 4) / (1024 * 1024)
    print(f"{'Estimated Memory (MB):':<50} {memory_mb:>10.2f}")
    
    return {
        'total': total_params,
        'trainable': trainable_params,
        'non_trainable': total_params - trainable_params,
        'memory_mb': memory_mb
    }


def get_model_summary(model):
    """Get a detailed breakdown of model components"""
    component_params = {}
    
    for name, param in model.named_parameters():
        # Group parameters by major components
        if 'cvt' in name:
            component = 'CVT Backbone'
        elif 'decoder' in name:
            component = 'CTC Decoder'
        elif 'classifier' in name or 'head' in name:
            component = 'Classification Head'
        elif 'embedding' in name or 'embed' in name:
            component = 'Embeddings'
        elif 'norm' in name or 'bn' in name:
            component = 'Normalization'
        else:
            component = 'Other'
            
        if component not in component_params:
            component_params[component] = 0
        component_params[component] += param.numel()
    
    print("\nParameter Breakdown by Component:")
    print("=" * 40)
    total = sum(component_params.values())
    for component, count in sorted(component_params.items(), key=lambda x: x[1], reverse=True):
        percentage = (count / total) * 100
        print(f"{component:<25} {count:>10,} ({percentage:5.1f}%)")
    
    return component_params


def test_htr_model():
    """Test the HTR model with 64x512 images"""

    # Model parameters
    vocab_size = len(DEFAULT_VOCAB)
    model = HTRModel(vocab_size=vocab_size, target_height=64, target_width=512)
    model.eval()

    print(f"Model created with vocab size: {vocab_size}")
    print(f"Target image size: 64x512")

    # Calculate and display model parameters
    param_info = calculate_model_parameters(model)
    
    # Get component breakdown
    component_breakdown = get_model_summary(model)

    # Test with dummy images
    batch_size = 2

    # Create dummy images of different sizes to test resizing
    dummy_images = []

    # Image 1: Different aspect ratio (will be resized)
    img1 = torch.randn(1, 40, 320)  # Original chunking size
    dummy_images.append(img1)

    # Image 2: Another different size
    img2 = torch.randn(1, 80, 640)  # Larger image
    dummy_images.append(img2)

    # Stack into batch (this will trigger resizing in forward pass)
    # For this test, we'll create properly sized images
    batch_images = torch.randn(batch_size, 1, 64, 512)

    print(f"Input batch shape: {batch_images.shape}")

    # Forward pass without targets (inference mode)
    with torch.no_grad():
        logits, input_lengths = model(batch_images)

    # Should be [W, B, vocab_size]
    print(f"Output logits shape: {logits.shape}")
    print(f"Input lengths: {input_lengths}")

    # Test decoding
    predictions = model.decode_predictions(logits, input_lengths)
    print(f"Decoded predictions: {predictions}")

    # Test with targets (training mode)
    dummy_targets = torch.tensor(
        [1, 2, 3, 4, 5, 6, 7, 8, 9, 10])  # Some character indices
    target_lengths = torch.tensor([5, 5])  # 5 characters each

    logits_train, loss = model(batch_images, dummy_targets, target_lengths)
    print(f"Training mode - Loss: {loss.item():.4f}")

    print("Test completed successfully!")


def test_image_resizer():
    """Test the ImageResizer class separately"""
    from model.HTR_3Stage import ImageResizer

    resizer = ImageResizer(target_height=64, target_width=512)

    # Test with numpy array
    dummy_np = np.random.randint(0, 255, (100, 200), dtype=np.uint8)
    resized_tensor = resizer.process_image(dummy_np)
    print(f"NumPy input shape: {dummy_np.shape}")
    # Should be [1, 1, 64, 512]
    print(f"Resized tensor shape: {resized_tensor.shape}")

    # Test with PIL Image
    dummy_pil = Image.fromarray(dummy_np)
    resized_tensor_pil = resizer.process_image(dummy_pil)
    print(f"PIL input size: {dummy_pil.size}")
    print(f"Resized PIL tensor shape: {resized_tensor_pil.shape}")

    print("ImageResizer test completed successfully!")


if __name__ == "__main__":
    print("Testing HTR Model with 64x512 resizing...")
    print("=" * 50)

    try:
        test_image_resizer()
        print()
        test_htr_model()
    except Exception as e:
        print(f"Test failed with error: {e}")
        import traceback
        traceback.print_exc()
