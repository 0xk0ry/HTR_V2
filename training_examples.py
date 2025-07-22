#!/usr/bin/env python3
"""
Example training script usage with data augmentation
"""

print("HTR Model Training with Data Augmentation")
print("=" * 50)

print("\n1. Basic training (no augmentation):")
print("python train.py --data_dir ./example_train_data --epochs 50 --batch_size 4")

print("\n2. Training with data augmentation:")
print("python train.py --data_dir ./example_train_data --epochs 50 --batch_size 4 --augment")

print("\n3. Training with validation and augmentation:")
print("python train.py --data_dir ./example_train_data --val_data_dir ./example_val_data --epochs 50 --batch_size 4 --augment")

print("\n4. Training with SAM optimizer and augmentation:")
print("python train.py --data_dir ./example_train_data --epochs 50 --batch_size 4 --augment --use_sam --sam_rho 0.05")

print("\n5. Resume training with augmentation:")
print("python train.py --data_dir ./example_train_data --epochs 100 --batch_size 4 --augment --resume ./checkpoints/best_model.pth")

print("\nData Augmentation Details:")
print("-" * 30)
print("✅ Random affine transforms (p=0.5)")
print("✅ Erosion & Dilation (p=0.5)")  
print("✅ Color jitter (brightness/contrast) (p=0.5)")
print("✅ Elastic distortion (p=0.5)")
print("✅ All transforms can be combined")
print("✅ Validation data is never augmented")

print("\nModel Specifications:")
print("-" * 20)
print("• Input: 64×512px chunks (greyscale)")
print("• Chunking: 320px first stride, 384px subsequent")
print("• 3-stage CvT architecture")
print("• ~31 tokens per chunk")
print("• CTC loss for sequence training")

print("\nRecommended Settings:")
print("-" * 20)
print("• Batch size: 4-8 (depending on GPU memory)")
print("• Learning rate: 1e-4")
print("• Epochs: 50-100")
print("• Use --augment for better generalization")
print("• Use --use_sam for improved convergence")
