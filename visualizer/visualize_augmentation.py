#!/usr/bin/env python3
"""
Visualize data augmentation effects on example training data
"""

from CvT3_V2.model.HTR_3Stage import DEFAULT_VOCAB
from data.transform import (
    Erosion, Dilation, ElasticDistortion,
    RandomTransform, GaussianNoise, SaltAndPepperNoise,
    Opening, Closing
)
import torch
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
import torchvision.transforms as transforms
from pathlib import Path
import sys
sys.path.append('.')


def load_example_images():
    """Load all images from example_train_data"""
    data_dir = Path("example_train_data")
    images = []
    texts = []

    # Find all PNG files
    for img_file in sorted(data_dir.glob("*.png")):
        # Load image
        img = Image.open(img_file).convert('L')

        # Load corresponding text
        txt_file = img_file.with_suffix('.txt')
        if txt_file.exists():
            with open(txt_file, 'r') as f:
                text = f.read().strip()
        else:
            text = img_file.stem

        images.append(img)
        texts.append(text)

    return images, texts


def resize_to_target_height(image, target_height=64):
    """Resize image to target height while maintaining aspect ratio"""
    w, h = image.size
    aspect_ratio = w / h
    new_width = int(target_height * aspect_ratio)
    return image.resize((new_width, target_height), Image.LANCZOS)


def apply_single_augmentation(image, aug_type):
    """Apply a single type of augmentation"""
    if aug_type == "original":
        return image
    elif aug_type == "affine":
        transform = transforms.RandomAffine(
            degrees=10,
            shear=5,
            interpolation=transforms.InterpolationMode.BILINEAR
        )
        return transform(image)
    elif aug_type == "perspective":
        transform = transforms.RandomPerspective(distortion_scale=0.4, p=1.0)
        return transform(image)
    elif aug_type == "blur":
        transform = transforms.GaussianBlur(kernel_size=(3, 7), sigma=(0.1, 1.5))
        return transform(image)
    elif aug_type == "salt_pepper":
        transform = SaltAndPepperNoise(prob=0.02)
        return transform(image)
    elif aug_type == "color_jitter":
        transform = transforms.ColorJitter(brightness=0.1, contrast=0.1)
        return transform(image)
    elif aug_type == "opening":
        transform = Opening(kernel=(3, 3))
        return transform(image)
    elif aug_type == "closing":
        transform = Closing(kernel=(3, 3))
        return transform(image)
    elif aug_type == "erosion":
        transform = Erosion(kernel=(2, 2), iterations=1)
        return transform(image)
    elif aug_type == "dilation":
        transform = Dilation(kernel=(2, 2), iterations=1)
        return transform(image)
    elif aug_type == "noise":
        transform = GaussianNoise(std=5)
        return transform(image)
    elif aug_type == "elastic":
        transform = ElasticDistortion(
            grid=(8, 8), magnitude=(1, 1), min_sep=(4, 4)
        )
        return transform(image)
    else:
        return image


def create_augmentation_grid():
    """Create a vertical grid showing different augmentations on a single image"""

    # Load example images
    images, texts = load_example_images()
    print(f"Loaded {len(images)} example images")

    # Use only the first image
    if not images:
        print("No example images found!")
        return

    image = images[0]
    text = texts[0]
    print(f"Using image: '{text}'")

    # Augmentation types to show (updated list without random_erasing)
    aug_types = ["original", "affine", "perspective", "blur", "salt_pepper", 
                 "color_jitter", "opening", "closing", 
                 "erosion", "dilation", "noise", "elastic"]
    aug_names = ["Original", "Random Affine", "Perspective Warp", "Gaussian Blur", 
                 "Salt & Pepper Noise", "Color Jitter", 
                 "Morphological Opening", "Morphological Closing", "Erosion", 
                 "Dilation", "Gaussian Noise", "Elastic Distortion"]

    # Create vertical figure (13 rows, 1 column)
    fig, axes = plt.subplots(len(aug_types), 1, figsize=(12, 26))
    fig.suptitle(
        f'Data Augmentation Examples: "{text}"', fontsize=16, fontweight='bold')

    # Resize image to target height
    image = resize_to_target_height(image, target_height=64)

    for aug_idx, (aug_type, aug_name) in enumerate(zip(aug_types, aug_names)):
        # Apply augmentation
        try:
            aug_image = apply_single_augmentation(image.copy(), aug_type)

            # Ensure the image is still the right size after augmentation
            if aug_image.size[1] != 64:
                aug_image = resize_to_target_height(
                    aug_image, target_height=64)

            # Convert to numpy for plotting
            img_array = np.array(aug_image)

        except Exception as e:
            print(f"Error applying {aug_type}: {e}")
            # Fallback to original image
            img_array = np.array(image)

        # Plot
        ax = axes[aug_idx]
        ax.imshow(img_array, cmap='gray', aspect='auto')
        ax.set_title(f"{aug_name}", fontsize=14, fontweight='bold', pad=10)
        ax.axis('off')

        # Add some stats about the image
        min_val, max_val = img_array.min(), img_array.max()
        mean_val, std_val = img_array.mean(), img_array.std()
        ax.text(0.02, 0.02, f"Min: {min_val:.3f}, Max: {max_val:.3f}, Mean: {mean_val:.3f}, Std: {std_val:.3f}",
                transform=ax.transAxes, fontsize=9,
                bbox=dict(boxstyle="round,pad=0.3", facecolor="white", alpha=0.8))

    # Adjust layout
    plt.tight_layout()
    plt.subplots_adjust(top=0.96)

    # Save the plot
    output_path = "augmentation_examples.png"
    plt.savefig(output_path, dpi=150, bbox_inches='tight', facecolor='white')
    print(f"Saved augmentation examples to: {output_path}")

    # Show the plot
    plt.show()

    return fig


def create_pipeline_comparison():
    """Compare old vs new augmentation pipeline effects"""
    
    # Load first example image
    images, texts = load_example_images()
    if not images:
        print("No example images found!")
        return

    image = resize_to_target_height(images[0], target_height=64)
    text = texts[0]

    # Define old and new pipeline effects
    pipelines = [
        ("Original", []),
        ("Old Pipeline", ["erosion", "dilation", "elastic", "noise"]),
        ("New Core Transforms", ["affine", "perspective", "blur", "salt_pepper"]),
        ("New Full Pipeline", ["affine", "perspective", "blur", "salt_pepper", 
                              "color_jitter", "opening"])
    ]

    # Create vertical figure (4 rows, 1 column)
    fig, axes = plt.subplots(len(pipelines), 1, figsize=(12, 12))
    fig.suptitle(
        f'Pipeline Comparison: "{text}"', fontsize=16, fontweight='bold')

    for idx, (pipeline_name, aug_list) in enumerate(pipelines):
        # Start with original image
        result_image = image.copy()

        # Apply each augmentation in sequence
        for aug_type in aug_list:
            try:
                result_image = apply_single_augmentation(result_image, aug_type)
                # Ensure size consistency
                if result_image.size[1] != 64:
                    result_image = resize_to_target_height(result_image, target_height=64)
            except Exception as e:
                print(f"Error in pipeline {pipeline_name} with {aug_type}: {e}")
                continue

        # Convert to numpy for plotting
        img_array = np.array(result_image)

        # Plot
        ax = axes[idx]
        ax.imshow(img_array, cmap='gray', aspect='auto')
        ax.set_title(pipeline_name, fontsize=14, fontweight='bold', pad=10)
        ax.axis('off')

        # Add some stats about the image
        min_val, max_val = img_array.min(), img_array.max()
        mean_val, std_val = img_array.mean(), img_array.std()
        ax.text(0.02, 0.02, f"Min: {min_val:.3f}, Max: {max_val:.3f}, Mean: {mean_val:.3f}, Std: {std_val:.3f}",
                transform=ax.transAxes, fontsize=9,
                bbox=dict(boxstyle="round,pad=0.3", facecolor="white", alpha=0.8))

    # Adjust layout
    plt.tight_layout()
    plt.subplots_adjust(top=0.94)

    # Save the plot
    output_path = "pipeline_comparison.png"
    plt.savefig(output_path, dpi=150, bbox_inches='tight', facecolor='white')
    print(f"Saved pipeline comparison to: {output_path}")

    plt.show()

    return fig


def create_combined_augmentation_example():
    """Show an example with multiple augmentations combined in vertical layout"""

    # Load first example image
    images, texts = load_example_images()
    if not images:
        print("No example images found!")
        return

    image = resize_to_target_height(images[0], target_height=64)
    text = texts[0]

    # Create combinations (updated with new augmentations)
    combinations = [
        ("Original", []),
        ("Affine + Blur", ["affine", "blur"]),
        ("Perspective + Salt&Pepper", ["perspective", "salt_pepper"]),
        ("Color Jitter + Erosion", ["color_jitter", "erosion"]),
        ("Opening + Dilation", ["opening", "dilation"]),
        ("Elastic + Noise + Dilation", ["elastic", "noise", "dilation"]),
        ("Full Pipeline (Light)", ["affine", "blur", "salt_pepper", "erosion"]),
        ("Full Pipeline (Heavy)", ["affine", "perspective", "blur", "salt_pepper", 
                                   "color_jitter", "opening", "noise"])
    ]

    # Create vertical figure (8 rows, 1 column)
    fig, axes = plt.subplots(len(combinations), 1, figsize=(12, 20))
    fig.suptitle(
        f'Combined Augmentation Examples: "{text}"', fontsize=16, fontweight='bold')

    for idx, (combo_name, aug_list) in enumerate(combinations):
        # Start with original image
        result_image = image.copy()

        # Apply each augmentation in sequence
        for aug_type in aug_list:
            try:
                result_image = apply_single_augmentation(
                    result_image, aug_type)
                # Ensure size consistency
                if result_image.size[1] != 64:
                    result_image = resize_to_target_height(
                        result_image, target_height=64)
            except Exception as e:
                print(
                    f"Error in combination {combo_name} with {aug_type}: {e}")
                continue

        # Convert to numpy for plotting
        img_array = np.array(result_image)

        # Plot
        ax = axes[idx]
        ax.imshow(img_array, cmap='gray', aspect='auto')
        ax.set_title(combo_name, fontsize=14, fontweight='bold', pad=10)
        ax.axis('off')

        # Add some stats about the image
        min_val, max_val = img_array.min(), img_array.max()
        mean_val, std_val = img_array.mean(), img_array.std()
        ax.text(0.02, 0.02, f"Min: {min_val:.3f}, Max: {max_val:.3f}, Mean: {mean_val:.3f}, Std: {std_val:.3f}",
                transform=ax.transAxes, fontsize=9,
                bbox=dict(boxstyle="round,pad=0.3", facecolor="white", alpha=0.8))

    # Adjust layout
    plt.tight_layout()
    plt.subplots_adjust(top=0.96)

    # Save the plot
    output_path = "combined_augmentation_examples.png"
    plt.savefig(output_path, dpi=150, bbox_inches='tight', facecolor='white')
    print(f"Saved combined augmentation examples to: {output_path}")

    plt.show()

    return fig


def main():
    """Main function to create all visualizations"""
    print("Creating Data Augmentation Visualizations")
    print("=" * 50)

    # Create individual augmentation grid (vertical layout, single image)
    print("\n1. Creating individual augmentation examples (vertical layout)...")
    create_augmentation_grid()

    # Create combined augmentation examples (vertical layout)
    print("\n2. Creating combined augmentation examples (vertical layout)...")
    create_combined_augmentation_example()

    # Create pipeline comparison
    print("\n3. Creating pipeline comparison (old vs new)...")
    create_pipeline_comparison()

    print("\n✅ Augmentation visualization completed!")
    print("\nGenerated files:")
    print("  - augmentation_examples.png (individual augmentations, vertical layout)")
    print("  - combined_augmentation_examples.png (combined effects, vertical layout)")
    print("  - pipeline_comparison.png (old vs new pipeline comparison)")


if __name__ == "__main__":
    main()
