"""
New HTR Image Chunking Visualizer
A clean implementation showing proper padding and ignored regions.
"""

from model.HTR_1Stage import ImageChunker
import torch
import torch.nn.functional as F
import torchvision.transforms as transforms
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from PIL import Image, ImageDraw, ImageFont
import numpy as np
import os
from pathlib import Path


def tensor_to_pil(tensor):
    """Convert normalized tensor back to PIL Image (handles both grayscale and RGB)"""
    # Handle different tensor shapes
    if len(tensor.shape) == 2:
        tensor = tensor.unsqueeze(0)  # [H, W] -> [1, H, W]
    
    num_channels = tensor.shape[0]
    
    # Denormalize based on number of channels
    if num_channels == 1:
        mean = torch.tensor([0.5]).view(1, 1, 1)
        std = torch.tensor([0.5]).view(1, 1, 1)
        tensor = tensor * std + mean
        tensor = torch.clamp(tensor, 0, 1)
        # Convert to PIL grayscale
        array = (tensor.squeeze(0).numpy() * 255).astype(np.uint8)
        return Image.fromarray(array, mode='L').convert('RGB')
    else:
        mean = torch.tensor([0.5, 0.5, 0.5]).view(3, 1, 1)
        std = torch.tensor([0.5, 0.5, 0.5]).view(3, 1, 1)
        tensor = tensor * std + mean
        tensor = torch.clamp(tensor, 0, 1)
        # Convert to PIL RGB
        array = (tensor.permute(1, 2, 0).numpy() * 255).astype(np.uint8)
        return Image.fromarray(array)


def add_overlay_to_chunk(chunk_pil, chunk_idx, total_chunks, chunk_positions):
    """Add proper overlays to show padding, overlap, and ignored regions"""
    chunk_with_overlay = chunk_pil.copy()
    draw = ImageDraw.Draw(chunk_with_overlay)

    width, height = chunk_pil.size
    start, end, left_pad, _ = chunk_positions[chunk_idx]

    # Parameters
    chunk_width = 320
    stride = 240
    padding = 40
    overlap_size = chunk_width - stride  # 80px
    ignored_size = 40  # 40px ignored from each side of overlap

    try:
        font = ImageFont.truetype("arial.ttf", 10)
        small_font = ImageFont.truetype("arial.ttf", 8)
    except:
        font = ImageFont.load_default()
        small_font = ImageFont.load_default()

    # 1. RED overlay for actual padding (first and last chunks only)
    if chunk_idx == 0 and left_pad > 0:
        # Left padding (40px) - RED
        padding_width = int((padding / chunk_width) * width)

        # Draw red semi-transparent overlay
        for y in range(height):
            for x in range(padding_width):
                original = chunk_pil.getpixel((x, y))
                blended = tuple(
                    int(original[i] * 0.5 + (255 if i == 0 else 0) * 0.5) for i in range(3))
                chunk_with_overlay.putpixel((x, y), blended)

        # Add text
        text = f"PAD\n{padding}px"
        text = ''
        draw.text((padding_width//2, height//2), text,
                  fill='white', font=font, anchor='mm')

    # Right padding for last chunk (if needed)
    if chunk_idx == total_chunks - 1:
        actual_content_width = end - start
        total_width_needed = actual_content_width + \
            (padding if chunk_idx == 0 else 0)

        if total_width_needed < chunk_width:
            right_pad_needed = chunk_width - total_width_needed
            right_pad_width = int((right_pad_needed / chunk_width) * width)
            right_start = width - right_pad_width

            # Draw red semi-transparent overlay for right padding
            for y in range(height):
                for x in range(right_start, width):
                    original = chunk_pil.getpixel((x, y))
                    blended = tuple(
                        int(original[i] * 0.5 + (255 if i == 0 else 0) * 0.5) for i in range(3))
                    chunk_with_overlay.putpixel((x, y), blended)

            # Add text
            text = f"PAD\n{right_pad_needed}px"
            text = ''
            draw.text((right_start + right_pad_width//2, height//2),
                      text, fill='white', font=font, anchor='mm')

    # 2. LIGHT BLUE overlays for overlap regions (80px each side) - Draw first

    # Left overlap region (80px from previous chunk)
    if chunk_idx > 0:
        # This chunk has 80px overlap with previous chunk at the beginning
        overlap_left_width = int((overlap_size / chunk_width) * width)

        # Calculate start position (after left padding if present)
        start_x = int((padding / chunk_width) * width) if chunk_idx == 0 else 0

        # Draw light blue semi-transparent overlay
        for y in range(height):
            for x in range(start_x, start_x + overlap_left_width):
                if x < width:
                    original = chunk_with_overlay.getpixel((x, y))
                    # Light blue (173, 216, 230)
                    blended = tuple(
                        int(original[i] * 0.6 + [173, 216, 230][i] * 0.4) for i in range(3))
                    chunk_with_overlay.putpixel((x, y), blended)

        # Add text
        text = f"OVERLAP\n{overlap_size}px"
        text = ''
        draw.text((start_x + overlap_left_width//2, height//6),
                  text, fill='navy', font=small_font, anchor='mm')

    # Right overlap region (80px with next chunk)
    if chunk_idx < total_chunks - 1:
        # This chunk has 80px overlap with next chunk at the end
        overlap_right_width = int((overlap_size / chunk_width) * width)

        # Calculate where the overlap region starts (from the right side)
        overlap_start = width - overlap_right_width

        # Draw light blue semi-transparent overlay
        for y in range(height):
            for x in range(overlap_start, width):
                original = chunk_with_overlay.getpixel((x, y))
                # Light blue (173, 216, 230)
                blended = tuple(
                    int(original[i] * 0.6 + [173, 216, 230][i] * 0.4) for i in range(3))
                chunk_with_overlay.putpixel((x, y), blended)

        # Add text
        text = f"OVERLAP\n{overlap_size}px"
        text = ''
        draw.text((overlap_start + overlap_right_width//2, height//6),
                  text, fill='navy', font=small_font, anchor='mm')

    # 3. GRAY overlays for ignored regions during merging (40px each side) - Draw on top

    # Left ignored region (40px that gets ignored during merging)
    if chunk_idx > 0:
        # During merging, we ignore the first 40px of overlap
        ignored_left_width = int((ignored_size / chunk_width) * width)

        # Calculate start position (after left padding if present)
        start_x = int((padding / chunk_width) * width) if chunk_idx == 0 else 0

        # Draw gray semi-transparent overlay on top of overlap
        for y in range(height):
            for x in range(start_x, start_x + ignored_left_width):
                if x < width:
                    original = chunk_with_overlay.getpixel((x, y))
                    blended = tuple(
                        int(original[i] * 0.5 + 128 * 0.5) for i in range(3))
                    chunk_with_overlay.putpixel((x, y), blended)

        # Add text
        text = f"IGNORE\n{ignored_size}px"
        text = ''
        draw.text((start_x + ignored_left_width//2, height//3),
                  text, fill='white', font=small_font, anchor='mm')

    # Right ignored region (40px that gets ignored during merging)
    if chunk_idx < total_chunks - 1:
        # During merging, we ignore the last 40px of overlap
        ignored_right_width = int((ignored_size / chunk_width) * width)

        # Calculate where the ignored region starts (from the right side)
        ignored_start = width - ignored_right_width

        # Draw gray semi-transparent overlay on top of overlap
        for y in range(height):
            for x in range(ignored_start, width):
                original = chunk_with_overlay.getpixel((x, y))
                blended = tuple(
                    int(original[i] * 0.5 + 128 * 0.5) for i in range(3))
                chunk_with_overlay.putpixel((x, y), blended)

        # Add text
        text = f"IGNORE\n{ignored_size}px"
        text = ''
        draw.text((ignored_start + ignored_right_width//2, 2*height//3),
                  text, fill='white', font=small_font, anchor='mm')

    return chunk_with_overlay


def create_visualization(original_image, preprocessed_image, chunks, chunk_positions, base_name, output_dir):
    """Create comprehensive visualization with matplotlib"""
    num_chunks = len(chunks)

    # Create merged visualization
    merged_image, useful_regions = create_merged_visualization(preprocessed_image, chunk_positions)

    # Create figure with one more row for merged image
    fig, axes = plt.subplots(
        3 + num_chunks, 1, figsize=(16, 6 + num_chunks * 2.5))

    if not isinstance(axes, (list, np.ndarray)):
        axes = [axes]

    # 1. Original image
    axes[0].imshow(original_image)
    axes[0].set_title(f"Original Image: {original_image.size[0]} × {original_image.size[1]} pixels",
                      fontweight='bold', fontsize=12)
    axes[0].axis('off')

    # 2. Preprocessed image with chunk boundaries
    axes[1].imshow(preprocessed_image)
    axes[1].set_title(f"Preprocessed Image: {preprocessed_image.size[0]} × {preprocessed_image.size[1]} pixels | "
                      f"Chunking: 320px width, 240px stride, 80px overlap",
                      fontweight='bold', fontsize=12)

    # Draw chunk boundaries
    colors = ['red', 'blue', 'green', 'orange', 'purple', 'brown', 'pink']
    for i, (start, end, left_pad, _) in enumerate(chunk_positions):
        color = colors[i % len(colors)]

        # Draw content boundary
        rect = patches.Rectangle((start, 0), end-start, preprocessed_image.size[1],
                                 linewidth=2, edgecolor=color, facecolor='none', linestyle='-')
        axes[1].add_patch(rect)

        # Add chunk label
        axes[1].text(start + 5, 10, f"C{i+1}", fontsize=10, fontweight='bold',
                     bbox=dict(boxstyle="round,pad=0.2", facecolor='white', edgecolor=color))

        # Show overlap regions (light blue)
        if i > 0:
            prev_start, prev_end, _, _ = chunk_positions[i-1]
            overlap_start = max(start, prev_start)
            overlap_end = min(end, prev_end)
            if overlap_end > overlap_start:
                overlap_rect = patches.Rectangle((overlap_start, 0), overlap_end-overlap_start,
                                                 preprocessed_image.size[1],
                                                 linewidth=1, edgecolor='lightblue', facecolor='lightblue',
                                                 alpha=0.4, linestyle='--')
                axes[1].add_patch(overlap_rect)

    axes[1].axis('off')

    # 3. Merged image (after applying merging logic)
    if merged_image:
        axes[2].imshow(merged_image)
        
        # Calculate total useful content
        total_useful_width = sum(end - start for start, end in useful_regions)
        efficiency = (total_useful_width / preprocessed_image.size[0]) * 100
        
        axes[2].set_title(f"After Merging: {merged_image.size[0]} × {merged_image.size[1]} pixels | "
                         f"Useful Content: {total_useful_width}px ({efficiency:.1f}% efficiency) | "
                         f"Removed: Padding + 40px from each overlap side",
                         fontweight='bold', fontsize=12)
        
        # Highlight the useful regions on the merged image
        for i, (region_start, region_end) in enumerate(useful_regions):
            # Calculate position in merged image
            region_offset = sum(useful_regions[j][1] - useful_regions[j][0] for j in range(i))
            region_width = region_end - region_start
            
            # Draw boundary for each useful region
            color = ['red', 'blue', 'green', 'orange', 'purple', 'brown', 'pink'][i % 7]
            rect = patches.Rectangle((region_offset, 0), region_width, merged_image.size[1],
                                   linewidth=2, edgecolor=color, facecolor='none', linestyle='-')
            axes[2].add_patch(rect)
            
            # Add region label
            axes[2].text(region_offset + 5, 10, f"R{i+1}", fontsize=10, fontweight='bold',
                        bbox=dict(boxstyle="round,pad=0.2", facecolor='white', edgecolor=color))
    else:
        axes[2].text(0.5, 0.5, "No useful content after merging", 
                    ha='center', va='center', transform=axes[2].transAxes, fontsize=12)
        axes[2].set_title("After Merging: No Content", fontweight='bold', fontsize=12)
    
    axes[2].axis('off')

    # 4. Individual chunks with overlays
    for i, chunk_tensor in enumerate(chunks):
        if i + 3 >= len(axes):
            break

        # Convert tensor to PIL
        if len(chunk_tensor.shape) == 4:
            chunk_tensor = chunk_tensor.squeeze(0)
        chunk_pil = tensor_to_pil(chunk_tensor)

        # Add overlays
        chunk_with_overlays = add_overlay_to_chunk(
            chunk_pil, i, num_chunks, chunk_positions)

        # Display
        axes[i + 3].imshow(chunk_with_overlays)

        # Create detailed title
        start, end, left_pad, _ = chunk_positions[i]
        content_width = end - start

        # Calculate overlaps and ignored regions
        overlap_left = 80 if i > 0 else 0
        overlap_right = 80 if i < num_chunks - 1 else 0
        ignored_left = 40 if i > 0 else 0
        ignored_right = 40 if i < num_chunks - 1 else 0
        useful_content = content_width - ignored_left - ignored_right

        title = (f"Chunk {i+1}: Content {start}→{end} ({content_width}px) | "
                 f"Left Pad: {left_pad}px | "
                 f"Overlap: L{overlap_left}px R{overlap_right}px | "
                 f"Ignored: L{ignored_left}px R{ignored_right}px | "
                 f"Useful: {useful_content}px")

        axes[i + 3].set_title(title, fontsize=10)
        axes[i + 3].axis('off')

        # Add color legend on the side
        # if i == 0:
        #     legend_elements = [
        #         plt.Rectangle((0, 0), 1, 1, facecolor='red', alpha=0.5,
        #                       label='Padding (added gray background)'),
        #         plt.Rectangle((0, 0), 1, 1, facecolor='lightblue',
        #                       alpha=0.4, label='Overlap regions (80px each side)'),
        #         plt.Rectangle((0, 0), 1, 1, facecolor='gray', alpha=0.5,
        #                       label='Ignored during merge (40px each side)'),
        #         plt.Rectangle((0, 0), 1, 1, facecolor='white',
        #                       alpha=1, label='Useful content')
        #     ]
        #     axes[i + 3].legend(handles=legend_elements,
        #                        loc='center left', bbox_to_anchor=(1, 0.5))

    plt.tight_layout()

    # Save
    viz_path = os.path.join(
        output_dir, f"{base_name}_new_chunking_visualization.png")
    plt.savefig(viz_path, dpi=150, bbox_inches='tight')
    plt.close()

    return viz_path


def create_merged_visualization(preprocessed_image, chunk_positions):
    """Create a visualization showing what the image looks like after merging (only useful content)"""
    # Parameters
    ignored_size = 40  # 40px ignored from each side of overlap

    # Calculate useful regions for each chunk
    useful_regions = []
    for i, (start, end, left_pad, _) in enumerate(chunk_positions):
        content_start = start
        content_end = end

        # Remove ignored regions based on merging logic
        if len(chunk_positions) == 1:
            # Single chunk - only remove left padding
            if left_pad > 0:
                content_start = start + left_pad
        else:
            if i == 0:
                # First chunk - remove left padding and right ignored region
                if left_pad > 0:
                    content_start = start + left_pad
                if i < len(chunk_positions) - 1:  # Not the last chunk
                    content_end = end - ignored_size
            elif i == len(chunk_positions) - 1:
                # Last chunk - remove left ignored region
                content_start = start + ignored_size
            else:
                # Middle chunk - remove ignored regions from both sides
                content_start = start + ignored_size
                content_end = end - ignored_size

        # Only add if there's useful content
        if content_end > content_start:
            useful_regions.append((content_start, content_end))

    # Create merged image by concatenating useful regions
    merged_parts = []
    for start, end in useful_regions:
        # Extract the useful portion from preprocessed image
        part = preprocessed_image.crop(
            (start, 0, end, preprocessed_image.size[1]))
        merged_parts.append(part)

    if merged_parts:
        # Calculate total width
        total_width = sum(part.size[0] for part in merged_parts)
        height = preprocessed_image.size[1]

        # Create new merged image
        merged_image = Image.new('RGB', (total_width, height), 'white')

        # Paste all useful parts
        x_offset = 0
        for part in merged_parts:
            merged_image.paste(part, (x_offset, 0))
            x_offset += part.size[0]

        return merged_image, useful_regions
    else:
        return None, useful_regions


def visualize_image_chunking(image_path, output_dir="prototype"):
    """Main visualization function"""
    print(f"🔍 Processing: {Path(image_path).name}")

    # Initialize chunker with correct parameters
    chunker = ImageChunker(
        target_height=40, chunk_width=320, stride=240)

    # Create output directory
    os.makedirs(output_dir, exist_ok=True)

    # Load and process image
    try:
        original_image = Image.open(image_path).convert('RGB')
        print(f"   📷 Original size: {original_image.size}")

        preprocessed_image = chunker.preprocess_image(original_image)
        print(f"   🔄 Preprocessed size: {preprocessed_image.size}")

        chunks, chunk_positions = chunker.create_chunks(preprocessed_image)
        print(f"   ✂️  Created {len(chunks)} chunks")

    except Exception as e:
        print(f"   ❌ Error processing image: {e}")
        return

    base_name = Path(image_path).stem

    # Save individual chunks with overlays
    try:
        for i, chunk_tensor in enumerate(chunks):
            if len(chunk_tensor.shape) == 4:
                chunk_tensor = chunk_tensor.squeeze(0)

            chunk_pil = tensor_to_pil(chunk_tensor)
            chunk_with_overlays = add_overlay_to_chunk(
                chunk_pil, i, len(chunks), chunk_positions)

            chunk_path = os.path.join(
                output_dir, f"{base_name}_new_chunk_{i+1:02d}.png")
            chunk_with_overlays.save(chunk_path)

            start, end, left_pad, _ = chunk_positions[i]
            print(f"   📦 Chunk {i+1}: {start}→{end} ({end-start}px content), "
                  f"left_pad={left_pad}px, saved as {Path(chunk_path).name}")

    except Exception as e:
        print(f"   ❌ Error saving chunks: {e}")

    # Create comprehensive visualization
    try:
        viz_path = create_visualization(original_image, preprocessed_image, chunks,
                                        chunk_positions, base_name, output_dir)
        print(f"   📊 Saved visualization: {Path(viz_path).name}")

    except Exception as e:
        print(f"   ❌ Error creating visualization: {e}")

    # Create merged image visualization
    try:
        merged_viz_path = os.path.join(
            output_dir, f"{base_name}_merged_visualization.png")
        merged_image, useful_regions = create_merged_visualization(
            preprocessed_image, chunk_positions)

        if merged_image:
            merged_image.save(merged_viz_path)
            print(f"   🖼️ Saved merged visualization: {Path(merged_viz_path).name}")

            # Optionally, display the merged image with matplotlib
            # plt.figure(figsize=(10, 5))
            # plt.imshow(merged_image)
            # plt.title("Merged Image Visualization")
            # plt.axis('off')
            # plt.show()

    except Exception as e:
        print(f"   ❌ Error creating merged visualization: {e}")

    # Print summary
    print(f"   📋 Summary:")
    print(f"      • Total chunks: {len(chunks)}")
    print(f"      • Chunk size: 320px width × 40px height")
    print(f"      • Stride: 240px (80px overlap between chunks)")
    print(f"      • First chunk: 40px left padding + content")
    print(f"      • Last chunk: right padding if needed")

    if len(chunks) > 1:
        total_content = sum(end - start for start, end,
                            _, _ in chunk_positions)
        total_useful = 0
        for i, (start, end, _, _) in enumerate(chunk_positions):
            content = end - start
            ignored_left = 40 if i > 0 else 0  # Only ignore 40px from each side
            # Only ignore 40px from each side
            ignored_right = 40 if i < len(chunks) - 1 else 0
            useful = content - ignored_left - ignored_right
            total_useful += useful

        efficiency = total_useful / preprocessed_image.size[0] * 100
        print(f"      • Useful content ratio: {efficiency:.1f}%")
        print(f"      • Ignored regions: 40px from each side of overlap")


def main():
    """Main function"""
    print("🎯 New HTR Image Chunking Visualizer")
    print("=" * 50)
    print("Features:")
    print("  🔴 RED overlays = Actual padding (added gray background)")
    print("  ⚪ GRAY overlays = Ignored regions during feature merging")
    print("  🔵 LIGHT BLUE overlays = Overlap regions (informational)")
    print("  ⚫ No overlay = Useful content that gets processed")
    print("=" * 50)

    # Look for images
    prototype_dir = "inference_data"
    if not os.path.exists(prototype_dir):
        print(f"❌ Directory not found: {prototype_dir}")
        return

    # Find images
    image_extensions = {'.png', '.jpg', '.jpeg', '.bmp', '.tiff'}
    image_files = []

    for file in os.listdir(prototype_dir):
        if Path(file).suffix.lower() in image_extensions:
            # Skip generated files
            if not any(keyword in file for keyword in ['_original', '_preprocessed', '_chunk', '_visualization', '_new_']):
                image_files.append(os.path.join(prototype_dir, file))

    if not image_files:
        print(f"❌ No original images found in {prototype_dir}")
        return

    print(f"📁 Found {len(image_files)} image(s) to process:\n")

    # Process each image
    for image_path in image_files:
        visualize_image_chunking(image_path, prototype_dir)
        print()

    print("✅ Processing complete!")
    print(f"🗂️  All output files saved in: {prototype_dir}/")


if __name__ == "__main__":
    main()
