#!/usr/bin/env python3
"""
Calculate the number of tokens (patches) for different image widths in the 3-stage CvT
"""

def calculate_tokens(width, height=64):
    """Calculate tokens through each stage of the 3-stage CvT"""
    
    print(f"Input image: {width}x{height}")
    print("="*50)
    
    # Stage 1: 3x3 conv, stride=2, padding=1
    h1 = (height - 3 + 2*1) // 2 + 1
    w1 = (width - 3 + 2*1) // 2 + 1
    tokens_stage1 = h1 * w1

    print(f"Stage 1 (3x3 conv, stride=2, pad=1, C=64):")
    print(f"  Output size: {h1} x {w1}")
    print(f"  Tokens: {tokens_stage1}")

    # Stage 2: 3x3 conv, stride=2, padding=1
    h2 = (h1 - 3 + 2*1) // 2 + 1
    w2 = (w1 - 3 + 2*1) // 2 + 1
    tokens_stage2 = h2 * w2

    print(f"\nStage 2 (3x3 conv, stride=2, pad=1, C=128):")
    print(f"  Output size: {h2} x {w2}")
    print(f"  Tokens: {tokens_stage2}")

    # Stage 3: 3x3 conv, stride=1, padding=1
    h3 = (h2 - 3 + 2*1) // 1 + 1
    w3 = (w2 - 3 + 2*1) // 1 + 1
    tokens_stage3 = h3 * w3

    print(f"\nStage 3 (3x3 conv, stride=1, pad=1, C=128):")  # Updated: 128 channels with new config
    print(f"  Output size: {h3} x {w3}")
    print(f"  Tokens: {tokens_stage3}")
    
    print(f"\nFinal sequence length for HTR: {w3} (height collapsed)")
    print(f"Overall spatial reduction: {width}x{height} -> {w3}x{h3}")
    print(f"Width reduction factor: {width/w3:.1f}x")
    
    return w3, tokens_stage3

if __name__ == "__main__":
    print("Token calculation for 3-stage CvT HTR model")
    print("="*60)
    
    # Standard chunk size
    print("\n1. Standard chunk (512px width):")
    w3_512, tokens_512 = calculate_tokens(512, 64)
    
    # Different image widths
    print("\n\n2. Different image widths:")
    widths = [320, 512, 640, 800, 1024]
    print("-"*40)
    
    for width in widths:
        print(f"\n{width}px width:")
        w3, tokens = calculate_tokens(width, 64)
    print("-"*40)

    print("\n\n3. Summary for chunked processing:")
    print("-"*40)
    print(f"Each 512px chunk produces {w3_512} time steps")
    print(f"For an 1600px image with ~3 chunks: ~{3 * w3_512} total time steps")
    print(f"(before merging overlaps)")
    print()
    print("NEW CONFIG IMPLEMENTED:")
    print("- Channels: 64→128→128 (memory efficient)")  
    print("- Heads: 1→2→4 (32-dim per head in Stage 3)")
    print("- Blocks: 1→2→6 (deep Stage 3 for long dependencies)")
    print("- MLP ratio: 2→2→4 (slim early, fuller Stage 3)")
    print("- Gradient checkpointing: Enabled")
    print("- Mixed precision: Enabled in training")
