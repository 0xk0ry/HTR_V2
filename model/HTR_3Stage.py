import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.transforms as transforms
from torch.nn import CTCLoss
import math
import numpy as np
from typing import List, Tuple, Optional, Dict
from PIL import Image
import cv2
from pyctcdecode import build_ctcdecoder


# Constants
DEFAULT_VOCAB = ['<blank>'] + \
    list('abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789.,!?;: ')
DEFAULT_NORMALIZATION = {
    'mean': [0.5],  # Greyscale normalization
    'std': [0.5]
}
DEFAULT_CVT_3STAGE_CONFIG = {
    'embed_dims': [64, 192, 390],  # 390 is divisible by 10 heads
    'num_heads': [1, 2, 10],
    'depths': [1, 2, 10],
    'patch_sizes': [7, 3, 3],
    'strides': [4, 2, 2],
    'kernel_sizes': [3, 3, 3],
    'mlp_ratios': [4, 4, 4]
}

# CvT (Convolutional Vision Transformer) Implementation


class PatchEmbed(nn.Module):
    """Image to Patch Embedding with convolution"""

    def __init__(self, img_size=224, patch_size=16, in_chans=3, embed_dim=768, stride=None):
        super().__init__()
        if stride is None:
            stride = patch_size

        # Handle both single int and tuple for img_size
        if isinstance(img_size, int):
            self.img_size = (img_size, img_size)
        else:
            self.img_size = img_size

        self.patch_size = patch_size
        self.stride = stride

        self.proj = nn.Conv2d(in_chans, embed_dim,
                              kernel_size=patch_size, stride=stride)
        self.norm = nn.LayerNorm(embed_dim)

    def forward(self, x):
        B, C, H, W = x.shape
        x = self.proj(x)  # [B, embed_dim, H', W']
        _, _, H_new, W_new = x.shape
        x = x.flatten(2).transpose(1, 2)  # [B, H'*W', embed_dim]
        x = self.norm(x)
        return x, (H_new, W_new)


class ConvolutionalAttention(nn.Module):
    """Convolutional Multi-head Attention"""

    def __init__(self, dim, num_heads=8, qkv_bias=False, attn_drop=0., proj_drop=0., kernel_size=3):
        super().__init__()
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.scale = head_dim ** -0.5

        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)

        # Convolutional projection for positional encoding
        self.conv_proj_q = nn.Conv2d(
            dim, dim, kernel_size=kernel_size, padding=kernel_size//2, groups=dim)
        self.conv_proj_k = nn.Conv2d(
            dim, dim, kernel_size=kernel_size, padding=kernel_size//2, groups=dim)
        self.conv_proj_v = nn.Conv2d(
            dim, dim, kernel_size=kernel_size, padding=kernel_size//2, groups=dim)

    def forward(self, x, H, W):
        B, N, C = x.shape

        # Reshape for conv operations
        x_2d = x.transpose(1, 2).reshape(B, C, H, W)

        # Apply convolutional projections
        q_conv = self.conv_proj_q(x_2d).flatten(2).transpose(1, 2)
        k_conv = self.conv_proj_k(x_2d).flatten(2).transpose(1, 2)
        v_conv = self.conv_proj_v(x_2d).flatten(2).transpose(1, 2)

        # Linear projections
        qkv = self.qkv(x + q_conv + k_conv + v_conv).reshape(B, N, 3,
                                                             self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]

        attn = (q @ k.transpose(-2, -1)) * self.scale
        attn = attn.softmax(dim=-1)
        attn = self.attn_drop(attn)

        x = (attn @ v).transpose(1, 2).reshape(B, N, C)
        x = self.proj(x)
        x = self.proj_drop(x)
        return x


class CvTBlock(nn.Module):
    """CvT Transformer Block"""

    def __init__(self, dim, num_heads, mlp_ratio=4., qkv_bias=False, drop=0., attn_drop=0., kernel_size=3):
        super().__init__()
        self.norm1 = nn.LayerNorm(dim)
        self.attn = ConvolutionalAttention(dim, num_heads=num_heads, qkv_bias=qkv_bias,
                                           attn_drop=attn_drop, proj_drop=drop, kernel_size=kernel_size)
        self.norm2 = nn.LayerNorm(dim)
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp = nn.Sequential(
            nn.Linear(dim, mlp_hidden_dim),
            nn.GELU(),
            nn.Dropout(drop),
            nn.Linear(mlp_hidden_dim, dim),
            nn.Dropout(drop)
        )

    def forward(self, x, H, W):
        x = x + self.attn(self.norm1(x), H, W)
        x = x + self.mlp(self.norm2(x))
        return x


class CvTStage(nn.Module):
    """CvT Stage with multiple blocks"""

    def __init__(self, patch_embed, blocks, norm=None):
        super().__init__()
        self.patch_embed = patch_embed
        self.blocks = nn.ModuleList(blocks)
        self.norm = norm

    def forward(self, x):
        x, (H, W) = self.patch_embed(x)
        for block in self.blocks:
            x = block(x, H, W)
        if self.norm is not None:
            x = self.norm(x)
        return x, (H, W)


class CvT3Stage(nn.Module):
    """3-Stage Convolutional Vision Transformer for HTR"""

    def __init__(self, img_size=512, in_chans=1, num_classes=1000):
        super().__init__()
        self.num_classes = num_classes

        # Configuration from specifications
        # Note: Adjusted embed_dims to be divisible by num_heads
        embed_dims = [64, 192, 390]  # 390 is divisible by 10 (390/10 = 39)
        num_heads = [1, 2, 10]
        depths = [1, 2, 10]
        patch_sizes = [7, 3, 3]
        strides = [4, 2, 2]
        kernel_sizes = [3, 3, 3]
        mlp_ratios = [4, 4, 4]

        # Stage 1: 7×7 conv, 64 channels, stride=4, 1 block
        self.stage1_embed = PatchEmbed(
            img_size=img_size,
            patch_size=patch_sizes[0],
            in_chans=in_chans,
            embed_dim=embed_dims[0],
            stride=strides[0]
        )

        self.stage1_blocks = nn.ModuleList()
        for i in range(depths[0]):
            self.stage1_blocks.append(CvTBlock(
                embed_dims[0],
                num_heads[0],
                mlp_ratios[0],
                qkv_bias=True,
                drop=0.1,
                attn_drop=0.1,
                kernel_size=kernel_sizes[0]
            ))

        # Stage 2: 3×3 conv, 192 channels, stride=2, 2 blocks
        # No need to specify img_size for intermediate stages
        self.stage2_embed = PatchEmbed(
            img_size=None,  # Will be determined dynamically
            patch_size=patch_sizes[1],
            in_chans=embed_dims[0],
            embed_dim=embed_dims[1],
            stride=strides[1]
        )

        self.stage2_blocks = nn.ModuleList()
        for i in range(depths[1]):
            self.stage2_blocks.append(CvTBlock(
                embed_dims[1],
                num_heads[1],
                mlp_ratios[1],
                qkv_bias=True,
                drop=0.1,
                attn_drop=0.1,
                kernel_size=kernel_sizes[1]
            ))

        # Stage 3: 3×3 conv, 384 channels, stride=2, 10 blocks
        self.stage3_embed = PatchEmbed(
            img_size=None,  # Will be determined dynamically
            patch_size=patch_sizes[2],
            in_chans=embed_dims[1],
            embed_dim=embed_dims[2],
            stride=strides[2]
        )

        self.stage3_blocks = nn.ModuleList()
        for i in range(depths[2]):
            self.stage3_blocks.append(CvTBlock(
                embed_dims[2],
                num_heads[2],
                mlp_ratios[2],
                qkv_bias=True,
                drop=0.1,
                attn_drop=0.1,
                kernel_size=kernel_sizes[2]
            ))

        self.norm = nn.LayerNorm(embed_dims[2])

        # Classification head (will be replaced in HTR model)
        self.head = nn.Linear(embed_dims[2], num_classes)

    def forward(self, x):
        """Forward pass through all 3 stages"""
        # Stage 1: [B, 1, 64, 512] -> [B, H1*W1, 64]
        x, (H1, W1) = self.stage1_embed(x)
        for block in self.stage1_blocks:
            x = block(x, H1, W1)

        # Reshape back to 2D for next stage
        x = x.transpose(1, 2).reshape(-1, 64, H1, W1)

        # Stage 2: [B, 64, H1, W1] -> [B, H2*W2, 192]
        x, (H2, W2) = self.stage2_embed(x)
        for block in self.stage2_blocks:
            x = block(x, H2, W2)

        # Reshape back to 2D for next stage
        x = x.transpose(1, 2).reshape(-1, 192, H2, W2)

        # Stage 3: [B, 192, H2, W2] -> [B, H3*W3, 390]
        x, (H3, W3) = self.stage3_embed(x)
        for block in self.stage3_blocks:
            x = block(x, H3, W3)

        # Global average pooling for classification
        x_pooled = x.mean(dim=1)
        x_cls = self.head(x_pooled)
        return x_cls

    def forward_features(self, x):
        """Return features without classification head"""
        # Stage 1: [B, 1, 64, 512] -> [B, H1*W1, 64]
        x, (H1, W1) = self.stage1_embed(x)
        for block in self.stage1_blocks:
            x = block(x, H1, W1)

        # Reshape back to 2D for next stage
        x = x.transpose(1, 2).reshape(-1, 64, H1, W1)

        # Stage 2: [B, 64, H1, W1] -> [B, H2*W2, 192]
        x, (H2, W2) = self.stage2_embed(x)
        for block in self.stage2_blocks:
            x = block(x, H2, W2)

        # Reshape back to 2D for next stage
        x = x.transpose(1, 2).reshape(-1, 192, H2, W2)

        # Stage 3: [B, 192, H2, W2] -> [B, H3*W3, 390]
        x, (H3, W3) = self.stage3_embed(x)
        for block in self.stage3_blocks:
            x = block(x, H3, W3)

        x = self.norm(x)
        return x, (H3, W3)


class ImageChunker:
    """Handles image chunking with overlapping and padding for 3-stage HTR"""

    def __init__(self, target_height=64, chunk_width=512, first_stride=320, stride=384):
        self.target_height = target_height
        self.chunk_width = chunk_width
        self.first_stride = first_stride  # 320px for first chunk
        self.stride = stride  # 384px for subsequent chunks
        self.padding = 40  # 40px padding as specified

    def preprocess_image(self, image):
        """Resize image to target height while preserving aspect ratio and convert to greyscale"""
        # Get dimensions based on image type
        if isinstance(image, np.ndarray):
            if len(image.shape) == 3:
                # Convert to greyscale if RGB
                image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
            h, w = image.shape[:2]
            is_numpy = True
        else:  # PIL Image
            # Convert to greyscale
            if image.mode != 'L':
                image = image.convert('L')
            w, h = image.size
            is_numpy = False

        # Calculate new dimensions
        aspect_ratio = w / h
        new_width = int(self.target_height * aspect_ratio)
        new_size = (new_width, self.target_height)

        # Resize based on available libraries and image type
        if is_numpy:
            image = cv2.resize(image, new_size)
        else:
            image = image.resize(new_size, Image.LANCZOS)

        return image

    def create_chunks(self, image):
        """Create overlapping chunks with grey padding for first and last chunks"""
        # Convert to tensor if needed
        image_tensor = self._convert_to_tensor(image)

        # Handle different tensor dimensions
        if image_tensor.dim() == 4:
            # Remove batch dimension if present
            image_tensor = image_tensor.squeeze(0)
        elif image_tensor.dim() == 2:
            # Add channel dimension for greyscale
            image_tensor = image_tensor.unsqueeze(0)

        C, H, W = image_tensor.shape

        chunks = []
        chunk_positions = []

        if W <= self.chunk_width - self.padding:
            # Single chunk case
            chunk, position = self._create_single_chunk(image_tensor, W)
            chunks.append(chunk)
            chunk_positions.append(position)
        else:
            # Multiple chunks case
            chunks, chunk_positions = self._create_multiple_chunks(
                image_tensor, W)

        return torch.cat(chunks, dim=0), chunk_positions

    def _convert_to_tensor(self, image):
        """Convert greyscale image to normalized tensor"""
        if isinstance(image, (np.ndarray, Image.Image)):
            transform = transforms.Compose([
                transforms.ToTensor(),
                transforms.Normalize(**DEFAULT_NORMALIZATION)
            ])
            if isinstance(image, np.ndarray):
                image = Image.fromarray(image)
            return transform(image)
        return image

    def _create_single_chunk(self, image_tensor, W):
        """Create a single chunk with padding"""
        # Add left padding
        chunk = F.pad(image_tensor, (self.padding, 0, 0, 0),
                      mode='constant', value=0.5)

        # Add right padding if needed
        current_width = chunk.size(2)
        if current_width < self.chunk_width:
            right_pad = self.chunk_width - current_width
            chunk = F.pad(chunk, (0, right_pad, 0, 0),
                          mode='constant', value=0.5)

        position = (0, W, self.padding, self.padding + W)
        return chunk.unsqueeze(0), position

    def _create_multiple_chunks(self, image_tensor, W):
        """Create multiple overlapping chunks with different stride patterns"""
        chunks = []
        chunk_positions = []
        start = 0
        chunk_idx = 0

        while start < W:
            is_first_chunk = (chunk_idx == 0)
            is_last_chunk = False

            # Calculate chunk end position based on stride pattern
            if is_first_chunk:
                content_width = self.chunk_width - self.padding  # 320 - 40 = 280px
                end = min(start + content_width, W)
                next_start = start + self.first_stride  # First stride is 200px
            else:
                end = min(start + self.chunk_width, W)
                next_start = start + self.stride  # Subsequent strides are 240px

            # Check if this is the last chunk
            if next_start >= W or end >= W:
                is_last_chunk = True

            # Create chunk
            chunk = image_tensor[:, :, start:end]
            chunk_width = end - start

            # Add padding
            if is_first_chunk:
                chunk = F.pad(chunk, (self.padding, 0, 0, 0),
                              mode='constant', value=0.5)

            # Add right padding for last chunk if needed
            current_width = chunk.size(2)
            if current_width < self.chunk_width:
                right_pad = self.chunk_width - current_width
                chunk = F.pad(chunk, (0, right_pad, 0, 0),
                              mode='constant', value=0.5)

            chunks.append(chunk.unsqueeze(0))

            # Store position info
            left_pad = self.padding if is_first_chunk else 0
            chunk_positions.append(
                (start, end, left_pad, left_pad + chunk_width))

            if is_last_chunk:
                break

            # Move to next chunk
            start = next_start
            chunk_idx += 1

        return chunks, chunk_positions


class HTRModel(nn.Module):
    """Handwritten Text Recognition Model with 3-stage CvT backbone"""

    def __init__(self, vocab_size, max_length=256, target_height=64, chunk_width=512,
                 first_stride=320, stride=384):
        super().__init__()

        self.vocab_size = vocab_size
        self.max_length = max_length
        self.chunker = ImageChunker(
            target_height, chunk_width, first_stride, stride)

        # 3-stage CvT backbone for feature extraction
        self.cvt = CvT3Stage(
            img_size=chunk_width,
            in_chans=1,  # Greyscale input
            num_classes=vocab_size
        )

        # MLP head for character prediction
        self.feature_dim = 390  # Final stage embedding dimension
        self.classifier = nn.Sequential(
            nn.Linear(self.feature_dim, self.feature_dim // 2),
            nn.GELU(),
            nn.Dropout(0.1),
            nn.Linear(self.feature_dim // 2, vocab_size)
        )

        # CTC Loss
        self.ctc_loss = CTCLoss(blank=0, reduction='none', zero_infinity=True)

    def _average_overlap_features(self, features1, features2, overlap_patches):
        """Average features in overlap regions for smoother transitions

        FIXED: Added proper overlap averaging instead of just cropping
        """
        if overlap_patches <= 0 or features1.size(0) < overlap_patches or features2.size(0) < overlap_patches:
            return features1, features2

        # Average the overlapping regions
        overlap1 = features1[-overlap_patches:]  # Last part of first chunk
        overlap2 = features2[:overlap_patches]   # First part of second chunk

        averaged_overlap = (overlap1 + overlap2) / 2.0

        # Replace overlapping regions with averaged values
        features1_modified = torch.cat(
            [features1[:-overlap_patches], averaged_overlap], dim=0)
        features2_modified = torch.cat(
            [averaged_overlap, features2[overlap_patches:]], dim=0)

        return features1_modified.contiguous(), features2_modified.contiguous()

    def forward(self, images, targets=None, target_lengths=None):
        """Forward pass through the model

        FIXED ISSUES:
        - Proper input length validation against target lengths
        - Correct log_softmax application before CTC loss
        - Better batch processing with length tracking
        - Tensor contiguity ensured throughout
        """
        batch_size = images.size(0)
        all_logits = []
        all_lengths = []

        for i in range(batch_size):
            image = images[i]

            # Create chunks from full image
            chunks, chunk_positions = self.chunker.create_chunks(image)

            # Process each chunk through CvT and extract time-sequence features
            chunk_features = []
            for chunk in chunks:
                # chunk has shape [C, H, W], need to add batch dimension
                chunk = chunk.unsqueeze(0)  # [1, C, H, W]
                features = self.forward_features(chunk)  # [W', C]
                chunk_features.append(features)

            # Merge chunks by removing padding and ignored regions
            merged_features = self._merge_chunk_features(
                chunk_features, chunk_positions)  # [T_total, C]

            # Apply classifier
            logits = self.classifier(merged_features)  # [T_total, vocab_size]
            all_logits.append(logits)
            all_lengths.append(logits.size(0))

        # FIXED: Better handling of empty sequences
        if not all_lengths or max(all_lengths) == 0:
            # Return minimal valid output for CTC
            max_seq_len = 1
            padded_logits = torch.zeros(
                batch_size, max_seq_len, self.vocab_size, device=images.device)
            all_lengths = [1] * batch_size
        else:
            # Pad sequences to same length
            max_seq_len = max(all_lengths)
            padded_logits = torch.zeros(
                batch_size, max_seq_len, self.vocab_size, device=images.device)

            for i, (logits, length) in enumerate(zip(all_logits, all_lengths)):
                if length > 0:
                    padded_logits[i, :length] = logits

        # FIXED: Ensure tensor is contiguous before transpose
        padded_logits = padded_logits.contiguous()

        # Transpose for CTC: [T_max, B, vocab_size] as required by CTC
        padded_logits = padded_logits.transpose(0, 1).contiguous()

        if self.training and targets is not None and target_lengths is not None:
            # FIXED: Proper input length validation and CTC loss computation
            input_lengths = torch.tensor(
                all_lengths, device=images.device, dtype=torch.long)

            # Validate that input sequences are long enough for targets
            for i in range(batch_size):
                if input_lengths[i] < target_lengths[i]:
                    print(
                        f"Warning: Input length {input_lengths[i]} < target length {target_lengths[i]} for sample {i}")
            # FIXED: Apply log_softmax before CTC loss (CTC expects log probabilities)
            log_probs = F.log_softmax(padded_logits, dim=-1)

            # Ensure targets and target_lengths are properly formatted
            if targets.dim() > 1:
                # If targets is 2D, flatten it (shouldn't happen with proper preprocessing)
                targets = targets.flatten()

            loss = self.ctc_loss(log_probs, targets,
                                 input_lengths, target_lengths)
            return padded_logits, loss
        else:
            return padded_logits, torch.tensor(all_lengths, device=images.device)

    def forward_features(self, x_chunk):
        """Extract features from a single chunk and convert to time sequence

        FIXED ISSUES:
        - Proper 2D → 1D conversion with height pooling
        - Explicit CLS token removal (though CvT doesn't use them)
        - Tensor contiguity ensured after reshaping
        """
        # x_chunk shape: [1, C, H, W] (single chunk with batch dim)
        features, (H_prime, W_prime) = self.cvt.forward_features(x_chunk)
        # features shape: [1, H'*W', C]

        # CvT doesn't use CLS tokens, but verify no extra tokens
        expected_patches = H_prime * W_prime
        actual_patches = features.shape[1]
        assert actual_patches == expected_patches, f"Unexpected token count: {actual_patches} vs {expected_patches}"

        # Reshape to separate spatial dimensions
        features = features.reshape(1, H_prime, W_prime, -1)  # [1, H', W', C]

        # Collapse height dimension (average pooling across height)
        # This converts 2D spatial features to 1D time sequence
        features = features.mean(dim=1)  # [1, W', C]

        # Squeeze batch dimension for consistency with merging
        features = features.squeeze(0)  # [W', C]

        # Ensure tensor is contiguous after reshaping operations
        features = features.contiguous()

        return features

    def _merge_chunk_features(self, chunk_features, chunk_positions):
        """Merge features from multiple chunks, removing padded and ignored regions during merging"""
        if not chunk_features:
            return torch.empty(0, self.feature_dim, device='cpu').contiguous()

        device = chunk_features[0].device

        # Calculate final patch stride through all 3 stages: 4 * 2 * 2 = 16
        final_patch_stride = 4 * 2 * 2  # Stage 1: 4, Stage 2: 2, Stage 3: 2
        
        #! ignored patches is correct to handle padding
        ignore_patches = max(1, self.chunker.padding // final_patch_stride) # 40px padding / 16px stride = 2.5 = 2 patches

        merged_features = []

        for i, (features, pos_info) in enumerate(zip(chunk_features, chunk_positions)):
            start_px, end_px, left_pad_px, _ = pos_info
            total_patches = features.size(0)

            # Calculate valid feature range based on chunk type
            start_idx, end_idx = self._calculate_chunk_bounds(
                i, len(chunk_positions), total_patches, left_pad_px,
                end_px - start_px, final_patch_stride, ignore_patches
            )

            # Extract valid features if range is valid
            if start_idx < end_idx and end_idx <= total_patches:
                valid_features = features[start_idx:end_idx].contiguous()
                if valid_features.size(0) > 0:
                    merged_features.append(valid_features)

        return torch.cat(merged_features, dim=0).contiguous() if merged_features else torch.empty(0, self.feature_dim, device=device).contiguous()

    def _calculate_chunk_bounds(self, chunk_idx, total_chunks, total_patches, left_pad_px, chunk_width_px, patch_stride, ignore_patches):
        """Calculate start and end indices for valid features in a chunk"""
        start_idx = 0
        end_idx = total_patches

        if total_chunks == 1:
            # Single chunk: remove padding from both sides
            if left_pad_px > 0:
                start_idx = min(left_pad_px // patch_stride, total_patches - 1)

            # Calculate actual content ratio for right padding removal
            if chunk_width_px < self.chunker.chunk_width - left_pad_px:
                content_ratio = (left_pad_px + chunk_width_px) / \
                    self.chunker.chunk_width
                end_idx = min(
                    max(1, int(content_ratio * total_patches)), total_patches)

        elif chunk_idx == 0:
            # First chunk: remove left padding and right ignore region
            if left_pad_px > 0:
                start_idx = min(left_pad_px // patch_stride, total_patches - 1)
            end_idx = max(start_idx + 1, total_patches - ignore_patches)

        elif chunk_idx == total_chunks - 1:
            # Last chunk: remove left ignore region and handle right padding
            if chunk_width_px < self.chunker.chunk_width:
                # Chunk has right padding
                content_ratio = chunk_width_px / self.chunker.chunk_width
                actual_patches = max(1, int(content_ratio * total_patches))
                start_idx = min(ignore_patches, actual_patches - 1)
                end_idx = actual_patches
            else:
                # No right padding
                start_idx = min(ignore_patches, total_patches - 1)

        else:
            # Middle chunk: remove ignore regions from both sides
            start_idx = min(ignore_patches, total_patches // 2)
            end_idx = max(start_idx + 1, total_patches - ignore_patches)

        return start_idx, end_idx
