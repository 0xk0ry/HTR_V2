import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.transforms as transforms
from torch.nn import CTCLoss
import math
import numpy as np
from typing import List, Tuple, Optional, Dict
from PIL import Image

try:
    import cv2
    CV2_AVAILABLE = True
except ImportError:
    CV2_AVAILABLE = False

try:
    import kenlm
    from pyctcdecode import build_ctcdecoder
    KENLM_AVAILABLE = True
except ImportError:
    KENLM_AVAILABLE = False
    print("Warning: KenLM and pyctcdecode not available. Install with: pip install kenlm pyctcdecode")

# Constants
DEFAULT_VOCAB = ['<blank>'] + \
    list('abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789.,!?;: ')
DEFAULT_NORMALIZATION = {
    'mean': [0.485, 0.456, 0.406],
    'std': [0.229, 0.224, 0.225]
}

# 3-Stage CvT Configuration optimized for HTR
DEFAULT_CVT_3STAGE_CONFIG = {
    'patch_sizes': [4, 3, 3],    # Gentler initial downsampling
    'strides': [2, 1, 1],        # Only first stage downsamples by 2, others preserve
    'kernel_sizes': [3, 3, 3],   # Convolutional attention kernel sizes
    'mlp_ratios': [4, 4, 4]      # MLP expansion ratios
}

# CvT (Convolutional Vision Transformer) Implementation - 3 Stage Version


class PatchEmbed(nn.Module):
    """Image to Patch Embedding with convolution"""

    def __init__(self, img_size=224, patch_size=16, in_chans=3, embed_dim=768, stride=None):
        super().__init__()
        if stride is None:
            stride = patch_size

        self.img_size = img_size
        self.patch_size = patch_size
        self.stride = stride

        self.proj = nn.Conv2d(in_chans, embed_dim,
                              kernel_size=patch_size, stride=stride)
        self.norm = nn.LayerNorm(embed_dim)

    def forward(self, x):
        if len(x.shape) != 4:
            raise ValueError(f"Expected 4D input tensor [B, C, H, W], got {x.shape}")
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
    """CvT Stage with patch embedding and multiple blocks"""

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
        
        # Convert back to 4D format for next stage: [B, H*W, C] -> [B, C, H, W]
        B, _, C = x.shape
        x = x.transpose(1, 2).reshape(B, C, H, W)
        
        return x, (H, W)


class CvT3Stage(nn.Module):
    """3-Stage Convolutional Vision Transformer optimized for HTR"""

    def __init__(self, img_size=320, in_chans=3, num_classes=1000, embed_dims=[64, 192, 384],
                 num_heads=[1, 3, 6], depths=[1, 2, 10], patch_sizes=[4, 3, 3],
                 strides=[2, 1, 1], kernel_sizes=[3, 3, 3], mlp_ratios=[4, 4, 4],
                 qkv_bias=True, drop_rate=0., attn_drop_rate=0.):
        super().__init__()
        self.num_classes = num_classes
        self.num_stages = 3

        # Build 3 stages
        self.stages = nn.ModuleList()
        
        for i in range(self.num_stages):
            # Input channels: 3 for first stage, previous embed_dim for subsequent stages
            in_chans_stage = in_chans if i == 0 else embed_dims[i-1]
            
            # Create patch embedding for this stage
            patch_embed = PatchEmbed(
                img_size=img_size,
                patch_size=patch_sizes[i],
                in_chans=in_chans_stage,
                embed_dim=embed_dims[i],
                stride=strides[i]
            )
            
            # Create blocks for this stage
            blocks = []
            for j in range(depths[i]):
                blocks.append(CvTBlock(
                    embed_dims[i],
                    num_heads[i],
                    mlp_ratios[i],
                    qkv_bias,
                    drop_rate,
                    attn_drop_rate,
                    kernel_sizes[i]
                ))
            
            # Add stage (no norm for intermediate stages, norm for final stage)
            norm = nn.LayerNorm(embed_dims[i]) if i == self.num_stages - 1 else None
            stage = CvTStage(patch_embed, blocks, norm)
            self.stages.append(stage)

        # Classification head (will be replaced in HTR model)
        self.head = nn.Linear(embed_dims[-1], num_classes)

    def forward(self, x):
        for stage in self.stages:
            x, (H, W) = stage(x)

        # Global average pooling
        x_pooled = x.mean(dim=1)
        x_cls = self.head(x_pooled)
        return x_cls

    def forward_features(self, x):
        """Return features without classification head"""
        stage_features = []
        
        for i, stage in enumerate(self.stages):
            # Process stage
            if i == len(self.stages) - 1:
                # For the last stage, don't convert back to 4D
                x, (H, W) = stage.patch_embed(x)
                for block in stage.blocks:
                    x = block(x, H, W)
                if stage.norm is not None:
                    x = stage.norm(x)
                # Keep as tokens for final output
            else:
                # For intermediate stages, convert back to 4D for next stage
                x, (H, W) = stage(x)
            
            stage_features.append({
                'features': x.clone(),
                'spatial_dims': (H, W),
                'stage_idx': i
            })
        
        return x, (H, W), stage_features

    def get_stage_features(self, x, stage_idx):
        """Get features and per-block features for a specific stage"""
        # Process through previous stages
        for i in range(stage_idx):
            x, (H, W) = self.stages[i](x)
        
        # Process target stage block by block
        stage = self.stages[stage_idx]
        x, (H, W) = stage.patch_embed(x)
        
        block_features = []
        for block in stage.blocks:
            x = block(x, H, W)
            block_features.append(x.clone())
        
        if stage.norm is not None:
            x = stage.norm(x)
            
        return x, (H, W), block_features


class ImageChunker:
    """Handles image chunking with overlapping and padding"""

    def __init__(self, target_height=40, chunk_width=320, stride=240):
        self.target_height = target_height
        self.chunk_width = chunk_width
        self.stride = stride
        self.padding = int((chunk_width - stride) / 2)

    def preprocess_image(self, image):
        """Resize image to target height while preserving aspect ratio"""
        # Get dimensions based on image type
        if isinstance(image, np.ndarray):
            h, w = image.shape[:2]
            is_numpy = True
        else:  # PIL Image
            w, h = image.size
            is_numpy = False

        # Calculate new dimensions
        aspect_ratio = w / h
        new_width = int(self.target_height * aspect_ratio)
        new_size = (new_width, self.target_height)

        # Resize based on available libraries and image type
        if is_numpy and CV2_AVAILABLE:
            image = cv2.resize(image, new_size)
            if len(image.shape) == 3:
                image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        else:
            # Convert to PIL and resize
            if is_numpy:
                image = Image.fromarray(image)
            image = image.resize(new_size, Image.LANCZOS)

        return image

    def create_chunks(self, image):
        """Create overlapping chunks with grey padding for first and last chunks"""
        # Convert to tensor if needed
        image_tensor = self._convert_to_tensor(image)
        C, H, W = image_tensor.shape

        chunks = []
        chunk_positions = []
        overlap_size = self.chunk_width - self.stride  # 320 - 240 = 80px

        if W <= self.chunk_width - self.padding:
            # Single chunk case
            chunk, position = self._create_single_chunk(image_tensor, W)
            chunks.append(chunk)
            chunk_positions.append(position)
        else:
            # Multiple chunks case
            chunks, chunk_positions = self._create_multiple_chunks(
                image_tensor, W, overlap_size)

        return torch.cat(chunks, dim=0), chunk_positions

    def _convert_to_tensor(self, image):
        """Convert image to normalized tensor"""
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

    def _create_multiple_chunks(self, image_tensor, W, overlap_size):
        """Create multiple overlapping chunks"""
        chunks = []
        chunk_positions = []
        start = 0
        chunk_idx = 0

        while start < W:
            is_first_chunk = (chunk_idx == 0)

            # Calculate chunk end position
            if is_first_chunk:
                content_width = self.chunk_width - self.padding  # 320 - 40 = 280px
                end = min(start + content_width, W)
            else:
                end = min(start + self.chunk_width, W)

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

            if end >= W:
                break

            # Next start maintains overlap
            start = end - overlap_size
            chunk_idx += 1

        return chunks, chunk_positions


class HTRModel(nn.Module):
    """3-Stage HTR Model with improved CvT backbone"""

    def __init__(self, vocab_size, max_length=256, target_height=40, chunk_width=320,
                 stride=240, embed_dims=[64, 192, 384], num_heads=[1, 3, 6],
                 depths=[1, 2, 10]):
        super().__init__()

        self.vocab_size = vocab_size
        self.max_length = max_length
        self.chunker = ImageChunker(target_height, chunk_width, stride)

        # 3-Stage CvT backbone for feature extraction
        self.cvt = CvT3Stage(
            img_size=chunk_width,
            in_chans=3,
            embed_dims=embed_dims,
            num_heads=num_heads,
            depths=depths,
            **DEFAULT_CVT_3STAGE_CONFIG,
            qkv_bias=True,
            drop_rate=0.1,
            attn_drop_rate=0.1
        )

        # MLP head for character prediction
        self.feature_dim = embed_dims[-1]
        self.classifier = nn.Sequential(
            nn.Linear(self.feature_dim, self.feature_dim // 2),
            nn.GELU(),
            nn.Dropout(0.1),
            nn.Linear(self.feature_dim // 2, vocab_size)
        )

        # CTC Loss (remove zero_infinity to see actual problematic cases)
        self.ctc_loss = CTCLoss(blank=0, reduction='mean', zero_infinity=False)

    def forward(self, images, targets=None, target_lengths=None):
        """Forward pass through the model"""
        batch_size = images.size(0)
        all_logits = []
        all_lengths = []

        for i in range(batch_size):
            image = images[i]

            # Create chunks
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

        # Handle empty sequences
        if not all_lengths or max(all_lengths) == 0:
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

        # Ensure tensor is contiguous before transpose
        padded_logits = padded_logits.contiguous()

        # Transpose for CTC: [T_max, B, vocab_size] as required by CTC
        padded_logits = padded_logits.transpose(0, 1).contiguous()

        if self.training and targets is not None and target_lengths is not None:
            input_lengths = torch.tensor(
                all_lengths, device=images.device, dtype=torch.long)

            # Apply log_softmax before CTC loss
            log_probs = F.log_softmax(padded_logits, dim=-1)

            # Ensure targets are properly formatted
            if targets.dim() > 1:
                targets = targets.flatten()

            # Additional validation and improved CTC loss calculation
            min_input_length = min(all_lengths)
            max_target_length = max(target_lengths.tolist())
            
            if min_input_length == 0:
                # Handle zero-length inputs
                print(f"Warning: Zero-length input sequence detected")
                loss = torch.tensor(1.0, device=images.device, requires_grad=True)
            elif max_target_length >= min_input_length:
                # Target longer than input - problematic for CTC
                print(f"Warning: Target length ({max_target_length}) >= input length ({min_input_length})")
                loss = torch.tensor(2.0, device=images.device, requires_grad=True)  # Penalty for invalid alignment
            else:
                try:
                    loss = self.ctc_loss(log_probs, targets,
                                         input_lengths, target_lengths)
                    
                    # Check for problematic loss values
                    if torch.isnan(loss) or torch.isinf(loss):
                        print(f"Warning: NaN/Inf loss detected")
                        print(f"  Input lengths: {input_lengths.tolist()}")
                        print(f"  Target lengths: {target_lengths.tolist()}")
                        loss = torch.tensor(2.0, device=images.device, requires_grad=True)
                    elif loss.item() == 0.0:
                        print(f"Warning: Exact zero loss detected - potential perfect alignment or numerical issue")
                        # Don't replace zero loss, but add small epsilon to avoid gradient issues
                        loss = loss + 1e-8
                        
                except Exception as e:
                    print(f"CTC Loss calculation failed: {e}")
                    print(f"  Input lengths: {input_lengths.tolist()}")
                    print(f"  Target lengths: {target_lengths.tolist()}")
                    loss = torch.tensor(2.0, device=images.device, requires_grad=True)
                    
            return padded_logits, loss
        else:
            return padded_logits, torch.tensor(all_lengths, device=images.device)

    def forward_features(self, x_chunk):
        """Extract features from a single chunk and convert to time sequence"""
        # x_chunk shape: [1, C, H, W] (single chunk with batch dim)
        features, (H_prime, W_prime), stage_features = self.cvt.forward_features(x_chunk)
        # features shape: [1, H'*W', C]

        # Verify no extra tokens
        expected_patches = H_prime * W_prime
        actual_patches = features.shape[1]
        assert actual_patches == expected_patches, f"Unexpected token count: {actual_patches} vs {expected_patches}"

        # Reshape to separate spatial dimensions
        features = features.reshape(1, H_prime, W_prime, -1)  # [1, H', W', C]

        # Collapse height dimension (average pooling across height)
        features = features.mean(dim=1)  # [1, W', C]

        # Squeeze batch dimension for consistency with merging
        features = features.squeeze(0)  # [W', C]

        # Ensure tensor is contiguous after reshaping operations
        features = features.contiguous()

        return features

    def extract_all_features(self, x_chunk):
        """Extract features from all stages and blocks for visualization"""
        with torch.no_grad():
            features_dict = {}
            
            # Get features from all stages
            final_features, (H_final, W_final), stage_features = self.cvt.forward_features(x_chunk)
            
            # Store stage features
            features_dict['stage_features'] = stage_features
            
            # Get per-block features for each stage
            per_stage_block_features = []
            x = x_chunk
            
            for stage_idx in range(self.cvt.num_stages):
                stage = self.cvt.stages[stage_idx]
                
                # Get block-by-block features for this stage
                if stage_idx == self.cvt.num_stages - 1:
                    # For the last stage, don't convert back to 4D
                    x, (H, W) = stage.patch_embed(x)
                    block_features = []
                    for block in stage.blocks:
                        x = block(x, H, W)
                        block_features.append(x.clone())
                    if stage.norm is not None:
                        x = stage.norm(x)
                else:
                    # For intermediate stages, process normally
                    x_stage_input = x
                    x, (H, W) = stage.patch_embed(x_stage_input)
                    block_features = []
                    for block in stage.blocks:
                        x = block(x, H, W)
                        block_features.append(x.clone())
                    if stage.norm is not None:
                        x = stage.norm(x)
                    
                    # Convert back to 4D format for next stage: [B, H*W, C] -> [B, C, H, W]
                    B, _, C = x.shape
                    x = x.transpose(1, 2).reshape(B, C, H, W)
                
                per_stage_block_features.append({
                    'stage_idx': stage_idx,
                    'spatial_dims': (H, W),
                    'block_features': block_features
                })
                
            
            features_dict['per_stage_block_features'] = per_stage_block_features
            
            # Final features and time sequence
            features_dict['final_features'] = {
                'features': final_features.clone(),
                'spatial_dims': (H_final, W_final)
            }
            
            # Convert to time sequence
            time_features = self.forward_features(x_chunk)
            features_dict['time_sequence'] = {
                'features': time_features.clone(),
                'spatial_dims': (time_features.shape[0],)
            }
            
            return features_dict

    def _merge_chunk_features(self, chunk_features, chunk_positions):
        """Merge features from multiple chunks, being more conservative with feature removal"""
        if not chunk_features:
            return torch.empty(0, self.feature_dim, device='cpu').contiguous()

        device = chunk_features[0].device
        
        # Be much more conservative with feature removal
        # Calculate stride from final stage
        final_stage = self.cvt.stages[-1]
        patch_stride = final_stage.patch_embed.stride
        
        # Reduce ignore regions significantly to preserve more sequence length
        ignore_patches = max(1, self.chunker.padding // (patch_stride * 4))  # Much smaller ignore region

        merged_features = []

        for i, (features, pos_info) in enumerate(zip(chunk_features, chunk_positions)):
            start_px, end_px, left_pad_px, _ = pos_info
            total_patches = features.size(0)

            # Be more conservative with bounds calculation
            start_idx, end_idx = self._calculate_chunk_bounds_conservative(
                i, len(chunk_positions), total_patches, left_pad_px,
                end_px - start_px, patch_stride, ignore_patches
            )

            # Extract valid features if range is valid
            if start_idx < end_idx and end_idx <= total_patches:
                valid_features = features[start_idx:end_idx].contiguous()
                if valid_features.size(0) > 0:
                    merged_features.append(valid_features)

        return torch.cat(merged_features, dim=0).contiguous() if merged_features else torch.empty(0, self.feature_dim, device=device).contiguous()

    def _calculate_chunk_bounds_conservative(self, chunk_idx, total_chunks, total_patches, left_pad_px, chunk_width_px, patch_stride, ignore_patches):
        """Calculate start and end indices for valid features in a chunk - more conservative approach"""
        start_idx = 0
        end_idx = total_patches

        if total_chunks == 1:
            # Single chunk: only remove significant padding
            if left_pad_px > patch_stride * 2:  # Only remove if padding is substantial
                start_idx = min(left_pad_px // (patch_stride * 2), total_patches - 1)
            
            # Be more conservative with right padding too
            if chunk_width_px < self.chunker.chunk_width - patch_stride * 2:
                content_ratio = (left_pad_px + chunk_width_px) / self.chunker.chunk_width
                end_idx = min(max(start_idx + 1, int(content_ratio * total_patches)), total_patches)

        elif chunk_idx == 0:
            # First chunk: only remove left padding if substantial
            if left_pad_px > patch_stride * 2:
                start_idx = min(left_pad_px // (patch_stride * 2), total_patches - 1)
            end_idx = max(start_idx + 1, total_patches - ignore_patches // 2)  # Smaller ignore region

        elif chunk_idx == total_chunks - 1:
            # Last chunk: be conservative with both sides
            if chunk_width_px < self.chunker.chunk_width - patch_stride * 2:
                content_ratio = chunk_width_px / self.chunker.chunk_width
                actual_patches = max(1, int(content_ratio * total_patches))
                start_idx = min(ignore_patches // 2, actual_patches - 1)
                end_idx = actual_patches
            else:
                start_idx = min(ignore_patches // 2, total_patches - 1)

        else:
            # Middle chunk: smaller ignore regions
            start_idx = min(ignore_patches // 2, total_patches // 4)
            end_idx = max(start_idx + 1, total_patches - ignore_patches // 2)

        return start_idx, end_idx

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


class CTCDecoder:
    """CTC Decoder with KenLM language model support"""

    def __init__(self, vocab, lm_path=None, alpha=0.5, beta=1.0):
        self.vocab = vocab
        self.blank_id = 0
        self.alpha = alpha
        self.beta = beta

        if KENLM_AVAILABLE and lm_path is not None:
            try:
                self.decoder = build_ctcdecoder(
                    labels=vocab,
                    kenlm_model_path=lm_path,
                    alpha=alpha,
                    beta=beta
                )
                self.use_lm = True
            except Exception as e:
                print(f"Failed to load language model: {e}")
                self.use_lm = False
        else:
            self.use_lm = False

    def greedy_decode(self, logits):
        """Simple greedy CTC decoding"""
        # logits: [seq_len, vocab_size]
        predictions = torch.argmax(logits, dim=-1)

        # Remove blanks and consecutive duplicates
        decoded = []
        prev = -1
        for pred in predictions:
            if pred != self.blank_id and pred != prev:
                decoded.append(pred.item())
            prev = pred

        return decoded

    def beam_search_decode(self, logits, beam_width=100):
        """Beam search decoding with optional language model"""
        if self.use_lm:
            # Use pyctcdecode with language model
            logits_np = F.softmax(logits, dim=-1).cpu().numpy()
            text = self.decoder.decode(logits_np, beam_width=beam_width)
            return text
        else:
            # Simple beam search without language model
            return self._simple_beam_search(logits, beam_width)

    def _simple_beam_search(self, logits, beam_width):
        """Simple beam search without language model"""
        seq_len, vocab_size = logits.shape
        log_probs = F.log_softmax(logits, dim=-1)

        # Initialize beam with empty sequence
        beams = [([], 0.0)]  # (sequence, log_prob)

        for t in range(seq_len):
            new_beams = []

            for sequence, log_prob in beams:
                for c in range(vocab_size):
                    new_log_prob = log_prob + log_probs[t, c].item()
                    new_sequence = self._update_sequence(sequence, c)
                    new_beams.append((new_sequence, new_log_prob))

            # Keep top beam_width beams
            new_beams.sort(key=lambda x: x[1], reverse=True)
            beams = new_beams[:beam_width]

        # Return best sequence as text
        best_sequence = beams[0][0]
        return ''.join([self.vocab[i] for i in best_sequence if i < len(self.vocab)])

    def _update_sequence(self, sequence, char_id):
        """Update sequence according to CTC rules"""
        if char_id == self.blank_id:
            # Blank token - don't extend sequence
            return sequence
        elif len(sequence) > 0 and sequence[-1] == char_id:
            # Same character as previous - don't extend
            return sequence
        else:
            # New character
            return sequence + [char_id]


def create_3stage_model_example():
    """Example of how to create and initialize the 3-stage model"""
    vocab_size = len(DEFAULT_VOCAB)

    # Create model with optimized 3-stage configuration
    model = HTRModel(
        vocab_size=vocab_size,
        max_length=256,
        target_height=40,
        chunk_width=320,
        stride=240,
        embed_dims=[64, 192, 384],
        num_heads=[1, 3, 6],
        depths=[1, 2, 10]
    )

    # Create decoder
    decoder = CTCDecoder(DEFAULT_VOCAB, lm_path=None)  # Set lm_path for KenLM

    return model, decoder, DEFAULT_VOCAB


if __name__ == "__main__":
    # Example usage
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    model, decoder, vocab = create_3stage_model_example()
    model.to(device)

    print(f"3-Stage Model created with {sum(p.numel() for p in model.parameters())} parameters")
    print(f"Vocabulary size: {len(vocab)}")

    # Show architecture details
    print("\n3-Stage CvT Architecture:")
    for i, stage in enumerate(model.cvt.stages):
        patch_embed = stage.patch_embed
        print(f"Stage {i+1}: {patch_embed.patch_size}x{patch_embed.patch_size} patch, "
              f"stride={patch_embed.stride}, embed_dim={patch_embed.proj.out_channels}, "
              f"blocks={len(stage.blocks)}")

    # Example forward pass
    batch_size = 2
    dummy_images = torch.randn(batch_size, 3, 40, 320).to(device)

    with torch.no_grad():
        logits, lengths = model(dummy_images)
        print(f"\nOutput shape: {logits.shape}")
        print(f"Sequence lengths: {lengths}")
        
        # Show feature map sizes through stages
        sample_chunk = dummy_images[0:1]
        features_dict = model.extract_all_features(sample_chunk)
        
        print("\nFeature map sizes through stages:")
        for stage_feat in features_dict['stage_features']:
            H, W = stage_feat['spatial_dims']
            print(f"Stage {stage_feat['stage_idx']+1}: {H}x{W}")
