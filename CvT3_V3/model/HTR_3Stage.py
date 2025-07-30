import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.transforms as transforms
from torch.nn import CTCLoss
from torch.utils.checkpoint import checkpoint
import math
import numpy as np
from typing import List, Tuple, Optional, Dict
from PIL import Image
import cv2
from pyctcdecode import build_ctcdecoder


# Constants
DEFAULT_VOCAB = ['<blank>'] + \
    list(
        'abcdefghijklmnopqrstuvwxyz'
        'ABCDEFGHIJKLMNOPQRSTUVWXYZ'
        '0123456789'
        '.,!?;: "#&\'()*+-/'
        'àáảãạăằắẳẵặâầấẩẫậ'
        'èéẻẽẹêềếểễệ'
        'ìíỉĩị'
        'òóỏõọôồốổỗộơờớởỡợ'
        'ùúủũụưừứửữự'
        'ỳýỷỹỵ'
        'đ'
        'ÀÁẢÃẠĂẰẮẲẴẶÂẦẤẨẪẬ'
        'ÈÉẺẼẸÊỀẾỂỄỆ'
        'ÌÍỈĨỊ'
        'ÒÓỎÕỌÔỒỐỔỖỘƠỜỚỞỠỢ'
        'ÙÚỦŨỤƯỪỨỬỮỰ'
        'ỲÝỶỸỴ'
        'Đ'
)
DEFAULT_NORMALIZATION = {
    'mean': [0.5],  # Greyscale normalization
    'std': [0.5]
}

class PatchEmbed(nn.Module):
    """Image to Patch Embedding with convolution"""

    def __init__(self, img_size=224, patch_size=16, in_chans=3, embed_dim=768, stride=None, padding=0):
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
        self.padding = padding

        # Ensure patch_size and stride are tuples for Conv2d
        if isinstance(patch_size, int):
            kernel_size = (patch_size, patch_size)
        else:
            kernel_size = tuple(patch_size)
        if isinstance(stride, int):
            stride_val = (stride, stride)
        else:
            stride_val = tuple(stride)
        self.proj = nn.Conv2d(in_chans, embed_dim,
                              kernel_size=kernel_size, stride=stride_val, padding=padding)
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
        # Handle both int and tuple kernel_size
        if isinstance(kernel_size, int):
            padding = kernel_size // 2
        else:  # tuple
            # For non-square kernels, use 'same' padding to maintain dimensions
            padding = (kernel_size[0] // 2, kernel_size[1] // 2)
            
        self.conv_proj_q = nn.Conv2d(
            dim, dim, kernel_size=kernel_size, padding=padding, groups=dim)
        self.conv_proj_k = nn.Conv2d(
            dim, dim, kernel_size=kernel_size, padding=padding, groups=dim)
        self.conv_proj_v = nn.Conv2d(
            dim, dim, kernel_size=kernel_size, padding=padding, groups=dim)

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
    """CvT Transformer Block with gradient checkpointing support"""

    def __init__(self, dim, num_heads, mlp_ratio=4., qkv_bias=False, drop=0., attn_drop=0., kernel_size=3, use_checkpoint=False):
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
        self.use_checkpoint = use_checkpoint

    def forward(self, x, H, W):
        if self.use_checkpoint and self.training:
            # Use gradient checkpointing to save memory during training
            def create_custom_forward(module):
                def custom_forward(*inputs):
                    return module(*inputs)
                return custom_forward

            x = x + \
                checkpoint(create_custom_forward(
                    lambda x: self.attn(self.norm1(x), H, W)), x)
            x = x + \
                checkpoint(create_custom_forward(
                    lambda x: self.mlp(self.norm2(x))), x)
        else:
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
    """3-Stage Convolutional Vision Transformer for HTR with 64x512 input"""

    def __init__(self, img_size=(64, 512), in_chans=1, num_classes=1000, use_checkpoint=False):
        super().__init__()
        self.num_classes = num_classes
        self.use_checkpoint = use_checkpoint

        # Configuration for 64x512 input
        embed_dims = [192, 384, 768] # 64→192→768 channels
        num_heads = [1, 2, 2]        # 1→2→4 heads
        depths = [1, 2, 3]           # 1→2→4 blocks
        patch_sizes = [3, 3, (16, 1)]
        strides = [(2, 2), (2, 2), (16, 1)]  # Progressive downsampling
        kernel_sizes = [3, 3, 3]  # Use 3x3 for all stages to avoid dimension issues
        mlp_ratios = [4, 4, 4]        # 4→4→4 MLP ratios


        # Stage 1: 3×3 conv, 192 channels, stride=2 (64×512 → 32×256)
        self.stage1_embed = PatchEmbed(
            img_size=img_size,
            patch_size=patch_sizes[0],
            in_chans=in_chans,
            embed_dim=embed_dims[0],
            stride=strides[0],
            padding=1
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
                kernel_size=kernel_sizes[0],
                # Only checkpoint later blocks
                use_checkpoint=use_checkpoint and i >= depths[0]//2
            ))

        # Stage 2: 3×3 conv, 384 channels, stride=2 (32×256 → 16×128)
        self.stage2_embed = PatchEmbed(
            img_size=None,  # Will be determined dynamically
            patch_size=patch_sizes[1],
            in_chans=embed_dims[0],
            embed_dim=embed_dims[1],
            stride=strides[1],
            padding=1
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
                kernel_size=kernel_sizes[1],
                # Only checkpoint later blocks
                use_checkpoint=use_checkpoint and i >= depths[1]//2
            ))

        # Stage 3: 3×3 conv, 768 channels, stride=2 (16×128 → 8×64)
        self.stage3_embed = PatchEmbed(
            img_size=None,  # Will be determined dynamically
            patch_size=patch_sizes[2],
            in_chans=embed_dims[1],
            embed_dim=embed_dims[2],
            stride=strides[2],
            padding=1
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
                kernel_size=kernel_sizes[2],
                # Only checkpoint later blocks
                use_checkpoint=use_checkpoint and i >= depths[2]//2
            ))

        self.norm = nn.LayerNorm(embed_dims[2])

        # Classification head (will be replaced in HTR model)
        self.head = nn.Linear(embed_dims[2], num_classes)

    def forward(self, x):
        """Forward pass through all 3 stages"""
        # Use the actual embed_dims from configuration
        embed_dims = [192, 384, 768]

        # Stage 1: [B, 1, 64, 512] -> [B, H1*W1, 192]
        x, (H1, W1) = self.stage1_embed(x)
        for block in self.stage1_blocks:
            x = block(x, H1, W1)

        # Reshape back to 2D for next stage
        x = x.transpose(1, 2).reshape(-1, embed_dims[0], H1, W1)

        # Stage 2: [B, 192, H1, W1] -> [B, H2*W2, 384]
        x, (H2, W2) = self.stage2_embed(x)
        for block in self.stage2_blocks:
            x = block(x, H2, W2)

        # Reshape back to 2D for next stage
        x = x.transpose(1, 2).reshape(-1, embed_dims[1], H2, W2)

        # Stage 3: [B, 384, H2, W2] -> [B, H3*W3, 768]
        x, (H3, W3) = self.stage3_embed(x)
        for block in self.stage3_blocks:
            x = block(x, H3, W3)

        # Global average pooling for classification
        x_pooled = x.mean(dim=1)
        x_cls = self.head(x_pooled)
        return x_cls

    def forward_features(self, x):
        """Return features without classification head"""
        # Use the actual embed_dims from configuration
        embed_dims = [192, 384, 768]

        # Stage 1: [B, 1, 64, 512] -> [B, H1*W1, 192]
        x, (H1, W1) = self.stage1_embed(x)
        for block in self.stage1_blocks:
            x = block(x, H1, W1)

        # Reshape back to 2D for next stage
        x = x.transpose(1, 2).reshape(-1, embed_dims[0], H1, W1)

        # Stage 2: [B, 192, H1, W1] -> [B, H2*W2, 384]
        x, (H2, W2) = self.stage2_embed(x)
        for block in self.stage2_blocks:
            x = block(x, H2, W2)

        # Reshape back to 2D for next stage
        x = x.transpose(1, 2).reshape(-1, embed_dims[1], H2, W2)

        # Stage 3: [B, 384, H2, W2] -> [B, H3*W3, 768]
        x, (H3, W3) = self.stage3_embed(x)
        for block in self.stage3_blocks:
            x = block(x, H3, W3)

        x = self.norm(x)
        return x, (H3, W3)


class ImageResizer:
    """Handles image resizing to 64x512 for HTR"""

    def __init__(self, target_height=64, target_width=512):
        self.target_height = target_height
        self.target_width = target_width

    def preprocess_image(self, image):
        """Resize image to 64x512 and convert to greyscale"""
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

        # Resize to target dimensions
        new_size = (self.target_width, self.target_height)

        # Resize based on available libraries and image type
        if is_numpy:
            image = cv2.resize(image, new_size)
        else:
            image = image.resize(new_size, Image.LANCZOS)

        return image

    def process_image(self, image):
        """Process image and return as tensor"""
        # Resize image
        resized_image = self.preprocess_image(image)

        # Convert to tensor
        image_tensor = self._convert_to_tensor(resized_image)

        # Add batch dimension if needed
        if image_tensor.dim() == 3:
            image_tensor = image_tensor.unsqueeze(0)
        elif image_tensor.dim() == 2:
            # Add channel and batch dimension for greyscale
            image_tensor = image_tensor.unsqueeze(0).unsqueeze(0)

        return image_tensor

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


class HTRModel(nn.Module):
    """Handwritten Text Recognition Model with 3-stage CvT backbone for 64x512 images"""

    def __init__(self, vocab_size, max_length=256, target_height=64, target_width=512):
        super().__init__()

        self.vocab_size = vocab_size
        self.max_length = max_length
        self.resizer = ImageResizer(target_height, target_width)

        # 3-stage CvT backbone for feature extraction
        self.cvt = CvT3Stage(
            img_size=(target_height, target_width),
            in_chans=1,  # Greyscale input
            num_classes=vocab_size,
            use_checkpoint=True  # Enable gradient checkpointing for memory efficiency
        )

        # MLP head for character prediction (768-dim features from Stage 3)
        self.feature_dim = 768  # Stage 3 outputs 768 channels
        self.classifier = nn.Sequential(
            # Hidden layer for memory efficiency
            nn.Linear(self.feature_dim, 256),
            nn.GELU(),
            nn.Dropout(0.1),
            nn.Linear(256, vocab_size)
        )

        # CTC Loss
        self.ctc_loss = CTCLoss(blank=0, reduction='mean', zero_infinity=True)

    def forward(self, images, targets=None, target_lengths=None):
        """Forward pass through the model for 64x512 images

        Args:
            images: Batch of images [B, C, H, W] or list of images
            targets: Target sequences for CTC loss (optional)
            target_lengths: Lengths of target sequences (optional)
        """
        # Handle different input formats
        if isinstance(images, list):
            # Process list of images
            processed_images = []
            for img in images:
                processed_img = self.resizer.process_image(img)
                processed_images.append(
                    processed_img.squeeze(0))  # Remove batch dim
            images = torch.stack(processed_images, dim=0)
        elif images.size(-1) != 512 or images.size(-2) != 64:
            # Need to resize images
            processed_images = []
            for i in range(images.size(0)):
                img = images[i]
                # Convert tensor back to PIL/numpy for resizing
                img_np = img.squeeze().cpu().numpy()
                if img_np.min() < 0:  # Denormalize if normalized
                    img_np = (img_np * 0.5) + 0.5
                img_np = (img_np * 255).astype(np.uint8)

                processed_img = self.resizer.process_image(img_np)
                processed_images.append(processed_img.squeeze(0))
            images = torch.stack(processed_images, dim=0)

        batch_size = images.size(0)
        device = images.device

        # Extract features through CvT backbone
        features, (H, W) = self.cvt.forward_features(images)  # [B, H*W, C]

        # Convert to time sequences by averaging over height dimension
        # Reshape to separate spatial dimensions: [B, H, W, C]
        features = features.reshape(batch_size, H, W, -1)

        # Average over height to get time sequence: [B, W, C]
        time_features = features.mean(dim=1)  # [B, W, C]

        # Apply classifier to get logits: [B, W, vocab_size]
        logits = self.classifier(time_features)

        # Transpose for CTC: [W, B, vocab_size]
        logits = logits.transpose(0, 1).contiguous()

        if targets is not None and target_lengths is not None:
            # Compute CTC loss
            input_lengths = torch.full(
                (batch_size,), W, device=device, dtype=torch.long)

            # Apply log_softmax before CTC loss
            log_probs = F.log_softmax(logits, dim=-1)

            # Ensure targets are properly formatted
            if targets.dim() > 1:
                targets = targets.flatten()

            loss = self.ctc_loss(log_probs, targets,
                                 input_lengths, target_lengths)
            return logits, loss
        else:
            input_lengths = torch.full(
                (batch_size,), W, device=device, dtype=torch.long)
            return logits, input_lengths

    def forward_features(self, x):
        """Extract features from images

        Args:
            x: Input images [B, C, H, W]

        Returns:
            features: [B, H'*W', C] feature tensor
            spatial_dims: (H', W') spatial dimensions after processing
        """
        features, spatial_dims = self.cvt.forward_features(x)
        return features, spatial_dims

    def decode_predictions(self, logits, input_lengths, vocab=None):
        """Decode CTC predictions to text

        Args:
            logits: [T, B, vocab_size] or [B, T, vocab_size]
            input_lengths: [B] lengths of input sequences
            vocab: Vocabulary list (optional, uses DEFAULT_VOCAB if None)

        Returns:
            List of decoded strings
        """
        if vocab is None:
            vocab = DEFAULT_VOCAB

        # Ensure logits are in [B, T, vocab_size] format
        if logits.size(1) != len(input_lengths):
            logits = logits.transpose(0, 1)

        batch_size = logits.size(0)
        predictions = []

        # Apply softmax to get probabilities
        probs = F.softmax(logits, dim=-1)

        for i in range(batch_size):
            seq_len = input_lengths[i]
            seq_probs = probs[i, :seq_len]  # [T, vocab_size]

            # Greedy decoding (take argmax)
            indices = torch.argmax(seq_probs, dim=-1)  # [T]

            # Remove blanks and duplicates (basic CTC decoding)
            decoded_indices = []
            prev_idx = None

            for idx in indices:
                idx_val = idx.item()
                if idx_val != 0 and idx_val != prev_idx:  # 0 is blank token
                    decoded_indices.append(idx_val)
                prev_idx = idx_val

            # Convert indices to characters
            if decoded_indices:
                decoded_text = ''.join(
                    [vocab[idx] for idx in decoded_indices if idx < len(vocab)])
            else:
                decoded_text = ""

            predictions.append(decoded_text)

        return predictions
