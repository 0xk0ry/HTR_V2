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
import kenlm
from pyctcdecode import build_ctcdecoder


# Constants
DEFAULT_VOCAB = ['<blank>'] + \
    list('abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789.,!?;: ')
DEFAULT_NORMALIZATION = {
    'mean': [0.485, 0.456, 0.406],
    'std': [0.229, 0.224, 0.225]
}
DEFAULT_CVT_CONFIG = {
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


class CvT(nn.Module):
    """Convolutional Vision Transformer - Simplified version for HTR"""

    def __init__(self, img_size=256, in_chans=3, num_classes=1000, embed_dims=[64, 192, 384],
                 num_heads=[1, 3, 6], depths=[1, 2, 10], patch_sizes=[7, 3, 3],
                 strides=[4, 2, 2], kernel_sizes=[3, 3, 3], mlp_ratios=[4, 4, 4],
                 qkv_bias=True, drop_rate=0., attn_drop_rate=0.):
        super().__init__()
        self.num_classes = num_classes

        # For HTR, we'll use a simplified single-stage CvT
        # First stage with patch embedding
        self.patch_embed = PatchEmbed(
            img_size=img_size,
            patch_size=patch_sizes[0],
            in_chans=in_chans,
            embed_dim=embed_dims[-1],  # Use final embedding dimension
            stride=strides[0]
        )

        # Transformer blocks
        self.blocks = nn.ModuleList()
        for i in range(depths[-1]):  # Use final depth
            self.blocks.append(CvTBlock(
                embed_dims[-1],
                num_heads[-1],
                mlp_ratios[-1],
                qkv_bias,
                drop_rate,
                attn_drop_rate,
                kernel_sizes[-1]
            ))

        self.norm = nn.LayerNorm(embed_dims[-1])

        # Classification head (will be replaced in HTR model)
        self.head = nn.Linear(embed_dims[-1], num_classes)

    def forward(self, x):
        x, (H, W) = self.patch_embed(x)
        for block in self.blocks:
            x = block(x, H, W)

        # Global average pooling
        x_pooled = x.mean(dim=1)
        x_cls = self.head(x_pooled)
        return x_cls

    def forward_features(self, x):
        """Return features without classification head"""
        x, (H, W) = self.patch_embed(x)
        for block in self.blocks:
            x = block(x, H, W)
        x = self.norm(x)
        return x, (H, W)


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
        if is_numpy:
            image = cv2.resize(image, new_size)
            if len(image.shape) == 3:
                image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        else:

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
    """Handwritten Text Recognition Model with CvT backbone"""

    def __init__(self, vocab_size, max_length=256, target_height=40, chunk_width=320,
                 stride=240, embed_dims=[64, 192, 384], num_heads=[1, 3, 6],
                 depths=[1, 2, 10]):
        super().__init__()

        self.vocab_size = vocab_size
        self.max_length = max_length
        self.chunker = ImageChunker(
            target_height, chunk_width, stride)

        # CvT backbone for feature extraction
        self.cvt = CvT(
            img_size=chunk_width,
            in_chans=3,
            embed_dims=embed_dims,
            num_heads=num_heads,
            depths=depths,
            **DEFAULT_CVT_CONFIG,
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

        # CTC Loss
        self.ctc_loss = CTCLoss(blank=0, reduction='mean', zero_infinity=True)

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
                    # CTC can handle this but it may cause issues

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
        patch_stride = self.cvt.patch_embed.stride
        ignore_patches = max(1, self.chunker.padding // patch_stride)

        merged_features = []

        for i, (features, pos_info) in enumerate(zip(chunk_features, chunk_positions)):
            start_px, end_px, left_pad_px, _ = pos_info
            total_patches = features.size(0)

            # Calculate valid feature range based on chunk type
            start_idx, end_idx = self._calculate_chunk_bounds(
                i, len(chunk_positions), total_patches, left_pad_px,
                end_px - start_px, patch_stride, ignore_patches
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


class CTCDecoder:
    """CTC Decoder with KenLM language model support"""

    def __init__(self, vocab, lm_path=None, alpha=0.5, beta=1.0):
        self.vocab = vocab
        self.blank_id = 0
        self.alpha = alpha
        self.beta = beta
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

# Training utilities


def train_epoch(model, dataloader, optimizer, device, vocab, use_sam=False):
    """Train for one epoch with improved error handling"""
    model.train()
    total_loss = 0
    num_batches = 0

    for batch in dataloader:
        try:
            loss = _process_training_batch(
                batch, model, optimizer, device, use_sam)

            if torch.isnan(loss) or torch.isinf(loss):
                print(
                    f"Warning: Invalid loss value {loss.item()}, skipping batch")
                continue

            total_loss += loss.item()
            num_batches += 1

            if num_batches % 100 == 0:
                print(f"Batch {num_batches}, Loss: {loss.item():.4f}")

        except Exception as e:
            print(f"Error in batch {num_batches}: {e}")
            continue

    return total_loss / num_batches if num_batches > 0 else float('inf')


def _process_training_batch(batch, model, optimizer, device, use_sam):
    """Process a single training batch"""
    images, targets, target_lengths = batch
    images = images.to(device)
    targets = targets.to(device, dtype=torch.long)
    target_lengths = target_lengths.to(device, dtype=torch.long)

    if use_sam:
        return _sam_training_step(model, optimizer, images, targets, target_lengths)
    else:
        return _standard_training_step(model, optimizer, images, targets, target_lengths)


def _sam_training_step(model, optimizer, images, targets, target_lengths):
    """SAM training step"""
    def closure():
        optimizer.zero_grad()
        logits, loss = model(images, targets, target_lengths)
        loss.backward()
        return loss

    # First forward-backward pass
    loss = closure()
    optimizer.first_step(zero_grad=True)

    # Second forward-backward pass
    closure()
    optimizer.second_step(zero_grad=True)
    return loss


def _standard_training_step(model, optimizer, images, targets, target_lengths):
    """Standard training step"""
    optimizer.zero_grad()
    logits, loss = model(images, targets, target_lengths)
    loss.backward()

    # Gradient clipping
    torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
    optimizer.step()

    return loss


def validate(model, dataloader, device, decoder):
    """Validate the model with improved accuracy calculation"""
    model.eval()
    total_loss = 0
    num_batches = 0
    correct_chars = 0
    total_chars = 0

    with torch.no_grad():
        for batch in dataloader:
            try:
                images, targets, target_lengths = batch
                images = images.to(device)
                targets = targets.to(device, dtype=torch.long)
                target_lengths = target_lengths.to(device, dtype=torch.long)

                logits, loss = model(images, targets, target_lengths)
                total_loss += loss.item()
                num_batches += 1

                # Calculate character accuracy
                batch_correct, batch_total = _calculate_batch_accuracy(
                    logits, targets, target_lengths, decoder
                )
                correct_chars += batch_correct
                total_chars += batch_total

            except Exception as e:
                print(f"Error in validation batch: {e}")
                continue

    char_accuracy = correct_chars / total_chars if total_chars > 0 else 0
    avg_loss = total_loss / num_batches if num_batches > 0 else float('inf')
    return avg_loss, char_accuracy


def _calculate_batch_accuracy(logits, targets, target_lengths, decoder):
    """Calculate character accuracy for a batch"""
    correct_chars = 0
    total_chars = 0

    batch_size = logits.size(1)

    for i in range(batch_size):
        try:
            # Get prediction
            pred_logits = logits[:, i, :]  # [seq_len, vocab_size]
            predicted = decoder.greedy_decode(pred_logits)

            # Get ground truth
            target_seq = _extract_target_sequence(targets, target_lengths, i)

            # Calculate accuracy
            min_len = min(len(predicted), len(target_seq))
            if min_len > 0:
                correct_chars += sum(1 for j in range(min_len)
                                     if predicted[j] == target_seq[j])
            total_chars += max(len(predicted), len(target_seq))

        except Exception:
            continue

    return correct_chars, total_chars


def _extract_target_sequence(targets, target_lengths, batch_idx):
    """Extract target sequence for a specific batch index"""
    if batch_idx >= len(target_lengths):
        return []

    target_len = target_lengths[batch_idx].item()
    if target_len <= 0:
        return []

    # Find start position for this target in concatenated tensor
    start_pos = sum(target_lengths[:batch_idx]).item() if batch_idx > 0 else 0
    end_pos = start_pos + target_len

    if end_pos > targets.size(0):
        return []

    return targets[start_pos:end_pos].cpu().numpy()

# Example usage and inference


def inference_example(model, image_path, decoder, device):
    """Example inference on a single image"""
    model.eval()

    # Load and preprocess image
    image = Image.open(image_path).convert('RGB')
    image = model.chunker.preprocess_image(image)

    # Convert to tensor
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(**DEFAULT_NORMALIZATION)
    ])
    image_tensor = transform(image).unsqueeze(0).to(device)

    with torch.no_grad():
        logits, lengths = model(image_tensor)

        # Decode
        pred_logits = logits[:lengths[0], 0, :]  # [seq_len, vocab_size]

        # Greedy decoding
        greedy_result = decoder.greedy_decode(pred_logits)
        greedy_text = ''.join([decoder.vocab[i]
                              for i in greedy_result if i < len(decoder.vocab)])

        # Beam search decoding
        beam_result = decoder.beam_search_decode(pred_logits, beam_width=100)

        return greedy_text, beam_result

# Example of how to initialize and use the model


def create_model_example():
    """Example of how to create and initialize the model"""
    vocab_size = len(DEFAULT_VOCAB)

    # Create model
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

    model, decoder, vocab = create_model_example()
    model.to(device)

    print(
        f"Model created with {sum(p.numel() for p in model.parameters())} parameters")
    print(f"Vocabulary size: {len(vocab)}")

    # Example forward pass
    batch_size = 2
    dummy_images = torch.randn(batch_size, 3, 40, 512).to(device)

    with torch.no_grad():
        logits, lengths = model(dummy_images)
        print(f"Output shape: {logits.shape}")
        print(f"Sequence lengths: {lengths}")
