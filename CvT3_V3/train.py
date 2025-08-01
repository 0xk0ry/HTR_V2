"""
Training script for HTR model with CvT backbone
"""

from utils.sam import SAM
import numpy as np
from PIL import Image
from model.HTR_3Stage import HTRModel, DEFAULT_VOCAB
import torch.nn.functional as F
import torch
import torch.optim as optim
import copy
from torch.utils.data import DataLoader, Dataset
import os
import json
import pickle
from pathlib import Path
import argparse
import sys
import torchvision.transforms as transforms
from transform import (
    Erosion, Dilation, ElasticDistortion,
    RandomTransform, GaussianNoise, SaltAndPepperNoise,
    Opening, Closing, Sharpen
)
import wandb
import editdistance

sys.path.append('.')


class EMA:
    """Exponential Moving Average for model parameters"""

    def __init__(self, model, decay=0.9999, warmup_steps=2000):
        self.decay = decay
        self.warmup_steps = warmup_steps
        self.num_updates = 0

        # Create EMA model (deep copy)
        self.ema_model = copy.deepcopy(model)
        self.ema_model.eval()

        # Disable gradients for EMA model
        for param in self.ema_model.parameters():
            param.requires_grad = False

    def update(self, model):
        """Update EMA parameters"""
        self.num_updates += 1

        # Compute decay factor with warmup
        decay = min(self.decay, (1 + self.num_updates) /
                    (10 + self.num_updates))
        if self.num_updates < self.warmup_steps:
            decay = self.num_updates / self.warmup_steps * self.decay

        # Update EMA parameters
        with torch.no_grad():
            for ema_param, model_param in zip(self.ema_model.parameters(), model.parameters()):
                ema_param.mul_(decay).add_(model_param, alpha=1 - decay)

    def state_dict(self):
        """Get EMA model state dict"""
        return {
            'ema_model': self.ema_model.state_dict(),
            'decay': self.decay,
            'num_updates': self.num_updates,
        }

    def load_state_dict(self, state_dict):
        """Load EMA model state dict"""
        self.ema_model.load_state_dict(state_dict['ema_model'])
        self.decay = state_dict.get('decay', self.decay)
        self.num_updates = state_dict.get('num_updates', 0)

    def apply_shadow(self, model):
        """Apply EMA weights to model (for inference)"""
        with torch.no_grad():
            for model_param, ema_param in zip(model.parameters(), self.ema_model.parameters()):
                model_param.copy_(ema_param)

    def restore(self, model, backup_params):
        """Restore original model weights"""
        with torch.no_grad():
            for model_param, backup_param in zip(model.parameters(), backup_params):
                model_param.copy_(backup_param)


class LabelSmoothingCTCLoss(torch.nn.Module):
    """CTC Loss with label smoothing"""

    def __init__(self, blank_idx=0, smoothing=0.1, reduction='mean', zero_infinity=True):
        super().__init__()
        self.blank_idx = blank_idx
        self.smoothing = smoothing
        self.reduction = reduction
        self.zero_infinity = zero_infinity
        self.ctc_loss = torch.nn.CTCLoss(
            blank=blank_idx, reduction='none', zero_infinity=zero_infinity)

    def forward(self, log_probs, targets, input_lengths, target_lengths):
        """
        Args:
            log_probs: [T, B, C] log probabilities
            targets: [sum(target_lengths)] target sequences
            input_lengths: [B] lengths of input sequences
            target_lengths: [B] lengths of target sequences
        """
        batch_size = log_probs.size(1)
        vocab_size = log_probs.size(2)

        # Compute standard CTC loss
        ctc_loss = self.ctc_loss(
            log_probs, targets, input_lengths, target_lengths)

        if self.smoothing > 0:
            # Apply label smoothing
            # Create uniform distribution over vocabulary (excluding blank)
            uniform_dist = torch.ones_like(log_probs) / (vocab_size - 1)
            uniform_dist[:, :, self.blank_idx] = 0  # Exclude blank token

            # Compute KL divergence with uniform distribution
            kl_loss = torch.nn.functional.kl_div(
                log_probs, uniform_dist, reduction='none', log_target=False
            ).sum(dim=-1)  # [T, B]

            # Average over time dimension, weighted by input lengths
            kl_loss_batch = []
            for i in range(batch_size):
                seq_len = input_lengths[i]
                kl_loss_batch.append(kl_loss[:seq_len, i].mean())
            kl_loss_batch = torch.stack(kl_loss_batch)

            # Combine CTC loss and smoothing loss
            loss = (1 - self.smoothing) * ctc_loss + \
                self.smoothing * kl_loss_batch
        else:
            loss = ctc_loss

        # Apply reduction
        if self.reduction == 'mean':
            return loss.mean()
        elif self.reduction == 'sum':
            return loss.sum()
        else:
            return loss


class HTRDataset(Dataset):
    """Dataset class for HTR training"""

    def __init__(self, data_dir, vocab, split='train', max_length=256, target_height=64, target_width=512, augment=False):
        self.data_dir = Path(data_dir)
        self.vocab = vocab
        self.char_to_idx = {char: idx for idx, char in enumerate(vocab)}
        self.max_length = max_length
        self.target_height = target_height
        self.target_width = target_width
        self.augment = augment
        self.split = split  # 'train', 'valid', or 'test'

        # Load dataset annotations
        self.samples = self._load_samples()

        # Setup data augmentation transforms
        self._setup_augmentation()

    def _load_samples(self):
        """Load image paths and corresponding texts from pickle file or fallback methods"""
        samples = []

        # First try to load from pickle file (labels.pkl)
        pkl_file = self.data_dir / "labels.pkl"
        if pkl_file.exists():
            print(f"Loading samples from pickle file: {pkl_file}")
            try:
                import pickle
                with open(pkl_file, 'rb') as f:
                    data = pickle.load(f)

                # Check if this pickle file has the expected structure
                if 'ground_truth' in data and self.split in data['ground_truth']:
                    ground_truth = data['ground_truth'][self.split]

                    for image_name, label_data in ground_truth.items():
                        # Handle both formats: direct text or nested dict with 'text' key
                        if isinstance(label_data, dict) and 'text' in label_data:
                            text = label_data['text']
                        elif isinstance(label_data, str):
                            text = label_data
                        else:
                            print(
                                f"Warning: Unexpected label format for {image_name}: {label_data}")
                            continue

                        # Construct image path
                        image_path = self.data_dir / self.split / image_name

                        # Check if image file exists
                        if image_path.exists():
                            samples.append({
                                'image_path': image_path,
                                'text': text
                            })
                        else:
                            print(
                                f"Warning: Image file not found: {image_path}")

                    print(
                        f"Loaded {len(samples)} samples from pickle file for split '{self.split}'")
                else:
                    print(
                        f"Warning: Split '{self.split}' not found in pickle file. Available keys: {list(data.get('ground_truth', {}).keys())}")

            except Exception as e:
                print(f"Error loading pickle file: {e}")
                print("Falling back to other loading methods...")

        if not samples:
            print("Loading samples from paired image/text files...")
            image_extensions = ['*.png', '*.jpg', '*.jpeg']
            for ext in image_extensions:
                for img_file in self.data_dir.glob(ext):
                    txt_file = img_file.with_suffix('.txt')
                    if txt_file.exists():
                        with open(txt_file, 'r', encoding='utf-8') as f:
                            text = f.read().strip()
                        samples.append({
                            'image_path': img_file,
                            'text': text
                        })

        print(
            f"Loaded {len(samples)} samples from {self.data_dir} (split: {self.split})")
        return samples

    def _setup_augmentation(self):
        """Setup data augmentation transforms"""
        if not self.augment:
            self.aug_transforms = []
            return

        # Each transform is applied with p=0.5 and can be combined
        self.aug_transforms = [
            # Random affine transforms (using RandomTransform from transform.py)
            transforms.RandomApply([RandomTransform(val=10)], p=0.5),

            # Erosion & Dilation
            transforms.RandomApply([
                transforms.RandomChoice([
                    Erosion(kernel=(2, 2), iterations=1),
                    Dilation(kernel=(2, 2), iterations=1)
                ])
            ], p=0.5),

            # Color jitter (using GaussianNoise for better greyscale compatibility)
            transforms.RandomApply([
                GaussianNoise(std=0.1)
            ], p=0.5),

            # Elastic distortion
            transforms.RandomApply([
                ElasticDistortion(
                    grid=(6, 6),
                    magnitude=(8, 8),
                    min_sep=(2, 2)
                )
            ], p=0.5),
        ]

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        sample = self.samples[idx]

        # Load image and convert to greyscale (as required by 3-stage model)
        image = Image.open(sample['image_path']).convert('L')

        # Resize to exact target dimensions (64x512)
        image = image.resize(
            (self.target_width, self.target_height), Image.LANCZOS)

        # Apply data augmentation if enabled
        if self.augment:
            for transform in self.aug_transforms:
                image = transform(image)

            # Ensure image is still the correct size after augmentation
            # Some transforms might change dimensions slightly
            current_w, current_h = image.size
            if current_h != self.target_height or current_w != self.target_width:
                # Resize back to target dimensions
                image = image.resize(
                    (self.target_width, self.target_height), Image.LANCZOS)

        # Convert to tensor and normalize for greyscale
        image_array = np.array(image) / 255.0
        image_tensor = torch.from_numpy(image_array).unsqueeze(
            0).float()  # Add channel dim

        # Normalize for greyscale (mean=0.5, std=0.5)
        image_tensor = (image_tensor - 0.5) / 0.5

        # Encode text
        text = sample['text']
        encoded_text = [self.char_to_idx.get(
            char, self.char_to_idx.get('<unk>', 1)) for char in text]
        encoded_text = encoded_text[:self.max_length]  # Truncate if too long

        return {
            'image': image_tensor,
            'text': encoded_text,
            'text_length': len(encoded_text),
            'original_text': text
        }


def collate_fn(batch):
    """Collate function for DataLoader"""
    images = [item['image'] for item in batch]
    texts = [item['text'] for item in batch]
    text_lengths = [item['text_length'] for item in batch]

    # Stack images directly since they're all the same size (64x512)
    images = torch.stack(images)

    # Pad texts to same length
    max_text_len = max(text_lengths)
    padded_texts = []
    for text in texts:
        padded = text + [0] * (max_text_len - len(text)
                               )  # Pad with blank token
        padded_texts.append(padded)

    texts = torch.tensor(padded_texts)
    text_lengths = torch.tensor(text_lengths)

    return images, texts, text_lengths


def create_vocab(data_dir=None):
    """Create vocabulary for the model"""

    # Try to load vocabulary from pickle file first
    if data_dir:
        pkl_file = Path(data_dir) / "labels.pkl"
        if pkl_file.exists():
            try:
                import pickle
                with open(pkl_file, 'rb') as f:
                    data = pickle.load(f)

                if 'charset' in data:
                    vocab = list(data['charset'])
                    # Ensure blank token is at index 0
                    if '' not in vocab:
                        vocab.insert(0, '')  # Add blank token at index 0
                    elif vocab[0] != '':
                        vocab.remove('')
                        vocab.insert(0, '')  # Move blank token to index 0

                    print(
                        f"Loaded vocabulary from pickle file: {len(vocab)} characters")
                    return vocab
                else:
                    print("Warning: 'charset' not found in pickle file")
            except Exception as e:
                print(f"Error loading vocabulary from pickle file: {e}")

    # Fallback to default vocabulary
    print("Using default vocabulary")
    return DEFAULT_VOCAB


def ctc_decode_greedy(logits, vocab):
    """Simple greedy CTC decoder"""
    # logits: [T, B, vocab_size]
    predictions = torch.argmax(logits, dim=-1)  # [T, B]
    decoded_texts = []

    for b in range(predictions.size(1)):
        pred_seq = predictions[:, b].cpu().numpy()

        # Remove consecutive duplicates and blanks
        decoded_chars = []
        prev_char = None
        for char_idx in pred_seq:
            if char_idx != 0 and char_idx != prev_char:  # 0 is blank
                if char_idx < len(vocab):
                    decoded_chars.append(vocab[char_idx])
            prev_char = char_idx

        decoded_texts.append(''.join(decoded_chars))

    return decoded_texts


def train_epoch(model, dataloader, optimizer, device, vocab, ema=None, criterion=None, use_sam=False, gradient_clip=1.0, print_frequency=10):
    """Train for one epoch with EMA and label smoothing"""
    model.train()
    total_loss = 0
    num_batches = 0

    # Enable gradient checkpointing only when needed for very large models
    if hasattr(model, 'cvt'):
        # Disable checkpointing for faster training with sufficient memory
        model.cvt.use_checkpoint = False

    # Use mixed precision for faster training
    scaler = torch.amp.GradScaler(
        'cuda') if torch.cuda.is_available() else None

    for batch_idx, (images, targets, target_lengths) in enumerate(dataloader):
        images = images.to(device, non_blocking=True)
        targets = targets.to(device, non_blocking=True)
        target_lengths = target_lengths.to(device, non_blocking=True)

        # Flatten targets for CTC loss
        targets_flat = []
        for i, length in enumerate(target_lengths):
            targets_flat.extend(targets[i][:length].tolist())
        targets_flat = torch.tensor(targets_flat, device=device)

        optimizer.zero_grad()

        if use_sam:
            # SAM training requires special handling
            def closure():
                optimizer.zero_grad()
                with torch.amp.autocast('cuda') if scaler else torch.no_grad():
                    if criterion is not None:
                        # Use custom loss function with label smoothing
                        logits, input_lengths = model(images)
                        log_probs = torch.nn.functional.log_softmax(
                            logits, dim=-1)
                        loss = criterion(log_probs, targets_flat,
                                         input_lengths, target_lengths)
                    else:
                        # Use model's built-in loss
                        logits, loss = model(
                            images, targets_flat, target_lengths)

                if scaler:
                    scaler.scale(loss).backward()
                else:
                    loss.backward()
                return loss

            # First forward-backward pass
            loss = closure()

            # SAM step with closure
            if scaler:
                scaler.unscale_(optimizer)
                if gradient_clip > 0:
                    torch.nn.utils.clip_grad_norm_(
                        model.parameters(), gradient_clip)
                scaler.step(optimizer, closure)
                scaler.update()
            else:
                if gradient_clip > 0:
                    torch.nn.utils.clip_grad_norm_(
                        model.parameters(), gradient_clip)
                optimizer.step(closure)
        else:
            # Standard training with mixed precision
            with torch.amp.autocast('cuda'):
                if criterion is not None:
                    # Use custom loss function with label smoothing
                    logits, input_lengths = model(images)
                    log_probs = torch.nn.functional.log_softmax(logits, dim=-1)
                    loss = criterion(log_probs, targets_flat,
                                     input_lengths, target_lengths)
                else:
                    # Use model's built-in loss
                    logits, loss = model(images, targets_flat, target_lengths)

            if scaler:
                scaler.scale(loss).backward()
                scaler.unscale_(optimizer)
                if gradient_clip > 0:
                    torch.nn.utils.clip_grad_norm_(
                        model.parameters(), gradient_clip)
                scaler.step(optimizer)
                scaler.update()
            else:
                loss.backward()
                if gradient_clip > 0:
                    torch.nn.utils.clip_grad_norm_(
                        model.parameters(), gradient_clip)
                optimizer.step()

        # Update EMA after each batch
        if ema is not None:
            ema.update(model)

        total_loss += loss.item()
        num_batches += 1

        # Print progress more frequently for feedback
        if print_frequency > 0 and batch_idx % print_frequency == 0:
            print(
                f"  Batch {batch_idx}/{len(dataloader)}, Loss: {loss.item():.4f}")

        # Less frequent cache clearing and optimized memory management
        if batch_idx % 100 == 0 and torch.cuda.is_available():
            torch.cuda.empty_cache()

    return total_loss / num_batches


def validate(model, dataloader, device, vocab, ema=None, use_ema=True):
    """Validate the model with optional EMA"""

    # Store original parameters if using EMA
    original_params = None
    if ema is not None and use_ema:
        original_params = [param.clone() for param in model.parameters()]
        ema.apply_shadow(model)

    model.eval()
    total_loss = 0
    total_chars = 0
    correct_chars = 0
    num_batches = 0
    cer_total = 0
    wer_total = 0

    with torch.no_grad():
        for images, targets, target_lengths in dataloader:
            images = images.to(device)
            targets = targets.to(device)
            target_lengths = target_lengths.to(device)

            # Flatten targets for CTC loss
            targets_flat = []
            for i, length in enumerate(target_lengths):
                targets_flat.extend(targets[i][:length].tolist())
            targets_flat = torch.tensor(targets_flat, device=device)

            logits, loss = model(images, targets_flat, target_lengths)
            total_loss += loss.item()
            num_batches += 1

            # Decode predictions for CER and WER calculation
            predictions = ctc_decode_greedy(logits, vocab)
            target_texts = [''.join([vocab[idx] for idx in targets[i][:target_lengths[i].item(
            )].cpu().numpy()]) for i in range(len(targets))]

            # Calculate CER and WER
            cer_total += calculate_cer(predictions, target_texts)
            wer_total += calculate_wer(predictions, target_texts)

            # Calculate character accuracy
            for i, (pred, target_len) in enumerate(zip(predictions, target_lengths)):
                target_chars = [vocab[idx]
                                for idx in targets[i][:target_len].cpu().numpy()]
                target_text = ''.join(target_chars)

                # Character-level accuracy
                for p_char, t_char in zip(pred, target_text):
                    if p_char == t_char:
                        correct_chars += 1
                    total_chars += 1

                # Add penalty for length mismatch
                if len(pred) != len(target_text):
                    total_chars += abs(len(pred) - len(target_text))

    char_accuracy = correct_chars / max(total_chars, 1)
    avg_loss = total_loss / num_batches
    avg_cer = cer_total / num_batches
    avg_wer = wer_total / num_batches

    # Restore original parameters if using EMA
    if ema is not None and use_ema and original_params is not None:
        ema.restore(model, original_params)

    return avg_loss, char_accuracy, avg_cer, avg_wer


def calculate_cer(predictions, targets):
    """Calculate Character Error Rate (CER)"""
    total_chars = 0
    total_errors = 0

    for pred, target in zip(predictions, targets):
        total_chars += len(target)
        total_errors += editdistance.eval(pred, target)

    return total_errors / total_chars if total_chars > 0 else 0


def calculate_wer(predictions, targets):
    """Calculate Word Error Rate (WER)"""
    total_words = 0
    total_errors = 0

    for pred, target in zip(predictions, targets):
        pred_words = pred.split()
        target_words = target.split()
        total_words += len(target_words)
        total_errors += editdistance.eval(pred_words, target_words)

    return total_errors / total_words if total_words > 0 else 0


def get_memory_usage():
    """Get current GPU memory usage in MB"""
    if torch.cuda.is_available():
        return torch.cuda.memory_allocated() / 1024**2
    return 0


def main():
    parser = argparse.ArgumentParser(description='Train HTR 3-Stage Model')

    # Data arguments
    parser.add_argument('--data_dir', type=str, required=True,
                        help='Data directory containing labels.pkl and train/valid/test splits')
    parser.add_argument('--augment', action='store_true',
                        help='Enable data augmentation')

    # Training arguments
    parser.add_argument('--epochs', type=int, default=100,
                        help='Number of epochs')
    parser.add_argument('--batch_size', type=int, default=16,
                        help='Batch size (increased for better GPU utilization)')  # Increased from 8
    parser.add_argument('--lr', type=float, default=5e-4,
                        help='Learning rate (increased for faster convergence)')
    parser.add_argument('--weight_decay', type=float,
                        default=0.05, help='Weight decay (increased for better regularization)')
    parser.add_argument('--betas', type=float, nargs=2,
                        default=[0.9, 0.95], help='Adam betas (optimized for transformers)')

    # Optimizer arguments
    parser.add_argument('--base_optimizer', choices=['adamw', 'adam', 'sgd'], default='adamw',
                        help='Base optimizer type')
    parser.add_argument('--use_sam', action='store_true',
                        help='Use SAM optimizer')
    parser.add_argument('--sam_rho', type=float,
                        default=0.05, help='SAM rho parameter')
    parser.add_argument('--sam_adaptive', action='store_true',
                        help='Use adaptive SAM')

    # EMA arguments
    parser.add_argument('--use_ema', action='store_true',
                        help='Use Exponential Moving Average')
    parser.add_argument('--ema_decay', type=float, default=0.9999,
                        help='EMA decay rate')
    parser.add_argument('--ema_warmup_steps', type=int, default=2000,
                        help='EMA warmup steps')

    # Label smoothing arguments
    parser.add_argument('--label_smoothing', type=float, default=0.0,
                        help='Label smoothing factor (0.0 = no smoothing)')

    # Scheduler arguments
    parser.add_argument('--use_scheduler', action='store_true',
                        help='Use learning rate scheduler')
    parser.add_argument('--scheduler_type', choices=['cosine', 'step'], default='cosine',
                        help='Type of learning rate scheduler')
    parser.add_argument('--warmup_epochs', type=int,
                        default=10, help='Warmup epochs')
    parser.add_argument('--scheduler_patience', type=int,
                        default=10, help='Scheduler patience for plateau')

    # Model arguments
    parser.add_argument('--resume', type=str, help='Resume from checkpoint')
    parser.add_argument('--save_dir', type=str,
                        default='./checkpoints', help='Checkpoint save directory')
    parser.add_argument('--save_frequency', type=int,
                        default=10, help='Save checkpoint every N epochs')

    # Advanced training options
    parser.add_argument('--gradient_clip', type=float,
                        default=0.5, help='Gradient clipping value (reduced for stability)')
    parser.add_argument('--early_stopping', type=int, default=15,
                        help='Early stopping patience (reduced for faster training)')
    parser.add_argument('--print_frequency', type=int,
                        default=0, help='Print frequency during training (more frequent feedback)')

    args = parser.parse_args()

    # Initialize wandb
    wandb.init(project="CVT3_V3", config={
        "epochs": args.epochs,
        "batch_size": args.batch_size,
        "learning_rate": args.lr,
        "weight_decay": args.weight_decay,
        "optimizer": args.base_optimizer,
        "gradient_clip": args.gradient_clip,
        "label_smoothing": args.label_smoothing,
        "use_ema": args.use_ema,
        "ema_decay": args.ema_decay,
        "ema_warmup_steps": args.ema_warmup_steps,
        "use_scheduler": args.use_scheduler,
        "scheduler_type": args.scheduler_type,
        "warmup_epochs": args.warmup_epochs
    })

    # Create save directory
    os.makedirs(args.save_dir, exist_ok=True)

    # Device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")

    # Print configuration
    print("\n" + "="*60)
    print("HTR 3-Stage Model Training Configuration")
    print("="*60)
    print(f"Data directory: {args.data_dir}")
    print(f"Save directory: {args.save_dir}")
    print(f"Epochs: {args.epochs}")
    print(f"Batch size: {args.batch_size}")
    print(f"Learning rate: {args.lr}")
    print(f"Weight decay: {args.weight_decay}")
    print(f"Optimizer: {args.base_optimizer.upper()}")
    print(f"Betas: {tuple(args.betas)}")
    print(f"Data augmentation: {'Enabled' if args.augment else 'Disabled'}")
    if args.use_sam:
        print(
            f"SAM optimizer: Enabled (rho={args.sam_rho}, adaptive={args.sam_adaptive})")
    if args.use_ema:
        print(
            f"EMA: Enabled (decay={args.ema_decay}, warmup={args.ema_warmup_steps})")
    if args.label_smoothing > 0:
        print(f"Label smoothing: Enabled (factor={args.label_smoothing})")
    if args.use_scheduler:
        print(
            f"Scheduler: {args.scheduler_type} (warmup_epochs={args.warmup_epochs})")
    print(f"Gradient clipping: {args.gradient_clip}")
    print("="*60)

    # Create vocabulary (load from pickle file if available)
    # vocab = create_vocab(data_dir=args.data_dir)
    vocab = DEFAULT_VOCAB
    print(f"Vocabulary size: {len(vocab)}")

    # Save vocabulary
    with open(os.path.join(args.save_dir, 'vocab.json'), 'w') as f:
        json.dump(vocab, f, indent=2)

    # Create model
    model = HTRModel(
        vocab_size=len(vocab),
        max_length=256,
        target_height=64,        # Updated to 64px height
        target_width=512         # Updated to 512px width
    )

    model.to(device)
    print(f"Model parameters: {sum(p.numel() for p in model.parameters()):,}")

    # Create EMA if enabled
    ema = None
    if args.use_ema:
        ema = EMA(model, decay=args.ema_decay,
                  warmup_steps=args.ema_warmup_steps)
        print(
            f"EMA enabled with decay={args.ema_decay}, warmup_steps={args.ema_warmup_steps}")

    # Create label smoothing loss if enabled
    criterion = None
    if args.label_smoothing > 0:
        criterion = LabelSmoothingCTCLoss(
            blank_idx=0,
            smoothing=args.label_smoothing,
            reduction='mean',
            zero_infinity=True
        )
        print(f"Label smoothing enabled with factor={args.label_smoothing}")

    # Create datasets with proper split specification
    train_dataset = HTRDataset(
        args.data_dir, vocab, split='train', target_height=64, target_width=512, augment=args.augment)
    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        collate_fn=collate_fn,
        num_workers=4,  # Increased from 8 for even faster data loading
        pin_memory=True,  # Faster GPU transfer
        persistent_workers=True,  # Keep workers alive between epochs
        prefetch_factor=4  # Prefetch more batches for smoother training
    )

    # Create validation dataset from the same data directory
    val_dataset = HTRDataset(
        args.data_dir, vocab, split='valid', target_height=64, target_width=512, augment=False)
    val_loader = DataLoader(
        val_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        collate_fn=collate_fn,
        num_workers=4,  # Fewer workers for validation
        pin_memory=True
    )

    # Base optimizer classes
    base_optimizers = {
        'adamw': optim.AdamW,
        'adam': optim.Adam,
        'sgd': optim.SGD
    }

    # Create optimizer with improved configuration
    if args.use_sam:
        print(f"Using SAM optimizer with {args.base_optimizer.upper()} base")

        if args.base_optimizer in ['adamw', 'adam']:
            optimizer = SAM(
                model.parameters(),
                base_optimizers[args.base_optimizer],
                lr=args.lr,
                betas=tuple(args.betas),
                weight_decay=args.weight_decay,
                rho=args.sam_rho,
                adaptive=args.sam_adaptive
            )
        else:  # SGD
            optimizer = SAM(
                model.parameters(),
                base_optimizers[args.base_optimizer],
                lr=args.lr,
                weight_decay=args.weight_decay,
                momentum=0.9,
                rho=args.sam_rho,
                adaptive=args.sam_adaptive
            )
    else:
        if args.base_optimizer == 'adamw':
            optimizer = optim.AdamW(
                model.parameters(),
                lr=args.lr,
                betas=tuple(args.betas),
                weight_decay=args.weight_decay
            )
        elif args.base_optimizer == 'adam':
            optimizer = optim.Adam(
                model.parameters(),
                lr=args.lr,
                betas=tuple(args.betas),
                weight_decay=args.weight_decay
            )
        elif args.base_optimizer == 'sgd':
            optimizer = optim.SGD(
                model.parameters(),
                lr=args.lr,
                weight_decay=args.weight_decay,
                momentum=0.9
            )

    # Learning rate scheduler
    scheduler = None
    if args.use_scheduler:
        if args.scheduler_type == 'cosine':
            scheduler = optim.lr_scheduler.CosineAnnealingLR(
                optimizer, T_max=args.epochs, eta_min=args.lr * 0.01
            )
        elif args.scheduler_type == 'step':
            scheduler = optim.lr_scheduler.StepLR(
                optimizer, step_size=args.epochs // 3, gamma=0.1
            )
        print(f"Using {args.scheduler_type} learning rate scheduler")
    else:
        scheduler = None
        print("No learning rate scheduler - using constant learning rate")

    # Resume from checkpoint if specified
    start_epoch = 0
    if args.resume:
        checkpoint = torch.load(args.resume, map_location=device)
        model.load_state_dict(checkpoint['model_state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        if scheduler and 'scheduler_state_dict' in checkpoint and checkpoint['scheduler_state_dict']:
            scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
        if ema and 'ema_state_dict' in checkpoint and checkpoint['ema_state_dict']:
            ema.load_state_dict(checkpoint['ema_state_dict'])
        start_epoch = checkpoint['epoch'] + 1
        print(f"Resumed from epoch {start_epoch}")
        if ema:
            print(f"EMA state restored from checkpoint")

    # Training loop with improved features
    best_val_loss = float('inf')
    patience_counter = 0
    train_losses = []
    val_losses = []

    print(f"\nStarting training for {args.epochs} epochs...")
    print(f"Training samples: {len(train_dataset)}")
    print(f"Validation samples: {len(val_dataset)}")

    for epoch in range(start_epoch, args.epochs):
        print(f"\nEpoch {epoch + 1}/{args.epochs}")
        print("-" * 50)

        # Train
        train_loss = train_epoch(
            model, train_loader, optimizer, device, vocab,
            ema=ema, criterion=criterion,
            use_sam=args.use_sam, gradient_clip=args.gradient_clip,
            print_frequency=args.print_frequency
        )
        train_losses.append(train_loss)
        print(f"Train Loss: {train_loss:.4f}")

        # Validate
        val_loss, char_acc, avg_cer, avg_wer = validate(
            model, val_loader, device, vocab, ema=ema, use_ema=args.use_ema)
        val_losses.append(val_loss)
        print(f"Val Loss: {val_loss:.4f}, Char Accuracy: {char_acc:.4f}")

        # Log CER and WER to wandb
        wandb.log({
            "train_loss": train_loss,
            "val_loss": val_loss,
            "char_accuracy": char_acc,
            "cer": avg_cer,
            "wer": avg_wer
        })

        # Print CER and WER
        print(f"CER: {avg_cer:.4f}, WER: {avg_wer:.4f}")

        # Learning rate scheduling
        if scheduler:
            if args.scheduler_type == 'plateau':
                scheduler.step(val_loss)
            else:
                scheduler.step()

            current_lr = scheduler.get_last_lr()[0] if hasattr(
                scheduler, 'get_last_lr') else args.lr
            print(f"Learning rate: {current_lr:.6f}")

        # Save checkpoint periodically
        if (epoch + 1) % args.save_frequency == 0:
            checkpoint = {
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'scheduler_state_dict': scheduler.state_dict() if scheduler else None,
                'ema_state_dict': ema.state_dict() if ema else None,
                'train_loss': train_loss,
                'val_loss': val_loss,
                'train_losses': train_losses,
                'val_losses': val_losses,
                'vocab': vocab,
                'args': vars(args)
            }

            torch.save(checkpoint, os.path.join(
                args.save_dir, f'checkpoint_epoch_{epoch+1}.pth'))

        # Save best model
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            patience_counter = 0

            best_checkpoint = {
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'scheduler_state_dict': scheduler.state_dict() if scheduler else None,
                'ema_state_dict': ema.state_dict() if ema else None,
                'train_loss': train_loss,
                'val_loss': val_loss,
                'train_losses': train_losses,
                'val_losses': val_losses,
                'vocab': vocab,
                'args': vars(args)
            }

            torch.save(best_checkpoint, os.path.join(
                args.save_dir, 'best_model.pth'))
            print(f"🎯 New best model saved! Val loss: {val_loss:.4f}")
        else:
            patience_counter += 1

        # Early stopping
        if args.early_stopping > 0 and patience_counter >= args.early_stopping:
            print(
                f"\n⏹️  Early stopping triggered after {patience_counter} epochs without improvement")
            break

    print(f"\n✅ Training completed!")
    print(f"Best validation loss: {best_val_loss:.4f}")
    print(f"Final model saved to: {args.save_dir}/best_model.pth")


if __name__ == "__main__":
    main()
