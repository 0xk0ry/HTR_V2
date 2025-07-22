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
from torch.utils.data import DataLoader, Dataset
import os
import json
from pathlib import Path
import argparse
import sys
import torchvision.transforms as transforms
from data.transform import (
    Erosion, Dilation, ElasticDistortion,
    RandomTransform, GaussianNoise
)
sys.path.append('.')


class HTRDataset(Dataset):
    """Dataset class for HTR training"""

    def __init__(self, data_dir, vocab, split='train', max_length=256, target_height=64, augment=False):
        self.data_dir = Path(data_dir)
        self.vocab = vocab
        self.char_to_idx = {char: idx for idx, char in enumerate(vocab)}
        self.max_length = max_length
        self.target_height = target_height
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

        # Resize to target height while maintaining aspect ratio
        w, h = image.size
        aspect_ratio = w / h
        new_width = int(self.target_height * aspect_ratio)
        image = image.resize((new_width, self.target_height), Image.LANCZOS)

        # Apply data augmentation if enabled
        if self.augment:
            for transform in self.aug_transforms:
                image = transform(image)

            # Ensure image is still the correct size after augmentation
            # Some transforms might change dimensions slightly
            current_w, current_h = image.size
            if current_h != self.target_height:
                # Resize back to target height, maintaining aspect ratio
                aspect_ratio = current_w / current_h
                new_width = int(self.target_height * aspect_ratio)
                image = image.resize(
                    (new_width, self.target_height), Image.LANCZOS)

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

    # Pad images to same width (greyscale: 1 channel)
    max_width = max(img.size(-1) for img in images)  # Last dimension is width
    padded_images = []
    for img in images:
        # img shape should be [1, H, W] for greyscale
        channels, height, width = img.shape
        padded = torch.zeros(channels, height, max_width)
        padded[:, :, :width] = img
        padded_images.append(padded)

    images = torch.stack(padded_images)

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


def train_epoch(model, dataloader, optimizer, device, vocab, use_sam=False, gradient_clip=1.0, print_frequency=10):
    """Train for one epoch"""
    model.train()
    total_loss = 0
    num_batches = 0

    for batch_idx, (images, targets, target_lengths) in enumerate(dataloader):
        images = images.to(device)
        targets = targets.to(device)
        target_lengths = target_lengths.to(device)

        # Flatten targets for CTC loss
        targets_flat = []
        for i, length in enumerate(target_lengths):
            targets_flat.extend(targets[i][:length].tolist())
        targets_flat = torch.tensor(targets_flat, device=device)

        if use_sam:
            # First forward-backward pass
            def closure():
                optimizer.zero_grad()
                logits, loss = model(images, targets_flat, target_lengths)
                loss.backward()
                # Gradient clipping
                if gradient_clip > 0:
                    torch.nn.utils.clip_grad_norm_(
                        model.parameters(), gradient_clip)
                return loss

            loss = optimizer.step(closure)

            # Second forward-backward pass
            logits, loss = model(images, targets_flat, target_lengths)
            optimizer.zero_grad()
            loss.backward()

            # Gradient clipping
            if gradient_clip > 0:
                torch.nn.utils.clip_grad_norm_(
                    model.parameters(), gradient_clip)

            optimizer.step()
        else:
            optimizer.zero_grad()
            logits, loss = model(images, targets_flat, target_lengths)
            loss.backward()

            # Gradient clipping
            if gradient_clip > 0:
                torch.nn.utils.clip_grad_norm_(
                    model.parameters(), gradient_clip)

            optimizer.step()

        total_loss += loss.item()
        num_batches += 1

        if batch_idx % print_frequency == 0:
            print(
                f'Batch {batch_idx}/{len(dataloader)}, Loss: {loss.item():.4f}')

    return total_loss / num_batches


def validate(model, dataloader, device, vocab):
    """Validate the model"""
    model.eval()
    total_loss = 0
    total_chars = 0
    correct_chars = 0
    num_batches = 0

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

            # Decode predictions for accuracy calculation
            predictions = ctc_decode_greedy(logits, vocab)

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

    return avg_loss, char_accuracy


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
    parser.add_argument('--batch_size', type=int, default=8, help='Batch size')
    parser.add_argument('--lr', type=float, default=1e-4, help='Learning rate')
    parser.add_argument('--weight_decay', type=float,
                        default=0.01, help='Weight decay')
    parser.add_argument('--betas', type=float, nargs=2,
                        default=[0.9, 0.999], help='Adam betas')

    # Optimizer arguments
    parser.add_argument('--base_optimizer', choices=['adamw', 'adam', 'sgd'], default='adamw',
                        help='Base optimizer type')
    parser.add_argument('--use_sam', action='store_true',
                        help='Use SAM optimizer')
    parser.add_argument('--sam_rho', type=float,
                        default=0.05, help='SAM rho parameter')
    parser.add_argument('--sam_adaptive', action='store_true',
                        help='Use adaptive SAM')

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
                        default=1.0, help='Gradient clipping value')
    parser.add_argument('--early_stopping', type=int, default=20,
                        help='Early stopping patience (0=disabled)')
    parser.add_argument('--print_frequency', type=int,
                        default=10, help='Print frequency during training')

    args = parser.parse_args()

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
    if args.use_scheduler:
        print(
            f"Scheduler: {args.scheduler_type} (warmup_epochs={args.warmup_epochs})")
    print(f"Gradient clipping: {args.gradient_clip}")
    print("="*60)

    # Create vocabulary (load from pickle file if available)
    vocab = create_vocab(data_dir=args.data_dir)
    print(f"Vocabulary size: {len(vocab)}")

    # Save vocabulary
    with open(os.path.join(args.save_dir, 'vocab.json'), 'w') as f:
        json.dump(vocab, f, indent=2)

    # Create model
    model = HTRModel(
        vocab_size=len(vocab),
        max_length=256,
        target_height=64,        # Updated to 64px height
        chunk_width=512,         # Updated to 512px chunks
        first_stride=320,        # Updated to 320px first stride
        stride=384               # Updated to 384px subsequent stride
    )

    model.to(device)
    print(f"Model parameters: {sum(p.numel() for p in model.parameters()):,}")

    # Create datasets with proper split specification
    train_dataset = HTRDataset(
        args.data_dir, vocab, split='train', augment=args.augment)
    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        collate_fn=collate_fn,
        num_workers=4
    )

    # Create validation dataset from the same data directory
    val_dataset = HTRDataset(
        args.data_dir, vocab, split='valid', augment=False)
    val_loader = DataLoader(
        val_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        collate_fn=collate_fn,
        num_workers=4
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
        # Default cosine scheduler for backward compatibility
        scheduler = optim.lr_scheduler.CosineAnnealingLR(
            optimizer, T_max=args.epochs, eta_min=args.lr * 0.01
        )

    # Resume from checkpoint if specified
    start_epoch = 0
    if args.resume:
        checkpoint = torch.load(args.resume, map_location=device)
        model.load_state_dict(checkpoint['model_state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        start_epoch = checkpoint['epoch'] + 1
        print(f"Resumed from epoch {start_epoch}")

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
            use_sam=args.use_sam, gradient_clip=args.gradient_clip,
            print_frequency=args.print_frequency
        )
        train_losses.append(train_loss)
        print(f"Train Loss: {train_loss:.4f}")

        # Validate
        val_loss, char_acc = validate(model, val_loader, device, vocab)
        val_losses.append(val_loss)
        print(f"Val Loss: {val_loss:.4f}, Char Accuracy: {char_acc:.4f}")

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
                'train_loss': train_loss,
                'val_loss': val_loss,
                'train_losses': train_losses,
                'val_losses': val_losses,
                'vocab': vocab,
                'args': vars(args)
            }

            torch.save(checkpoint, os.path.join(
                args.save_dir, f'checkpoint_epoch_{epoch}.pth'))

        # Save best model
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            patience_counter = 0

            best_checkpoint = {
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'scheduler_state_dict': scheduler.state_dict() if scheduler else None,
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
