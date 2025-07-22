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

    def __init__(self, data_dir, vocab, max_length=256, target_height=64, augment=False):
        self.data_dir = Path(data_dir)
        self.vocab = vocab
        self.char_to_idx = {char: idx for idx, char in enumerate(vocab)}
        self.max_length = max_length
        self.target_height = target_height
        self.augment = augment

        # Load dataset annotations
        self.samples = self._load_samples()
        
        # Setup data augmentation transforms
        self._setup_augmentation()

    def _load_samples(self):
        """Load image paths and corresponding texts"""
        samples = []

        # Look for annotation file
        annotation_file = self.data_dir / "annotations.json"
        if annotation_file.exists():
            with open(annotation_file, 'r', encoding='utf-8') as f:
                annotations = json.load(f)
                for item in annotations:
                    samples.append({
                        'image_path': self.data_dir / item['image'],
                        'text': item['text']
                    })
        else:
            # Fallback: look for paired image/txt files (.png, .jpg, .jpeg)
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

        print(f"Loaded {len(samples)} samples from {self.data_dir}")
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
                image = image.resize((new_width, self.target_height), Image.LANCZOS)

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


def create_vocab():
    """Create vocabulary for the model"""
    # Use the default vocabulary from the 3-stage model
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


def train_epoch(model, dataloader, optimizer, device, vocab, use_sam=False):
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
                return loss

            loss = optimizer.step(closure)

            # Second forward-backward pass
            logits, loss = model(images, targets_flat, target_lengths)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
        else:
            optimizer.zero_grad()
            logits, loss = model(images, targets_flat, target_lengths)
            loss.backward()
            optimizer.step()

        total_loss += loss.item()
        num_batches += 1

        if batch_idx % 10 == 0:
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
    parser = argparse.ArgumentParser(description='Train HTR model')
    parser.add_argument('--data_dir', type=str,
                        required=True, help='Path to training data')
    parser.add_argument('--val_data_dir', type=str,
                        help='Path to validation data')
    parser.add_argument('--output_dir', type=str,
                        default='./checkpoints', help='Output directory')
    parser.add_argument('--epochs', type=int, default=100,
                        help='Number of epochs')
    parser.add_argument('--batch_size', type=int, default=8, help='Batch size')
    parser.add_argument('--lr', type=float, default=1e-4, help='Learning rate')
    parser.add_argument('--resume', type=str, help='Resume from checkpoint')
    parser.add_argument('--lm_path', type=str,
                        help='Path to KenLM language model')

    # Data augmentation options
    parser.add_argument('--augment', action='store_true',
                        help='Enable data augmentation for training')

    # SAM optimizer options
    parser.add_argument('--use_sam', action='store_true',
                        help='Use SAM (Sharpness-Aware Minimization) optimizer')
    parser.add_argument('--sam_rho', type=float, default=0.05,
                        help='SAM rho parameter (neighborhood size)')
    parser.add_argument('--sam_adaptive', action='store_true',
                        help='Use adaptive SAM')
    parser.add_argument('--base_optimizer', type=str, default='adamw',
                        choices=['adamw', 'adam', 'sgd'],
                        help='Base optimizer for SAM')

    args = parser.parse_args()

    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)

    # Device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")

    # Create vocabulary
    vocab = create_vocab()
    print(f"Vocabulary size: {len(vocab)}")

    # Save vocabulary
    with open(os.path.join(args.output_dir, 'vocab.json'), 'w') as f:
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

    # Create datasets
    print(f"Data augmentation: {'Enabled' if args.augment else 'Disabled'}")
    train_dataset = HTRDataset(args.data_dir, vocab, augment=args.augment)
    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        collate_fn=collate_fn,
        num_workers=4
    )

    val_loader = None
    if args.val_data_dir:
        # Validation dataset should not use augmentation
        val_dataset = HTRDataset(args.val_data_dir, vocab, augment=False)
        val_loader = DataLoader(
            val_dataset,
            batch_size=args.batch_size,
            shuffle=False,
            collate_fn=collate_fn,
            num_workers=4
        )

    # Optimizer and scheduler
    if args.use_sam:
        print(f"Using SAM optimizer with {args.base_optimizer.upper()} base")
        print(f"SAM rho: {args.sam_rho}, adaptive: {args.sam_adaptive}")

        # Create base optimizer class
        base_optimizers = {
            'adamw': optim.AdamW,
            'adam': optim.Adam,
            'sgd': optim.SGD
        }
        base_optimizer_class = base_optimizers[args.base_optimizer]

        # Create SAM optimizer
        optimizer = SAM(
            model.parameters(),
            base_optimizer_class,
            lr=args.lr,
            weight_decay=0.01,
            rho=args.sam_rho,
            adaptive=args.sam_adaptive
        )
    else:
        print("Using standard AdamW optimizer")
        optimizer = optim.AdamW(
            model.parameters(), lr=args.lr, weight_decay=0.01)

    scheduler = optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=args.epochs)

    # Resume from checkpoint if specified
    start_epoch = 0
    if args.resume:
        checkpoint = torch.load(args.resume, map_location=device)
        model.load_state_dict(checkpoint['model_state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        start_epoch = checkpoint['epoch'] + 1
        print(f"Resumed from epoch {start_epoch}")

    # Training loop
    best_val_loss = float('inf')

    for epoch in range(start_epoch, args.epochs):
        print(f"\\nEpoch {epoch + 1}/{args.epochs}")
        print("-" * 50)

        # Train
        train_loss = train_epoch(
            model, train_loader, optimizer, device, vocab, use_sam=args.use_sam)
        print(f"Train Loss: {train_loss:.4f}")

        # Validate
        if val_loader:
            val_loss, char_acc = validate(model, val_loader, device, vocab)
            print(f"Val Loss: {val_loss:.4f}, Char Accuracy: {char_acc:.4f}")
        else:
            val_loss = train_loss

        # Save checkpoint
        checkpoint = {
            'epoch': epoch,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'train_loss': train_loss,
            'val_loss': val_loss,
            'vocab': vocab
        }

        torch.save(checkpoint, os.path.join(
            args.output_dir, f'checkpoint_epoch_{epoch}.pth'))

        # Save best model
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save(checkpoint, os.path.join(
                args.output_dir, 'best_model.pth'))
            print(f"New best model saved with val loss: {val_loss:.4f}")

        # Update learning rate
        scheduler.step()
        print(f"Learning rate: {scheduler.get_last_lr()[0]:.6f}")


if __name__ == "__main__":
    main()
