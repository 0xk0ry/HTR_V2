# Training Script Updates Summary

## Added Features

### 1. EMA (Exponential Moving Average)
- **Class**: `EMA` in train.py
- **Purpose**: Maintains shadow weights that are exponentially averaged versions of the model parameters
- **Benefits**: 
  - Better inference performance (typically 1-3% improvement)
  - More stable training
  - Reduces variance in model predictions
- **Arguments**:
  - `--use_ema`: Enable EMA
  - `--ema_decay`: Decay rate (default: 0.9999)
  - `--ema_warmup_steps`: Warmup steps (default: 2000)

### 2. Enhanced SAM Optimizer
- **Already supported**: SAM was already imported and used
- **Enhanced**: Better integration with the training loop
- **Purpose**: Sharpness-Aware Minimization for better generalization
- **Benefits**:
  - Finds flatter minima
  - Better generalization performance
  - Reduces overfitting
- **Cost**: ~2x training time due to double forward pass
- **Arguments**:
  - `--use_sam`: Enable SAM
  - `--sam_rho`: Perturbation radius (default: 0.05)
  - `--sam_adaptive`: Use adaptive SAM

### 3. Label Smoothing
- **Class**: `LabelSmoothingCTCLoss` in train.py
- **Purpose**: Regularization technique that prevents overconfident predictions
- **Benefits**:
  - Reduces overfitting
  - Better calibrated predictions
  - Improved generalization
- **How it works**: Mixes ground truth with uniform distribution
- **Arguments**:
  - `--label_smoothing`: Smoothing factor (default: 0.0, recommended: 0.05-0.15)

### 4. Updated to 64x512 Image Processing
- **Changed**: From chunking approach to simple resizing
- **Benefits**:
  - Much simpler code
  - Faster training
  - More predictable memory usage
  - Fixed-size batches
- **Updated**:
  - `HTRDataset` constructor and `__getitem__` method
  - `collate_fn` simplified (no padding needed)
  - Model creation parameters

## Key Code Changes

### Model Creation (Updated)
```python
# Before (chunking)
model = HTRModel(
    vocab_size=len(vocab),
    target_height=40,
    chunk_width=320,
    first_stride=200,
    stride=240
)

# After (resizing)
model = HTRModel(
    vocab_size=len(vocab),
    target_height=64,
    target_width=512
)
```

### Dataset Creation (Updated)
```python
# Before
train_dataset = HTRDataset(
    args.data_dir, vocab, split='train', augment=args.augment)

# After
train_dataset = HTRDataset(
    args.data_dir, vocab, split='train', 
    target_height=64, target_width=512, augment=args.augment)
```

### Training Loop (Enhanced)
```python
# Enhanced train_epoch function with EMA and label smoothing
train_loss = train_epoch(
    model, train_loader, optimizer, device, vocab,
    ema=ema, criterion=criterion,  # New parameters
    use_sam=args.use_sam, gradient_clip=args.gradient_clip,
    print_frequency=args.print_frequency
)

# Enhanced validation with EMA
val_loss, char_acc = validate(
    model, val_loader, device, vocab, 
    ema=ema, use_ema=args.use_ema  # New parameters
)
```

### Checkpoint Saving (Enhanced)
```python
checkpoint = {
    'epoch': epoch,
    'model_state_dict': model.state_dict(),
    'optimizer_state_dict': optimizer.state_dict(),
    'scheduler_state_dict': scheduler.state_dict() if scheduler else None,
    'ema_state_dict': ema.state_dict() if ema else None,  # New
    'train_loss': train_loss,
    'val_loss': val_loss,
    # ... other fields
}
```

## Usage Examples

### Full Feature Training
```bash
python train.py \
    --data_dir /path/to/data \
    --epochs 100 \
    --batch_size 16 \
    --lr 5e-4 \
    --augment \
    --use_sam \
    --sam_rho 0.05 \
    --sam_adaptive \
    --use_ema \
    --ema_decay 0.9999 \
    --label_smoothing 0.1 \
    --use_scheduler \
    --scheduler_type cosine
```

### Conservative Training (EMA + Label Smoothing only)
```bash
python train.py \
    --data_dir /path/to/data \
    --epochs 50 \
    --batch_size 32 \
    --use_ema \
    --label_smoothing 0.05
```

## Compatibility
- ✅ Backward compatible with existing arguments
- ✅ All new features are optional (disabled by default)
- ✅ Existing checkpoints can be loaded (EMA will start fresh)
- ✅ Can mix and match features as needed

## Performance Impact
- **EMA**: Minimal training overhead (~1%), significant inference improvement
- **SAM**: ~2x training time, better generalization
- **Label Smoothing**: Minimal overhead, prevents overfitting
- **64x512 Resizing**: Faster training overall due to simpler processing

## Recommended Settings for Different Scenarios

### High-Quality Training (Best Results)
```bash
--use_ema --ema_decay 0.9999 --use_sam --sam_rho 0.05 --label_smoothing 0.1
```

### Fast Training (Good Results)
```bash
--use_ema --ema_decay 0.999 --label_smoothing 0.05
```

### Experimental/Research
```bash
--use_sam --sam_rho 0.1 --sam_adaptive --label_smoothing 0.15
```
