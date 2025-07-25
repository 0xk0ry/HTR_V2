"""
Inference script for HTR model with CvT backbone
"""

from CvT3_V1.model.CTC_Decoder import CTCDecoder
from CvT3_V1.model.HTR_3Stage import HTRModel, DEFAULT_VOCAB
import torch
import json
import argparse
from pathlib import Path
from PIL import Image
import sys
import re
sys.path.append('.')
import editdistance


def format_string_for_wer(str_input):
    """Format string for WER calculation by adding spaces around punctuation"""
    str_input = re.sub(
        '([\[\]{}/\\()\"\'&+*=<>?.;:,!\-—_€#%°])', r' \1 ', str_input)
    str_input = re.sub('([ \n])+', " ", str_input).strip()
    return str_input


def calculate_cer(predicted, ground_truth):
    """Calculate Character Error Rate (CER) using edit distance"""
    if len(ground_truth) == 0:
        return 0.0 if len(predicted) == 0 else 1.0

    # Calculate edit distance between characters
    distance = editdistance.eval(predicted, ground_truth)
    return distance / len(ground_truth)


def calculate_wer(predicted, ground_truth):
    """Calculate Word Error Rate (WER) using edit distance"""
    # Format strings for WER calculation (add spaces around punctuation)
    pred_formatted = format_string_for_wer(predicted)
    gt_formatted = format_string_for_wer(ground_truth)

    # Split into words
    pred_words = pred_formatted.split()
    gt_words = gt_formatted.split()

    if len(gt_words) == 0:
        return 0.0 if len(pred_words) == 0 else 1.0

    # Calculate edit distance between word lists
    distance = editdistance.eval(pred_words, gt_words)
    return distance / len(gt_words)


def calculate_metrics_batch(predictions_greedy, predictions_beam, ground_truths):
    """
    Calculate CER and WER using the same method as HTR_VT validation
    Accumulates total edit distances and lengths, then calculates final metrics
    """
    # Initialize accumulators
    total_cer_distance_greedy = 0
    total_cer_distance_beam = 0
    total_wer_distance_greedy = 0
    total_wer_distance_beam = 0
    total_char_length = 0
    total_word_length = 0

    for pred_greedy, pred_beam, gt in zip(predictions_greedy, predictions_beam, ground_truths):
        # CER calculation
        cer_dist_greedy = editdistance.eval(pred_greedy, gt)
        cer_dist_beam = editdistance.eval(pred_beam, gt)
        total_cer_distance_greedy += cer_dist_greedy
        total_cer_distance_beam += cer_dist_beam
        total_char_length += len(gt)

        # WER calculation
        pred_greedy_formatted = format_string_for_wer(pred_greedy)
        pred_beam_formatted = format_string_for_wer(pred_beam)
        gt_formatted = format_string_for_wer(gt)

        pred_greedy_words = pred_greedy_formatted.split()
        pred_beam_words = pred_beam_formatted.split()
        gt_words = gt_formatted.split()

        wer_dist_greedy = editdistance.eval(pred_greedy_words, gt_words)
        wer_dist_beam = editdistance.eval(pred_beam_words, gt_words)
        total_wer_distance_greedy += wer_dist_greedy
        total_wer_distance_beam += wer_dist_beam
        total_word_length += len(gt_words)

    # Calculate final metrics
    if total_char_length > 0:
        cer_greedy = total_cer_distance_greedy / total_char_length
        cer_beam = total_cer_distance_beam / total_char_length
    else:
        cer_greedy = cer_beam = 0.0

    if total_word_length > 0:
        wer_greedy = total_wer_distance_greedy / total_word_length
        wer_beam = total_wer_distance_beam / total_word_length
    else:
        wer_greedy = wer_beam = 0.0

    return cer_greedy, cer_beam, wer_greedy, wer_beam


def load_ground_truth(image_path):
    """Load ground truth text for an image"""
    image_path = Path(image_path)
    # Replace image extension with .txt
    gt_path = image_path.with_suffix('.txt')

    if gt_path.exists():
        with open(gt_path, 'r', encoding='utf-8') as f:
            return f.read().strip()
    return None


def load_model_and_vocab(checkpoint_path, device):
    """Load trained model and vocabulary"""
    checkpoint = torch.load(checkpoint_path, map_location=device)
    # vocab = checkpoint['vocab']
    vocab=DEFAULT_VOCAB

    model = HTRModel(
        vocab_size=len(vocab),
        max_length=256,
        target_height=40,        # Updated to 40px height
        chunk_width=320,         # Updated to 320px chunks
        first_stride=200,        # Updated to 200px first stride
        stride=240               # Updated to 240px subsequent stride
    )

    model.load_state_dict(checkpoint['model_state_dict'])
    model.to(device)
    model.eval()

    return model, vocab


def predict_single_image(model, decoder, image_path, device, decode_mode='both', beam_width=100):
    """Predict text from a single image"""
    # Load and preprocess image
    image = Image.open(image_path).convert('RGB')

    # Use the model's chunker for preprocessing
    preprocessed_image = model.chunker.preprocess_image(image).convert('L')

    # Convert to tensor
    import torchvision.transforms as transforms
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5], std=[0.5])
    ])

    image_tensor = transform(preprocessed_image).unsqueeze(0).to(device)

    with torch.no_grad():
        logits, lengths = model(image_tensor)

        # Get logits for the first (and only) sample
        pred_logits = logits[:lengths[0], 0, :]  # [seq_len, vocab_size]

        result = {}
        if decode_mode in ('greedy', 'both'):
            greedy_result = decoder.greedy_decode(pred_logits)
            greedy_text = ''.join([decoder.vocab[i]
                                  for i in greedy_result if i < len(decoder.vocab)])
            result['greedy'] = greedy_text
        if decode_mode in ('beam', 'both'):
            beam_result = decoder.beam_search_decode(pred_logits, beam_width=beam_width)
            result['beam_search'] = beam_result
        result['confidence'] = torch.softmax(pred_logits, dim=-1).max(dim=-1)[0].mean().item()
        return result


def batch_inference(model, decoder, image_dir, output_file, device, decode_mode='both', beam_width=100):
    """Run inference on all images in a directory"""
    image_dir = Path(image_dir)
    results = []

    # Collect predictions and ground truths for batch metric calculation
    predictions_greedy = []
    predictions_beam = []
    ground_truths = []

    # Supported image extensions
    extensions = ['.jpg', '.jpeg', '.png', '.bmp', '.tiff']

    image_files = []
    for ext in extensions:
        image_files.extend(image_dir.glob(f"*{ext}"))

    print(f"Found {len(image_files)} images to process")

    for image_file in image_files:
        print(f"Processing: {image_file.name}")

        try:
            prediction = predict_single_image(
                model, decoder, image_file, device, decode_mode=decode_mode, beam_width=beam_width)

            # Load ground truth
            ground_truth = load_ground_truth(image_file)

            result = {
                'image_path': str(image_file),
                'image_name': image_file.name,
                'confidence': prediction['confidence'],
                'ground_truth': ground_truth
            }
            if 'greedy' in prediction:
                result['greedy_prediction'] = prediction['greedy']
            if 'beam_search' in prediction:
                result['beam_search_prediction'] = prediction['beam_search']

            # Calculate individual metrics for display
            if ground_truth is not None:
                if 'greedy' in prediction:
                    cer_greedy = calculate_cer(prediction['greedy'], ground_truth)
                    wer_greedy = calculate_wer(prediction['greedy'], ground_truth)
                    result['cer_greedy'] = cer_greedy
                    result['wer_greedy'] = wer_greedy
                    predictions_greedy.append(prediction['greedy'])
                if 'beam_search' in prediction:
                    cer_beam = calculate_cer(prediction['beam_search'], ground_truth)
                    wer_beam = calculate_wer(prediction['beam_search'], ground_truth)
                    result['cer_beam_search'] = cer_beam
                    result['wer_beam_search'] = wer_beam
                    predictions_beam.append(prediction['beam_search'])
                ground_truths.append(ground_truth)

                if 'greedy' in prediction:
                    print(f"  Greedy: {prediction['greedy']}")
                if 'beam_search' in prediction:
                    print(f"  Beam Search: {prediction['beam_search']}")
                print(f"  Ground Truth: {ground_truth}")
                if 'greedy' in prediction and 'beam_search' in prediction:
                    print(f"  CER (Greedy/Beam): {result.get('cer_greedy', 0):.3f} / {result.get('cer_beam_search', 0):.3f}")
                    print(f"  WER (Greedy/Beam): {result.get('wer_greedy', 0):.3f} / {result.get('wer_beam_search', 0):.3f}")
                elif 'greedy' in prediction:
                    print(f"  CER (Greedy): {result.get('cer_greedy', 0):.3f}")
                    print(f"  WER (Greedy): {result.get('wer_greedy', 0):.3f}")
                elif 'beam_search' in prediction:
                    print(f"  CER (Beam): {result.get('cer_beam_search', 0):.3f}")
                    print(f"  WER (Beam): {result.get('wer_beam_search', 0):.3f}")
                print(f"  Confidence: {prediction['confidence']:.3f}")
            else:
                if 'greedy' in prediction:
                    print(f"  Greedy: {prediction['greedy']}")
                if 'beam_search' in prediction:
                    print(f"  Beam Search: {prediction['beam_search']}")
                print(f"  Confidence: {prediction['confidence']:.3f}")
                print(f"  Warning: No ground truth found for {image_file.name}")

            results.append(result)

        except Exception as e:
            print(f"  Error processing {image_file.name}: {e}")
            results.append({
                'image_path': str(image_file),
                'image_name': image_file.name,
                'error': str(e)
            })

    # Calculate batch metrics using HTR_VT method
    if len(ground_truths) > 0:
        # Only calculate metrics if predictions exist for that mode
        batch_cer_greedy = batch_cer_beam = batch_wer_greedy = batch_wer_beam = None
        if predictions_greedy:
            batch_cer_greedy, _, batch_wer_greedy, _ = calculate_metrics_batch(
                predictions_greedy, predictions_beam if predictions_beam else predictions_greedy, ground_truths)
        if predictions_beam:
            _, batch_cer_beam, _, batch_wer_beam = calculate_metrics_batch(
                predictions_greedy if predictions_greedy else predictions_beam, predictions_beam, ground_truths)

        avg_metrics = {
            'batch_cer_greedy': batch_cer_greedy,
            'batch_cer_beam_search': batch_cer_beam,
            'batch_wer_greedy': batch_wer_greedy,
            'batch_wer_beam_search': batch_wer_beam,
            'samples_with_ground_truth': len(ground_truths),
            'total_samples': len(image_files),
            'calculation_method': 'HTR_VT_style_batch_calculation'
        }

        # Add summary to results
        final_results = {
            'summary': avg_metrics,
            'detailed_results': results
        }
    else:
        final_results = {
            'summary': {'note': 'No ground truth files found for metric calculation'},
            'detailed_results': results
        }

    # Save results
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(final_results, f, indent=2, ensure_ascii=False)

    print(f"\nResults saved to: {output_file}")

    # Print summary
    if len(ground_truths) > 0:
        print(f"\nBatch Metrics (HTR_VT style, based on {len(ground_truths)} samples):")
        if batch_cer_greedy is not None:
            print(f"CER - Greedy: {batch_cer_greedy:.3f}")
        if batch_cer_beam is not None:
            print(f"CER - Beam Search: {batch_cer_beam:.3f}")
        if batch_wer_greedy is not None:
            print(f"WER - Greedy: {batch_wer_greedy:.3f}")
        if batch_wer_beam is not None:
            print(f"WER - Beam Search: {batch_wer_beam:.3f}")

    return results


def main():
    parser = argparse.ArgumentParser(description='HTR Model Inference')
    parser.add_argument('--checkpoint', type=str,
                        required=True, help='Path to model checkpoint')
    parser.add_argument('--image', type=str,
                        help='Path to single image for inference')
    parser.add_argument('--image_dir', type=str,
                        help='Directory containing images for batch inference')
    parser.add_argument(
        '--output', type=str, default='inference_results.json', help='Output file for results')
    parser.add_argument('--lm_path', type=str,
                        help='Path to KenLM language model')
    parser.add_argument('--beam_width', type=int, default=100,
                        help='Beam width for beam search')
    parser.add_argument('--decode_mode', type=str, default='both', choices=['greedy', 'beam', 'both'],
                        help='Decoding mode: greedy, beam, or both (default: both)')

    args = parser.parse_args()

    if not args.image and not args.image_dir:
        print("Please provide either --image or --image_dir")
        return

    # Device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")

    # Load model and vocabulary
    print("Loading model...")
    model, vocab = load_model_and_vocab(args.checkpoint, device)
    print(f"Model loaded with vocabulary size: {len(vocab)}")

    # Create decoder
    decoder = CTCDecoder(vocab, lm_path=args.lm_path)
    if decoder.use_lm:
        print("Language model loaded successfully")
    else:
        print("Using decoder without language model")

    if args.image:
        # Single image inference
        print(f"\nProcessing single image: {args.image}")
        prediction = predict_single_image(model, decoder, args.image, device, decode_mode=args.decode_mode, beam_width=args.beam_width)

        # Try to load ground truth
        ground_truth = load_ground_truth(args.image)

        print("\nResults:")
        if 'greedy' in prediction:
            print(f"Greedy Decoding: {prediction['greedy']}")
        if 'beam_search' in prediction:
            print(f"Beam Search: {prediction['beam_search']}")
        print(f"Confidence: {prediction['confidence']:.3f}")

        if ground_truth is not None:
            if 'greedy' in prediction:
                cer_greedy = calculate_cer(prediction['greedy'], ground_truth)
                wer_greedy = calculate_wer(prediction['greedy'], ground_truth)
                print(f"CER (Greedy): {cer_greedy:.3f}")
                print(f"WER (Greedy): {wer_greedy:.3f}")
            if 'beam_search' in prediction:
                cer_beam = calculate_cer(prediction['beam_search'], ground_truth)
                wer_beam = calculate_wer(prediction['beam_search'], ground_truth)
                print(f"CER (Beam Search): {cer_beam:.3f}")
                print(f"WER (Beam Search): {wer_beam:.3f}")
            print(f"Ground Truth: {ground_truth}")

        # Save single result
        result = {
            'image_path': args.image,
            'confidence': prediction['confidence']
        }
        if 'greedy' in prediction:
            result['greedy_prediction'] = prediction['greedy']
        if 'beam_search' in prediction:
            result['beam_search_prediction'] = prediction['beam_search']
        if ground_truth is not None:
            result['ground_truth'] = ground_truth
            if 'greedy' in prediction:
                result['cer_greedy'] = cer_greedy
                result['wer_greedy'] = wer_greedy
            if 'beam_search' in prediction:
                result['cer_beam_search'] = cer_beam
                result['wer_beam_search'] = wer_beam

        with open(args.output, 'w', encoding='utf-8') as f:
            json.dump(result, f, indent=2, ensure_ascii=False)

    elif args.image_dir:
        # Batch inference
        print(f"\nProcessing images in directory: {args.image_dir}")
        results = batch_inference(
            model, decoder, args.image_dir, args.output, device, decode_mode=args.decode_mode, beam_width=args.beam_width)

        # Print summary
        successful = sum(1 for r in results if 'error' not in r)
        print(f"\nSummary:")
        print(f"Total images: {len(results)}")
        print(f"Successful: {successful}")
        print(f"Failed: {len(results) - successful}")


if __name__ == "__main__":
    main()
